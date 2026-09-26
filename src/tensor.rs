//! Tensors with named indices, slot symmetries, and an integer coefficient.

use crate::canonicalization::{canonicalize, canonicalize_in, SlotGroup};
use crate::dummy::{classify, Metric};
use crate::error::{validate_permutation, ButlerPortugalError, Result};
use crate::index::{IndexKind, TensorIndex};
use crate::permutation::identity;
use crate::symmetry::Symmetry;
use crate::young_tableaux::{young_symmetrizer_permutations, StandardTableau};
use std::collections::btree_map::{BTreeMap, Entry};
use std::fmt;

/// A named tensor with ordered indices and slot symmetries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tensor {
    name: String,
    indices: Vec<TensorIndex>,
    symmetries: Vec<Symmetry>,
    metrics: BTreeMap<String, Metric>,
    factors: Vec<(String, usize)>,
    coefficient: i32,
}

impl Tensor {
    /// Creates a tensor with coefficient `1`.
    ///
    /// ```rust
    /// use butler_portugal::{Tensor, TensorIndex};
    ///
    /// let tensor = Tensor::new(
    ///     "g",
    ///     vec![TensorIndex::new("mu", 0), TensorIndex::new("nu", 1)],
    /// );
    /// ```
    pub fn new(name: &str, indices: Vec<TensorIndex>) -> Self {
        Self::with_coefficient(name, indices, 1)
    }

    /// Creates a tensor with the given coefficient.
    pub fn with_coefficient(name: &str, indices: Vec<TensorIndex>, coefficient: i32) -> Self {
        Self {
            name: name.to_string(),
            indices,
            symmetries: Vec::new(),
            metrics: BTreeMap::new(),
            factors: Vec::new(),
            coefficient,
        }
    }

    /// The product of `factors` as one tensor. Indices are concatenated in
    /// factor order, each factor keeps its symmetries on its own slots, and
    /// identical factors (same name, rank and symmetries) commute.
    /// Coefficients multiply and metrics merge. Conflicting metrics for one
    /// index type are an error.
    ///
    /// ```rust
    /// use butler_portugal::{canonicalize, Symmetry, Tensor, TensorIndex};
    ///
    /// let f = |a: TensorIndex, b: TensorIndex| {
    ///     let mut t = Tensor::new("F", vec![a, b]);
    ///     t.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));
    ///     t
    /// };
    /// let ff = Tensor::product(&[
    ///     f(TensorIndex::covariant("b", 0), TensorIndex::covariant("a", 1)),
    ///     f(TensorIndex::contravariant("a", 0), TensorIndex::contravariant("b", 1)),
    /// ])?;
    /// assert_eq!(ff.to_string(), "F_b_a F^a^b");
    /// assert_eq!(canonicalize(&ff)?.to_string(), "-F_a_b F^a^b");
    /// # Ok::<(), butler_portugal::ButlerPortugalError>(())
    /// ```
    pub fn product(factors: &[Tensor]) -> Result<Self> {
        let rank: usize = factors.iter().map(Tensor::rank).sum();
        let mut product = Self::new("", Vec::with_capacity(rank));
        let mut offsets = Vec::with_capacity(factors.len());
        for (i, factor) in factors.iter().enumerate() {
            let offset = product.indices.len();
            offsets.push(offset);
            let embed = |p: &[usize]| {
                let mut q = identity(rank);
                for (slot, &image) in p.iter().enumerate() {
                    q[offset + slot] = offset + image;
                }
                q
            };
            let (perms, signs): (Vec<_>, Vec<_>) = factor
                .symmetries
                .iter()
                .map(|s| s.generators(factor.rank()))
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flatten()
                .map(|(p, sign)| (embed(&p), sign))
                .unzip();
            if !perms.is_empty() {
                product.add_symmetry(Symmetry::custom(perms, signs));
            }
            if let Some(j) = factors[..i].iter().position(|g| g.commutes_with(factor)) {
                let mut swap = identity(rank);
                for slot in 0..factor.rank() {
                    swap.swap(offsets[j] + slot, offset + slot);
                }
                product.add_symmetry(Symmetry::custom(vec![swap], vec![1]));
            }
            for (index_type, &metric) in &factor.metrics {
                match product.metrics.entry(index_type.clone()) {
                    Entry::Vacant(e) => {
                        e.insert(metric);
                    }
                    Entry::Occupied(e) if *e.get() != metric => {
                        return Err(ButlerPortugalError::IncompatibleTensors(format!(
                            "conflicting metrics for index type {index_type:?}"
                        )));
                    }
                    Entry::Occupied(_) => {}
                }
            }
            product.indices.extend(
                factor
                    .indices
                    .iter()
                    .enumerate()
                    .map(|(k, index)| index.with_position(offset + k)),
            );
            product.factors.extend(factor.segments());
            product.coefficient *= factor.coefficient;
        }
        product.name = product
            .factors
            .iter()
            .map(|(n, _)| n.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        Ok(product)
    }

    fn commutes_with(&self, other: &Tensor) -> bool {
        self.name == other.name
            && self.rank() == other.rank()
            && self.symmetries == other.symmetries
            && self.factors == other.factors
    }

    fn segments(&self) -> Vec<(String, usize)> {
        if self.factors.is_empty() {
            vec![(self.name.clone(), self.rank())]
        } else {
            self.factors.clone()
        }
    }

    /// The tensor's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The indices in slot order.
    pub fn indices(&self) -> &[TensorIndex] {
        &self.indices
    }

    /// Mutable access to the indices.
    pub fn indices_mut(&mut self) -> &mut Vec<TensorIndex> {
        &mut self.indices
    }

    /// The declared slot symmetries.
    pub fn symmetries(&self) -> &[Symmetry] {
        &self.symmetries
    }

    /// The coefficient.
    pub fn coefficient(&self) -> i32 {
        self.coefficient
    }

    /// Sets the coefficient.
    pub fn set_coefficient(&mut self, coefficient: i32) {
        self.coefficient = coefficient;
    }

    /// Declares a slot symmetry.
    pub fn add_symmetry(&mut self, symmetry: Symmetry) {
        self.symmetries.push(symmetry);
    }

    /// Removes every declared symmetry.
    pub fn clear_symmetries(&mut self) {
        self.symmetries.clear();
    }

    /// Sets the metric of an index type, which decides whether the ends of
    /// its contractions can swap. The default index type is `""` and every
    /// type defaults to [`Metric::Symmetric`].
    pub fn set_metric(&mut self, index_type: &str, metric: Metric) {
        self.metrics.insert(index_type.to_string(), metric);
    }

    /// The metric of an index type.
    pub fn metric(&self, index_type: &str) -> Metric {
        self.metrics.get(index_type).copied().unwrap_or_default()
    }

    /// Classifies each slot as free or as one end of a contraction. A name
    /// occurring once covariant and once contravariant within an index type
    /// is a contraction, numbered in order of type and name. A contracted
    /// name occurring more than twice is an error.
    pub fn index_kinds(&self) -> Result<Vec<IndexKind>> {
        classify(&self.indices)
    }

    /// Number of indices.
    pub fn rank(&self) -> usize {
        self.indices.len()
    }

    /// The signed slot symmetry group generated by the declared symmetries.
    pub fn slot_group(&self) -> Result<SlotGroup> {
        SlotGroup::new(self)
    }

    /// Rearranges the slots so that new slot `i` holds old slot `perm[i]`, without
    /// touching the coefficient. `perm` must be a valid permutation of the slots.
    pub(crate) fn reorder(&self, perm: &[usize]) -> Self {
        let indices = perm
            .iter()
            .enumerate()
            .map(|(i, &p)| self.indices[p].clone().with_position(i))
            .collect();
        Self {
            indices,
            ..self.clone()
        }
    }

    /// The factor the coefficient picks up under `perm`: the symmetry sign if
    /// `perm` is a symmetry, `0` if the symmetries force the tensor to vanish,
    /// and `1` otherwise.
    fn permutation_sign(&self, perm: &[usize]) -> Result<i32> {
        let group = self.slot_group()?;
        Ok(if group.is_zero() {
            0
        } else {
            group.sign(perm).unwrap_or(1)
        })
    }

    /// Returns the tensor with new slot `i` holding old slot `perm[i]`.
    ///
    /// When `perm` is one of the tensor's symmetries the coefficient is
    /// multiplied by the symmetry's sign, so the result denotes the same
    /// quantity. Otherwise the coefficient is unchanged. A tensor that
    /// vanishes by symmetry comes back with coefficient `0`.
    pub fn permute(&self, perm: &[usize]) -> Result<Self> {
        validate_permutation(perm, self.rank())?;
        let sign = self.permutation_sign(perm)?;
        let mut t = self.reorder(perm);
        t.coefficient *= sign;
        Ok(t)
    }

    /// Swaps two slots in place and returns the factor applied to the
    /// coefficient, as for [`Tensor::permute`]. Out-of-range or equal slots
    /// leave the tensor untouched and return `1`.
    pub fn swap_indices(&mut self, i: usize, j: usize) -> i32 {
        if i >= self.rank() || j >= self.rank() || i == j {
            return 1;
        }
        let mut perm = identity(self.rank());
        perm.swap(i, j);
        let sign = self.permutation_sign(&perm).unwrap_or(1);
        *self = self.reorder(&perm);
        self.coefficient *= sign;
        sign
    }

    /// True if the coefficient is zero or the symmetries force the tensor to vanish.
    pub fn is_zero(&self) -> bool {
        self.coefficient == 0 || canonicalize(self).is_ok_and(|t| t.coefficient == 0)
    }

    /// Applies the Young symmetrizer of `tableau` to the slots: symmetrizes
    /// over each row, then antisymmetrizes over each column. Each resulting
    /// term is canonicalized under the tensor's own symmetries and like terms
    /// are merged, so the result is a sum of distinct nonzero tensors with
    /// unnormalized integer coefficients. An empty result means the projection
    /// vanishes.
    ///
    /// ```rust
    /// use butler_portugal::young_tableaux::{Shape, StandardTableau};
    /// use butler_portugal::{Tensor, TensorIndex};
    ///
    /// let tensor = Tensor::new(
    ///     "T",
    ///     vec![TensorIndex::new("a", 0), TensorIndex::new("b", 1)],
    /// );
    /// let column = StandardTableau::new(Shape(vec![1, 1]), vec![vec![1], vec![2]]).unwrap();
    /// let terms = tensor.project_with_tableau(&column)?;
    /// let shown: Vec<String> = terms.iter().map(|t| t.to_string()).collect();
    /// assert_eq!(shown, ["T_a_b", "-T_b_a"]);
    /// # Ok::<(), butler_portugal::ButlerPortugalError>(())
    /// ```
    pub fn project_with_tableau(&self, tableau: &StandardTableau) -> Result<Vec<Tensor>> {
        let group = self.slot_group()?;
        let perms = young_symmetrizer_permutations(tableau, self.rank())?;
        let term = |(perm, sign): &(Vec<usize>, i32)| {
            let mut t = self.reorder(perm);
            t.coefficient *= sign;
            canonicalize_in(&t, &group)
        };
        #[cfg(feature = "parallel")]
        let canonical: Vec<Tensor> = {
            use rayon::prelude::*;
            perms.par_iter().map(term).collect::<Result<_>>()?
        };
        #[cfg(not(feature = "parallel"))]
        let canonical: Vec<Tensor> = perms.iter().map(term).collect::<Result<_>>()?;
        let mut terms: BTreeMap<Vec<(String, bool)>, Tensor> = BTreeMap::new();
        for term in canonical.into_iter().filter(|t| t.coefficient != 0) {
            let key = term
                .indices
                .iter()
                .map(|i| (i.name().to_string(), i.is_contravariant()))
                .collect();
            match terms.entry(key) {
                Entry::Occupied(mut e) => e.get_mut().coefficient += term.coefficient,
                Entry::Vacant(e) => {
                    e.insert(term);
                }
            }
        }
        Ok(terms.into_values().filter(|t| t.coefficient != 0).collect())
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.coefficient == 0 {
            return write!(f, "0");
        }
        if self.coefficient < 0 {
            write!(f, "-")?;
        }
        if self.coefficient.abs() != 1 {
            write!(f, "{}", self.coefficient.abs())?;
        }
        let mut indices = self.indices.iter();
        for (k, (name, rank)) in self.segments().iter().enumerate() {
            if k > 0 {
                write!(f, " ")?;
            }
            write!(f, "{name}")?;
            for index in indices.by_ref().take(*rank) {
                write!(f, "{index}")?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::young_tableaux::Shape;

    fn tensor(name: &str, names: &[&str]) -> Tensor {
        Tensor::new(
            name,
            names
                .iter()
                .enumerate()
                .map(|(i, n)| TensorIndex::new(n, i))
                .collect(),
        )
    }

    #[test]
    fn display() {
        assert_eq!(tensor("g", &["mu", "nu"]).to_string(), "g_mu_nu");
        let mut t = Tensor::with_coefficient(
            "F",
            vec![
                TensorIndex::contravariant("mu", 0),
                TensorIndex::covariant("nu", 1),
            ],
            -3,
        );
        assert_eq!(t.to_string(), "-3F^mu_nu");
        t.set_coefficient(0);
        assert_eq!(t.to_string(), "0");
    }

    #[test]
    fn swap_reports_symmetry_sign() {
        let mut a = tensor("A", &["i", "j"]);
        a.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));
        assert_eq!(a.swap_indices(0, 1), -1);
        assert_eq!(a.to_string(), "-A_j_i");
        assert_eq!(a.indices()[0].position(), 0);

        let mut c = tensor("C", &["a", "b", "c"]);
        c.add_symmetry(Symmetry::cyclic(vec![0, 1, 2]));
        assert_eq!(c.swap_indices(0, 1), 1);

        let mut z = tensor("Z", &["a", "b", "c"]);
        z.add_symmetry(Symmetry::symmetric(vec![0, 1]));
        z.add_symmetry(Symmetry::antisymmetric(vec![1, 2]));
        assert_eq!(z.swap_indices(0, 1), 0);
        assert!(z.is_zero());
    }

    #[test]
    fn permute_validates_and_signs() {
        let t = tensor("T", &["a"]);
        assert!(t.permute(&[0, 1]).is_err());
        assert!(t.permute(&[1]).is_err());

        let mut a = tensor("A", &["a", "b", "c"]);
        a.add_symmetry(Symmetry::antisymmetric(vec![0, 1, 2]));
        assert_eq!(a.permute(&[1, 2, 0]).unwrap().to_string(), "A_b_c_a");
        assert_eq!(a.permute(&[1, 0, 2]).unwrap().to_string(), "-A_b_a_c");
        let plain = tensor("T", &["a", "b"]).permute(&[1, 0]).unwrap();
        assert_eq!(plain.to_string(), "T_b_a");
    }

    #[test]
    fn projection_terms() {
        let row = StandardTableau::new(Shape(vec![2]), vec![vec![1, 2]]).unwrap();
        let column = StandardTableau::new(Shape(vec![1, 1]), vec![vec![1], vec![2]]).unwrap();

        let t = tensor("T", &["a", "b"]);
        let shown = |terms: Vec<Tensor>| terms.iter().map(|t| t.to_string()).collect::<Vec<_>>();
        assert_eq!(
            shown(t.project_with_tableau(&row).unwrap()),
            ["T_a_b", "T_b_a"]
        );

        let mut a = tensor("A", &["a", "b"]);
        a.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));
        assert!(a.project_with_tableau(&row).unwrap().is_empty());
        assert_eq!(shown(a.project_with_tableau(&column).unwrap()), ["2A_a_b"]);

        let hook = StandardTableau::new(Shape(vec![2, 1]), vec![vec![1, 2], vec![3]]).unwrap();
        assert_eq!(
            shown(
                tensor("T", &["a", "b", "c"])
                    .project_with_tableau(&hook)
                    .unwrap()
            ),
            ["T_a_b_c", "T_b_a_c", "-T_b_c_a", "-T_c_b_a"]
        );

        assert!(t.project_with_tableau(&hook).is_err());
    }
}
