//! Canonicalization of a tensor under its slot symmetries.
//!
//! The slot symmetries generate a group of signed permutations. Signs are
//! encoded by acting on two extra points `n` and `n + 1`, which a sign of
//! `-1` swaps, so an ordinary Schreier-Sims chain over `n + 2` points
//! describes the signed group. The tensor is identically zero exactly when
//! that group contains the element that swaps only the two sign points.
//!
//! Because the base is the slot order `0, 1, ..., n - 1`, the lexicographically
//! minimal index arrangement can be found by walking the stabilizer chain one
//! slot at a time, keeping every partial product that achieves the minimum so
//! far. Only repeated index names cause branching.
//!
//! Dummy (contracted) index renaming is not modelled. Only slot symmetries are
//! applied.

use crate::error::{ButlerPortugalError, Result};
use crate::index::TensorIndex;
use crate::permutation::{compose, identity, is_identity, Permutation};
use crate::schreier_sims::{schreier_sims, BSGS};
use crate::tensor::Tensor;

/// The signed slot symmetry group of a tensor.
#[derive(Debug, Clone)]
pub struct SlotGroup {
    rank: usize,
    bsgs: BSGS,
}

impl SlotGroup {
    /// Builds the slot symmetry group of `tensor`.
    pub fn new(tensor: &Tensor) -> Result<Self> {
        let rank = tensor.rank();
        let mut generators = Vec::new();
        for symmetry in tensor.symmetries() {
            for (perm, sign) in symmetry.generators(rank)? {
                generators.push(signed(&perm, sign));
            }
        }
        Ok(Self {
            rank,
            bsgs: schreier_sims(&generators, rank + 2),
        })
    }

    /// The underlying stabilizer chain on `rank + 2` points.
    pub fn bsgs(&self) -> &BSGS {
        &self.bsgs
    }

    /// True if the symmetries force the tensor to vanish, because some slot
    /// permutation is a symmetry with both signs.
    pub fn is_zero(&self) -> bool {
        self.bsgs.contains(&signed(&identity(self.rank), -1))
    }

    /// Number of distinct slot permutations in the group.
    pub fn order(&self) -> usize {
        if self.is_zero() {
            self.bsgs.order() / 2
        } else {
            self.bsgs.order()
        }
    }

    /// The sign the tensor picks up under `perm`, or `None` if `perm` is not a symmetry.
    pub fn sign(&self, perm: &[usize]) -> Option<i32> {
        if perm.len() != self.rank {
            return None;
        }
        let residual = self.bsgs.sift(&signed(perm, 1));
        if is_identity(&residual) {
            Some(1)
        } else if residual == signed(&identity(self.rank), -1) {
            Some(-1)
        } else {
            None
        }
    }

    /// Canonicalizes `tensor` with this group instead of rebuilding it, which
    /// is most of the cost of [`canonicalize`]. The tensor's own symmetries are
    /// ignored; only its rank is checked.
    pub fn canonicalize(&self, tensor: &Tensor) -> Result<Tensor> {
        if tensor.rank() != self.rank {
            return Err(ButlerPortugalError::IncompatibleTensors(format!(
                "group has rank {} but tensor has rank {}",
                self.rank,
                tensor.rank()
            )));
        }
        Ok(canonicalize_in(tensor, self))
    }

    /// Every signed slot permutation in the group.
    pub fn elements(&self) -> Vec<(Permutation, i32)> {
        self.bsgs.elements().iter().map(|e| unsigned(e)).collect()
    }

    /// The group element producing the lexicographically minimal arrangement of
    /// `indices`, with its sign. Returns `None` if the arrangement is reached
    /// with both signs, which means the tensor vanishes.
    pub fn minimal(&self, indices: &[TensorIndex]) -> Option<(Permutation, i32)> {
        let key = |slot: usize| (indices[slot].name(), indices[slot].is_contravariant());
        let mut partial = vec![identity(self.rank + 2)];
        for (slot, level) in self.bsgs.levels().iter().enumerate().take(self.rank) {
            let mut best = None;
            let mut next = Vec::new();
            for p in &partial {
                for u in level.transversal.values() {
                    let q = compose(u, p);
                    let k = key(q[slot]);
                    match best {
                        Some(b) if k > b => {}
                        Some(b) if k == b => next.push(q),
                        _ => {
                            best = Some(k);
                            next = vec![q];
                        }
                    }
                }
            }
            partial = next;
        }
        let (perm, sign) = unsigned(&partial[0]);
        partial
            .iter()
            .all(|q| unsigned(q).1 == sign)
            .then_some((perm, sign))
    }
}

/// Encodes a signed permutation of `n` points as a permutation of `n + 2` points.
fn signed(perm: &[usize], sign: i32) -> Permutation {
    let n = perm.len();
    let mut p = perm.to_vec();
    if sign == 1 {
        p.extend([n, n + 1]);
    } else {
        p.extend([n + 1, n]);
    }
    p
}

/// Decodes a permutation of `n + 2` points into a signed permutation of `n` points.
fn unsigned(perm: &[usize]) -> (Permutation, i32) {
    let n = perm.len() - 2;
    let sign = if perm[n] == n { 1 } else { -1 };
    (perm[..n].to_vec(), sign)
}

/// Brings a tensor into canonical form.
///
/// The result has the lexicographically smallest index arrangement reachable
/// by the tensor's slot symmetries, ordered by index name and then variance
/// with covariant first, and its coefficient carries the accumulated sign.
/// A tensor that vanishes by symmetry comes back with coefficient `0`.
///
/// ```rust
/// use butler_portugal::{canonicalize, Symmetry, Tensor, TensorIndex};
///
/// let mut tensor = Tensor::new(
///     "R",
///     vec![
///         TensorIndex::new("d", 0),
///         TensorIndex::new("c", 1),
///         TensorIndex::new("b", 2),
///         TensorIndex::new("a", 3),
///     ],
/// );
/// tensor.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));
/// tensor.add_symmetry(Symmetry::antisymmetric(vec![2, 3]));
/// tensor.add_symmetry(Symmetry::symmetric_pairs(vec![(0, 1), (2, 3)]));
///
/// let canonical = canonicalize(&tensor)?;
/// assert_eq!(canonical.to_string(), "R_a_b_c_d");
/// # Ok::<(), butler_portugal::ButlerPortugalError>(())
/// ```
pub fn canonicalize(tensor: &Tensor) -> Result<Tensor> {
    let group = SlotGroup::new(tensor)?;
    Ok(canonicalize_in(tensor, &group))
}

/// Canonicalizes `tensor` using an already computed slot group.
pub(crate) fn canonicalize_in(tensor: &Tensor, group: &SlotGroup) -> Tensor {
    let zero = || {
        let mut t = tensor.clone();
        t.set_coefficient(0);
        t
    };
    if tensor.coefficient() == 0 || group.is_zero() {
        return zero();
    }
    match group.minimal(tensor.indices()) {
        Some((perm, sign)) => {
            let mut t = tensor.reorder(&perm);
            t.set_coefficient(t.coefficient() * sign);
            t
        }
        None => zero(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symmetry::Symmetry;

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
    fn trivial() {
        let t = tensor("T", &["i"]);
        assert_eq!(canonicalize(&t).unwrap(), t);
        let t = Tensor::new("s", vec![]);
        assert_eq!(canonicalize(&t).unwrap(), t);
    }

    #[test]
    fn symmetric() {
        let mut t = tensor("S", &["b", "a"]);
        t.add_symmetry(Symmetry::symmetric(vec![0, 1]));
        assert_eq!(canonicalize(&t).unwrap().to_string(), "S_a_b");
    }

    #[test]
    fn antisymmetric() {
        let mut t = tensor("A", &["b", "a"]);
        t.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));
        assert_eq!(canonicalize(&t).unwrap().to_string(), "-A_a_b");
    }

    #[test]
    fn antisymmetric_repeated_index_vanishes() {
        let mut t = tensor("A", &["a", "a"]);
        t.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));
        assert_eq!(canonicalize(&t).unwrap().coefficient(), 0);
    }

    #[test]
    fn inconsistent_symmetries_vanish() {
        let mut t = tensor("T", &["a", "b", "c"]);
        t.add_symmetry(Symmetry::symmetric(vec![0, 1]));
        t.add_symmetry(Symmetry::antisymmetric(vec![1, 2]));
        assert!(SlotGroup::new(&t).unwrap().is_zero());
        assert_eq!(canonicalize(&t).unwrap().coefficient(), 0);
    }

    #[test]
    fn group_order_and_signs() {
        let mut t = tensor("R", &["a", "b", "c", "d"]);
        t.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));
        t.add_symmetry(Symmetry::antisymmetric(vec![2, 3]));
        t.add_symmetry(Symmetry::symmetric_pairs(vec![(0, 1), (2, 3)]));
        let g = SlotGroup::new(&t).unwrap();
        assert_eq!(g.order(), 8);
        assert_eq!(g.sign(&[1, 0, 2, 3]), Some(-1));
        assert_eq!(g.sign(&[2, 3, 0, 1]), Some(1));
        assert_eq!(g.sign(&[3, 2, 1, 0]), Some(1));
        assert_eq!(g.sign(&[3, 2, 0, 1]), Some(-1));
        assert_eq!(g.sign(&[0, 2, 1, 3]), None);
        assert_eq!(g.elements().len(), 8);
    }

    #[test]
    fn cyclic_is_idempotent() {
        let mut t = tensor("T", &["a", "b", "c"]);
        t.add_symmetry(Symmetry::cyclic(vec![0, 1, 2]));
        let once = canonicalize(&t).unwrap();
        assert_eq!(once.to_string(), "T_a_b_c");
        assert_eq!(canonicalize(&once).unwrap(), once);
        let mut t = tensor("T", &["c", "a", "b"]);
        t.add_symmetry(Symmetry::cyclic(vec![0, 1, 2]));
        assert_eq!(canonicalize(&t).unwrap().to_string(), "T_a_b_c");
        let mut t = tensor("T", &["b", "a", "c"]);
        t.add_symmetry(Symmetry::cyclic(vec![0, 1, 2]));
        assert_eq!(canonicalize(&t).unwrap().to_string(), "T_a_c_b");
    }

    #[test]
    fn name_order_matches_index_ordering() {
        let mut t = tensor("T", &["a1", "a"]);
        t.add_symmetry(Symmetry::symmetric(vec![0, 1]));
        assert_eq!(canonicalize(&t).unwrap().to_string(), "T_a_a1");
    }

    #[test]
    fn covariant_sorts_before_contravariant() {
        let mut t = Tensor::new(
            "S",
            vec![
                TensorIndex::contravariant("a", 0),
                TensorIndex::covariant("a", 1),
            ],
        );
        t.add_symmetry(Symmetry::symmetric(vec![0, 1]));
        assert_eq!(canonicalize(&t).unwrap().to_string(), "S_a^a");
    }

    #[test]
    fn reused_group_matches_fresh_canonicalization() {
        let mut template = tensor("R", &["a", "b", "c", "d"]);
        template.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));
        template.add_symmetry(Symmetry::antisymmetric(vec![2, 3]));
        template.add_symmetry(Symmetry::symmetric_pairs(vec![(0, 1), (2, 3)]));
        let group = template.slot_group().unwrap();
        for names in [
            ["d", "c", "b", "a"],
            ["b", "a", "c", "d"],
            ["a", "b", "a", "b"],
        ] {
            let mut t = tensor("R", &names);
            t.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));
            t.add_symmetry(Symmetry::antisymmetric(vec![2, 3]));
            t.add_symmetry(Symmetry::symmetric_pairs(vec![(0, 1), (2, 3)]));
            assert_eq!(group.canonicalize(&t).unwrap(), canonicalize(&t).unwrap());
        }
        assert!(group.canonicalize(&tensor("T", &["a"])).is_err());
    }

    #[test]
    fn invalid_symmetry_is_an_error() {
        let mut t = tensor("T", &["a", "b"]);
        t.add_symmetry(Symmetry::symmetric(vec![0, 7]));
        assert!(canonicalize(&t).is_err());
    }
}
