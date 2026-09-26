//! Dummy (contracted) indices and the double coset search.
//!
//! A tensor is written as a map `g` from slots to labels. Slot symmetries `s`
//! act on the right and renaming of dummies `d` acts on the left, so every
//! equivalent way of writing the tensor is some `d . g . s` in the double
//! coset `D g S`. The canonical form is the lexicographically smallest element
//! of that double coset, and the tensor vanishes when one arrangement appears
//! with both signs.
//!
//! Labels number the indices: free indices first in canonical order, then
//! each dummy pair as covariant followed by contravariant, with pairs ordered
//! by index type and name. Contracted names are therefore interchangeable and
//! the canonical form renames them to the sorted dummy names in order of first
//! appearance.

use crate::canonicalization::{signed, SlotGroup};
use crate::error::{ButlerPortugalError, Result};
use crate::index::{IndexKind, TensorIndex};
use crate::permutation::{compose, identity, Permutation};
use crate::schreier_sims::schreier_sims;
use crate::tensor::Tensor;
use std::collections::hash_map::{Entry, HashMap};
use std::collections::BTreeMap;

/// The metric of an index type, which decides whether the two ends of a
/// contraction may trade places.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Metric {
    /// `A^a B_a = A_a B^a`, as for the spacetime metric.
    #[default]
    Symmetric,
    /// `A^a B_a = -A_a B^a`, as for the spinor metric.
    Antisymmetric,
    /// Indices cannot be raised or lowered.
    Absent,
}

/// One factor of the dummy group.
#[derive(Debug, Clone, PartialEq, Eq)]
enum DummySet {
    /// Contracted pairs `(covariant, contravariant)` of one index type, in label order.
    Pairs {
        pairs: Vec<(usize, usize)>,
        metric: Metric,
    },
    /// Labels of a name repeated with a single variance, freely interchangeable.
    Repeated(Vec<usize>),
}

/// The group of relabellings that leave a tensor unchanged: exchange of
/// dummy pairs, exchange of the ends of a pair when the metric allows it,
/// and permutation of repeated indices. It acts on labels, not slots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DummyGroup {
    degree: usize,
    sets: Vec<DummySet>,
}

impl DummyGroup {
    /// Builds the dummy group of `tensor`.
    pub fn new(tensor: &Tensor) -> Result<Self> {
        Ok(Labels::new(tensor)?.group)
    }

    /// Number of contracted pairs.
    pub fn pairs(&self) -> usize {
        self.sets
            .iter()
            .map(|set| match set {
                DummySet::Pairs { pairs, .. } => pairs.len(),
                DummySet::Repeated(_) => 0,
            })
            .sum()
    }

    /// Signed generators acting on labels.
    pub fn generators(&self) -> Vec<(Permutation, i32)> {
        let n = self.degree;
        let swap = |pairs: &[(usize, usize)]| {
            let mut p = identity(n);
            for &(a, b) in pairs {
                p.swap(a, b);
            }
            p
        };
        let mut gens = Vec::new();
        for set in &self.sets {
            match set {
                DummySet::Pairs { pairs, metric } => {
                    for w in pairs.windows(2) {
                        gens.push((swap(&[(w[0].0, w[1].0), (w[0].1, w[1].1)]), 1));
                    }
                    let sign = match metric {
                        Metric::Symmetric => 1,
                        Metric::Antisymmetric => -1,
                        Metric::Absent => continue,
                    };
                    gens.extend(pairs.iter().map(|&p| (swap(&[p]), sign)));
                }
                DummySet::Repeated(labels) => {
                    gens.extend(labels.windows(2).map(|w| (swap(&[(w[0], w[1])]), 1)));
                }
            }
        }
        gens
    }

    /// Number of relabellings in the group, counted by Schreier-Sims.
    pub fn order(&self) -> usize {
        let gens: Vec<Permutation> = self
            .generators()
            .iter()
            .map(|(p, s)| signed(p, *s))
            .collect();
        schreier_sims(&gens, self.degree + 2).order()
    }

    /// Smallest label in the orbit of `label`.
    fn orbit_min(&self, label: usize) -> usize {
        for set in &self.sets {
            match set {
                DummySet::Pairs { pairs, metric } => {
                    if let Some(&(lo, _)) =
                        pairs.iter().find(|&&(lo, up)| label == lo || label == up)
                    {
                        let first = pairs[0];
                        return if *metric != Metric::Absent || label == lo {
                            first.0
                        } else {
                            first.1
                        };
                    }
                }
                DummySet::Repeated(labels) if labels.contains(&label) => return labels[0],
                DummySet::Repeated(_) => {}
            }
        }
        label
    }

    /// A signed group element on `degree + 2` points taking `from` to `to`,
    /// which must lie in the orbit of `from`.
    fn transport(&self, from: usize, to: usize) -> Permutation {
        let n = self.degree;
        let mut t = identity(n + 2);
        for set in &self.sets {
            match set {
                DummySet::Pairs { pairs, metric } => {
                    let find = |x| pairs.iter().position(|&(lo, up)| x == lo || x == up);
                    let (Some(a), Some(b)) = (find(from), find(to)) else {
                        continue;
                    };
                    let ((la, ua), (lb, ub)) = (pairs[a], pairs[b]);
                    t.swap(la, lb);
                    t.swap(ua, ub);
                    if (from == la) != (to == lb) {
                        for v in t.iter_mut() {
                            if *v == lb {
                                *v = ub;
                            } else if *v == ub {
                                *v = lb;
                            }
                        }
                        if *metric == Metric::Antisymmetric {
                            t.swap(n, n + 1);
                        }
                    }
                    return t;
                }
                DummySet::Repeated(labels) if labels.contains(&from) => {
                    t.swap(from, to);
                    return t;
                }
                DummySet::Repeated(_) => {}
            }
        }
        t
    }

    /// Restricts to the stabilizer of `label`. Fixing one end of a pair fixes
    /// the whole pair, so the pair is dropped.
    fn stabilize(&mut self, label: usize) {
        for set in &mut self.sets {
            match set {
                DummySet::Pairs { pairs, .. } => {
                    pairs.retain(|&(lo, up)| label != lo && label != up)
                }
                DummySet::Repeated(labels) => labels.retain(|&l| l != label),
            }
        }
        self.sets.retain(|set| match set {
            DummySet::Pairs { pairs, .. } => !pairs.is_empty(),
            DummySet::Repeated(labels) => labels.len() > 1,
        });
    }
}

/// Classifies each slot as free or as one end of a numbered contraction.
/// Pairs are numbered in order of index type and name. A name that occurs
/// with both variances must occur exactly twice.
pub(crate) fn classify(indices: &[TensorIndex]) -> Result<Vec<IndexKind>> {
    let mut groups: BTreeMap<(&str, &str), Vec<usize>> = BTreeMap::new();
    for (slot, index) in indices.iter().enumerate() {
        groups
            .entry((index.index_type(), index.name()))
            .or_default()
            .push(slot);
    }
    let mut kinds = vec![IndexKind::Free; indices.len()];
    let mut id = 0;
    for ((_, name), slots) in groups {
        let up = slots
            .iter()
            .filter(|&&s| indices[s].is_contravariant())
            .count();
        if up == 0 || up == slots.len() {
            continue;
        }
        if slots.len() != 2 {
            return Err(ButlerPortugalError::InvalidTensor(format!(
                "contracted index {name} appears {} times",
                slots.len()
            )));
        }
        for s in slots {
            kinds[s] = IndexKind::Dummy(id);
        }
        id += 1;
    }
    Ok(kinds)
}

/// A tensor's indices numbered for the double coset search.
pub(crate) struct Labels {
    /// Label held by each slot.
    pub slots: Vec<usize>,
    /// Index written for each label, with dummies named by pair order.
    pub names: Vec<TensorIndex>,
    pub group: DummyGroup,
}

impl Labels {
    pub fn new(tensor: &Tensor) -> Result<Self> {
        let indices = tensor.indices();
        let n = indices.len();
        let kinds = classify(indices)?;
        let mut free: Vec<usize> = (0..n).filter(|&s| kinds[s] == IndexKind::Free).collect();
        free.sort_by(|&a, &b| indices[a].canonical_cmp(&indices[b]));
        let mut pairs: Vec<[Option<usize>; 2]> = Vec::new();
        for (slot, kind) in kinds.iter().enumerate() {
            if let IndexKind::Dummy(id) = *kind {
                if pairs.len() <= id {
                    pairs.resize(id + 1, [None, None]);
                }
                pairs[id][usize::from(indices[slot].is_contravariant())] = Some(slot);
            }
        }
        let pair_slots: Vec<usize> = pairs.iter().flatten().flatten().copied().collect();

        let mut slots = vec![0; n];
        let mut names = Vec::with_capacity(n);
        for (label, &slot) in free.iter().chain(&pair_slots).enumerate() {
            slots[slot] = label;
            names.push(indices[slot].clone());
        }

        let mut sets: Vec<DummySet> = free
            .chunk_by(|&a, &b| indices[a].label_cmp(&indices[b]).is_eq())
            .filter(|run| run.len() > 1)
            .map(|run| DummySet::Repeated(run.iter().map(|&s| slots[s]).collect()))
            .collect();
        let f = free.len();
        let mut by_type: BTreeMap<&str, Vec<(usize, usize)>> = BTreeMap::new();
        for (k, pair) in pairs.iter().enumerate() {
            if let Some(slot) = pair[0] {
                by_type
                    .entry(indices[slot].index_type())
                    .or_default()
                    .push((f + 2 * k, f + 2 * k + 1));
            }
        }
        for (index_type, pairs) in by_type {
            sets.push(DummySet::Pairs {
                pairs,
                metric: tensor.metric(index_type),
            });
        }
        Ok(Self {
            slots,
            names,
            group: DummyGroup { degree: n, sets },
        })
    }

    pub fn has_dummies(&self) -> bool {
        self.group.pairs() > 0
    }
}

/// Butler-Portugal double coset search. Returns the lexicographically
/// smallest element of `D g S` as a slot to label map with its sign, or
/// `None` if the tensor vanishes.
///
/// Slots are fixed in base order. For each partial solution `(s, d)`, every
/// transversal element `u` of the slot stabilizer offers the label
/// `d g s u (i)`, which the dummy stabilizer can lower to the minimum of its
/// orbit. The smallest such minimum is the canonical label for slot `i`, and
/// every way of reaching it survives to the next slot. Survivors that write
/// the tensor identically are merged, and if two of them differ only in sign
/// the tensor is zero.
pub(crate) fn double_coset_rep(
    slot_group: &SlotGroup,
    labels: &Labels,
) -> Option<(Vec<usize>, i32)> {
    let n = labels.slots.len();
    let g = signed(&labels.slots, 1);
    let mut dummies = labels.group.clone();
    let mut tab = vec![(identity(n + 2), identity(n + 2))];
    for (slot, level) in slot_group.bsgs().levels().iter().enumerate().take(n) {
        let mut best = usize::MAX;
        let mut candidates = Vec::new();
        for (s, d) in &tab {
            for u in level.transversal.values() {
                let s1 = compose(u, s);
                let label = d[g[s1[slot]]];
                let min = dummies.orbit_min(label);
                if min < best {
                    best = min;
                    candidates.clear();
                }
                if min == best {
                    candidates.push((s1, d, label));
                }
            }
        }
        let mut seen: HashMap<Vec<usize>, usize> = HashMap::new();
        let mut next = Vec::new();
        for (s1, d, label) in candidates {
            let d1 = compose(d, &dummies.transport(label, best));
            let h = compose(&compose(&s1, &g), &d1);
            match seen.entry(h[..n].to_vec()) {
                Entry::Occupied(e) if *e.get() != h[n] => return None,
                Entry::Occupied(_) => {}
                Entry::Vacant(e) => {
                    e.insert(h[n]);
                    next.push((s1, d1));
                }
            }
        }
        dummies.stabilize(best);
        tab = next;
    }
    let (s, d) = tab.first()?;
    let h = compose(&compose(s, &g), d);
    Some((h[..n].to_vec(), if h[n] == n { 1 } else { -1 }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(spec: &[(&str, bool)]) -> Tensor {
        Tensor::new(
            "T",
            spec.iter()
                .enumerate()
                .map(|(i, &(n, up))| {
                    if up {
                        TensorIndex::contravariant(n, i)
                    } else {
                        TensorIndex::covariant(n, i)
                    }
                })
                .collect(),
        )
    }

    #[test]
    fn classification() {
        let t = tensor(&[
            ("b", true),
            ("x", false),
            ("a", false),
            ("b", false),
            ("a", true),
        ]);
        assert_eq!(
            t.index_kinds().unwrap(),
            [
                IndexKind::Dummy(1),
                IndexKind::Free,
                IndexKind::Dummy(0),
                IndexKind::Dummy(1),
                IndexKind::Dummy(0)
            ]
        );
        let t = tensor(&[("a", false), ("a", false)]);
        assert_eq!(t.index_kinds().unwrap(), [IndexKind::Free; 2]);
        let t = tensor(&[("a", false), ("a", true), ("a", false)]);
        assert!(t.index_kinds().is_err());
        let spinor = Tensor::new(
            "T",
            vec![
                TensorIndex::covariant("a", 0),
                TensorIndex::contravariant("a", 1).of_type("spinor"),
            ],
        );
        assert_eq!(spinor.index_kinds().unwrap(), [IndexKind::Free; 2]);
    }

    #[test]
    fn dummy_group_orders() {
        let three = tensor(&[
            ("a", false),
            ("b", false),
            ("c", false),
            ("a", true),
            ("b", true),
            ("c", true),
        ]);
        assert_eq!(DummyGroup::new(&three).unwrap().order(), 48);
        let mut t = three.clone();
        t.set_metric("", Metric::Antisymmetric);
        assert_eq!(DummyGroup::new(&t).unwrap().order(), 48);
        t.set_metric("", Metric::Absent);
        assert_eq!(DummyGroup::new(&t).unwrap().order(), 6);
        let repeated = tensor(&[("a", false), ("a", false), ("a", false)]);
        let group = DummyGroup::new(&repeated).unwrap();
        assert_eq!((group.order(), group.pairs()), (6, 0));
    }

    #[test]
    fn transport_hits_target_with_metric_sign() {
        let mut t = tensor(&[("a", false), ("a", true), ("b", false), ("b", true)]);
        t.set_metric("", Metric::Antisymmetric);
        let group = DummyGroup::new(&t).unwrap();
        for from in 0..4 {
            let to = group.orbit_min(from);
            let p = group.transport(from, to);
            assert_eq!(p[from], to);
            let flips = from % 2 != to % 2;
            assert_eq!(p[4] == 5, flips);
        }
    }
}
