//! Regression tests for canonicalization behaviour that used to be wrong.

use butler_portugal::young_tableaux::{Shape, StandardTableau};
use butler_portugal::{canonicalize, SlotGroup, Symmetry, Tensor, TensorIndex};

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
fn no_symmetry_leaves_order_alone() {
    let t = tensor("T", &["b", "a"]);
    assert_eq!(canonicalize(&t).unwrap(), t);
}

#[test]
fn pair_exchange_alone_does_not_sort_within_pairs() {
    let mut t = tensor("T", &["b", "a", "c", "d"]);
    t.add_symmetry(Symmetry::symmetric_pairs(vec![(0, 1), (2, 3)]));
    assert_eq!(canonicalize(&t).unwrap().to_string(), "T_b_a_c_d");

    let mut t = tensor("T", &["c", "d", "a", "b"]);
    t.add_symmetry(Symmetry::symmetric_pairs(vec![(0, 1), (2, 3)]));
    assert_eq!(canonicalize(&t).unwrap().to_string(), "T_a_b_c_d");
}

#[test]
fn antisymmetric_with_unsorted_slot_list() {
    let mut t = tensor("A", &["b", "c", "a"]);
    t.add_symmetry(Symmetry::antisymmetric(vec![2, 0]));
    assert_eq!(canonicalize(&t).unwrap().to_string(), "-A_a_c_b");
}

#[test]
fn cyclic_symmetry_is_idempotent_and_signed_correctly() {
    let mut t = tensor("T", &["a", "b", "c"]);
    t.add_symmetry(Symmetry::cyclic(vec![0, 1, 2]));
    let once = canonicalize(&t).unwrap();
    assert_eq!(once, t);
    assert_eq!(canonicalize(&once).unwrap(), once);

    let group = SlotGroup::new(&t).unwrap();
    assert_eq!(group.order(), 3);
    assert_eq!(group.sign(&[0, 1, 2]), Some(1));
    assert_eq!(group.sign(&[2, 0, 1]), Some(1));
    assert_eq!(group.sign(&[1, 0, 2]), None);
}

#[test]
fn custom_symmetry_composes_generators() {
    let mut t = tensor("T", &["c", "b", "a", "d"]);
    t.add_symmetry(Symmetry::custom(
        vec![vec![1, 0, 2, 3], vec![0, 2, 1, 3]],
        vec![-1, -1],
    ));
    let group = SlotGroup::new(&t).unwrap();
    assert_eq!(group.order(), 6);
    assert_eq!(group.sign(&[0, 1, 2, 3]), Some(1));
    assert_eq!(group.sign(&[2, 0, 1, 3]), Some(1));
    assert_eq!(group.sign(&[2, 1, 0, 3]), Some(-1));
    assert_eq!(canonicalize(&t).unwrap().to_string(), "-T_a_b_c_d");
}

#[test]
fn inconsistent_symmetries_give_zero() {
    let mut t = tensor("T", &["a", "b", "c"]);
    t.add_symmetry(Symmetry::symmetric(vec![0, 1]));
    t.add_symmetry(Symmetry::antisymmetric(vec![1, 2]));
    assert!(t.is_zero());
    assert_eq!(canonicalize(&t).unwrap().coefficient(), 0);

    let mut t = tensor("T", &["a", "b", "c"]);
    t.add_symmetry(Symmetry::symmetric(vec![0, 1, 2]));
    t.add_symmetry(Symmetry::antisymmetric(vec![1, 2]));
    assert!(t.is_zero());

    // Cyclic invariance on three slots of a Riemann tensor is consistent
    let mut t = tensor("R", &["a", "b", "c", "d"]);
    t.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));
    t.add_symmetry(Symmetry::antisymmetric(vec![2, 3]));
    t.add_symmetry(Symmetry::symmetric_pairs(vec![(0, 1), (2, 3)]));
    t.add_symmetry(Symmetry::cyclic(vec![1, 2, 3]));
    assert!(!t.is_zero());
    assert_eq!(SlotGroup::new(&t).unwrap().order(), 24);
}

#[test]
fn repeated_names_with_signs() {
    let mut t = tensor("R", &["a", "b", "a", "b"]);
    t.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));
    t.add_symmetry(Symmetry::antisymmetric(vec![2, 3]));
    t.add_symmetry(Symmetry::symmetric_pairs(vec![(0, 1), (2, 3)]));
    assert_eq!(canonicalize(&t).unwrap().to_string(), "R_a_b_a_b");

    let mut t = tensor("R", &["a", "b", "b", "a"]);
    t.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));
    t.add_symmetry(Symmetry::antisymmetric(vec![2, 3]));
    t.add_symmetry(Symmetry::symmetric_pairs(vec![(0, 1), (2, 3)]));
    assert_eq!(canonicalize(&t).unwrap().to_string(), "-R_a_b_a_b");

    let mut t = tensor("S", &["a", "a", "b"]);
    t.add_symmetry(Symmetry::antisymmetric(vec![0, 1, 2]));
    assert_eq!(canonicalize(&t).unwrap().coefficient(), 0);
}

#[test]
fn young_projection_of_riemann_tensor() {
    let mut r = tensor("R", &["a", "b", "c", "d"]);
    r.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));
    r.add_symmetry(Symmetry::antisymmetric(vec![2, 3]));
    r.add_symmetry(Symmetry::symmetric_pairs(vec![(0, 1), (2, 3)]));

    let window = StandardTableau::new(Shape(vec![2, 2]), vec![vec![1, 3], vec![2, 4]]).unwrap();
    let terms = r.project_with_tableau(&window).unwrap();
    let shown: Vec<String> = terms.iter().map(|t| t.to_string()).collect();
    assert_eq!(shown, ["8R_a_b_c_d", "4R_a_c_b_d", "-4R_a_d_b_c"]);

    let row = StandardTableau::new(Shape(vec![4]), vec![vec![1, 2, 3, 4]]).unwrap();
    assert!(r.project_with_tableau(&row).unwrap().is_empty());
}
