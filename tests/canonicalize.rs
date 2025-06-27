//! Integration tests for canonicalization methods

use butler_portugal::young_tableaux::{Shape, StandardTableau};
use butler_portugal::{
    canonicalize_with_optimizations, CanonicalizationMethod, Symmetry, Tensor, TensorIndex,
};

#[test]
fn test_symmetric_tensor_canonicalization_methods() {
    // Symmetric tensor S_ab = S_ba
    let mut tensor = Tensor::new(
        "S",
        vec![TensorIndex::new("b", 0), TensorIndex::new("a", 1)],
    );
    tensor.add_symmetry(Symmetry::symmetric(vec![0, 1]));

    // Schreier-Sims method
    let canonical_schreier =
        canonicalize_with_optimizations(&tensor, None, CanonicalizationMethod::SchreierSims)
            .unwrap();
    assert_eq!(canonical_schreier.indices()[0].name(), "a");
    assert_eq!(canonical_schreier.indices()[1].name(), "b");

    // Young symmetrizer method
    let shape = Shape(vec![2]);
    let tableau = StandardTableau::new(shape, vec![vec![1, 2]]).unwrap();
    let canonical_young = canonicalize_with_optimizations(
        &tensor,
        Some(&tableau),
        CanonicalizationMethod::YoungSymmetrizer,
    )
    .unwrap();
    assert_eq!(canonical_young.indices()[0].name(), "a");
    assert_eq!(canonical_young.indices()[1].name(), "b");
}

#[test]
fn test_antisymmetric_tensor_canonicalization_methods() {
    // Antisymmetric tensor A_ab = -A_ba
    let mut tensor = Tensor::new(
        "A",
        vec![TensorIndex::new("b", 0), TensorIndex::new("a", 1)],
    );
    tensor.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));

    // Schreier-Sims method
    let canonical_schreier =
        canonicalize_with_optimizations(&tensor, None, CanonicalizationMethod::SchreierSims)
            .unwrap();
    assert_eq!(canonical_schreier.indices()[0].name(), "a");
    assert_eq!(canonical_schreier.indices()[1].name(), "b");
    assert_eq!(canonical_schreier.coefficient(), -1);

    // Young symmetrizer method
    let shape = Shape(vec![1, 1]);
    let tableau = StandardTableau::new(shape, vec![vec![1], vec![2]]).unwrap();
    let canonical_young = canonicalize_with_optimizations(
        &tensor,
        Some(&tableau),
        CanonicalizationMethod::YoungSymmetrizer,
    )
    .unwrap();
    println!(
        "Young result: indices = {:?}, coefficient = {}",
        canonical_young
            .indices()
            .iter()
            .map(|i| i.name())
            .collect::<Vec<_>>(),
        canonical_young.coefficient()
    );
    assert_eq!(canonical_young.indices()[0].name(), "a");
    assert_eq!(canonical_young.indices()[1].name(), "b");
    assert_eq!(canonical_young.coefficient(), -1);
}

#[test]
fn test_mixed_symmetry_tensor_canonicalization_methods() {
    // Tensor with no symmetry
    let tensor = Tensor::new(
        "T",
        vec![TensorIndex::new("b", 0), TensorIndex::new("a", 1)],
    );

    // Schreier-Sims method (should just sort by name)
    let canonical_schreier =
        canonicalize_with_optimizations(&tensor, None, CanonicalizationMethod::SchreierSims)
            .unwrap();
    assert_eq!(canonical_schreier.indices()[0].name(), "a");
    assert_eq!(canonical_schreier.indices()[1].name(), "b");

    // Young symmetrizer method with shape [2] (symmetric)
    let shape = Shape(vec![2]);
    let tableau = StandardTableau::new(shape, vec![vec![1, 2]]).unwrap();
    let canonical_young = canonicalize_with_optimizations(
        &tensor,
        Some(&tableau),
        CanonicalizationMethod::YoungSymmetrizer,
    )
    .unwrap();
    // Should be symmetric, so indices sorted
    assert_eq!(canonical_young.indices()[0].name(), "a");
    assert_eq!(canonical_young.indices()[1].name(), "b");
}
