//! Integration tests for the Butler-Portugal library
//!
//! These tests verify the complete functionality of the library
//! including complex tensor canonicalization scenarios.

use butler_portugal::canonicalize_with_optimizations;
use butler_portugal::young_tableaux::{Shape, StandardTableau};
use butler_portugal::*;

#[test]
fn test_riemann_tensor_canonicalization() {
    // Test Riemann tensor with all its symmetries
    // R_abcd = -R_bacd = -R_abdc = R_badc = R_cdab

    let mut tensor = Tensor::new(
        "R",
        vec![
            TensorIndex::new("mu", 0),
            TensorIndex::new("nu", 1),
            TensorIndex::new("rho", 2),
            TensorIndex::new("sigma", 3),
        ],
    );

    // Add Riemann tensor symmetries
    tensor.add_symmetry(Symmetry::antisymmetric(vec![0, 1])); // R_[ab]cd
    tensor.add_symmetry(Symmetry::antisymmetric(vec![2, 3])); // R_ab[cd]
    tensor.add_symmetry(Symmetry::symmetric_pairs(vec![(0, 1), (2, 3)])); // R_(ab)(cd)

    let canonical = canonicalize(&tensor).unwrap();

    // The canonical form should have indices in alphabetical order
    assert_eq!(canonical.indices()[0].name(), "mu");
    assert_eq!(canonical.indices()[1].name(), "nu");
    assert_eq!(canonical.indices()[2].name(), "rho");
    assert_eq!(canonical.indices()[3].name(), "sigma");
}

#[test]
fn test_christoffel_symbol_canonicalization() {
    // Christoffel symbols are symmetric in the lower indices
    // Γ^μ_νρ = Γ^μ_ρν

    let mut gamma = Tensor::new(
        "Gamma",
        vec![
            TensorIndex::contravariant("mu", 0),
            TensorIndex::new("rho", 1),
            TensorIndex::new("nu", 2),
        ],
    );

    gamma.add_symmetry(Symmetry::symmetric(vec![1, 2]));

    let canonical = canonicalize(&gamma).unwrap();

    // Lower indices should be in alphabetical order
    assert_eq!(canonical.indices()[1].name(), "nu");
    assert_eq!(canonical.indices()[2].name(), "rho");
}

#[test]
fn test_electromagnetic_field_tensor() {
    // Electromagnetic field tensor F_μν = -F_νμ

    let mut f_tensor = Tensor::new(
        "F",
        vec![TensorIndex::new("nu", 0), TensorIndex::new("mu", 1)],
    );

    f_tensor.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));

    let canonical = canonicalize(&f_tensor).unwrap();

    // Should be in alphabetical order with correct sign
    assert_eq!(canonical.indices()[0].name(), "mu");
    assert_eq!(canonical.indices()[1].name(), "nu");
    assert_eq!(canonical.coefficient(), -1); // Sign change from swap
}

#[test]
fn test_stress_energy_tensor() {
    // Stress-energy tensor T_μν = T_νμ (symmetric)

    let mut t_tensor = Tensor::new(
        "T",
        vec![TensorIndex::new("nu", 0), TensorIndex::new("mu", 1)],
    );

    t_tensor.add_symmetry(Symmetry::symmetric(vec![0, 1]));

    let canonical = canonicalize(&t_tensor).unwrap();

    // Should be in alphabetical order with same sign
    assert_eq!(canonical.indices()[0].name(), "mu");
    assert_eq!(canonical.indices()[1].name(), "nu");
    assert_eq!(canonical.coefficient(), 1); // No sign change
}

#[test]
fn test_zero_tensor_from_antisymmetry() {
    // Antisymmetric tensor with repeated indices should be zero

    let mut tensor = Tensor::new(
        "A",
        vec![
            TensorIndex::new("mu", 0),
            TensorIndex::new("mu", 1), // Same index name
        ],
    );

    tensor.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));

    let canonical = canonicalize(&tensor).unwrap();
    assert_eq!(canonical.coefficient(), 0);
}

#[test]
fn test_mixed_variance_tensor() {
    // Test tensor with both upper and lower indices

    let mut tensor = Tensor::new(
        "R",
        vec![
            TensorIndex::contravariant("nu", 0),
            TensorIndex::new("mu", 1),
            TensorIndex::contravariant("sigma", 2),
            TensorIndex::new("rho", 3),
        ],
    );

    // Mixed Riemann tensor: R^μ_ν^ρ_σ = -R^μ_ν^σ_ρ
    tensor.add_symmetry(Symmetry::antisymmetric(vec![2, 3]));

    let canonical = canonicalize(&tensor).unwrap();

    // Should maintain variance structure while canonicalizing
    // The exact order may change but variance should be preserved for each position
    let contravariant_count = canonical
        .indices()
        .iter()
        .filter(|i| i.is_contravariant())
        .count();
    let covariant_count = canonical
        .indices()
        .iter()
        .filter(|i| i.is_covariant())
        .count();
    assert_eq!(contravariant_count, 2);
    assert_eq!(covariant_count, 2);
}

#[test]
fn test_complex_tensor_with_multiple_symmetries() {
    // Test a tensor with multiple different symmetry properties

    let mut tensor = Tensor::new(
        "C",
        vec![
            TensorIndex::new("d", 0),
            TensorIndex::new("a", 1),
            TensorIndex::new("c", 2),
            TensorIndex::new("b", 3),
        ],
    );

    // Add multiple symmetries
    tensor.add_symmetry(Symmetry::symmetric(vec![1, 3])); // positions 1,3 symmetric
    tensor.add_symmetry(Symmetry::antisymmetric(vec![0, 2])); // positions 0,2 antisymmetric

    let canonical = canonicalize(&tensor).unwrap();

    // Verify the result respects all symmetries
    assert!(!canonical.is_zero());

    // The exact form depends on the algorithm, but it should be consistent
    let result_str = format!("{canonical}");
    assert!(result_str.contains("C"));
}

#[test]
fn test_tensor_coefficient_handling() {
    // Test that coefficients are properly handled during canonicalization

    let mut tensor = Tensor::with_coefficient(
        "T",
        vec![TensorIndex::new("b", 0), TensorIndex::new("a", 1)],
        3,
    );

    tensor.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));

    let canonical = canonicalize(&tensor).unwrap();

    // Should have alphabetical order with coefficient adjusted for sign change
    assert_eq!(canonical.indices()[0].name(), "a");
    assert_eq!(canonical.indices()[1].name(), "b");
    assert_eq!(canonical.coefficient(), -3); // Original coefficient times sign change
}

#[test]
fn test_single_index_tensor() {
    // Single index tensors should remain unchanged

    let tensor = Tensor::new("v", vec![TensorIndex::new("mu", 0)]);
    let canonical = canonicalize(&tensor).unwrap();

    assert_eq!(canonical, tensor);
}

#[test]
fn test_identity_tensor() {
    // Test tensor that's already in canonical form

    let mut tensor = Tensor::new(
        "g",
        vec![TensorIndex::new("mu", 0), TensorIndex::new("nu", 1)],
    );

    tensor.add_symmetry(Symmetry::symmetric(vec![0, 1]));

    let canonical = canonicalize(&tensor).unwrap();

    // Should be unchanged
    assert_eq!(canonical.indices()[0].name(), "mu");
    assert_eq!(canonical.indices()[1].name(), "nu");
    assert_eq!(canonical.coefficient(), 1);
}

#[test]
fn test_error_handling() {
    // Test various error conditions

    let tensor = Tensor::new("T", vec![TensorIndex::new("a", 0)]);

    // Invalid permutation
    let result = tensor.permute(&[0, 1]); // Wrong size
    assert!(result.is_err());

    let result = tensor.permute(&[1]); // Out of bounds
    assert!(result.is_err());
}

#[test]
fn test_optimization_paths() {
    // Test that optimized canonicalization gives same results

    let mut riemann = Tensor::new(
        "R",
        vec![
            TensorIndex::new("d", 0),
            TensorIndex::new("c", 1),
            TensorIndex::new("b", 2),
            TensorIndex::new("a", 3),
        ],
    );

    riemann.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));
    riemann.add_symmetry(Symmetry::antisymmetric(vec![2, 3]));

    let standard = canonicalize(&riemann).unwrap();
    let optimized = canonicalize_with_optimizations(&riemann, None).unwrap();

    // Both should give the same result
    assert_eq!(standard.indices()[0].name(), optimized.indices()[0].name());
    assert_eq!(standard.indices()[1].name(), optimized.indices()[1].name());
    assert_eq!(standard.indices()[2].name(), optimized.indices()[2].name());
    assert_eq!(standard.indices()[3].name(), optimized.indices()[3].name());
    assert_eq!(standard.coefficient(), optimized.coefficient());
}

#[test]
fn test_large_tensor_performance() {
    // Test with a larger tensor to verify reasonable performance

    let mut tensor = Tensor::new(
        "T",
        vec![
            TensorIndex::new("f", 0),
            TensorIndex::new("e", 1),
            TensorIndex::new("d", 2),
            TensorIndex::new("c", 3),
            TensorIndex::new("b", 4),
            TensorIndex::new("a", 5),
        ],
    );

    // Add some symmetries to make it non-trivial
    tensor.add_symmetry(Symmetry::symmetric(vec![0, 1]));
    tensor.add_symmetry(Symmetry::antisymmetric(vec![2, 3]));
    tensor.add_symmetry(Symmetry::symmetric(vec![4, 5]));

    let start = std::time::Instant::now();
    let canonical = canonicalize(&tensor).unwrap();
    let duration = start.elapsed();

    // Should complete in reasonable time (less than 1 second for this size)
    assert!(duration.as_secs() < 1);
    assert!(!canonical.is_zero());
}

#[test]
fn test_tensor_projection_with_tableau() {
    use butler_portugal::{Tensor, TensorIndex};
    // Symmetric tableau shape for 2 indices
    let shape = Shape(vec![2]);
    let tableau = StandardTableau::new(shape, vec![vec![1, 2]]).unwrap();
    let tensor = Tensor::new(
        "S",
        vec![TensorIndex::new("b", 0), TensorIndex::new("a", 1)],
    );
    let projected = canonicalize_with_optimizations(&tensor, Some(&tableau)).unwrap();
    // The result should be symmetric in a and b, so indices should be sorted
    assert_eq!(projected.indices()[0].name(), "a");
    assert_eq!(projected.indices()[1].name(), "b");
}
