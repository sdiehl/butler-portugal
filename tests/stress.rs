//! Stress tests for tensor canonicalization at scale

use butler_portugal::young_tableaux::{Shape, StandardTableau};
use butler_portugal::{
    canonicalize, canonicalize_with_optimizations, CanonicalizationMethod, Symmetry, Tensor,
    TensorIndex,
};
use std::time::Instant;

/// Creates a large tensor with many indices for stress testing
fn create_large_tensor(rank: usize, name: &str) -> Tensor {
    let mut indices = Vec::with_capacity(rank);
    for i in 0..rank {
        let index_name = format!("i{i}");
        indices.push(TensorIndex::new(&index_name, i));
    }
    Tensor::new(name, indices)
}

/// Creates a large tensor with symmetric pairs for stress testing
fn create_large_symmetric_tensor(rank: usize, name: &str) -> Tensor {
    let mut tensor = create_large_tensor(rank, name);

    // Add symmetric pairs for all adjacent indices
    for i in (0..rank).step_by(2) {
        if i + 1 < rank {
            tensor.add_symmetry(Symmetry::symmetric(vec![i, i + 1]));
        }
    }

    tensor
}

/// Creates a large tensor with antisymmetric pairs for stress testing
fn create_large_antisymmetric_tensor(rank: usize, name: &str) -> Tensor {
    let mut tensor = create_large_tensor(rank, name);

    // Add antisymmetric pairs for all adjacent indices
    for i in (0..rank).step_by(2) {
        if i + 1 < rank {
            tensor.add_symmetry(Symmetry::antisymmetric(vec![i, i + 1]));
        }
    }

    tensor
}

/// Creates a large tensor with mixed symmetries for stress testing
fn create_large_mixed_symmetry_tensor(rank: usize, name: &str) -> Tensor {
    let mut tensor = create_large_tensor(rank, name);

    // Add symmetric pairs for even indices
    for i in (0..rank).step_by(4) {
        if i + 1 < rank {
            tensor.add_symmetry(Symmetry::symmetric(vec![i, i + 1]));
        }
    }

    // Add antisymmetric pairs for odd indices
    for i in (2..rank).step_by(4) {
        if i + 1 < rank {
            tensor.add_symmetry(Symmetry::antisymmetric(vec![i, i + 1]));
        }
    }

    tensor
}

/// Creates a large tensor with cyclic symmetries for stress testing
fn create_large_cyclic_tensor(rank: usize, name: &str) -> Tensor {
    let mut tensor = create_large_tensor(rank, name);

    // Add cyclic symmetries for groups of 3 indices
    for i in (0..rank).step_by(3) {
        if i + 2 < rank {
            tensor.add_symmetry(Symmetry::cyclic(vec![i, i + 1, i + 2]));
        }
    }

    tensor
}

/// Stress test for large symmetric tensors
#[test]
fn test_large_symmetric_tensor_stress() {
    let rank = 12; // Large enough to be challenging but not overwhelming
    let tensor = create_large_symmetric_tensor(rank, "S");

    println!("Testing large symmetric tensor with rank {rank}");
    let start = Instant::now();

    let result = canonicalize(&tensor).expect("Canonicalization failed");

    let duration = start.elapsed();
    println!("Large symmetric tensor canonicalization took: {duration:?}");

    // Verify the result is a permutation of the input indices
    let mut indices = result
        .indices()
        .iter()
        .map(|i| i.name())
        .collect::<Vec<_>>();
    let mut expected = tensor
        .indices()
        .iter()
        .map(|i| i.name())
        .collect::<Vec<_>>();
    indices.sort();
    expected.sort();
    assert_eq!(
        indices, expected,
        "Result should be a permutation of the input indices"
    );
    assert_eq!(
        result.coefficient(),
        1,
        "Coefficient should be 1 for symmetric tensor"
    );
}

/// Stress test for large antisymmetric tensors
#[test]
fn test_large_antisymmetric_tensor_stress() {
    let rank = 10; // Slightly smaller due to complexity
    let tensor = create_large_antisymmetric_tensor(rank, "A");

    println!("Testing large antisymmetric tensor with rank {rank}");
    let start = Instant::now();

    let result = canonicalize(&tensor).expect("Canonicalization failed");

    let duration = start.elapsed();
    println!("Large antisymmetric tensor canonicalization took: {duration:?}");

    // Verify the result is canonical
    let indices: Vec<_> = result.indices().iter().map(|i| i.name()).collect();
    let mut sorted_indices = indices.clone();
    sorted_indices.sort();

    assert_eq!(
        indices, sorted_indices,
        "Result should be in canonical order"
    );
}

/// Stress test for large mixed symmetry tensors
#[test]
fn test_large_mixed_symmetry_tensor_stress() {
    let rank = 8; // Moderate size for mixed symmetries
    let tensor = create_large_mixed_symmetry_tensor(rank, "M");

    println!("Testing large mixed symmetry tensor with rank {rank}");
    let start = Instant::now();

    let result = canonicalize(&tensor).expect("Canonicalization failed");

    let duration = start.elapsed();
    println!("Large mixed symmetry tensor canonicalization took: {duration:?}");

    // Verify the result is canonical
    let indices: Vec<_> = result.indices().iter().map(|i| i.name()).collect();
    let mut sorted_indices = indices.clone();
    sorted_indices.sort();

    assert_eq!(
        indices, sorted_indices,
        "Result should be in canonical order"
    );
}

/// Stress test for large cyclic tensors
#[test]
fn test_large_cyclic_tensor_stress() {
    let rank = 9; // Multiple of 3 for cyclic symmetries
    let tensor = create_large_cyclic_tensor(rank, "C");

    println!("Testing large cyclic tensor with rank {rank}");
    let start = Instant::now();

    let result = canonicalize(&tensor).expect("Canonicalization failed");

    let duration = start.elapsed();
    println!("Large cyclic tensor canonicalization took: {duration:?}");

    // For cyclic tensors, verify that the result is a valid tensor
    assert_eq!(result.rank(), rank, "Result should have the same rank");
    assert!(!result.is_zero(), "Result should not be zero");

    // Print the result for inspection
    let indices: Vec<_> = result.indices().iter().map(|i| i.name()).collect();
    println!("Canonical result indices: {indices:?}");
}

/// Stress test comparing both canonicalization methods
#[test]
fn test_canonicalization_methods_comparison() {
    let rank = 8;
    let tensor = create_large_mixed_symmetry_tensor(rank, "T");

    println!("Comparing canonicalization methods for tensor with rank {rank}");

    // Test Schreier-Sims method
    let start = Instant::now();
    let schreier_result =
        canonicalize_with_optimizations(&tensor, None, &CanonicalizationMethod::SchreierSims)
            .expect("Schreier-Sims canonicalization failed");
    let schreier_duration = start.elapsed();

    // Test Young symmetrizer method with symmetric tableau
    let shape = Shape(vec![rank]);
    let tableau = StandardTableau::new(shape, vec![(1..=rank).collect()]).unwrap();
    let start = Instant::now();
    let young_result = canonicalize_with_optimizations(
        &tensor,
        Some(&tableau),
        &CanonicalizationMethod::YoungSymmetrizer,
    )
    .expect("Young symmetrizer canonicalization failed");
    let young_duration = start.elapsed();

    println!("Schreier-Sims took: {schreier_duration:?}");
    println!("Young symmetrizer took: {young_duration:?}");

    // Both methods should produce canonical results
    let schreier_indices: Vec<_> = schreier_result.indices().iter().map(|i| i.name()).collect();
    let young_indices: Vec<_> = young_result.indices().iter().map(|i| i.name()).collect();

    let mut sorted_indices = schreier_indices.clone();
    sorted_indices.sort();

    assert_eq!(
        schreier_indices, sorted_indices,
        "Schreier-Sims result should be canonical"
    );
    assert_eq!(
        young_indices, sorted_indices,
        "Young symmetrizer result should be canonical"
    );
}

/// Stress test for very large tensor (if system can handle it)
#[test]
fn test_very_large_tensor_stress() {
    let rank = 16; // Very large tensor
    let tensor = create_large_symmetric_tensor(rank, "VL");

    println!("Testing very large tensor with rank {rank}");
    let start = Instant::now();

    let result = canonicalize(&tensor).expect("Canonicalization failed");

    let duration = start.elapsed();
    println!("Very large tensor canonicalization took: {duration:?}");

    // Verify the result is a permutation of the input indices
    let mut indices = result
        .indices()
        .iter()
        .map(|i| i.name())
        .collect::<Vec<_>>();
    let mut expected = tensor
        .indices()
        .iter()
        .map(|i| i.name())
        .collect::<Vec<_>>();
    indices.sort();
    expected.sort();
    assert_eq!(
        indices, expected,
        "Result should be a permutation of the input indices"
    );
    assert_eq!(
        result.coefficient(),
        1,
        "Coefficient should be 1 for symmetric tensor"
    );
}

/// Performance benchmark test
#[test]
fn test_performance_benchmark() {
    let ranks = vec![4, 6, 8, 10, 12];

    println!("Performance benchmark across different tensor ranks:");
    println!("Rank\tSymmetric\tAntisymmetric\tMixed");
    println!("----\t----------\t-------------\t----");

    for &rank in &ranks {
        let symmetric_tensor = create_large_symmetric_tensor(rank, "S");
        let antisymmetric_tensor = create_large_antisymmetric_tensor(rank, "A");
        let mixed_tensor = create_large_mixed_symmetry_tensor(rank, "M");

        let start = Instant::now();
        canonicalize(&symmetric_tensor).expect("Symmetric canonicalization failed");
        let symmetric_time = start.elapsed();

        let start = Instant::now();
        canonicalize(&antisymmetric_tensor).expect("Antisymmetric canonicalization failed");
        let antisymmetric_time = start.elapsed();

        let start = Instant::now();
        canonicalize(&mixed_tensor).expect("Mixed canonicalization failed");
        let mixed_time = start.elapsed();

        println!("{rank}\t{symmetric_time:?}\t{antisymmetric_time:?}\t{mixed_time:?}");
    }
}
