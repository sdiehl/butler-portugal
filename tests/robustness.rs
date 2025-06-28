//! Robustness tests for the Butler-Portugal library
//!
//! These tests verify error handling, validation, and performance optimizations.

use butler_portugal::canonicalization::BSGS;
use butler_portugal::error::{validate_symmetry_config, validate_tensor_for_canonicalization};
use butler_portugal::*;

#[test]
fn test_performance_configuration() {
    let config = CanonicalizationConfig::default();
    assert!(config.enable_caching);
    assert!(config.early_termination);
    assert_eq!(config.max_permutations, Some(10000));
    assert!(!config.parallel_processing);
}

#[test]
fn test_cache_functionality() {
    let cache = CanonicalizationCache::new();

    // Test BSGS caching
    let bsgs = BSGS::identity(3);
    cache.insert_bsgs("test_bsgs".to_string(), bsgs.clone());

    let retrieved = cache.get_bsgs("test_bsgs");
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap(), bsgs);

    // Test permutation caching
    let perms = vec![vec![0, 1, 2], vec![1, 0, 2]];
    cache.insert_permutations("test_perms".to_string(), perms.clone());

    let retrieved_perms = cache.get_permutations("test_perms");
    assert!(retrieved_perms.is_some());
    assert_eq!(retrieved_perms.unwrap(), perms);
}

#[test]
fn test_enhanced_canonicalization_with_config() {
    let mut tensor = Tensor::new(
        "T",
        vec![TensorIndex::new("b", 0), TensorIndex::new("a", 1)],
    );
    tensor.add_symmetry(Symmetry::symmetric(vec![0, 1]));

    let config = CanonicalizationConfig::default();
    let cache = CanonicalizationCache::new();

    let result = canonicalize_with_config(&tensor, &config, Some(&cache)).unwrap();

    // Should be in canonical form (alphabetical order)
    assert_eq!(result.indices()[0].name(), "a");
    assert_eq!(result.indices()[1].name(), "b");
}

#[test]
fn test_early_termination() {
    let mut tensor = Tensor::new(
        "A",
        vec![TensorIndex::new("a", 0), TensorIndex::new("a", 1)], // Same index name
    );
    tensor.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));

    let mut config = CanonicalizationConfig::default();
    config.early_termination = true;

    let result = canonicalize_with_config(&tensor, &config, None).unwrap();
    assert_eq!(result.coefficient(), 0); // Should be zero due to antisymmetry
}

#[test]
fn test_permutation_limit() {
    let mut tensor = Tensor::new(
        "T",
        vec![
            TensorIndex::new("a", 0),
            TensorIndex::new("b", 1),
            TensorIndex::new("c", 2),
            TensorIndex::new("d", 3),
        ],
    );
    // Add many symmetries to generate many permutations
    tensor.add_symmetry(Symmetry::symmetric(vec![0, 1, 2, 3]));

    let mut config = CanonicalizationConfig::default();
    config.max_permutations = Some(5); // Limit to 5 permutations

    let result = canonicalize_with_config(&tensor, &config, None).unwrap();
    assert!(!result.is_zero());
}

#[test]
fn test_symmetry_validation() {
    // Valid symmetry
    let valid_symmetries = vec![Symmetry::symmetric(vec![0, 1])];
    assert!(validate_symmetry_config(&valid_symmetries, 2).is_ok());

    // Invalid: index out of bounds
    let invalid_symmetries = vec![Symmetry::symmetric(vec![0, 2])]; // index 2 doesn't exist in rank 2
    assert!(validate_symmetry_config(&invalid_symmetries, 2).is_err());

    // Invalid: duplicate indices
    let duplicate_symmetries = vec![Symmetry::symmetric(vec![0, 0])];
    assert!(validate_symmetry_config(&duplicate_symmetries, 2).is_err());
}

#[test]
fn test_tensor_validation() {
    // Valid tensor
    let mut valid_tensor = Tensor::new(
        "T",
        vec![TensorIndex::new("a", 0), TensorIndex::new("b", 1)],
    );
    valid_tensor.add_symmetry(Symmetry::symmetric(vec![0, 1]));
    assert!(validate_tensor_for_canonicalization(&valid_tensor).is_ok());

    // Invalid: empty indices
    let invalid_tensor = Tensor::new("T", vec![]);
    assert!(validate_tensor_for_canonicalization(&invalid_tensor).is_err());
}

#[test]
fn test_symmetry_conflicts() {
    let mut tensor = Tensor::new(
        "T",
        vec![TensorIndex::new("a", 0), TensorIndex::new("b", 1)],
    );

    // Conflicting symmetries: symmetric and antisymmetric on same indices
    tensor.add_symmetry(Symmetry::symmetric(vec![0, 1]));
    tensor.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));

    assert!(validate_tensor_for_canonicalization(&tensor).is_err());
}

#[test]
fn test_custom_symmetry_validation() {
    let valid_permutations = vec![vec![0, 1, 2], vec![1, 0, 2]];
    let signs = vec![1, -1];

    let custom_symmetry = Symmetry::custom(valid_permutations, signs);
    let symmetries = vec![custom_symmetry];

    assert!(validate_symmetry_config(&symmetries, 3).is_ok());

    // Invalid: mismatched permutation and sign counts
    let invalid_permutations = vec![vec![0, 1, 2]];
    let invalid_signs = vec![1, -1]; // More signs than permutations

    let invalid_custom_symmetry = Symmetry::custom(invalid_permutations, invalid_signs);
    let invalid_symmetries = vec![invalid_custom_symmetry];

    assert!(validate_symmetry_config(&invalid_symmetries, 3).is_err());
}

#[test]
fn test_cyclic_symmetry_validation() {
    // Valid cyclic symmetry
    let valid_cyclic = vec![Symmetry::cyclic(vec![0, 1, 2])];
    assert!(validate_symmetry_config(&valid_cyclic, 3).is_ok());

    // Invalid: too few indices for cyclic
    let invalid_cyclic = vec![Symmetry::cyclic(vec![0])];
    assert!(validate_symmetry_config(&invalid_cyclic, 2).is_err());
}

#[test]
fn test_error_recovery() {
    let mut tensor = Tensor::new(
        "T",
        vec![TensorIndex::new("a", 0), TensorIndex::new("b", 1)],
    );

    // Add a symmetry that will cause the tensor to be zero
    tensor.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));

    // Create a tensor with repeated indices (should be zero)
    let mut zero_tensor = Tensor::new(
        "A",
        vec![TensorIndex::new("a", 0), TensorIndex::new("a", 1)],
    );
    zero_tensor.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));

    let config = CanonicalizationConfig::default();

    // Both should return zero tensors without panicking
    let result1 = canonicalize_with_config(&tensor, &config, None).unwrap();
    let result2 = canonicalize_with_config(&zero_tensor, &config, None).unwrap();

    assert_eq!(result1.coefficient(), 1); // Original tensor is valid
    assert_eq!(result2.coefficient(), 0); // Zero tensor due to repeated indices
}

#[test]
fn test_large_tensor_handling() {
    // Create a large tensor with many symmetries
    let mut indices = Vec::new();
    for i in 0..8 {
        indices.push(TensorIndex::new(&format!("i{i}"), i));
    }

    let mut tensor = Tensor::new("L", indices);

    // Add multiple symmetries
    tensor.add_symmetry(Symmetry::symmetric(vec![0, 1, 2]));
    tensor.add_symmetry(Symmetry::antisymmetric(vec![3, 4]));
    tensor.add_symmetry(Symmetry::symmetric(vec![5, 6, 7]));

    let config = CanonicalizationConfig::default();
    let cache = CanonicalizationCache::new();

    // Should complete without panicking
    let result = canonicalize_with_config(&tensor, &config, Some(&cache)).unwrap();
    assert!(!result.is_zero());
}

#[test]
fn test_memory_efficiency() {
    // Test that the cache doesn't grow unbounded
    let cache = CanonicalizationCache::new();

    // Insert many items
    for i in 0..100 {
        let bsgs = BSGS::identity(3);
        cache.insert_bsgs(format!("bsgs_{i}"), bsgs);

        let perms = vec![vec![0, 1, 2]];
        cache.insert_permutations(format!("perms_{i}"), perms);
    }

    // Verify we can still retrieve items
    assert!(cache.get_bsgs("bsgs_0").is_some());
    assert!(cache.get_permutations("perms_0").is_some());
}
