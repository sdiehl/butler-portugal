//! Butler-Portugal tensor canonicalization algorithm
//!
//! This module implements the core Butler-Portugal algorithm for bringing
//! tensors into canonical form by systematically applying symmetry operations.

use crate::error::Result;
use crate::index::TensorIndex;
use crate::symmetry::Symmetry;
use crate::tensor::Tensor;

/// Canonicalizes a tensor using the Butler-Portugal algorithm
///
/// The Butler-Portugal algorithm works by:
/// 1. Identifying all possible index permutations
/// 2. Applying symmetry constraints to reduce the search space
/// 3. Finding the lexicographically minimal form
/// 4. Returning the canonical tensor with appropriate coefficient
///
/// # Arguments
/// * `tensor` - The tensor to canonicalize
///
/// # Returns
/// * `Ok(Tensor)` - The canonicalized tensor
/// * `Err(ButlerPortugalError)` - If canonicalization fails
///
/// # Example
/// ```rust
/// use butler_portugal::{Tensor, TensorIndex, Symmetry, canonicalize};
///
/// let mut tensor = Tensor::new("R", vec![
///     TensorIndex::new("a", 0),
///     TensorIndex::new("b", 1),
///     TensorIndex::new("c", 2),
///     TensorIndex::new("d", 3),
/// ]);
///
/// // Riemann tensor symmetries
/// tensor.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));
/// tensor.add_symmetry(Symmetry::antisymmetric(vec![2, 3]));
///
/// let canonical = canonicalize(&tensor).unwrap();
/// ```
pub fn canonicalize(tensor: &Tensor) -> Result<Tensor> {
    // Handle trivial cases
    if tensor.is_zero() {
        let mut zero_tensor = tensor.clone();
        zero_tensor.set_coefficient(0);
        return Ok(zero_tensor);
    }

    if tensor.rank() <= 1 {
        return Ok(tensor.clone());
    }

    // Generate all relevant permutations considering symmetries
    let permutations = generate_canonical_permutations(tensor)?;

    if permutations.is_empty() {
        return Ok(tensor.clone());
    }

    // Find the lexicographically minimal tensor form
    let mut best_tensor = None;
    let mut best_key = None;

    for perm in permutations {
        let candidate = tensor.permute(&perm)?;

        // Skip if this permutation makes the tensor zero
        if candidate.is_zero() {
            continue;
        }

        let key = canonical_key(&candidate);

        if best_key.as_ref().is_none_or(|bk| key < *bk) {
            best_tensor = Some(candidate);
            best_key = Some(key);
        }
    }

    match best_tensor {
        Some(tensor) => Ok(tensor),
        None => {
            // All permutations resulted in zero tensor
            let mut zero_tensor = tensor.clone();
            zero_tensor.set_coefficient(0);
            Ok(zero_tensor)
        }
    }
}

/// Generates all canonical permutations for a tensor considering its symmetries
fn generate_canonical_permutations(tensor: &Tensor) -> Result<Vec<Vec<usize>>> {
    let n = tensor.rank();
    let mut all_perms = generate_all_permutations(n);

    // Filter permutations based on symmetry constraints
    filter_by_symmetries(&mut all_perms, tensor.symmetries());

    Ok(all_perms)
}

/// Generates all permutations of indices 0..n
fn generate_all_permutations(n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return vec![vec![]];
    }
    if n == 1 {
        return vec![vec![0]];
    }

    let mut result = Vec::new();
    let indices: Vec<usize> = (0..n).collect();
    generate_permutations_recursive(&indices, 0, &mut result);
    result
}

/// Recursive helper for generating permutations
fn generate_permutations_recursive(arr: &[usize], start: usize, result: &mut Vec<Vec<usize>>) {
    if start == arr.len() {
        result.push(arr.to_vec());
        return;
    }

    for i in start..arr.len() {
        let mut arr_copy = arr.to_vec();
        arr_copy.swap(start, i);
        generate_permutations_recursive(&arr_copy, start + 1, result);
    }
}

/// Filters permutations based on tensor symmetries
fn filter_by_symmetries(permutations: &mut Vec<Vec<usize>>, symmetries: &[Symmetry]) {
    if symmetries.is_empty() {
        return;
    }

    permutations.retain(|perm| {
        for symmetry in symmetries {
            if !symmetry.is_valid_permutation(perm) {
                return false;
            }
        }
        true
    });
}

/// Creates a canonical key for comparing tensors
fn canonical_key(tensor: &Tensor) -> String {
    let mut key = String::new();

    // Add tensor name
    key.push_str(tensor.name());
    key.push('_');

    // Add index information in canonical order
    for index in tensor.indices() {
        key.push_str(index.name());
        if index.is_contravariant() {
            key.push('^');
        } else {
            key.push('_');
        }
    }

    key
}

/// Advanced canonicalization with optimization for specific tensor types
pub fn canonicalize_with_optimizations(tensor: &Tensor) -> Result<Tensor> {
    // Check for common tensor patterns and apply optimized algorithms
    if is_riemann_like(tensor) {
        canonicalize_riemann_tensor(tensor)
    } else if is_symmetric_tensor(tensor) {
        canonicalize_symmetric_tensor(tensor)
    } else if is_antisymmetric_tensor(tensor) {
        canonicalize_antisymmetric_tensor(tensor)
    } else {
        canonicalize(tensor)
    }
}

/// Checks if tensor has Riemann-like symmetries
fn is_riemann_like(tensor: &Tensor) -> bool {
    if tensor.rank() != 4 {
        return false;
    }

    let symmetries = tensor.symmetries();
    let has_first_antisym = symmetries.iter().any(|s| s.is_antisymmetric_pair(0, 1));
    let has_second_antisym = symmetries.iter().any(|s| s.is_antisymmetric_pair(2, 3));

    has_first_antisym && has_second_antisym
}

/// Optimized canonicalization for Riemann tensors
fn canonicalize_riemann_tensor(tensor: &Tensor) -> Result<Tensor> {
    // For Riemann tensors, we can use the specific symmetry structure
    // R_abcd = -R_bacd = -R_abdc = R_badc
    // R_abcd = R_cdab (Bianchi identity constraint can be added)

    // Generate only the minimal set of permutations needed
    let permutations = vec![
        vec![0, 1, 2, 3], // original
        vec![1, 0, 2, 3], // swap first pair
        vec![0, 1, 3, 2], // swap second pair
        vec![1, 0, 3, 2], // swap both pairs
        vec![2, 3, 0, 1], // exchange pairs
        vec![3, 2, 0, 1], // exchange pairs + swap first
        vec![2, 3, 1, 0], // exchange pairs + swap second
        vec![3, 2, 1, 0], // exchange pairs + swap both
    ];

    let mut best_tensor = None;
    let mut best_key = None;

    for perm in permutations {
        let candidate = tensor.permute(&perm)?;

        if candidate.is_zero() {
            continue;
        }

        let key = canonical_key(&candidate);

        if best_key.as_ref().is_none_or(|bk| key < *bk) {
            best_tensor = Some(candidate);
            best_key = Some(key);
        }
    }

    match best_tensor {
        Some(tensor) => Ok(tensor),
        None => {
            let mut zero_tensor = tensor.clone();
            zero_tensor.set_coefficient(0);
            Ok(zero_tensor)
        }
    }
}

/// Checks if tensor is purely symmetric
fn is_symmetric_tensor(tensor: &Tensor) -> bool {
    tensor.symmetries().iter().all(|s| s.is_symmetric())
}

/// Optimized canonicalization for symmetric tensors
fn canonicalize_symmetric_tensor(tensor: &Tensor) -> Result<Tensor> {
    // For symmetric tensors, canonical form has indices in alphabetical order
    let mut indices_with_positions: Vec<(usize, &TensorIndex)> =
        tensor.indices().iter().enumerate().collect();

    indices_with_positions.sort_by(|a, b| a.1.canonical_cmp(b.1));

    let permutation: Vec<usize> = indices_with_positions.iter().map(|(pos, _)| *pos).collect();
    tensor.permute(&permutation)
}

/// Checks if tensor is purely antisymmetric
fn is_antisymmetric_tensor(tensor: &Tensor) -> bool {
    tensor.symmetries().iter().all(|s| s.is_antisymmetric())
}

/// Optimized canonicalization for antisymmetric tensors
fn canonicalize_antisymmetric_tensor(tensor: &Tensor) -> Result<Tensor> {
    // For antisymmetric tensors, canonical form has indices in alphabetical order
    // but we need to track the sign from the permutation
    let mut indices_with_positions: Vec<(usize, &TensorIndex)> =
        tensor.indices().iter().enumerate().collect();

    indices_with_positions.sort_by(|a, b| a.1.canonical_cmp(b.1));

    let permutation: Vec<usize> = indices_with_positions.iter().map(|(pos, _)| *pos).collect();
    tensor.permute(&permutation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symmetry::Symmetry;

    #[test]
    fn test_trivial_canonicalization() {
        let tensor = Tensor::new("T", vec![TensorIndex::new("i", 0)]);
        let result = canonicalize(&tensor).unwrap();
        assert_eq!(result, tensor);
    }

    #[test]
    fn test_symmetric_tensor_canonicalization() {
        let mut tensor = Tensor::new(
            "S",
            vec![TensorIndex::new("b", 0), TensorIndex::new("a", 1)],
        );

        tensor.add_symmetry(Symmetry::symmetric(vec![0, 1]));

        let result = canonicalize(&tensor).unwrap();
        assert_eq!(result.indices()[0].name(), "a");
        assert_eq!(result.indices()[1].name(), "b");
    }

    #[test]
    fn test_antisymmetric_tensor_canonicalization() {
        let mut tensor = Tensor::new(
            "A",
            vec![TensorIndex::new("b", 0), TensorIndex::new("a", 1)],
        );

        tensor.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));

        let result = canonicalize(&tensor).unwrap();
        assert_eq!(result.indices()[0].name(), "a");
        assert_eq!(result.indices()[1].name(), "b");
        assert_eq!(result.coefficient(), -1); // Sign change from swap
    }

    #[test]
    fn test_zero_tensor_canonicalization() {
        let mut tensor = Tensor::new(
            "A",
            vec![TensorIndex::new("a", 0), TensorIndex::new("a", 1)],
        );

        tensor.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));

        let result = canonicalize(&tensor).unwrap();
        assert_eq!(result.coefficient(), 0);
    }

    #[test]
    fn test_riemann_tensor_canonicalization() {
        let mut tensor = Tensor::new(
            "R",
            vec![
                TensorIndex::new("d", 0),
                TensorIndex::new("c", 1),
                TensorIndex::new("b", 2),
                TensorIndex::new("a", 3),
            ],
        );

        tensor.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));
        tensor.add_symmetry(Symmetry::antisymmetric(vec![2, 3]));

        let result = canonicalize(&tensor).unwrap();

        // Should be in canonical order
        assert_eq!(result.indices()[0].name(), "a");
        assert_eq!(result.indices()[1].name(), "b");
        assert_eq!(result.indices()[2].name(), "c");
        assert_eq!(result.indices()[3].name(), "d");
    }
}
