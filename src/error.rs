//! Error types for the Butler-Portugal library
//!
//! This module defines error types that can occur during tensor
//! canonicalization and related operations.

use std::error::Error;
use std::fmt;

/// Result type for Butler-Portugal operations
pub type Result<T> = std::result::Result<T, ButlerPortugalError>;

/// Errors that can occur during tensor canonicalization
#[derive(Debug, Clone, PartialEq)]
pub enum ButlerPortugalError {
    /// Invalid permutation provided
    InvalidPermutation(String),

    /// Invalid tensor structure
    InvalidTensor(String),

    /// Invalid symmetry specification
    InvalidSymmetry(String),

    /// Index out of bounds
    IndexOutOfBounds { index: usize, max: usize },

    /// Incompatible tensor operations
    IncompatibleTensors(String),

    /// Mathematical error (division by zero, etc.)
    MathematicalError(String),

    /// Generic computation error
    ComputationError(String),

    /// Performance/optimization error
    PerformanceError(String),

    /// Cache error
    CacheError(String),

    /// Memory allocation error
    MemoryError(String),

    /// Configuration error
    ConfigurationError(String),
}

impl fmt::Display for ButlerPortugalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ButlerPortugalError::InvalidPermutation(msg) => {
                write!(f, "Invalid permutation: {msg}")
            }
            ButlerPortugalError::InvalidTensor(msg) => {
                write!(f, "Invalid tensor: {msg}")
            }
            ButlerPortugalError::InvalidSymmetry(msg) => {
                write!(f, "Invalid symmetry: {msg}")
            }
            ButlerPortugalError::IndexOutOfBounds { index, max } => {
                write!(f, "Index {index} out of bounds (max: {max})")
            }
            ButlerPortugalError::IncompatibleTensors(msg) => {
                write!(f, "Incompatible tensors: {msg}")
            }
            ButlerPortugalError::MathematicalError(msg) => {
                write!(f, "Mathematical error: {msg}")
            }
            ButlerPortugalError::ComputationError(msg) => {
                write!(f, "Computation error: {msg}")
            }
            ButlerPortugalError::PerformanceError(msg) => {
                write!(f, "Performance error: {msg}")
            }
            ButlerPortugalError::CacheError(msg) => {
                write!(f, "Cache error: {msg}")
            }
            ButlerPortugalError::MemoryError(msg) => {
                write!(f, "Memory error: {msg}")
            }
            ButlerPortugalError::ConfigurationError(msg) => {
                write!(f, "Configuration error: {msg}")
            }
        }
    }
}

impl Error for ButlerPortugalError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }
}

impl From<std::num::ParseIntError> for ButlerPortugalError {
    fn from(err: std::num::ParseIntError) -> Self {
        ButlerPortugalError::ComputationError(format!("Parse error: {err}"))
    }
}

impl From<std::string::FromUtf8Error> for ButlerPortugalError {
    fn from(err: std::string::FromUtf8Error) -> Self {
        ButlerPortugalError::ComputationError(format!("UTF-8 error: {err}"))
    }
}

/// Helper macro for creating errors with formatted messages
#[macro_export]
macro_rules! bp_error {
    ($variant:ident, $fmt:expr) => {
        $crate::ButlerPortugalError::$variant($fmt.to_string())
    };
    ($variant:ident, $fmt:expr, $($arg:tt)*) => {
        $crate::ButlerPortugalError::$variant(format!($fmt, $($arg)*))
    };
}

/// Helper macro for creating Results with formatted error messages
#[macro_export]
macro_rules! bp_bail {
    ($variant:ident, $fmt:expr) => {
        return Err($crate::bp_error!($variant, $fmt))
    };
    ($variant:ident, $fmt:expr, $($arg:tt)*) => {
        return Err($crate::bp_error!($variant, $fmt, $($arg)*))
    };
}

/// Helper function for validating index bounds
pub fn validate_index_bounds(index: usize, max: usize) -> Result<()> {
    if index >= max {
        Err(ButlerPortugalError::IndexOutOfBounds { index, max })
    } else {
        Ok(())
    }
}

/// Helper function for validating permutation
pub fn validate_permutation(permutation: &[usize], expected_size: usize) -> Result<()> {
    if permutation.len() != expected_size {
        return Err(ButlerPortugalError::InvalidPermutation(format!(
            "Expected permutation of size {}, got {}",
            expected_size,
            permutation.len()
        )));
    }

    let mut seen = vec![false; expected_size];
    for &p in permutation {
        if p >= expected_size {
            return Err(ButlerPortugalError::InvalidPermutation(format!(
                "Permutation element {} out of bounds (max: {})",
                p,
                expected_size - 1
            )));
        }
        if seen[p] {
            return Err(ButlerPortugalError::InvalidPermutation(format!(
                "Duplicate element {p} in permutation"
            )));
        }
        seen[p] = true;
    }

    Ok(())
}

/// Helper function for validating tensor indices
pub fn validate_tensor_indices(indices: &[crate::TensorIndex]) -> Result<()> {
    if indices.is_empty() {
        return Err(ButlerPortugalError::InvalidTensor(
            "Tensor must have at least one index".to_string(),
        ));
    }

    // Check for position consistency
    for (i, index) in indices.iter().enumerate() {
        if index.position() != i {
            return Err(ButlerPortugalError::InvalidTensor(format!(
                "Index at position {} has incorrect position value {}",
                i,
                index.position()
            )));
        }
    }

    Ok(())
}

/// Validates symmetry configuration for a tensor
pub fn validate_symmetry_config(symmetries: &[crate::Symmetry], tensor_rank: usize) -> Result<()> {
    for (i, symmetry) in symmetries.iter().enumerate() {
        match symmetry {
            crate::Symmetry::Symmetric { indices } | crate::Symmetry::Antisymmetric { indices } => {
                for &idx in indices {
                    if idx >= tensor_rank {
                        return Err(ButlerPortugalError::InvalidSymmetry(format!(
                            "Symmetry {i} references index {idx} but tensor has rank {tensor_rank}"
                        )));
                    }
                }
                // Check for duplicate indices within the same symmetry
                let mut seen = std::collections::HashSet::new();
                for &idx in indices {
                    if !seen.insert(idx) {
                        return Err(ButlerPortugalError::InvalidSymmetry(format!(
                            "Symmetry {i} has duplicate index {idx}"
                        )));
                    }
                }
            }
            crate::Symmetry::SymmetricPairs { pairs } => {
                for &(i, j) in pairs {
                    if i >= tensor_rank || j >= tensor_rank {
                        return Err(ButlerPortugalError::InvalidSymmetry(format!(
                            "SymmetricPairs in symmetry {i} references indices ({i}, {j}) but tensor has rank {tensor_rank}"
                        )));
                    }
                    if i == j {
                        return Err(ButlerPortugalError::InvalidSymmetry(format!(
                            "SymmetricPairs in symmetry {i} has identical indices ({i}, {j})"
                        )));
                    }
                }
            }
            crate::Symmetry::Cyclic { indices } => {
                for &idx in indices {
                    if idx >= tensor_rank {
                        return Err(ButlerPortugalError::InvalidSymmetry(format!(
                            "Cyclic symmetry {i} references index {idx} but tensor has rank {tensor_rank}"
                        )));
                    }
                }
                if indices.len() < 2 {
                    return Err(ButlerPortugalError::InvalidSymmetry(format!(
                        "Cyclic symmetry {} must have at least 2 indices, got {}",
                        i,
                        indices.len()
                    )));
                }
            }
            crate::Symmetry::Custom {
                valid_permutations,
                signs,
            } => {
                if valid_permutations.len() != signs.len() {
                    return Err(ButlerPortugalError::InvalidSymmetry(format!(
                        "Custom symmetry {} has {} permutations but {} signs",
                        i,
                        valid_permutations.len(),
                        signs.len()
                    )));
                }
                for (j, perm) in valid_permutations.iter().enumerate() {
                    if perm.len() != tensor_rank {
                        return Err(ButlerPortugalError::InvalidSymmetry(format!(
                            "Custom symmetry {} permutation {} has length {} but tensor has rank {}",
                            i, j, perm.len(), tensor_rank
                        )));
                    }
                    if let Err(e) = validate_permutation(perm, tensor_rank) {
                        return Err(ButlerPortugalError::InvalidSymmetry(format!(
                            "Custom symmetry {i} permutation {j} is invalid: {e}"
                        )));
                    }
                }
            }
        }
    }
    Ok(())
}

/// Validates tensor configuration for canonicalization
pub fn validate_tensor_for_canonicalization(tensor: &crate::Tensor) -> Result<()> {
    // Check basic tensor validity
    validate_tensor_indices(tensor.indices())?;

    // Check symmetry validity
    validate_symmetry_config(tensor.symmetries(), tensor.rank())?;

    // Check for conflicting symmetries
    check_symmetry_conflicts(tensor.symmetries())?;

    Ok(())
}

/// Checks for conflicts between different symmetries
fn check_symmetry_conflicts(symmetries: &[crate::Symmetry]) -> Result<()> {
    for (i, sym1) in symmetries.iter().enumerate() {
        for (j, sym2) in symmetries.iter().enumerate().skip(i + 1) {
            if let Some(conflict) = find_symmetry_conflict(sym1, sym2) {
                return Err(ButlerPortugalError::InvalidSymmetry(format!(
                    "Conflicting symmetries {i} and {j}: {conflict}"
                )));
            }
        }
    }
    Ok(())
}

/// Finds conflicts between two symmetries
fn find_symmetry_conflict(sym1: &crate::Symmetry, sym2: &crate::Symmetry) -> Option<String> {
    // Extract indices involved in each symmetry
    let indices1 = get_symmetry_indices(sym1);
    let indices2 = get_symmetry_indices(sym2);

    // Check for overlapping indices
    let overlap: Vec<usize> = indices1.intersection(&indices2).cloned().collect();
    if overlap.is_empty() {
        return None; // No overlap, no conflict
    }

    // Check if the overlapping indices have compatible symmetry types
    match (sym1, sym2) {
        (crate::Symmetry::Symmetric { .. }, crate::Symmetry::Symmetric { .. }) => None,
        (crate::Symmetry::Antisymmetric { .. }, crate::Symmetry::Antisymmetric { .. }) => None,
        (crate::Symmetry::Symmetric { .. }, crate::Symmetry::Antisymmetric { .. })
        | (crate::Symmetry::Antisymmetric { .. }, crate::Symmetry::Symmetric { .. }) => Some(
            format!("Symmetric and antisymmetric symmetries overlap on indices {overlap:?}"),
        ),
        _ => Some(format!(
            "Incompatible symmetry types overlap on indices {overlap:?}"
        )),
    }
}

/// Extracts all indices involved in a symmetry
fn get_symmetry_indices(symmetry: &crate::Symmetry) -> std::collections::HashSet<usize> {
    match symmetry {
        crate::Symmetry::Symmetric { indices } | crate::Symmetry::Antisymmetric { indices } => {
            indices.iter().cloned().collect()
        }
        crate::Symmetry::SymmetricPairs { pairs } => {
            pairs.iter().flat_map(|(i, j)| vec![*i, *j]).collect()
        }
        crate::Symmetry::Cyclic { indices } => indices.iter().cloned().collect(),
        crate::Symmetry::Custom {
            valid_permutations, ..
        } => {
            if let Some(first_perm) = valid_permutations.first() {
                first_perm.iter().cloned().collect()
            } else {
                std::collections::HashSet::new()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = ButlerPortugalError::InvalidPermutation("test message".to_string());
        let display = format!("{err}");
        assert!(display.contains("Invalid permutation"));
        assert!(display.contains("test message"));
    }

    #[test]
    fn test_index_bounds_validation() {
        assert!(validate_index_bounds(0, 5).is_ok());
        assert!(validate_index_bounds(4, 5).is_ok());
        assert!(validate_index_bounds(5, 5).is_err());
        assert!(validate_index_bounds(10, 5).is_err());
    }

    #[test]
    fn test_permutation_validation() {
        assert!(validate_permutation(&[0, 1, 2], 3).is_ok());
        assert!(validate_permutation(&[2, 1, 0], 3).is_ok());
        assert!(validate_permutation(&[0, 1], 3).is_err()); // Wrong size
        assert!(validate_permutation(&[0, 1, 3], 3).is_err()); // Out of bounds
        assert!(validate_permutation(&[0, 1, 1], 3).is_err()); // Duplicate
    }

    #[test]
    fn test_bp_error_macro() {
        let err = bp_error!(InvalidTensor, "test");
        assert_eq!(err, ButlerPortugalError::InvalidTensor("test".to_string()));

        let err = bp_error!(InvalidTensor, "test {}", 42);
        assert_eq!(
            err,
            ButlerPortugalError::InvalidTensor("test 42".to_string())
        );
    }
}
