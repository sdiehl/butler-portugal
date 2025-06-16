//! Tensor representation and manipulation
//!
//! This module provides the core `Tensor` struct and associated methods
//! for representing tensors with indices and symmetry properties.

use crate::index::TensorIndex;
use crate::symmetry::Symmetry;
use std::fmt;

/// Represents a tensor with indices and symmetry properties
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    /// The name/symbol of the tensor (e.g., "R" for Riemann tensor)
    name: String,
    /// The indices of the tensor
    indices: Vec<TensorIndex>,
    /// Symmetry properties of the tensor
    symmetries: Vec<Symmetry>,
    /// Optional coefficient (default is 1)
    coefficient: i32,
}

impl Tensor {
    /// Creates a new tensor with the given name and indices
    ///
    /// # Arguments
    /// * `name` - The name/symbol of the tensor
    /// * `indices` - Vector of tensor indices
    ///
    /// # Example
    /// ```rust
    /// use butler_portugal::{Tensor, TensorIndex};
    ///
    /// let tensor = Tensor::new("g", vec![
    ///     TensorIndex::new("mu", 0),
    ///     TensorIndex::new("nu", 1),
    /// ]);
    /// ```
    pub fn new(name: &str, indices: Vec<TensorIndex>) -> Self {
        Self {
            name: name.to_string(),
            indices,
            symmetries: Vec::new(),
            coefficient: 1,
        }
    }

    /// Creates a new tensor with a coefficient
    pub fn with_coefficient(name: &str, indices: Vec<TensorIndex>, coefficient: i32) -> Self {
        Self {
            name: name.to_string(),
            indices,
            symmetries: Vec::new(),
            coefficient,
        }
    }

    /// Returns the name of the tensor
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns a reference to the tensor indices
    pub fn indices(&self) -> &[TensorIndex] {
        &self.indices
    }

    /// Returns a mutable reference to the tensor indices
    pub fn indices_mut(&mut self) -> &mut Vec<TensorIndex> {
        &mut self.indices
    }

    /// Returns a reference to the symmetries
    pub fn symmetries(&self) -> &[Symmetry] {
        &self.symmetries
    }

    /// Returns the coefficient
    pub fn coefficient(&self) -> i32 {
        self.coefficient
    }

    /// Sets the coefficient
    pub fn set_coefficient(&mut self, coefficient: i32) {
        self.coefficient = coefficient;
    }

    /// Adds a symmetry property to the tensor
    ///
    /// # Arguments
    /// * `symmetry` - The symmetry to add
    pub fn add_symmetry(&mut self, symmetry: Symmetry) {
        self.symmetries.push(symmetry);
    }

    /// Removes all symmetries
    pub fn clear_symmetries(&mut self) {
        self.symmetries.clear();
    }

    /// Returns the rank (number of indices) of the tensor
    pub fn rank(&self) -> usize {
        self.indices.len()
    }

    /// Swaps two indices at the given positions
    ///
    /// # Arguments
    /// * `i` - First index position
    /// * `j` - Second index position
    ///
    /// Returns the sign change (1 or -1) based on symmetry properties
    pub fn swap_indices(&mut self, i: usize, j: usize) -> i32 {
        if i >= self.indices.len() || j >= self.indices.len() || i == j {
            return 1;
        }

        self.indices.swap(i, j);

        // Calculate sign change based on symmetries
        let mut sign = 1;
        for symmetry in &self.symmetries {
            sign *= symmetry.sign_change_for_swap(i, j);
        }

        self.coefficient *= sign;
        sign
    }

    /// Creates a copy of the tensor with permuted indices
    ///
    /// # Arguments
    /// * `permutation` - Array representing the permutation
    pub fn permute(&self, permutation: &[usize]) -> crate::Result<Self> {
        if permutation.len() != self.indices.len() {
            return Err(crate::ButlerPortugalError::InvalidPermutation(format!(
                "Permutation length {} doesn't match tensor rank {}",
                permutation.len(),
                self.indices.len()
            )));
        }

        let mut new_indices = Vec::with_capacity(self.indices.len());
        for &p in permutation {
            if p >= self.indices.len() {
                return Err(crate::ButlerPortugalError::InvalidPermutation(format!(
                    "Permutation index {p} out of bounds"
                )));
            }
            new_indices.push(self.indices[p].clone());
        }

        let mut new_tensor = Self {
            name: self.name.clone(),
            indices: new_indices,
            symmetries: self.symmetries.clone(),
            coefficient: self.coefficient,
        };

        // Calculate sign change for this permutation
        let sign = self.permutation_sign(permutation);
        new_tensor.coefficient *= sign;

        Ok(new_tensor)
    }

    /// Calculates the sign of a permutation based on tensor symmetries
    fn permutation_sign(&self, permutation: &[usize]) -> i32 {
        let mut sign = 1;

        // Count inversions for each symmetry group
        for symmetry in &self.symmetries {
            sign *= symmetry.permutation_sign(permutation);
        }

        sign
    }

    /// Checks if the tensor is zero due to symmetry constraints
    pub fn is_zero(&self) -> bool {
        self.coefficient == 0
            || self
                .symmetries
                .iter()
                .any(|s| s.makes_tensor_zero(&self.indices))
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.coefficient == 0 {
            return write!(f, "0");
        }

        let sign = if self.coefficient < 0 { "-" } else { "" };
        let coeff = if self.coefficient.abs() == 1 {
            String::new()
        } else {
            self.coefficient.abs().to_string()
        };

        write!(f, "{}{}{}", sign, coeff, self.name)?;

        if !self.indices.is_empty() {
            write!(f, "_")?;
            for (i, index) in self.indices.iter().enumerate() {
                if i > 0 {
                    write!(f, " ")?;
                }
                write!(f, "{index}")?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symmetry::Symmetry;

    #[test]
    fn test_tensor_creation() {
        let tensor = Tensor::new(
            "R",
            vec![TensorIndex::new("a", 0), TensorIndex::new("b", 1)],
        );

        assert_eq!(tensor.name(), "R");
        assert_eq!(tensor.rank(), 2);
        assert_eq!(tensor.coefficient(), 1);
    }

    #[test]
    fn test_tensor_with_coefficient() {
        let tensor = Tensor::with_coefficient("T", vec![TensorIndex::new("i", 0)], -3);

        assert_eq!(tensor.coefficient(), -3);
    }

    #[test]
    fn test_index_swapping() {
        let mut tensor = Tensor::new(
            "A",
            vec![TensorIndex::new("i", 0), TensorIndex::new("j", 1)],
        );

        tensor.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));

        let sign = tensor.swap_indices(0, 1);
        assert_eq!(sign, -1);
        assert_eq!(tensor.coefficient(), -1);
    }

    #[test]
    fn test_tensor_display() {
        let tensor = Tensor::new(
            "g",
            vec![TensorIndex::new("mu", 0), TensorIndex::new("nu", 1)],
        );

        let display = format!("{}", tensor);
        assert!(display.contains("g"));
        assert!(display.contains("mu"));
        assert!(display.contains("nu"));
    }
}
