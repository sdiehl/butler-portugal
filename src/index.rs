//! Tensor index representation and manipulation
//!
//! This module provides the `TensorIndex` struct for representing
//! individual tensor indices with names and positions.

use std::fmt;

/// Represents a single tensor index
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorIndex {
    /// The name/label of the index (e.g., "mu", "nu", "a", "b")
    name: String,
    /// The position of the index in the tensor
    position: usize,
    /// Whether the index is contravariant (true) or covariant (false)
    contravariant: bool,
    /// The index type, such as a spacetime or spinor index. Empty by default.
    index_type: String,
}

/// How an index takes part in a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IndexKind {
    /// Appears once, or repeated with a single variance.
    Free,
    /// One end of a contraction, carrying the pair's id.
    Dummy(usize),
}

impl TensorIndex {
    /// Creates a new tensor index
    ///
    /// # Arguments
    /// * `name` - The name of the index
    /// * `position` - The position in the tensor
    ///
    /// # Example
    /// ```rust
    /// use butler_portugal::TensorIndex;
    ///
    /// let index = TensorIndex::new("mu", 0);
    /// ```
    pub fn new(name: &str, position: usize) -> Self {
        Self {
            name: name.to_string(),
            position,
            contravariant: false,
            index_type: String::new(),
        }
    }

    /// Creates a new contravariant tensor index
    ///
    /// # Arguments
    /// * `name` - The name of the index
    /// * `position` - The position in the tensor
    pub fn contravariant(name: &str, position: usize) -> Self {
        Self {
            name: name.to_string(),
            position,
            contravariant: true,
            index_type: String::new(),
        }
    }

    /// Creates a new covariant tensor index
    ///
    /// # Arguments
    /// * `name` - The name of the index
    /// * `position` - The position in the tensor
    pub fn covariant(name: &str, position: usize) -> Self {
        Self {
            name: name.to_string(),
            position,
            contravariant: false,
            index_type: String::new(),
        }
    }

    /// Returns a copy of this index with the given index type. Indices of
    /// different types never contract, and each type has its own metric.
    ///
    /// ```rust
    /// use butler_portugal::TensorIndex;
    ///
    /// let spinor = TensorIndex::new("A", 0).of_type("spinor");
    /// assert_eq!(spinor.index_type(), "spinor");
    /// ```
    pub fn of_type(mut self, index_type: &str) -> Self {
        self.index_type = index_type.to_string();
        self
    }

    /// The index type.
    pub fn index_type(&self) -> &str {
        &self.index_type
    }

    /// Returns the name of the index
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the position of the index
    pub fn position(&self) -> usize {
        self.position
    }

    /// Sets the position of the index
    pub fn set_position(&mut self, position: usize) {
        self.position = position;
    }

    /// Returns true if the index is contravariant
    pub fn is_contravariant(&self) -> bool {
        self.contravariant
    }

    /// Returns true if the index is covariant
    pub fn is_covariant(&self) -> bool {
        !self.contravariant
    }

    /// Sets the variance of the index
    pub fn set_contravariant(&mut self, contravariant: bool) {
        self.contravariant = contravariant;
    }

    /// Creates a copy with a new name
    pub fn with_name(&self, name: &str) -> Self {
        Self {
            name: name.to_string(),
            ..self.clone()
        }
    }

    /// Creates a copy with a new position
    pub fn with_position(&self, position: usize) -> Self {
        Self {
            position,
            ..self.clone()
        }
    }

    /// Checks if two indices can be contracted: same type and name, opposite variance.
    pub fn can_contract_with(&self, other: &TensorIndex) -> bool {
        self.index_type == other.index_type
            && self.name == other.name
            && self.contravariant != other.contravariant
    }

    /// Canonical ordering: type, then name, then covariant before
    /// contravariant, then position.
    pub fn canonical_cmp(&self, other: &TensorIndex) -> std::cmp::Ordering {
        self.label_cmp(other)
            .then(self.position.cmp(&other.position))
    }

    /// Ordering ignoring position: type, then name, then variance.
    pub(crate) fn label_cmp(&self, other: &TensorIndex) -> std::cmp::Ordering {
        (&self.index_type, &self.name, self.contravariant).cmp(&(
            &other.index_type,
            &other.name,
            other.contravariant,
        ))
    }
}

impl fmt::Display for TensorIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.contravariant {
            write!(f, "^{}", self.name)
        } else {
            write!(f, "_{}", self.name)
        }
    }
}

impl PartialOrd for TensorIndex {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(std::cmp::Ord::cmp(self, other))
    }
}

impl Ord for TensorIndex {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.canonical_cmp(other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_creation() {
        let index = TensorIndex::new("mu", 0);
        assert_eq!(index.name(), "mu");
        assert_eq!(index.position(), 0);
        assert!(index.is_covariant());
        assert!(!index.is_contravariant());
    }

    #[test]
    fn test_contravariant_index() {
        let index = TensorIndex::contravariant("nu", 1);
        assert_eq!(index.name(), "nu");
        assert_eq!(index.position(), 1);
        assert!(index.is_contravariant());
        assert!(!index.is_covariant());
    }

    #[test]
    fn test_index_contraction() {
        let index1 = TensorIndex::covariant("a", 0);
        let index2 = TensorIndex::contravariant("a", 1);
        let index3 = TensorIndex::covariant("b", 2);

        assert!(index1.can_contract_with(&index2));
        assert!(!index1.can_contract_with(&index3));
        assert!(!index1.can_contract_with(&index1));
    }

    #[test]
    fn test_canonical_ordering() {
        let index1 = TensorIndex::covariant("a", 0);
        let index2 = TensorIndex::contravariant("a", 1);
        let index3 = TensorIndex::covariant("b", 2);

        assert!(index1 < index2); // covariant comes before contravariant
        assert!(index1 < index3); // "a" comes before "b"
        assert!(index2 < index3); // "a" comes before "b"
    }

    #[test]
    fn test_index_display() {
        let covariant = TensorIndex::covariant("mu", 0);
        let contravariant = TensorIndex::contravariant("nu", 1);

        assert_eq!(format!("{covariant}"), "_mu");
        assert_eq!(format!("{contravariant}"), "^nu");
    }
}
