//! # Butler-Portugal Tensor Canonicalization Library
//!
//! Brings tensors into a canonical form: the lexicographically smallest index
//! arrangement reachable by the slot symmetries and by renaming contracted
//! (dummy) indices, with the accumulated sign folded into the coefficient.
//! The symmetries are handled as signed permutation groups through the
//! Schreier-Sims algorithm, and dummies through the Butler-Portugal double
//! coset search, so large symmetry groups never need to be enumerated.
//!
//! ## Example
//! ```rust
//! use butler_portugal::{canonicalize, Symmetry, Tensor, TensorIndex};
//!
//! let mut tensor = Tensor::new(
//!     "R",
//!     vec![
//!         TensorIndex::new("b", 0),
//!         TensorIndex::new("a", 1),
//!         TensorIndex::new("d", 2),
//!         TensorIndex::new("c", 3),
//!     ],
//! );
//!
//! // Riemann tensor symmetries
//! tensor.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));
//! tensor.add_symmetry(Symmetry::antisymmetric(vec![2, 3]));
//! tensor.add_symmetry(Symmetry::symmetric_pairs(vec![(0, 1), (2, 3)]));
//!
//! let canonical = canonicalize(&tensor)?;
//! assert_eq!(canonical.to_string(), "R_a_b_c_d");
//! # Ok::<(), butler_portugal::ButlerPortugalError>(())
//! ```

pub mod canonicalization;
pub mod dummy;
pub mod error;
pub mod ffi;
pub mod index;
pub mod permutation;
pub mod schreier_sims;
pub mod symmetry;
pub mod tensor;
pub mod young_tableaux;

pub use canonicalization::{canonicalize, SlotGroup};
pub use dummy::{DummyGroup, Metric};
pub use error::{ButlerPortugalError, Result};
pub use index::{IndexKind, TensorIndex};
pub use symmetry::Symmetry;
pub use tensor::Tensor;

#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;
