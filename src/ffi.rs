//! C Foreign Function Interface for Butler-Portugal
//!
//! Provides C-compatible bindings for tensor canonicalization functionality.
//! All types are exposed as opaque pointers with explicit lifetime management.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

use crate::canonicalization::canonicalize;
use crate::dummy::Metric;
use crate::index::TensorIndex;
use crate::symmetry::Symmetry;
use crate::tensor::Tensor;

// Opaque handle types for C
/// Opaque handle to a Tensor
pub type TensorHandle = *mut Tensor;
/// Opaque handle to a TensorIndex
pub type TensorIndexHandle = *mut TensorIndex;
/// Opaque handle to a Symmetry
pub type SymmetryHandle = *mut Symmetry;

/// Result codes for FFI operations
#[repr(C)]
pub enum BPResult {
    /// Operation succeeded
    Success = 0,
    /// Null pointer passed
    NullPointer = 1,
    /// Invalid argument
    InvalidArgument = 2,
    /// Canonicalization failed
    CanonicalizationError = 3,
    /// Memory allocation failed
    AllocationError = 4,
}

/// Metric of an index type, deciding whether contracted pairs may swap ends.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BPMetric {
    /// Raising and lowering is free
    Symmetric = 0,
    /// Raising and lowering flips the sign
    Antisymmetric = 1,
    /// No metric, contraction ends are fixed
    Absent = 2,
}

fn metric_from_c(metric: i32) -> Option<Metric> {
    match metric {
        m if m == BPMetric::Symmetric as i32 => Some(Metric::Symmetric),
        m if m == BPMetric::Antisymmetric as i32 => Some(Metric::Antisymmetric),
        m if m == BPMetric::Absent as i32 => Some(Metric::Absent),
        _ => None,
    }
}

unsafe fn c_str<'a>(s: *const c_char) -> Option<&'a str> {
    if s.is_null() {
        None
    } else {
        CStr::from_ptr(s).to_str().ok()
    }
}

unsafe fn set_error(error_out: *mut BPResult, result: BPResult) {
    if !error_out.is_null() {
        *error_out = result;
    }
}

// -----------------------------------------------------------------------------
// TensorIndex Functions
// -----------------------------------------------------------------------------

/// Create a new covariant tensor index.
///
/// # Safety
/// - `name` must be a valid null-terminated C string.
/// - The returned handle must be freed with `bp_index_free`.
#[no_mangle]
pub unsafe extern "C" fn bp_index_new(name: *const c_char, position: usize) -> TensorIndexHandle {
    if name.is_null() {
        return ptr::null_mut();
    }
    let Ok(name_str) = CStr::from_ptr(name).to_str() else {
        return ptr::null_mut();
    };
    Box::into_raw(Box::new(TensorIndex::new(name_str, position)))
}

/// Create a new contravariant tensor index.
///
/// # Safety
/// - `name` must be a valid null-terminated C string.
/// - The returned handle must be freed with `bp_index_free`.
#[no_mangle]
pub unsafe extern "C" fn bp_index_contravariant(
    name: *const c_char,
    position: usize,
) -> TensorIndexHandle {
    if name.is_null() {
        return ptr::null_mut();
    }
    let Ok(name_str) = CStr::from_ptr(name).to_str() else {
        return ptr::null_mut();
    };
    Box::into_raw(Box::new(TensorIndex::contravariant(name_str, position)))
}

/// Set the index type of an index. Indices only contract within one type and
/// each type has its own metric. The default type is the empty string.
///
/// # Safety
/// - `index` must be a valid non-null handle.
/// - `index_type` must be a valid null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn bp_index_set_type(
    index: TensorIndexHandle,
    index_type: *const c_char,
) -> BPResult {
    if index.is_null() || index_type.is_null() {
        return BPResult::NullPointer;
    }
    let Some(index_type) = c_str(index_type) else {
        return BPResult::InvalidArgument;
    };
    *index = (*index).clone().of_type(index_type);
    BPResult::Success
}

/// Free a tensor index.
///
/// # Safety
/// - `index` must be a valid handle returned by `bp_index_new` or `bp_index_contravariant`,
///   or null (in which case this is a no-op).
#[no_mangle]
pub unsafe extern "C" fn bp_index_free(index: TensorIndexHandle) {
    if !index.is_null() {
        drop(Box::from_raw(index));
    }
}

/// Clone a tensor index.
///
/// # Safety
/// - `index` must be a valid non-null handle.
/// - The returned handle must be freed with `bp_index_free`.
#[no_mangle]
pub unsafe extern "C" fn bp_index_clone(index: TensorIndexHandle) -> TensorIndexHandle {
    if index.is_null() {
        return ptr::null_mut();
    }
    Box::into_raw(Box::new((*index).clone()))
}

// -----------------------------------------------------------------------------
// Symmetry Functions
// -----------------------------------------------------------------------------

/// Create a symmetric symmetry for the given indices.
///
/// # Safety
/// - `indices` must point to a valid array of `len` elements.
/// - The returned handle must be freed with `bp_symmetry_free`.
#[no_mangle]
pub unsafe extern "C" fn bp_symmetry_symmetric(
    indices: *const usize,
    len: usize,
) -> SymmetryHandle {
    if indices.is_null() && len > 0 {
        return ptr::null_mut();
    }
    let indices_vec = if len > 0 {
        std::slice::from_raw_parts(indices, len).to_vec()
    } else {
        Vec::new()
    };
    Box::into_raw(Box::new(Symmetry::symmetric(indices_vec)))
}

/// Create an antisymmetric symmetry for the given indices.
///
/// # Safety
/// - `indices` must point to a valid array of `len` elements.
/// - The returned handle must be freed with `bp_symmetry_free`.
#[no_mangle]
pub unsafe extern "C" fn bp_symmetry_antisymmetric(
    indices: *const usize,
    len: usize,
) -> SymmetryHandle {
    if indices.is_null() && len > 0 {
        return ptr::null_mut();
    }
    let indices_vec = if len > 0 {
        std::slice::from_raw_parts(indices, len).to_vec()
    } else {
        Vec::new()
    };
    Box::into_raw(Box::new(Symmetry::antisymmetric(indices_vec)))
}

/// Create a symmetric pairs symmetry.
///
/// # Safety
/// - `pairs` must point to a valid array of `len` pairs (2 * len usize values).
///   Pairs are passed as [a0, b0, a1, b1, ...].
/// - The returned handle must be freed with `bp_symmetry_free`.
#[no_mangle]
pub unsafe extern "C" fn bp_symmetry_symmetric_pairs(
    pairs: *const usize,
    len: usize,
) -> SymmetryHandle {
    if pairs.is_null() && len > 0 {
        return ptr::null_mut();
    }
    let pairs_vec: Vec<(usize, usize)> = if len > 0 {
        let flat = std::slice::from_raw_parts(pairs, len * 2);
        flat.chunks(2).map(|c| (c[0], c[1])).collect()
    } else {
        Vec::new()
    };
    Box::into_raw(Box::new(Symmetry::symmetric_pairs(pairs_vec)))
}

/// Create a cyclic symmetry for the given indices.
///
/// # Safety
/// - `indices` must point to a valid array of `len` elements.
/// - The returned handle must be freed with `bp_symmetry_free`.
#[no_mangle]
pub unsafe extern "C" fn bp_symmetry_cyclic(indices: *const usize, len: usize) -> SymmetryHandle {
    if indices.is_null() && len > 0 {
        return ptr::null_mut();
    }
    let indices_vec = if len > 0 {
        std::slice::from_raw_parts(indices, len).to_vec()
    } else {
        Vec::new()
    };
    Box::into_raw(Box::new(Symmetry::cyclic(indices_vec)))
}

/// Create a symmetry from explicit signed generators. `permutations` holds
/// `num_generators` permutations of length `rank` back to back, where new slot
/// `i` takes old slot `p[i]`, and `signs` holds one sign (`1` or `-1`) each.
///
/// # Safety
/// - `permutations` must point to `num_generators * rank` elements.
/// - `signs` must point to `num_generators` elements.
/// - The returned handle must be freed with `bp_symmetry_free`.
#[no_mangle]
pub unsafe extern "C" fn bp_symmetry_custom(
    permutations: *const usize,
    signs: *const i32,
    num_generators: usize,
    rank: usize,
) -> SymmetryHandle {
    if num_generators > 0 && (permutations.is_null() || signs.is_null()) {
        return ptr::null_mut();
    }
    if num_generators == 0 {
        return Box::into_raw(Box::new(Symmetry::custom(Vec::new(), Vec::new())));
    }
    let flat = std::slice::from_raw_parts(permutations, num_generators * rank);
    let perms = flat.chunks(rank.max(1)).map(<[usize]>::to_vec).collect();
    let signs = std::slice::from_raw_parts(signs, num_generators).to_vec();
    Box::into_raw(Box::new(Symmetry::custom(perms, signs)))
}

/// Free a symmetry.
///
/// # Safety
/// - `symmetry` must be a valid handle or null.
#[no_mangle]
pub unsafe extern "C" fn bp_symmetry_free(symmetry: SymmetryHandle) {
    if !symmetry.is_null() {
        drop(Box::from_raw(symmetry));
    }
}

/// Clone a symmetry.
///
/// # Safety
/// - `symmetry` must be a valid non-null handle.
/// - The returned handle must be freed with `bp_symmetry_free`.
#[no_mangle]
pub unsafe extern "C" fn bp_symmetry_clone(symmetry: SymmetryHandle) -> SymmetryHandle {
    if symmetry.is_null() {
        return ptr::null_mut();
    }
    Box::into_raw(Box::new((*symmetry).clone()))
}

// -----------------------------------------------------------------------------
// Tensor Functions
// -----------------------------------------------------------------------------

/// Create a new tensor with the given name and indices.
///
/// # Safety
/// - `name` must be a valid null-terminated C string.
/// - `indices` must point to a valid array of `num_indices` TensorIndexHandle pointers.
///   The indices are cloned, so the caller retains ownership of the original handles.
/// - The returned handle must be freed with `bp_tensor_free`.
#[no_mangle]
pub unsafe extern "C" fn bp_tensor_new(
    name: *const c_char,
    indices: *const TensorIndexHandle,
    num_indices: usize,
) -> TensorHandle {
    if name.is_null() {
        return ptr::null_mut();
    }
    if indices.is_null() && num_indices > 0 {
        return ptr::null_mut();
    }

    let Ok(name_str) = CStr::from_ptr(name).to_str() else {
        return ptr::null_mut();
    };

    let indices_vec: Vec<TensorIndex> = if num_indices > 0 {
        let handles = std::slice::from_raw_parts(indices, num_indices);
        let mut vec = Vec::with_capacity(num_indices);
        for &handle in handles {
            if handle.is_null() {
                return ptr::null_mut();
            }
            vec.push((*handle).clone());
        }
        vec
    } else {
        Vec::new()
    };

    Box::into_raw(Box::new(Tensor::new(name_str, indices_vec)))
}

/// Create a new tensor with a coefficient.
///
/// # Safety
/// - Same requirements as `bp_tensor_new`.
#[no_mangle]
pub unsafe extern "C" fn bp_tensor_with_coefficient(
    name: *const c_char,
    indices: *const TensorIndexHandle,
    num_indices: usize,
    coefficient: i32,
) -> TensorHandle {
    if name.is_null() {
        return ptr::null_mut();
    }
    if indices.is_null() && num_indices > 0 {
        return ptr::null_mut();
    }

    let Ok(name_str) = CStr::from_ptr(name).to_str() else {
        return ptr::null_mut();
    };

    let indices_vec: Vec<TensorIndex> = if num_indices > 0 {
        let handles = std::slice::from_raw_parts(indices, num_indices);
        let mut vec = Vec::with_capacity(num_indices);
        for &handle in handles {
            if handle.is_null() {
                return ptr::null_mut();
            }
            vec.push((*handle).clone());
        }
        vec
    } else {
        Vec::new()
    };

    Box::into_raw(Box::new(Tensor::with_coefficient(
        name_str,
        indices_vec,
        coefficient,
    )))
}

/// Free a tensor.
///
/// # Safety
/// - `tensor` must be a valid handle or null.
#[no_mangle]
pub unsafe extern "C" fn bp_tensor_free(tensor: TensorHandle) {
    if !tensor.is_null() {
        drop(Box::from_raw(tensor));
    }
}

/// Clone a tensor.
///
/// # Safety
/// - `tensor` must be a valid non-null handle.
/// - The returned handle must be freed with `bp_tensor_free`.
#[no_mangle]
pub unsafe extern "C" fn bp_tensor_clone(tensor: TensorHandle) -> TensorHandle {
    if tensor.is_null() {
        return ptr::null_mut();
    }
    Box::into_raw(Box::new((*tensor).clone()))
}

/// Add a symmetry to a tensor.
///
/// # Safety
/// - Both `tensor` and `symmetry` must be valid non-null handles.
/// - The symmetry is cloned, so the caller retains ownership of the original.
#[no_mangle]
pub unsafe extern "C" fn bp_tensor_add_symmetry(
    tensor: TensorHandle,
    symmetry: SymmetryHandle,
) -> BPResult {
    if tensor.is_null() || symmetry.is_null() {
        return BPResult::NullPointer;
    }
    (*tensor).add_symmetry((*symmetry).clone());
    BPResult::Success
}

/// Set the metric of an index type on a tensor. `metric` is a `BPMetric`
/// value, anything else is `InvalidArgument`.
///
/// # Safety
/// - `tensor` must be a valid non-null handle.
/// - `index_type` must be a valid null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn bp_tensor_set_metric(
    tensor: TensorHandle,
    index_type: *const c_char,
    metric: i32,
) -> BPResult {
    if tensor.is_null() || index_type.is_null() {
        return BPResult::NullPointer;
    }
    let (Some(index_type), Some(metric)) = (c_str(index_type), metric_from_c(metric)) else {
        return BPResult::InvalidArgument;
    };
    (*tensor).set_metric(index_type, metric);
    BPResult::Success
}

/// Multiply tensors into one tensor, see `Tensor::product`. Identical factors
/// commute. Returns null on a null factor or conflicting metrics.
///
/// # Safety
/// - `factors` must point to `num_factors` valid non-null handles.
/// - The factors are cloned, the caller keeps ownership.
/// - The returned handle must be freed with `bp_tensor_free`.
#[no_mangle]
pub unsafe extern "C" fn bp_tensor_product(
    factors: *const TensorHandle,
    num_factors: usize,
    error_out: *mut BPResult,
) -> TensorHandle {
    if factors.is_null() && num_factors > 0 {
        set_error(error_out, BPResult::NullPointer);
        return ptr::null_mut();
    }
    let handles = if num_factors > 0 {
        std::slice::from_raw_parts(factors, num_factors)
    } else {
        &[]
    };
    if handles.iter().any(|h| h.is_null()) {
        set_error(error_out, BPResult::NullPointer);
        return ptr::null_mut();
    }
    let owned: Vec<Tensor> = handles.iter().map(|&h| (*h).clone()).collect();
    let Ok(product) = Tensor::product(&owned) else {
        set_error(error_out, BPResult::InvalidArgument);
        return ptr::null_mut();
    };
    set_error(error_out, BPResult::Success);
    Box::into_raw(Box::new(product))
}

/// Get the rank (number of indices) of a tensor.
///
/// # Safety
/// - `tensor` must be a valid non-null handle.
#[no_mangle]
pub unsafe extern "C" fn bp_tensor_rank(tensor: TensorHandle) -> usize {
    if tensor.is_null() {
        return 0;
    }
    (*tensor).rank()
}

/// Get the coefficient of a tensor.
///
/// # Safety
/// - `tensor` must be a valid non-null handle.
#[no_mangle]
pub unsafe extern "C" fn bp_tensor_coefficient(tensor: TensorHandle) -> i32 {
    if tensor.is_null() {
        return 0;
    }
    (*tensor).coefficient()
}

/// Check if a tensor is zero due to symmetry constraints.
///
/// # Safety
/// - `tensor` must be a valid non-null handle.
#[no_mangle]
pub unsafe extern "C" fn bp_tensor_is_zero(tensor: TensorHandle) -> bool {
    if tensor.is_null() {
        return true;
    }
    (*tensor).is_zero()
}

/// Get a string representation of the tensor.
/// Returns a newly allocated C string that must be freed with `bp_string_free`.
///
/// # Safety
/// - `tensor` must be a valid non-null handle.
/// - The returned string must be freed with `bp_string_free`.
#[no_mangle]
pub unsafe extern "C" fn bp_tensor_to_string(tensor: TensorHandle) -> *mut c_char {
    if tensor.is_null() {
        return ptr::null_mut();
    }
    let s = format!("{}", *tensor);
    match CString::new(s) {
        Ok(cstr) => cstr.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

/// Free a string returned by the library.
///
/// # Safety
/// - `s` must be a string returned by this library or null.
#[no_mangle]
pub unsafe extern "C" fn bp_string_free(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}

// -----------------------------------------------------------------------------
// Canonicalization Functions
// -----------------------------------------------------------------------------

/// Canonicalize a tensor using the Butler-Portugal algorithm.
/// Returns a new tensor handle representing the canonical form.
///
/// # Safety
/// - `tensor` must be a valid non-null handle.
/// - The returned handle must be freed with `bp_tensor_free`.
/// - On error, returns null and sets `error_out` if provided.
#[no_mangle]
pub unsafe extern "C" fn bp_canonicalize(
    tensor: TensorHandle,
    error_out: *mut BPResult,
) -> TensorHandle {
    if tensor.is_null() {
        if !error_out.is_null() {
            *error_out = BPResult::NullPointer;
        }
        return ptr::null_mut();
    }

    if let Ok(canonical) = canonicalize(&*tensor) {
        if !error_out.is_null() {
            *error_out = BPResult::Success;
        }
        Box::into_raw(Box::new(canonical))
    } else {
        if !error_out.is_null() {
            *error_out = BPResult::CanonicalizationError;
        }
        ptr::null_mut()
    }
}

// -----------------------------------------------------------------------------
// Version Information
// -----------------------------------------------------------------------------

/// Get the library version string.
/// Returns a static string that should NOT be freed.
#[no_mangle]
pub extern "C" fn bp_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn test_ffi_basic_flow() {
        unsafe {
            // Create indices
            let name_a = CString::new("a").expect("CString::new failed");
            let name_b = CString::new("b").expect("CString::new failed");

            let idx_a = bp_index_new(name_a.as_ptr(), 0);
            let idx_b = bp_index_new(name_b.as_ptr(), 1);

            assert!(!idx_a.is_null());
            assert!(!idx_b.is_null());

            // Create tensor
            let tensor_name = CString::new("T").expect("CString::new failed");
            let indices = [idx_a, idx_b];
            let tensor = bp_tensor_new(tensor_name.as_ptr(), indices.as_ptr(), 2);

            assert!(!tensor.is_null());
            assert_eq!(bp_tensor_rank(tensor), 2);
            assert_eq!(bp_tensor_coefficient(tensor), 1);

            // Add symmetry
            let sym_indices: [usize; 2] = [0, 1];
            let symmetry = bp_symmetry_symmetric(sym_indices.as_ptr(), 2);
            assert!(!symmetry.is_null());

            let result = bp_tensor_add_symmetry(tensor, symmetry);
            assert!(matches!(result, BPResult::Success));

            // Canonicalize
            let mut error = BPResult::Success;
            let canonical = bp_canonicalize(tensor, &mut error);
            assert!(!canonical.is_null());
            assert!(matches!(error, BPResult::Success));

            // Get string representation
            let s = bp_tensor_to_string(canonical);
            assert!(!s.is_null());

            // Cleanup
            bp_string_free(s);
            bp_tensor_free(canonical);
            bp_symmetry_free(symmetry);
            bp_tensor_free(tensor);
            bp_index_free(idx_a);
            bp_index_free(idx_b);
        }
    }

    #[test]
    fn test_ffi_dummies_and_products() {
        unsafe {
            let name = |s: &str| CString::new(s).expect("CString::new failed");
            let (a, b, t, f, spin) = (name("a"), name("b"), name("t"), name("F"), name("spin"));
            let lower = [bp_index_new(b.as_ptr(), 0), bp_index_new(a.as_ptr(), 1)];
            let upper = [
                bp_index_contravariant(a.as_ptr(), 0),
                bp_index_contravariant(b.as_ptr(), 1),
            ];
            for &i in lower.iter().chain(&upper) {
                assert!(matches!(
                    bp_index_set_type(i, spin.as_ptr()),
                    BPResult::Success
                ));
            }
            let swap: [usize; 2] = [1, 0];
            let anti = bp_symmetry_custom(swap.as_ptr(), [-1].as_ptr(), 1, 2);
            let f_lower = bp_tensor_new(f.as_ptr(), lower.as_ptr(), 2);
            let f_upper = bp_tensor_new(f.as_ptr(), upper.as_ptr(), 2);
            for &x in &[f_lower, f_upper] {
                assert!(matches!(bp_tensor_add_symmetry(x, anti), BPResult::Success));
                assert!(matches!(
                    bp_tensor_set_metric(x, spin.as_ptr(), 1),
                    BPResult::Success
                ));
            }
            assert!(matches!(
                bp_tensor_set_metric(f_lower, t.as_ptr(), 7),
                BPResult::InvalidArgument
            ));

            let mut error = BPResult::NullPointer;
            let product = bp_tensor_product([f_lower, f_upper].as_ptr(), 2, &mut error);
            assert!(matches!(error, BPResult::Success));
            let canonical = bp_canonicalize(product, &mut error);
            let s = bp_tensor_to_string(canonical);
            assert_eq!(CStr::from_ptr(s).to_str(), Ok("-F_a_b F^a^b"));

            assert!(
                bp_tensor_product([f_lower, ptr::null_mut()].as_ptr(), 2, &mut error).is_null()
            );
            assert!(matches!(error, BPResult::NullPointer));

            bp_string_free(s);
            bp_tensor_free(canonical);
            bp_tensor_free(product);
            bp_tensor_free(f_lower);
            bp_tensor_free(f_upper);
            bp_symmetry_free(anti);
            for i in lower.into_iter().chain(upper) {
                bp_index_free(i);
            }
        }
    }

    #[test]
    fn test_ffi_null_safety() {
        unsafe {
            // All functions should handle null gracefully
            bp_index_free(ptr::null_mut());
            bp_symmetry_free(ptr::null_mut());
            bp_tensor_free(ptr::null_mut());
            bp_string_free(ptr::null_mut());

            assert!(bp_index_new(ptr::null(), 0).is_null());
            assert!(bp_tensor_new(ptr::null(), ptr::null(), 0).is_null());
            assert_eq!(bp_tensor_rank(ptr::null_mut()), 0);
            assert!(bp_tensor_is_zero(ptr::null_mut()));
        }
    }
}
