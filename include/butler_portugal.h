/**
 * This header provides C bindings for the Butler-Portugal tensor
 * canonicalization library. All types are exposed as opaque pointers with
 * explicit lifetime management.
 */

#ifndef BUTLER_PORTUGAL_H
#define BUTLER_PORTUGAL_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle types */
typedef void *BPTensorHandle;
typedef void *BPTensorIndexHandle;
typedef void *BPSymmetryHandle;

/* Result codes */
typedef enum {
  BP_SUCCESS = 0,
  BP_NULL_POINTER = 1,
  BP_INVALID_ARGUMENT = 2,
  BP_CANONICALIZATION_ERROR = 3,
  BP_ALLOCATION_ERROR = 4,
} BPResult;

/* -------------------------------------------------------------------------- */
/* TensorIndex Functions */
/* -------------------------------------------------------------------------- */

/**
 * Create a new covariant tensor index.
 *
 * @param name      Null-terminated string for the index name (e.g., "mu", "a")
 * @param position  Position of the index in the tensor (0-indexed)
 * @return          Handle to the new index, or NULL on failure
 *
 * The returned handle must be freed with bp_index_free().
 */
BPTensorIndexHandle bp_index_new(const char *name, size_t position);

/**
 * Create a new contravariant tensor index.
 *
 * @param name      Null-terminated string for the index name
 * @param position  Position of the index in the tensor (0-indexed)
 * @return          Handle to the new index, or NULL on failure
 *
 * The returned handle must be freed with bp_index_free().
 */
BPTensorIndexHandle bp_index_contravariant(const char *name, size_t position);

/**
 * Free a tensor index.
 *
 * @param index  Handle to free (may be NULL, which is a no-op)
 */
void bp_index_free(BPTensorIndexHandle index);

/**
 * Clone a tensor index.
 *
 * @param index  Handle to clone (must not be NULL)
 * @return       Handle to the cloned index, or NULL on failure
 *
 * The returned handle must be freed with bp_index_free().
 */
BPTensorIndexHandle bp_index_clone(BPTensorIndexHandle index);

/* -------------------------------------------------------------------------- */
/* Symmetry Functions */
/* -------------------------------------------------------------------------- */

/**
 * Create a symmetric symmetry for the given indices.
 * T_{abc} = T_{bac} = T_{cab} = ...
 *
 * @param indices  Array of index positions that are symmetric
 * @param len      Number of elements in the indices array
 * @return         Handle to the new symmetry, or NULL on failure
 *
 * The returned handle must be freed with bp_symmetry_free().
 */
BPSymmetryHandle bp_symmetry_symmetric(const size_t *indices, size_t len);

/**
 * Create an antisymmetric symmetry for the given indices.
 * T_{ab} = -T_{ba}
 *
 * @param indices  Array of index positions that are antisymmetric
 * @param len      Number of elements in the indices array
 * @return         Handle to the new symmetry, or NULL on failure
 *
 * The returned handle must be freed with bp_symmetry_free().
 */
BPSymmetryHandle bp_symmetry_antisymmetric(const size_t *indices, size_t len);

/**
 * Create a symmetric pairs symmetry.
 * Used for tensors like the Riemann tensor: R_{abcd} = R_{cdab}
 *
 * @param pairs    Array of index pairs as [a0, b0, a1, b1, ...]
 * @param len      Number of pairs (array has 2*len elements)
 * @return         Handle to the new symmetry, or NULL on failure
 *
 * The returned handle must be freed with bp_symmetry_free().
 */
BPSymmetryHandle bp_symmetry_symmetric_pairs(const size_t *pairs, size_t len);

/**
 * Create a cyclic symmetry for the given indices.
 * T_{abc} = T_{bca} = T_{cab}
 *
 * @param indices  Array of index positions with cyclic symmetry
 * @param len      Number of elements in the indices array
 * @return         Handle to the new symmetry, or NULL on failure
 *
 * The returned handle must be freed with bp_symmetry_free().
 */
BPSymmetryHandle bp_symmetry_cyclic(const size_t *indices, size_t len);

/**
 * Free a symmetry.
 *
 * @param symmetry  Handle to free (may be NULL, which is a no-op)
 */
void bp_symmetry_free(BPSymmetryHandle symmetry);

/**
 * Clone a symmetry.
 *
 * @param symmetry  Handle to clone (must not be NULL)
 * @return          Handle to the cloned symmetry, or NULL on failure
 *
 * The returned handle must be freed with bp_symmetry_free().
 */
BPSymmetryHandle bp_symmetry_clone(BPSymmetryHandle symmetry);

/* -------------------------------------------------------------------------- */
/* Tensor Functions */
/* -------------------------------------------------------------------------- */

/**
 * Create a new tensor with the given name and indices.
 *
 * @param name         Null-terminated string for the tensor name (e.g., "R",
 * "g")
 * @param indices      Array of TensorIndexHandle pointers
 * @param num_indices  Number of indices
 * @return             Handle to the new tensor, or NULL on failure
 *
 * The indices are cloned, so the caller retains ownership of the original
 * handles. The returned handle must be freed with bp_tensor_free().
 */
BPTensorHandle bp_tensor_new(const char *name,
                             const BPTensorIndexHandle *indices,
                             size_t num_indices);

/**
 * Create a new tensor with a coefficient.
 *
 * @param name         Null-terminated string for the tensor name
 * @param indices      Array of TensorIndexHandle pointers
 * @param num_indices  Number of indices
 * @param coefficient  Numerical coefficient for the tensor
 * @return             Handle to the new tensor, or NULL on failure
 *
 * The returned handle must be freed with bp_tensor_free().
 */
BPTensorHandle bp_tensor_with_coefficient(const char *name,
                                          const BPTensorIndexHandle *indices,
                                          size_t num_indices,
                                          int32_t coefficient);

/**
 * Free a tensor.
 *
 * @param tensor  Handle to free (may be NULL, which is a no-op)
 */
void bp_tensor_free(BPTensorHandle tensor);

/**
 * Clone a tensor.
 *
 * @param tensor  Handle to clone (must not be NULL)
 * @return        Handle to the cloned tensor, or NULL on failure
 *
 * The returned handle must be freed with bp_tensor_free().
 */
BPTensorHandle bp_tensor_clone(BPTensorHandle tensor);

/**
 * Add a symmetry to a tensor.
 *
 * @param tensor    Handle to the tensor
 * @param symmetry  Handle to the symmetry to add
 * @return          BP_SUCCESS on success, error code otherwise
 *
 * The symmetry is cloned, so the caller retains ownership of the original.
 */
BPResult bp_tensor_add_symmetry(BPTensorHandle tensor,
                                BPSymmetryHandle symmetry);

/**
 * Get the rank (number of indices) of a tensor.
 *
 * @param tensor  Handle to the tensor
 * @return        Number of indices, or 0 if tensor is NULL
 */
size_t bp_tensor_rank(BPTensorHandle tensor);

/**
 * Get the coefficient of a tensor.
 *
 * @param tensor  Handle to the tensor
 * @return        The coefficient, or 0 if tensor is NULL
 */
int32_t bp_tensor_coefficient(BPTensorHandle tensor);

/**
 * Check if a tensor is zero due to symmetry constraints.
 *
 * @param tensor  Handle to the tensor
 * @return        true if tensor is zero, false otherwise
 */
bool bp_tensor_is_zero(BPTensorHandle tensor);

/**
 * Get a string representation of the tensor.
 *
 * @param tensor  Handle to the tensor
 * @return        Newly allocated C string, or NULL on failure
 *
 * The returned string must be freed with bp_string_free().
 */
char *bp_tensor_to_string(BPTensorHandle tensor);

/**
 * Free a string returned by the library.
 *
 * @param s  String to free (may be NULL, which is a no-op)
 */
void bp_string_free(char *s);

/* -------------------------------------------------------------------------- */
/* Canonicalization Functions */
/* -------------------------------------------------------------------------- */

/**
 * Canonicalize a tensor using the Butler-Portugal algorithm.
 *
 * @param tensor     Handle to the tensor to canonicalize
 * @param error_out  Optional pointer to receive error code (may be NULL)
 * @return           Handle to the canonical form tensor, or NULL on failure
 *
 * The returned handle must be freed with bp_tensor_free().
 */
BPTensorHandle bp_canonicalize(BPTensorHandle tensor, BPResult *error_out);

/* -------------------------------------------------------------------------- */
/* Version Information */
/* -------------------------------------------------------------------------- */

/**
 * Get the library version string.
 *
 * @return  Static version string (do NOT free)
 */
const char *bp_version(void);

#ifdef __cplusplus
}
#endif

#endif /* BUTLER_PORTUGAL_H */
