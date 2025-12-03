/**
 * Compile with:
 *   clang -I../../include -L../../target/release -lbutler_portugal example.c -o
 * example
 *
 * Run with:
 *   LD_LIBRARY_PATH=../../target/release ./example     (Linux)
 *   DYLD_LIBRARY_PATH=../../target/release ./example   (macOS)
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "butler_portugal.h"

static void test_version(void) {
  const char *version = bp_version();
  assert(version != NULL);
  assert(strlen(version) > 0);
  printf("Library version: %s\n", version);
}

static void test_index_creation(void) {
  printf("Testing index creation...\n");

  /* Create covariant index */
  BPTensorIndexHandle idx = bp_index_new("mu", 0);
  assert(idx != NULL);

  /* Clone and free */
  BPTensorIndexHandle idx_clone = bp_index_clone(idx);
  assert(idx_clone != NULL);

  bp_index_free(idx_clone);
  bp_index_free(idx);

  /* Create contravariant index */
  BPTensorIndexHandle contra_idx = bp_index_contravariant("nu", 1);
  assert(contra_idx != NULL);
  bp_index_free(contra_idx);

  /* Null safety */
  bp_index_free(NULL);

  printf("  PASSED\n");
}

static void test_symmetry_creation(void) {
  printf("Testing symmetry creation...\n");

  /* Symmetric */
  size_t sym_indices[] = {0, 1};
  BPSymmetryHandle sym = bp_symmetry_symmetric(sym_indices, 2);
  assert(sym != NULL);
  bp_symmetry_free(sym);

  /* Antisymmetric */
  BPSymmetryHandle asym = bp_symmetry_antisymmetric(sym_indices, 2);
  assert(asym != NULL);
  bp_symmetry_free(asym);

  /* Symmetric pairs (Riemann-like) */
  size_t pairs[] = {0, 1, 2, 3};
  BPSymmetryHandle sym_pairs = bp_symmetry_symmetric_pairs(pairs, 2);
  assert(sym_pairs != NULL);
  bp_symmetry_free(sym_pairs);

  /* Cyclic */
  size_t cyc_indices[] = {0, 1, 2};
  BPSymmetryHandle cyc = bp_symmetry_cyclic(cyc_indices, 3);
  assert(cyc != NULL);

  /* Clone */
  BPSymmetryHandle cyc_clone = bp_symmetry_clone(cyc);
  assert(cyc_clone != NULL);
  bp_symmetry_free(cyc_clone);
  bp_symmetry_free(cyc);

  /* Null safety */
  bp_symmetry_free(NULL);

  printf("  PASSED\n");
}

static void test_tensor_creation(void) {
  printf("Testing tensor creation...\n");

  /* Create indices */
  BPTensorIndexHandle idx_a = bp_index_new("a", 0);
  BPTensorIndexHandle idx_b = bp_index_new("b", 1);
  assert(idx_a != NULL && idx_b != NULL);

  /* Create tensor */
  BPTensorIndexHandle indices[] = {idx_a, idx_b};
  BPTensorHandle tensor = bp_tensor_new("T", indices, 2);
  assert(tensor != NULL);

  /* Check properties */
  assert(bp_tensor_rank(tensor) == 2);
  assert(bp_tensor_coefficient(tensor) == 1);
  assert(!bp_tensor_is_zero(tensor));

  /* Get string representation */
  char *str = bp_tensor_to_string(tensor);
  assert(str != NULL);
  printf("  Tensor: %s\n", str);
  bp_string_free(str);

  /* Clone */
  BPTensorHandle tensor_clone = bp_tensor_clone(tensor);
  assert(tensor_clone != NULL);
  bp_tensor_free(tensor_clone);

  /* Cleanup */
  bp_tensor_free(tensor);
  bp_index_free(idx_a);
  bp_index_free(idx_b);

  /* Null safety */
  bp_tensor_free(NULL);
  assert(bp_tensor_rank(NULL) == 0);

  printf("  PASSED\n");
}

static void test_tensor_with_coefficient(void) {
  printf("Testing tensor with coefficient...\n");

  BPTensorIndexHandle idx = bp_index_new("i", 0);
  BPTensorIndexHandle indices[] = {idx};

  BPTensorHandle tensor = bp_tensor_with_coefficient("A", indices, 1, -3);
  assert(tensor != NULL);
  assert(bp_tensor_coefficient(tensor) == -3);

  char *str = bp_tensor_to_string(tensor);
  assert(str != NULL);
  printf("  Tensor: %s\n", str);
  bp_string_free(str);

  bp_tensor_free(tensor);
  bp_index_free(idx);

  printf("  PASSED\n");
}

static void test_symmetry_addition(void) {
  printf("Testing symmetry addition...\n");

  /* Create a 2-index tensor */
  BPTensorIndexHandle idx_a = bp_index_new("a", 0);
  BPTensorIndexHandle idx_b = bp_index_new("b", 1);
  BPTensorIndexHandle indices[] = {idx_a, idx_b};

  BPTensorHandle tensor = bp_tensor_new("S", indices, 2);
  assert(tensor != NULL);

  /* Add symmetric symmetry */
  size_t sym_indices[] = {0, 1};
  BPSymmetryHandle sym = bp_symmetry_symmetric(sym_indices, 2);
  assert(sym != NULL);

  BPResult result = bp_tensor_add_symmetry(tensor, sym);
  assert(result == BP_SUCCESS);

  /* Test null handling */
  result = bp_tensor_add_symmetry(NULL, sym);
  assert(result == BP_NULL_POINTER);

  result = bp_tensor_add_symmetry(tensor, NULL);
  assert(result == BP_NULL_POINTER);

  bp_symmetry_free(sym);
  bp_tensor_free(tensor);
  bp_index_free(idx_a);
  bp_index_free(idx_b);

  printf("  PASSED\n");
}

static void test_canonicalization(void) {
  printf("Testing canonicalization...\n");

  /* Create Riemann-like tensor R_abcd */
  BPTensorIndexHandle idx_a = bp_index_new("a", 0);
  BPTensorIndexHandle idx_b = bp_index_new("b", 1);
  BPTensorIndexHandle idx_c = bp_index_new("c", 2);
  BPTensorIndexHandle idx_d = bp_index_new("d", 3);
  BPTensorIndexHandle indices[] = {idx_a, idx_b, idx_c, idx_d};

  BPTensorHandle tensor = bp_tensor_new("R", indices, 4);
  assert(tensor != NULL);

  /* Add Riemann symmetries */
  /* Antisymmetric in first pair */
  size_t asym1[] = {0, 1};
  BPSymmetryHandle sym1 = bp_symmetry_antisymmetric(asym1, 2);
  bp_tensor_add_symmetry(tensor, sym1);
  bp_symmetry_free(sym1);

  /* Antisymmetric in second pair */
  size_t asym2[] = {2, 3};
  BPSymmetryHandle sym2 = bp_symmetry_antisymmetric(asym2, 2);
  bp_tensor_add_symmetry(tensor, sym2);
  bp_symmetry_free(sym2);

  /* Symmetric exchange of pairs */
  size_t pairs[] = {0, 1, 2, 3};
  BPSymmetryHandle sym3 = bp_symmetry_symmetric_pairs(pairs, 2);
  bp_tensor_add_symmetry(tensor, sym3);
  bp_symmetry_free(sym3);

  /* Print original */
  char *orig_str = bp_tensor_to_string(tensor);
  printf("  Original: %s\n", orig_str);
  bp_string_free(orig_str);

  /* Canonicalize */
  BPResult error = BP_SUCCESS;
  BPTensorHandle canonical = bp_canonicalize(tensor, &error);
  assert(error == BP_SUCCESS);
  assert(canonical != NULL);

  /* Print canonical form */
  char *canon_str = bp_tensor_to_string(canonical);
  printf("  Canonical: %s\n", canon_str);
  bp_string_free(canon_str);

  /* Cleanup */
  bp_tensor_free(canonical);
  bp_tensor_free(tensor);
  bp_index_free(idx_a);
  bp_index_free(idx_b);
  bp_index_free(idx_c);
  bp_index_free(idx_d);

  /* Test null handling */
  BPTensorHandle null_result = bp_canonicalize(NULL, &error);
  assert(null_result == NULL);
  assert(error == BP_NULL_POINTER);

  printf("  PASSED\n");
}

static void test_zero_tensor(void) {
  printf("Testing zero tensor detection...\n");

  /* Create antisymmetric tensor with repeated index (should be zero) */
  BPTensorIndexHandle idx_a1 = bp_index_new("a", 0);
  BPTensorIndexHandle idx_a2 = bp_index_new("a", 1); /* Same name! */
  BPTensorIndexHandle indices[] = {idx_a1, idx_a2};

  BPTensorHandle tensor = bp_tensor_new("A", indices, 2);
  assert(tensor != NULL);

  /* Add antisymmetric symmetry */
  size_t asym_indices[] = {0, 1};
  BPSymmetryHandle asym = bp_symmetry_antisymmetric(asym_indices, 2);
  bp_tensor_add_symmetry(tensor, asym);
  bp_symmetry_free(asym);

  /* Should be zero due to antisymmetry with repeated indices */
  assert(bp_tensor_is_zero(tensor));

  char *str = bp_tensor_to_string(tensor);
  printf("  Zero tensor: %s (is_zero=%d)\n", str, bp_tensor_is_zero(tensor));
  bp_string_free(str);

  bp_tensor_free(tensor);
  bp_index_free(idx_a1);
  bp_index_free(idx_a2);

  printf("  PASSED\n");
}

int main(void) {
  printf("=== Butler-Portugal C FFI Tests ===\n\n");

  test_version();
  test_index_creation();
  test_symmetry_creation();
  test_tensor_creation();
  test_tensor_with_coefficient();
  test_symmetry_addition();
  test_canonicalization();
  test_zero_tensor();

  printf("\n=== All tests passed! ===\n");
  return 0;
}
