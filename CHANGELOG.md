# Changelog

## 0.3.0 (2026-09-26)

Finish Schreier-Sims. Canonicalization now runs on a signed permutation group.

- Schreier-Sims: stabilizer chain, transversals, sifting, order, enumeration.
- Signed slot group encodes sign as two extra points.
- Lexicographic minimum found by walking the stabilizer chain.
- No more group enumeration, fully symmetric rank 12 canonicalizes quickly.
- Tensors vanishing by conflicting symmetries now return zero.
- Cyclic symmetry gives correct signs and is idempotent.
- Pair exchange no longer sorts within pairs.
- Antisymmetric slot lists in any order now sign correctly.
- Custom symmetries are generators, identity and products recognized.
- Out of range or repeated symmetry slots are errors.
- Young symmetrizer works for multi-row tableaux
- `project_with_tableau` returns a `Vec<Tensor>` of canonical terms.
- Removed `canonicalize_with_optimizations` and `CanonicalizationMethod`.
- Removed the recursive enumeration that overflowed at rank 8.
- Added public `SlotGroup` with `order`, `sign`, `is_zero`, `elements`.
- Added `permutation` module, deleted duplicated compose and parity helpers.
- `Symmetry::Custom` fields renamed to `permutations` and `signs`.
- Display prints `R_mu_nu` instead of `R__mu _nu`.
- `swap_indices` returns `0` when the tensor vanishes.
- Regression tests for every fixed defect and group order checks.
- Optional `parallel` feature: rayon parallelizes Young projection.
- `SlotGroup::canonicalize` reuses one group across many tensors.
- Four ignored relativity tests now run with correct expectations.

## 0.2.0 (2026-05-30)

- C FFI with opaque handles in `src/ffi.rs`.
- C header `include/butler_portugal.h` and C example.
- Build as `cdylib` alongside `rlib`.
- Bump criterion to 0.8 and refresh dependencies.
- Clippy cleanups in benchmarks, example, and tableaux.

## 0.1.3 (2025-06-28)

- Schreier-Sims module with orbits and sift attempt.
- Young tableaux: shapes, standard and semistandard tableaux, RSK.
- `Tensor::project_with_tableau` Young symmetrizer projection.
- `canonicalize_with_optimizations` with method selection.
- Stress test suite for large tensors.
- Canonicalization method comparison tests.
- Clippy configuration denying unwrap and panic.
- Roadmap document.

## 0.1.2 (2025-06-23)

- Canonicalization returns `Result` instead of `Option`.
- Error type with validation helpers.
- Basic example reworked around `Result`.
- rustfmt configuration.
- Bump criterion to 0.6.

## 0.1.1 (2025-06-16)

- Criterion benchmark suite for physics tensors.
- Crate metadata for publishing.

## 0.1.0 (2025-06-16)

- Initial release.
- `Tensor`, `TensorIndex`, and `Symmetry` types.
- Symmetric, antisymmetric, pair, cyclic, and custom symmetries.
- Brute-force canonicalization over generated permutations.
- Riemann tensor example and README.
- CI, dependabot, publish workflow, MIT license.
