# Butler-Portugal

A Rust library for bringing tensors into canonical form.

Given a tensor, its slot symmetries, and its contracted (dummy) indices, the library finds the lexicographically smallest equivalent index arrangement and folds the accumulated sign into the coefficient. Tensors that vanish by symmetry come back with coefficient zero. Slot symmetries are handled as signed permutation groups through the Schreier-Sims algorithm, and dummy renaming through the Butler-Portugal double coset search, so no group is ever enumerated.

It was adapted from the excellent (but bit-rotted) [xPerm](https://www.xact.es/xPerm/index.html) and ported to modern Rust.

## Install

```bash
cargo add butler-portugal
```

## Example

The Ricci contraction $R^a{}_{bac}$ of the Riemann tensor:

```rust
use butler_portugal::*;

let mut r = Tensor::new(
    "R",
    vec![
        TensorIndex::contravariant("a", 0),
        TensorIndex::covariant("c", 1),
        TensorIndex::covariant("a", 2),
        TensorIndex::covariant("b", 3),
    ],
);
r.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));
r.add_symmetry(Symmetry::antisymmetric(vec![2, 3]));
r.add_symmetry(Symmetry::symmetric_pairs(vec![(0, 1), (2, 3)]));

assert_eq!(canonicalize(&r).unwrap().to_string(), "R_b_a_c^a");
```

## C

Build the shared library, then link against it with clang:

```bash
cargo build --release
clang -Iinclude -Ltarget/release -lbutler_portugal app.c -o app
```

Include `butler_portugal.h`; see `examples/c/example.c` for usage.

## Python

Use the `ctypes` wrapper over the C library:

```python
from butler_portugal import Tensor

r = Tensor("R", ["^a", "c", "a", "b"]).antisymmetric(0, 1).antisymmetric(2, 3)
print(r.symmetric_pairs((0, 1), (2, 3)).canonicalize())  # R_b_a_c^a
```

Build with `cargo build --release` and put `python/` on `PYTHONPATH`.

## References

1. Butler, G. (1991). Fundamental Algorithms for Permutation Groups. Lecture Notes in Computer Science 559. Springer.
1. Portugal, R. (1999). Algorithmic simplification of tensor expressions. Journal of Physics A: Mathematical and General, 32(44), 7779.
1. Manssur, L. R., Portugal, R., & Svaiter, B. F. (2002). Group-theoretic approach for symbolic tensor manipulation. International Journal of Modern Physics C, 13(07), 859-879.
1. Martin-Garcia, J. M. (2008). xPerm: Fast index canonicalization for tensor computer algebra. Computer Physics Communications, 179(8), 597-603.
1. Niehoff, B. E. (2018). Faster tensor canonicalization. Computer Physics Communications, 228, 123-145.
1. Holt, D. F., Eick, B., & O'Brien, E. A. (2005). Handbook of Computational Group Theory. Chapman and Hall/CRC.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.md) file for details.
