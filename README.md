# Butler-Portugal

A Rust library for bringing tensors with slot symmetries into canonical form.

Given a tensor such as the Riemann tensor $R_{\mu\nu\rho\sigma}$ and its slot symmetries, the library finds the lexicographically smallest index arrangement reachable by those symmetries and folds the accumulated sign into the coefficient. Tensors that vanish by symmetry, for example an antisymmetric pair carrying the same index name or a slot that is both symmetric and antisymmetric with another, come back with coefficient zero.

The symmetries are treated as a group of signed permutations of the slots. The library builds a base and strong generating set for that group with the [Schreier-Sims algorithm](https://en.wikipedia.org/wiki/Schreier%E2%80%93Sims_algorithm) and walks the resulting stabilizer chain slot by slot to find the minimum, so the group is never enumerated. A fully symmetric rank 12 tensor, whose group has $12!$ elements, canonicalizes in about a millisecond.

Only slot symmetries are handled. Renaming of contracted (dummy) indices, and hence the double coset search of the full Butler-Portugal algorithm, is not implemented.

## Usage

```bash
cargo add butler-portugal
```

For a longer walkthrough see [examples/basic.rs](examples/basic.rs).

## Example

The Riemann tensor is antisymmetric in each pair of slots and symmetric under exchange of the two pairs:

$$R_{\mu\nu\rho\sigma} = -R_{\nu\mu\rho\sigma} = -R_{\mu\nu\sigma\rho} = R_{\rho\sigma\mu\nu}$$

```rust
use butler_portugal::*;

let mut riemann = Tensor::new(
    "R",
    vec![
        TensorIndex::new("sigma", 0),
        TensorIndex::new("rho", 1),
        TensorIndex::new("nu", 2),
        TensorIndex::new("mu", 3),
    ],
);

riemann.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));
riemann.add_symmetry(Symmetry::antisymmetric(vec![2, 3]));
riemann.add_symmetry(Symmetry::symmetric_pairs(vec![(0, 1), (2, 3)]));

let canonical = canonicalize(&riemann).unwrap();
assert_eq!(canonical.to_string(), "R_mu_nu_rho_sigma");
```

Indices are ordered by name, then covariant before contravariant. Symmetries can be declared as `symmetric`, `antisymmetric`, `symmetric_pairs` (exchange of whole pairs, implying nothing about the order inside a pair), `cyclic`, or `custom` (explicit signed generators).

## Group queries

`SlotGroup` exposes the signed symmetry group directly:

```rust
use butler_portugal::*;

let mut r = Tensor::new("R", (0..4).map(|i| TensorIndex::new("a", i)).collect());
r.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));
r.add_symmetry(Symmetry::antisymmetric(vec![2, 3]));
r.add_symmetry(Symmetry::symmetric_pairs(vec![(0, 1), (2, 3)]));

let group = SlotGroup::new(&r).unwrap();
assert_eq!(group.order(), 8);
assert_eq!(group.sign(&[1, 0, 2, 3]), Some(-1));
assert_eq!(group.sign(&[0, 2, 1, 3]), None);
```

Building the group is most of the cost of `canonicalize`, so when many tensors share the same symmetries build it once and call `group.canonicalize(&tensor)` on each.

## Young symmetrizers

`Tensor::project_with_tableau` applies the Young symmetrizer of a standard tableau (symmetrize rows, then antisymmetrize columns) and returns the resulting sum as a list of distinct canonical tensors with integer coefficients. Projecting the Riemann tensor onto the window tableau gives the familiar three-term combination, and projecting it onto a single row gives nothing:

```rust
use butler_portugal::young_tableaux::{Shape, StandardTableau};
use butler_portugal::*;

let mut r = Tensor::new("R", ["a", "b", "c", "d"].iter().enumerate().map(|(i, n)| TensorIndex::new(n, i)).collect());
r.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));
r.add_symmetry(Symmetry::antisymmetric(vec![2, 3]));
r.add_symmetry(Symmetry::symmetric_pairs(vec![(0, 1), (2, 3)]));

let window = StandardTableau::new(Shape(vec![2, 2]), vec![vec![1, 3], vec![2, 4]]).unwrap();
let terms: Vec<String> = r.project_with_tableau(&window).unwrap().iter().map(|t| t.to_string()).collect();
assert_eq!(terms, ["8R_a_b_c_d", "4R_a_c_b_d", "-4R_a_d_b_c"]);
```

The symmetrizer has one term per row and column permutation, so it grows factorially with the tableau. The optional `parallel` feature canonicalizes those terms on a rayon thread pool:

```toml
butler-portugal = { version = "0.3", features = ["parallel"] }
```

## References

1. Portugal, R. (1999). Algorithmic simplification of tensor expressions. Journal of Physics A: Mathematical and General, 32(44), 7779.
1. Manssur, L. R., Portugal, R., & Svaiter, B. F. (2002). Group-theoretic approach for symbolic tensor manipulation. International Journal of Modern Physics C, 13(07), 859-879.
1. Martin-Garcia, J. M. (2008). xPerm: Fast index canonicalization for tensor computer algebra. Computer Physics Communications, 179(8), 597-603.
1. Niehoff, B. E. (2018). Faster tensor canonicalization. Computer Physics Communications, 228, 123-145.
1. Holt, D. F., Eick, B., & O'Brien, E. A. (2005). Handbook of Computational Group Theory. Chapman and Hall/CRC. Chapter 4, Schreier-Sims.

## License

Released under the MIT License. See the [LICENSE](LICENSE) file for details.
