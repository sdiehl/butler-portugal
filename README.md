# Butler-Portugal

A Rust implementation of the Butler-Portugal algorithm for tensor canonicalization.

The Butler–Portugal algorithm is for bringing tensors with arbitrary symmetries into canonical form. It systematically applies all slot and dummy symmetries by finding a canonical representative in the double coset $D g S$ (where $S$ is the slot symmetry group and $D$ is the dummy index symmetry group), using the Schreier–Sims algorithm to handle large permutation groups and ensure the minimal (canonical) index arrangement is found under all allowed symmetries.

We provide two canonicalization methods:

- **Schreier–Sims (Group-theoretic):**
  Uses the [Schreier–Sims algorithm](https://en.wikipedia.org/wiki/Schreier%E2%80%93Sims_algorithm) to efficiently enumerate all index permutations allowed by the tensor's symmetries, finding the lexicographically minimal representative. This is the default and most general method.

- **Young Symmetrizer (Tableau-based):**
  Projects the tensor onto an irreducible symmetry type using a [Young tableau](https://en.wikipedia.org/wiki/Young_tableau), symmetrizing and antisymmetrizing indices according to the tableau's rows and columns. This is useful for explicit irreducible decomposition.

## Usage

To add the crate to your project, run:

```bash
cargo add butler-portugal
```

For example usage, see the [basic.rs](examples/basic.rs) example.

## Example

The Riemann curvature tensor $R_{\mu\nu\rho\sigma}$ satisfies the following symmetries:

1.  **Antisymmetry in the first two indices:**

    $$R_{\mu\nu\rho\sigma} = -R_{\nu\mu\rho\sigma}$$

2.  **Antisymmetry in the last two indices:**

    $$R_{\mu\nu\rho\sigma} = -R_{\mu\nu\sigma\rho}$$

3.  **Pairwise interchange symmetry:**

    $$R_{\mu\nu\rho\sigma} = R_{\rho\sigma\mu\nu}$$

4.  **First Bianchi Identity (cyclic symmetry on the first three indices):**

    $$R_{\mu\nu\rho\sigma} + R_{\mu\rho\sigma\nu} + R_{\mu\sigma\nu\rho} = 0$$

We can use the crate to canonicalize the Riemann tensor:

```rust
use butler_portugal::*;

let mut riemann = Tensor::new(
    "R",
    vec![
        TensorIndex::new("mu", 0),
        TensorIndex::new("nu", 1),
        TensorIndex::new("rho", 2),
        TensorIndex::new("sigma", 3),
    ],
);

// Antisymmetry in first pair
riemann.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));
// Antisymmetry in second pair
riemann.add_symmetry(Symmetry::antisymmetric(vec![2, 3]));
// Symmetric exchange of pairs
riemann.add_symmetry(Symmetry::symmetric_pairs(vec![(0, 1), (2, 3)]));

let canonical = canonicalize(&riemann);
```

## References

1. Martin-García, J. M. (2008). xPerm: Fast index canonicalization for tensor computer algebra. Computer Physics Communications, 179(8), 597–603.
1. Niehoff, B. E. (2018). Faster tensor canonicalization. Computer Physics Communications, 228, 123-145.
1. Niehoff, B. (2017). Efficient algorithms for tensor canonicalization with general index symmetries. Computer Physics Communications, 220, 1–9.
1. Gonçalves, L. (n.d.). Young Diagrams and Tensors: The Particle Physics Dream Team. Retrieved June 27, 2025, from https://www.math.tecnico.ulisboa.pt/~jnatar/MAGEF-24/trabalhos/Leonor.pdf
1. Kessler, D., Kvinge, H., & Wilson, J. B. (2018). A Frobenius-Schreier-Sims algorithm to decompose associative algebras. Journal of Symbolic Computation, 87, 1–19.
1. Welsh, T. A. (1992). Young tableaux as explicit bases for the irreducible modules of the classical Lie groups and algebras. Journal of Algebra, 148(2), 377–404.
1. Some Notes on Young Tableaux as useful for irreps of su(n). (n.d.). from https://www.physics.mcgill.ca/~keshav/673IV/youngtableaux.pdf

## License

Released under the MIT License. See the [LICENSE](LICENSE) file for details.
