# Butler-Portugal

A Rust implementation of the Butler-Portugal algorithm for tensor canonicalization.

The Butler-Portugal algorithm is a systematic method for bringing tensors into canonical form by applying symmetry operations. We use the double coset approach where a tensor with slot symmetries $S$ and dummy symmetries $D$ is canonicalized by finding the minimal representative in the double coset $D \cdot g \cdot S$. We use the Schreier-Sims algorithm for symmetry group generation.

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

1. Niehoff, B. E. (2018). Faster tensor canonicalization. Computer Physics Communications, 228, 123-145.
1. Gonçalves, L. (n.d.). Young Diagrams and Tensors: The Particle Physics Dream Team. Retrieved June 27, 2025, from https://www.math.tecnico.ulisboa.pt/~jnatar/MAGEF-24/trabalhos/Leonor.pdf
1. Martin-García, J. M. (2008). xPerm: Fast index canonicalization for tensor computer algebra. Computer Physics Communications, 179(8), 597–603. https://doi.org/10.1016/j.cpc.2008.04.018
1. Niehoff, B. (2017). Efficient algorithms for tensor canonicalization with general index symmetries. Computer Physics Communications, 220, 1–9. https://doi.org/10.1016/j.cpc.2017.06.017
1. Kessler, D., Kvinge, H., & Wilson, J. B. (2018). A Frobenius-Schreier-Sims algorithm to decompose associative algebras. Journal of Symbolic Computation, 87, 1–19. https://doi.org/10.1016/j.jsc.2017.08.003
1. Welsh, T. A. (1992). Young tableaux as explicit bases for the irreducible modules of the classical Lie groups and algebras. Journal of Algebra, 148(2), 377–404. https://doi.org/10.1016/0021-8693(92)90112-6
1. Rainbird, J., & Craw, A. (2019). Young tableaux, flag varieties, and Grassmannians. The American Mathematical Monthly, 126(7), 601–616. https://doi.org/10.1080/00029890.2019.1624162
1. Particle Data Group. (2024). Young tableaux and tensor product decomposition in SU(N). In Review of Particle Physics. https://pdg.lbl.gov/2024/reviews/young-tableaux.html

## License

Released under the MIT License. See the [LICENSE](LICENSE) file for details.
