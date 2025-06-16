# Butler-Portugal

An implementation of the Butler-Portugal algorithm for tensor canonicalization.

The Butler-Portugal algorithm is a systematic method for bringing tensors into canonical form by applying symmetry operations.

## Usage

To add the crate to your project, run:

```bash
cargo add butler-portugal
```

For example, the Riemann curvature tensor $R_{\mu\nu\rho\sigma}$ satisfies the following symmetries:

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

let canonical = canonicalize(&riemann).unwrap();
```

See [examples/basic_usage.rs](examples/basic_usage.rs) for more examples.

## LICENSE

Released under the MIT License. See the LICENSE file for details.


