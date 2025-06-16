# Butler-Portugal

An implementation of the Butler-Portugal algorithm for tensor canonicalization.

The Butler-Portugal algorithm is a systematic method for bringing tensors into canonical form by applying symmetry operations.

## Usage

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

## LICENSE

MIT License
