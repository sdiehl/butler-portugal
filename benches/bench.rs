//! Criterion benchmarks for complex symmetry canonicalization
//!
//! This benchmark suite tests the performance of the Butler-Portugal algorithm
//! on various complex tensor types commonly encountered in theoretical physics,
//! including general relativity, quantum field theory, and string theory.

use butler_portugal::*;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

/// Benchmark simple symmetric and antisymmetric tensors
fn bench_basic_symmetries(c: &mut Criterion) {
    let mut group = c.benchmark_group("basic_symmetries");

    for &size in &[2, 3, 4, 5] {
        // Symmetric tensor benchmark
        group.bench_with_input(BenchmarkId::new("symmetric", size), &size, |b, &size| {
            b.iter(|| {
                let indices: Vec<TensorIndex> = (0..size)
                    .rev() // Reverse order to force canonicalization work
                    .map(|i| TensorIndex::new(&format!("idx{}", i), i))
                    .collect();

                let mut tensor = Tensor::new("S", indices);
                tensor.add_symmetry(Symmetry::symmetric((0..size).collect()));

                black_box(canonicalize(&tensor).unwrap())
            })
        });

        // Antisymmetric tensor benchmark
        group.bench_with_input(
            BenchmarkId::new("antisymmetric", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let indices: Vec<TensorIndex> = (0..size)
                        .rev() // Reverse order to force canonicalization work
                        .map(|i| TensorIndex::new(&format!("idx{}", i), i))
                        .collect();

                    let mut tensor = Tensor::new("A", indices);
                    tensor.add_symmetry(Symmetry::antisymmetric((0..size).collect()));

                    black_box(canonicalize(&tensor).unwrap())
                })
            },
        );
    }

    group.finish();
}

/// Benchmark Riemann curvature tensor with full symmetries
fn bench_riemann_tensor(c: &mut Criterion) {
    c.bench_function("riemann_tensor_full_symmetries", |b| {
        b.iter(|| {
            // Create Riemann tensor with indices in worst-case order
            let mut riemann = Tensor::new(
                "R",
                vec![
                    TensorIndex::new("sigma", 0),
                    TensorIndex::new("rho", 1),
                    TensorIndex::new("nu", 2),
                    TensorIndex::new("mu", 3),
                ],
            );

            // Add all Riemann tensor symmetries
            riemann.add_symmetry(Symmetry::antisymmetric(vec![0, 1])); // R_[ab]cd
            riemann.add_symmetry(Symmetry::antisymmetric(vec![2, 3])); // R_ab[cd]
            riemann.add_symmetry(Symmetry::symmetric_pairs(vec![(0, 1), (2, 3)])); // R_(ab)(cd)

            black_box(canonicalize(&riemann).unwrap())
        })
    });
}

/// Benchmark Weyl tensor (same symmetries as Riemann)
fn bench_weyl_tensor(c: &mut Criterion) {
    c.bench_function("weyl_tensor_canonicalization", |b| {
        b.iter(|| {
            let mut weyl = Tensor::new(
                "C",
                vec![
                    TensorIndex::new("d", 0),
                    TensorIndex::new("c", 1),
                    TensorIndex::new("b", 2),
                    TensorIndex::new("a", 3),
                ],
            );

            weyl.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));
            weyl.add_symmetry(Symmetry::antisymmetric(vec![2, 3]));
            weyl.add_symmetry(Symmetry::symmetric_pairs(vec![(0, 1), (2, 3)]));

            black_box(canonicalize(&weyl).unwrap())
        })
    });
}

/// Benchmark mixed variance tensors
fn bench_mixed_variance_tensors(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_variance");

    // Mixed Riemann tensor R^μ_ν^ρ_σ
    group.bench_function("mixed_riemann", |b| {
        b.iter(|| {
            let mut mixed_riemann = Tensor::new(
                "R",
                vec![
                    TensorIndex::contravariant("sigma", 0),
                    TensorIndex::covariant("rho", 1),
                    TensorIndex::contravariant("nu", 2),
                    TensorIndex::covariant("mu", 3),
                ],
            );

            mixed_riemann.add_symmetry(Symmetry::antisymmetric(vec![1, 3])); // Lower indices antisymmetric
            mixed_riemann.add_symmetry(Symmetry::antisymmetric(vec![0, 2])); // Upper indices antisymmetric

            black_box(canonicalize(&mixed_riemann).unwrap())
        })
    });

    // Christoffel symbols Γ^μ_νρ
    group.bench_function("christoffel_symbols", |b| {
        b.iter(|| {
            let mut gamma = Tensor::new(
                "Gamma",
                vec![
                    TensorIndex::contravariant("mu", 0),
                    TensorIndex::covariant("sigma", 1),
                    TensorIndex::covariant("nu", 2),
                ],
            );

            gamma.add_symmetry(Symmetry::symmetric(vec![1, 2])); // Symmetric in lower indices

            black_box(canonicalize(&gamma).unwrap())
        })
    });

    group.finish();
}

/// Benchmark complex multi-symmetry tensors
fn bench_complex_multi_symmetry(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex_multi_symmetry");

    // 6-index tensor with multiple different symmetry types
    group.bench_function("six_index_mixed_symmetries", |b| {
        b.iter(|| {
            let mut complex_tensor = Tensor::new(
                "Complex",
                vec![
                    TensorIndex::new("f", 0),
                    TensorIndex::new("e", 1),
                    TensorIndex::new("d", 2),
                    TensorIndex::new("c", 3),
                    TensorIndex::new("b", 4),
                    TensorIndex::new("a", 5),
                ],
            );

            // Multiple symmetry types
            complex_tensor.add_symmetry(Symmetry::symmetric(vec![0, 1])); // Symmetric in first pair
            complex_tensor.add_symmetry(Symmetry::antisymmetric(vec![2, 3])); // Antisymmetric in middle pair
            complex_tensor.add_symmetry(Symmetry::symmetric(vec![4, 5])); // Symmetric in last pair

            black_box(canonicalize(&complex_tensor).unwrap())
        })
    });

    // 8-index tensor (like curvature squared terms)
    group.bench_function("eight_index_curvature_squared", |b| {
        b.iter(|| {
            let mut riemann_squared = Tensor::new(
                "RR",
                vec![
                    TensorIndex::new("h", 0),
                    TensorIndex::new("g", 1),
                    TensorIndex::new("f", 2),
                    TensorIndex::new("e", 3),
                    TensorIndex::new("d", 4),
                    TensorIndex::new("c", 5),
                    TensorIndex::new("b", 6),
                    TensorIndex::new("a", 7),
                ],
            );

            // First Riemann tensor symmetries (indices 0-3)
            riemann_squared.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));
            riemann_squared.add_symmetry(Symmetry::antisymmetric(vec![2, 3]));
            riemann_squared.add_symmetry(Symmetry::symmetric_pairs(vec![(0, 1), (2, 3)]));

            // Second Riemann tensor symmetries (indices 4-7)
            riemann_squared.add_symmetry(Symmetry::antisymmetric(vec![4, 5]));
            riemann_squared.add_symmetry(Symmetry::antisymmetric(vec![6, 7]));
            riemann_squared.add_symmetry(Symmetry::symmetric_pairs(vec![(4, 5), (6, 7)]));

            black_box(canonicalize(&riemann_squared).unwrap())
        })
    });

    group.finish();
}

/// Benchmark string theory and higher-dimensional tensors
fn bench_advanced_physics_tensors(c: &mut Criterion) {
    let mut group = c.benchmark_group("advanced_physics");

    // Non-abelian gauge field strength F^a_μν
    group.bench_function("nonabelian_field_strength", |b| {
        b.iter(|| {
            let mut nonabelian_field = Tensor::new(
                "F",
                vec![
                    TensorIndex::contravariant("a", 0), // Lie algebra index
                    TensorIndex::covariant("nu", 1),
                    TensorIndex::covariant("mu", 2),
                ],
            );

            nonabelian_field.add_symmetry(Symmetry::antisymmetric(vec![1, 2]));

            black_box(canonicalize(&nonabelian_field).unwrap())
        })
    });

    // Bianchi identity structure (5-index tensor)
    group.bench_function("bianchi_identity_structure", |b| {
        b.iter(|| {
            let mut bianchi = Tensor::new(
                "B",
                vec![
                    TensorIndex::new("e", 0),
                    TensorIndex::new("d", 1),
                    TensorIndex::new("c", 2),
                    TensorIndex::new("b", 3),
                    TensorIndex::new("a", 4),
                ],
            );

            // Antisymmetric in last three indices (Bianchi-like)
            bianchi.add_symmetry(Symmetry::antisymmetric(vec![2, 3, 4]));
            // Symmetric in first two
            bianchi.add_symmetry(Symmetry::symmetric(vec![0, 1]));

            black_box(canonicalize(&bianchi).unwrap())
        })
    });

    group.finish();
}

/// Benchmark optimization paths comparison
fn bench_optimization_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("optimization_comparison");

    // Compare standard vs optimized canonicalization for Riemann tensor
    let riemann_setup = || {
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

        riemann
    };

    group.bench_function("standard_canonicalization", |b| {
        b.iter(|| {
            let riemann = riemann_setup();
            black_box(canonicalize(&riemann).unwrap())
        })
    });

    group.bench_function("optimized_canonicalization", |b| {
        b.iter(|| {
            let riemann = riemann_setup();
            black_box(canonicalize_with_optimizations(&riemann).unwrap())
        })
    });

    group.finish();
}

/// Comprehensive benchmark suite
fn bench_comprehensive_suite(c: &mut Criterion) {
    let mut group = c.benchmark_group("comprehensive_suite");

    // Test various common physics tensors all together
    group.bench_function("all_common_physics_tensors", |b| {
        b.iter(|| {
            let test_cases = vec![
                // Metric tensor
                (
                    "g",
                    vec![TensorIndex::new("nu", 0), TensorIndex::new("mu", 1)],
                    vec![Symmetry::symmetric(vec![0, 1])],
                ),
                // Electromagnetic field
                (
                    "F",
                    vec![TensorIndex::new("nu", 0), TensorIndex::new("mu", 1)],
                    vec![Symmetry::antisymmetric(vec![0, 1])],
                ),
                // Ricci tensor
                (
                    "Ric",
                    vec![TensorIndex::new("nu", 0), TensorIndex::new("mu", 1)],
                    vec![Symmetry::symmetric(vec![0, 1])],
                ),
                // Riemann tensor
                (
                    "R",
                    vec![
                        TensorIndex::new("sigma", 0),
                        TensorIndex::new("rho", 1),
                        TensorIndex::new("nu", 2),
                        TensorIndex::new("mu", 3),
                    ],
                    vec![
                        Symmetry::antisymmetric(vec![0, 1]),
                        Symmetry::antisymmetric(vec![2, 3]),
                        Symmetry::symmetric_pairs(vec![(0, 1), (2, 3)]),
                    ],
                ),
            ];

            for (name, indices, symmetries) in test_cases {
                let mut tensor = Tensor::new(name, indices);
                for symmetry in symmetries {
                    tensor.add_symmetry(symmetry);
                }
                black_box(canonicalize(&tensor).unwrap());
            }
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_basic_symmetries,
    bench_riemann_tensor,
    bench_weyl_tensor,
    bench_mixed_variance_tensors,
    bench_complex_multi_symmetry,
    bench_advanced_physics_tensors,
    bench_optimization_comparison,
    bench_comprehensive_suite
);

criterion_main!(benches);
