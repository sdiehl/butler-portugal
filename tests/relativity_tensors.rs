//! Comprehensive test suite for relativity tensors
//!
//! This module tests the Butler-Portugal library with extensive examples
//! from general relativity, including all major tensor types used in
//! Einstein's field equations and related physics.

use butler_portugal::*;

#[test]
fn test_riemann_curvature_tensor_full_symmetries() {
    // Riemann tensor R_abcd with all symmetries:
    // R_abcd = -R_bacd = -R_abdc = R_badc = R_cdab
    // + Bianchi identity: R_a[bcd] = 0

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

    // Should be in lexicographical order
    assert_eq!(canonical.indices()[0].name(), "mu");
    assert_eq!(canonical.indices()[1].name(), "nu");
    assert_eq!(canonical.indices()[2].name(), "rho");
    assert_eq!(canonical.indices()[3].name(), "sigma");
    assert!(!canonical.is_zero());
}

#[test]
fn test_ricci_tensor_symmetry() {
    // Ricci tensor R_ab = R_ba (symmetric)

    let mut ricci = Tensor::new(
        "Ric",
        vec![TensorIndex::new("nu", 0), TensorIndex::new("mu", 1)],
    );

    ricci.add_symmetry(Symmetry::symmetric(vec![0, 1]));

    let canonical = canonicalize(&ricci).unwrap();

    // Should be alphabetically ordered
    assert_eq!(canonical.indices()[0].name(), "mu");
    assert_eq!(canonical.indices()[1].name(), "nu");
    assert_eq!(canonical.coefficient(), 1);
}

#[test]
fn test_einstein_tensor_symmetry() {
    // Einstein tensor G_ab = R_ab - (1/2)g_ab R (symmetric)

    let mut einstein = Tensor::new(
        "G",
        vec![TensorIndex::new("beta", 0), TensorIndex::new("alpha", 1)],
    );

    einstein.add_symmetry(Symmetry::symmetric(vec![0, 1]));

    let canonical = canonicalize(&einstein).unwrap();

    assert_eq!(canonical.indices()[0].name(), "alpha");
    assert_eq!(canonical.indices()[1].name(), "beta");
    assert_eq!(canonical.coefficient(), 1);
}

#[test]
fn test_stress_energy_tensor_symmetry() {
    // Stress-energy tensor T_ab = T_ba (symmetric)

    let mut stress_energy = Tensor::new(
        "T",
        vec![TensorIndex::new("1", 0), TensorIndex::new("0", 1)],
    );

    stress_energy.add_symmetry(Symmetry::symmetric(vec![0, 1]));

    let canonical = canonicalize(&stress_energy).unwrap();

    // Numerical ordering
    assert_eq!(canonical.indices()[0].name(), "0");
    assert_eq!(canonical.indices()[1].name(), "1");
}

#[test]
fn test_weyl_tensor_symmetries() {
    // Weyl tensor C_abcd has same symmetries as Riemann tensor
    // but is traceless: C_a^a_cd = 0

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

    let canonical = canonicalize(&weyl).unwrap();

    assert_eq!(canonical.indices()[0].name(), "a");
    assert_eq!(canonical.indices()[1].name(), "b");
    assert_eq!(canonical.indices()[2].name(), "c");
    assert_eq!(canonical.indices()[3].name(), "d");
}

#[test]
fn test_electromagnetic_tensor_antisymmetry() {
    // Electromagnetic field tensor F_ab = -F_ba

    let mut em_field = Tensor::new(
        "F",
        vec![TensorIndex::new("nu", 0), TensorIndex::new("mu", 1)],
    );

    em_field.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));

    let canonical = canonicalize(&em_field).unwrap();

    assert_eq!(canonical.indices()[0].name(), "mu");
    assert_eq!(canonical.indices()[1].name(), "nu");
    assert_eq!(canonical.coefficient(), -1); // Sign change from swap
}

#[test]
fn test_electromagnetic_dual_tensor() {
    // Electromagnetic dual tensor *F_ab = -F^cd ε_cdab

    let mut em_dual = Tensor::new(
        "starF",
        vec![TensorIndex::new("beta", 0), TensorIndex::new("alpha", 1)],
    );

    em_dual.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));

    let canonical = canonicalize(&em_dual).unwrap();

    assert_eq!(canonical.indices()[0].name(), "alpha");
    assert_eq!(canonical.indices()[1].name(), "beta");
    assert_eq!(canonical.coefficient(), -1);
}

#[test]
fn test_christoffel_symbols_first_kind() {
    // Christoffel symbols of first kind: Γ_abc = (1/2)(∂_c g_ab + ∂_b g_ac - ∂_a g_bc)
    // Symmetric in first two indices: Γ_abc = Γ_bac

    let mut christoffel1 = Tensor::new(
        "Gamma1",
        vec![
            TensorIndex::new("nu", 0),
            TensorIndex::new("mu", 1),
            TensorIndex::new("rho", 2),
        ],
    );

    christoffel1.add_symmetry(Symmetry::symmetric(vec![0, 1]));

    let canonical = canonicalize(&christoffel1).unwrap();

    assert_eq!(canonical.indices()[0].name(), "mu");
    assert_eq!(canonical.indices()[1].name(), "nu");
    assert_eq!(canonical.indices()[2].name(), "rho");
    assert_eq!(canonical.coefficient(), 1);
}

#[test]
fn test_christoffel_symbols_second_kind() {
    // Christoffel symbols of second kind: Γ^a_bc
    // Symmetric in lower indices: Γ^a_bc = Γ^a_cb

    let mut christoffel2 = Tensor::new(
        "Gamma",
        vec![
            TensorIndex::contravariant("mu", 0),
            TensorIndex::new("rho", 1),
            TensorIndex::new("nu", 2),
        ],
    );

    christoffel2.add_symmetry(Symmetry::symmetric(vec![1, 2]));

    let canonical = canonicalize(&christoffel2).unwrap();

    // Upper index stays in position 0, lower indices ordered
    assert!(canonical.indices()[0].is_contravariant());
    assert_eq!(canonical.indices()[0].name(), "mu");
    assert_eq!(canonical.indices()[1].name(), "nu");
    assert_eq!(canonical.indices()[2].name(), "rho");
}

#[test]
fn test_levi_civita_tensor_antisymmetry() {
    // Levi-Civita tensor ε_abcd (totally antisymmetric)

    let mut levi_civita = Tensor::new(
        "epsilon",
        vec![
            TensorIndex::new("d", 0),
            TensorIndex::new("c", 1),
            TensorIndex::new("b", 2),
            TensorIndex::new("a", 3),
        ],
    );

    // Totally antisymmetric
    levi_civita.add_symmetry(Symmetry::antisymmetric(vec![0, 1, 2, 3]));

    let canonical = canonicalize(&levi_civita).unwrap();

    // Should be in alphabetical order with appropriate sign
    assert_eq!(canonical.indices()[0].name(), "a");
    assert_eq!(canonical.indices()[1].name(), "b");
    assert_eq!(canonical.indices()[2].name(), "c");
    assert_eq!(canonical.indices()[3].name(), "d");
}

#[test]
fn test_metric_tensor_symmetry() {
    // Metric tensor g_ab = g_ba

    let mut metric = Tensor::new(
        "g",
        vec![TensorIndex::new("nu", 0), TensorIndex::new("mu", 1)],
    );

    metric.add_symmetry(Symmetry::symmetric(vec![0, 1]));

    let canonical = canonicalize(&metric).unwrap();

    assert_eq!(canonical.indices()[0].name(), "mu");
    assert_eq!(canonical.indices()[1].name(), "nu");
    assert_eq!(canonical.coefficient(), 1);
}

#[test]
fn test_inverse_metric_tensor() {
    // Inverse metric tensor g^ab = g^ba

    let mut inv_metric = Tensor::new(
        "g",
        vec![
            TensorIndex::contravariant("nu", 0),
            TensorIndex::contravariant("mu", 1),
        ],
    );

    inv_metric.add_symmetry(Symmetry::symmetric(vec![0, 1]));

    let canonical = canonicalize(&inv_metric).unwrap();

    assert!(canonical.indices()[0].is_contravariant());
    assert!(canonical.indices()[1].is_contravariant());
    assert_eq!(canonical.indices()[0].name(), "mu");
    assert_eq!(canonical.indices()[1].name(), "nu");
}

#[test]
fn test_mixed_riemann_tensor() {
    // Mixed Riemann tensor R^a_bcd with mixed indices
    // R^a_bcd = -R^a_cbd (antisymmetric in last two indices)

    let mut mixed_riemann = Tensor::new(
        "R",
        vec![
            TensorIndex::contravariant("mu", 0),
            TensorIndex::new("nu", 1),
            TensorIndex::new("sigma", 2),
            TensorIndex::new("rho", 3),
        ],
    );

    mixed_riemann.add_symmetry(Symmetry::antisymmetric(vec![2, 3]));

    let canonical = canonicalize(&mixed_riemann).unwrap();

    assert!(canonical.indices()[0].is_contravariant());
    assert!(canonical.indices()[1].is_covariant());
    assert!(canonical.indices()[2].is_covariant());
    assert!(canonical.indices()[3].is_covariant());
}

#[test]
fn test_torsion_tensor_antisymmetry() {
    // Torsion tensor T^a_bc = -T^a_cb (antisymmetric in lower indices)

    let mut torsion = Tensor::new(
        "T",
        vec![
            TensorIndex::contravariant("alpha", 0),
            TensorIndex::new("gamma", 1),
            TensorIndex::new("beta", 2),
        ],
    );

    torsion.add_symmetry(Symmetry::antisymmetric(vec![1, 2]));

    let canonical = canonicalize(&torsion).unwrap();

    assert!(canonical.indices()[0].is_contravariant());
    assert_eq!(canonical.indices()[0].name(), "alpha");
    assert_eq!(canonical.indices()[1].name(), "beta");
    assert_eq!(canonical.indices()[2].name(), "gamma");
    assert_eq!(canonical.coefficient(), -1);
}

#[test]
fn test_bianchi_identity_structure() {
    // Test tensor with Bianchi-like structure: T_a[bcd] = 0
    // This means T_abcd + T_acdb + T_adbc = 0

    let mut bianchi = Tensor::new(
        "B",
        vec![
            TensorIndex::new("a", 0),
            TensorIndex::new("d", 1),
            TensorIndex::new("c", 2),
            TensorIndex::new("b", 3),
        ],
    );

    // Antisymmetric in last three indices
    bianchi.add_symmetry(Symmetry::antisymmetric(vec![1, 2, 3]));

    let canonical = canonicalize(&bianchi).unwrap();

    assert_eq!(canonical.indices()[0].name(), "a");
    assert_eq!(canonical.indices()[1].name(), "b");
    assert_eq!(canonical.indices()[2].name(), "c");
    assert_eq!(canonical.indices()[3].name(), "d");
}

#[test]
fn test_zero_tensor_from_repeated_antisymmetric_indices() {
    // Antisymmetric tensor with repeated indices should vanish

    let mut zero_tensor = Tensor::new(
        "A",
        vec![
            TensorIndex::new("mu", 0),
            TensorIndex::new("nu", 1),
            TensorIndex::new("mu", 2), // Repeated index
        ],
    );

    zero_tensor.add_symmetry(Symmetry::antisymmetric(vec![0, 1, 2]));

    let canonical = canonicalize(&zero_tensor).unwrap();
    assert_eq!(canonical.coefficient(), 0);
}

#[test]
fn test_energy_momentum_tensor_conservation() {
    // Energy-momentum tensor with symmetry T_ab = T_ba

    let mut energy_momentum = Tensor::new(
        "Tmunu",
        vec![TensorIndex::new("3", 0), TensorIndex::new("2", 1)],
    );

    energy_momentum.add_symmetry(Symmetry::symmetric(vec![0, 1]));

    let canonical = canonicalize(&energy_momentum).unwrap();

    assert_eq!(canonical.indices()[0].name(), "2");
    assert_eq!(canonical.indices()[1].name(), "3");
}

#[test]
fn test_maxwell_stress_tensor() {
    // Maxwell stress tensor (symmetric)

    let mut maxwell_stress = Tensor::new(
        "MaxT",
        vec![TensorIndex::new("j", 0), TensorIndex::new("i", 1)],
    );

    maxwell_stress.add_symmetry(Symmetry::symmetric(vec![0, 1]));

    let canonical = canonicalize(&maxwell_stress).unwrap();

    assert_eq!(canonical.indices()[0].name(), "i");
    assert_eq!(canonical.indices()[1].name(), "j");
}

#[test]
fn test_curvature_scalar_contraction() {
    // Test tensor representing scalar curvature R = g^ab R_ab
    // This is just a scalar (rank-0 tensor)

    let scalar_curvature = Tensor::new("R", vec![]);
    let canonical = canonicalize(&scalar_curvature).unwrap();

    assert_eq!(canonical.rank(), 0);
    assert_eq!(canonical.name(), "R");
}

#[test]
fn test_complex_mixed_symmetries() {
    // Test tensor with multiple different symmetry types

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

    // Mixed symmetries: symmetric in (0,1), antisymmetric in (2,3), symmetric in (4,5)
    complex_tensor.add_symmetry(Symmetry::symmetric(vec![0, 1]));
    complex_tensor.add_symmetry(Symmetry::antisymmetric(vec![2, 3]));
    complex_tensor.add_symmetry(Symmetry::symmetric(vec![4, 5]));

    let canonical = canonicalize(&complex_tensor).unwrap();

    assert!(!canonical.is_zero());
    assert_eq!(canonical.rank(), 6);
}

#[test]
fn test_spin_connection_antisymmetry() {
    // Spin connection ω^ab_c antisymmetric in first two indices

    let mut spin_connection = Tensor::new(
        "omega",
        vec![
            TensorIndex::contravariant("beta", 0),
            TensorIndex::contravariant("alpha", 1),
            TensorIndex::new("mu", 2),
        ],
    );

    spin_connection.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));

    let canonical = canonicalize(&spin_connection).unwrap();

    assert!(canonical.indices()[0].is_contravariant());
    assert!(canonical.indices()[1].is_contravariant());
    assert!(canonical.indices()[2].is_covariant());
    assert_eq!(canonical.indices()[0].name(), "alpha");
    assert_eq!(canonical.indices()[1].name(), "beta");
    assert_eq!(canonical.coefficient(), -1);
}

#[test]
fn test_field_strength_tensor_yang_mills() {
    // Yang-Mills field strength tensor F^a_μν antisymmetric in spacetime indices

    let mut yang_mills = Tensor::new(
        "F",
        vec![
            TensorIndex::contravariant("a", 0),
            TensorIndex::new("nu", 1),
            TensorIndex::new("mu", 2),
        ],
    );

    yang_mills.add_symmetry(Symmetry::antisymmetric(vec![1, 2]));

    let canonical = canonicalize(&yang_mills).unwrap();

    assert!(canonical.indices()[0].is_contravariant());
    assert_eq!(canonical.indices()[0].name(), "a");
    assert_eq!(canonical.indices()[1].name(), "mu");
    assert_eq!(canonical.indices()[2].name(), "nu");
    assert_eq!(canonical.coefficient(), -1);
}

#[test]
fn test_performance_large_riemann_variations() {
    // Test performance with multiple Riemann tensor variations

    for i in 0..10 {
        let mut riemann = Tensor::new(
            "R",
            vec![
                TensorIndex::new(&format!("i{}", i), 0),
                TensorIndex::new(&format!("j{}", i), 1),
                TensorIndex::new(&format!("k{}", i), 2),
                TensorIndex::new(&format!("l{}", i), 3),
            ],
        );

        riemann.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));
        riemann.add_symmetry(Symmetry::antisymmetric(vec![2, 3]));

        let start = std::time::Instant::now();
        let canonical = canonicalize(&riemann).unwrap();
        let duration = start.elapsed();

        assert!(duration.as_millis() < 100); // Should be fast
        assert!(!canonical.is_zero());
    }
}

#[test]
fn test_all_relativity_tensors_comprehensive() {
    // Final comprehensive test ensuring all common relativity tensors work

    let test_cases = vec![
        (
            "Metric",
            vec!["mu", "nu"],
            vec![Symmetry::symmetric(vec![0, 1])],
        ),
        (
            "Ricci",
            vec!["a", "b"],
            vec![Symmetry::symmetric(vec![0, 1])],
        ),
        (
            "Einstein",
            vec!["alpha", "beta"],
            vec![Symmetry::symmetric(vec![0, 1])],
        ),
        (
            "EM_Field",
            vec!["mu", "nu"],
            vec![Symmetry::antisymmetric(vec![0, 1])],
        ),
        (
            "Riemann",
            vec!["a", "b", "c", "d"],
            vec![
                Symmetry::antisymmetric(vec![0, 1]),
                Symmetry::antisymmetric(vec![2, 3]),
            ],
        ),
    ];

    for (name, indices, symmetries) in test_cases {
        let index_objects: Vec<TensorIndex> = indices
            .iter()
            .enumerate()
            .map(|(i, &name)| TensorIndex::new(name, i))
            .collect();

        let mut tensor = Tensor::new(name, index_objects);

        for symmetry in symmetries {
            tensor.add_symmetry(symmetry);
        }

        let canonical = canonicalize(&tensor).unwrap();

        // Basic sanity checks
        assert_eq!(canonical.name(), name);
        assert_eq!(canonical.rank(), indices.len());

        // Verify canonical form is valid
        if !canonical.is_zero() {
            assert!(canonical.coefficient() != 0);
        }
    }
}
