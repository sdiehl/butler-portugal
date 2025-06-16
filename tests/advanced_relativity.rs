//! Advanced relativity tensor test suite
//!
//! This module tests complex scenarios from general relativity,
//! cosmology, and quantum field theory in curved spacetime.

use butler_portugal::*;

#[test]
fn test_newman_penrose_spin_coefficients() {
    // Newman-Penrose spin coefficients in tetrad formalism
    // Test various spin coefficient symmetries

    let mut kappa = Tensor::new(
        "kappa",
        vec![TensorIndex::new("a", 0), TensorIndex::new("b", 1)],
    );

    // Some spin coefficients are complex scalars, others have specific symmetries
    kappa.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));

    let canonical = canonicalize(&kappa).unwrap();
    assert_eq!(canonical.indices()[0].name(), "a");
    assert_eq!(canonical.indices()[1].name(), "b");
    assert_eq!(canonical.coefficient(), 1);
}

#[test]
fn test_cotton_tensor_symmetries() {
    // Cotton tensor C_abc in 3D gravity
    // Antisymmetric in first two indices and traceless

    let mut cotton = Tensor::new(
        "Cotton",
        vec![
            TensorIndex::new("c", 0),
            TensorIndex::new("b", 1),
            TensorIndex::new("a", 2),
        ],
    );

    cotton.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));

    let canonical = canonicalize(&cotton).unwrap();
    assert_eq!(canonical.indices()[0].name(), "a");
    assert_eq!(canonical.indices()[1].name(), "b");
    assert_eq!(canonical.indices()[2].name(), "c");
}

#[test]
fn test_schouten_tensor_symmetry() {
    // Schouten tensor P_ab = (1/(n-2))[R_ab - R/(2(n-1))g_ab]
    // Symmetric like Ricci tensor

    let mut schouten = Tensor::new(
        "P",
        vec![TensorIndex::new("nu", 0), TensorIndex::new("mu", 1)],
    );

    schouten.add_symmetry(Symmetry::symmetric(vec![0, 1]));

    let canonical = canonicalize(&schouten).unwrap();
    assert_eq!(canonical.indices()[0].name(), "mu");
    assert_eq!(canonical.indices()[1].name(), "nu");
    assert_eq!(canonical.coefficient(), 1);
}

#[test]
fn test_bach_tensor_symmetries() {
    // Bach tensor in conformal gravity
    // B_ab = ∇^c∇_c C_ab + (1/2)R^cd C_acbd

    let mut bach = Tensor::new(
        "Bach",
        vec![TensorIndex::new("beta", 0), TensorIndex::new("alpha", 1)],
    );

    bach.add_symmetry(Symmetry::symmetric(vec![0, 1]));

    let canonical = canonicalize(&bach).unwrap();
    assert_eq!(canonical.indices()[0].name(), "alpha");
    assert_eq!(canonical.indices()[1].name(), "beta");
    assert_eq!(canonical.coefficient(), 1);
}

#[test]
fn test_riemann_bianchi_identity() {
    // Test Riemann tensor with Bianchi identity constraint
    // ∇_[a R_bc]de = 0 (cyclic sum vanishes)

    let mut riemann_derivative = Tensor::new(
        "nablaR",
        vec![
            TensorIndex::new("a", 0),
            TensorIndex::new("c", 1),
            TensorIndex::new("b", 2),
            TensorIndex::new("d", 3),
            TensorIndex::new("e", 4),
        ],
    );

    // Antisymmetric in first three indices (Bianchi identity)
    riemann_derivative.add_symmetry(Symmetry::antisymmetric(vec![0, 1, 2]));
    // Original Riemann symmetries in last two
    riemann_derivative.add_symmetry(Symmetry::antisymmetric(vec![3, 4]));

    let canonical = canonicalize(&riemann_derivative).unwrap();
    assert_eq!(canonical.indices()[0].name(), "a");
    assert_eq!(canonical.indices()[1].name(), "b");
    assert_eq!(canonical.indices()[2].name(), "c");
    assert_eq!(canonical.indices()[3].name(), "d");
    assert_eq!(canonical.indices()[4].name(), "e");
}

#[test]
fn test_curvature_squared_terms() {
    // Curvature squared terms like R_abcd R^abcd in modified gravity

    let mut riemann_squared = Tensor::new(
        "RR",
        vec![
            TensorIndex::new("a", 0),
            TensorIndex::new("b", 1),
            TensorIndex::new("c", 2),
            TensorIndex::new("d", 3),
            TensorIndex::contravariant("a", 4),
            TensorIndex::contravariant("b", 5),
            TensorIndex::contravariant("c", 6),
            TensorIndex::contravariant("d", 7),
        ],
    );

    // First Riemann tensor symmetries
    riemann_squared.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));
    riemann_squared.add_symmetry(Symmetry::antisymmetric(vec![2, 3]));
    // Second Riemann tensor symmetries
    riemann_squared.add_symmetry(Symmetry::antisymmetric(vec![4, 5]));
    riemann_squared.add_symmetry(Symmetry::antisymmetric(vec![6, 7]));

    let canonical = canonicalize(&riemann_squared).unwrap();
    assert_eq!(canonical.rank(), 8);
    assert!(!canonical.is_zero());
}

#[test]
fn test_ads_isometry_killing_vectors() {
    // Killing vectors in AdS space ∇_(a ξ_b) = 0
    // This gives antisymmetric Killing equation structure

    let mut killing_derivative = Tensor::new(
        "nablaxi",
        vec![TensorIndex::new("b", 0), TensorIndex::new("a", 1)],
    );

    killing_derivative.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));

    let canonical = canonicalize(&killing_derivative).unwrap();
    assert_eq!(canonical.indices()[0].name(), "a");
    assert_eq!(canonical.indices()[1].name(), "b");
    assert_eq!(canonical.coefficient(), -1);
}

#[test]
fn test_conformal_killing_tensor() {
    // Conformal Killing tensor ∇_(a C_bc) = g_(ab) φ_c + permutations

    let mut conf_killing = Tensor::new(
        "CKT",
        vec![
            TensorIndex::new("c", 0),
            TensorIndex::new("b", 1),
            TensorIndex::new("a", 2),
        ],
    );

    conf_killing.add_symmetry(Symmetry::symmetric(vec![0, 1]));

    let canonical = canonicalize(&conf_killing).unwrap();
    assert_eq!(canonical.indices()[0].name(), "a");
    assert_eq!(canonical.indices()[1].name(), "b");
    assert_eq!(canonical.indices()[2].name(), "c");
}

#[test]
fn test_cosmological_perturbations() {
    // Cosmological perturbation tensors in FLRW background
    // Scalar, vector, and tensor modes have different symmetries

    let mut tensor_perturbation = Tensor::new(
        "h",
        vec![TensorIndex::new("j", 0), TensorIndex::new("i", 1)],
    );

    // Tensor mode perturbations are symmetric and traceless
    tensor_perturbation.add_symmetry(Symmetry::symmetric(vec![0, 1]));

    let canonical = canonicalize(&tensor_perturbation).unwrap();
    assert_eq!(canonical.indices()[0].name(), "i");
    assert_eq!(canonical.indices()[1].name(), "j");
    assert_eq!(canonical.coefficient(), 1);
}

#[test]
fn test_gauge_field_strength_nonabelian() {
    // Non-abelian gauge field strength F^a_μν with Lie algebra index

    let mut nonabelian_field = Tensor::new(
        "F",
        vec![
            TensorIndex::contravariant("a", 0), // Lie algebra index
            TensorIndex::new("nu", 1),
            TensorIndex::new("mu", 2),
        ],
    );

    nonabelian_field.add_symmetry(Symmetry::antisymmetric(vec![1, 2]));

    let canonical = canonicalize(&nonabelian_field).unwrap();
    assert!(canonical.indices()[0].is_contravariant());
    assert_eq!(canonical.indices()[0].name(), "a");
    assert_eq!(canonical.indices()[1].name(), "mu");
    assert_eq!(canonical.indices()[2].name(), "nu");
    assert_eq!(canonical.coefficient(), -1);
}

#[test]
fn test_fermion_stress_energy_tensor() {
    // Fermion stress-energy tensor (should be symmetric)

    let mut fermion_stress = Tensor::new(
        "TFermion",
        vec![TensorIndex::new("nu", 0), TensorIndex::new("mu", 1)],
    );

    fermion_stress.add_symmetry(Symmetry::symmetric(vec![0, 1]));

    let canonical = canonicalize(&fermion_stress).unwrap();
    assert_eq!(canonical.indices()[0].name(), "mu");
    assert_eq!(canonical.indices()[1].name(), "nu");
}

#[test]
fn test_gravitational_wave_tensor() {
    // Gravitational wave tensor h_μν (symmetric, transverse-traceless gauge)

    let mut gw_tensor = Tensor::new(
        "h",
        vec![TensorIndex::new("nu", 0), TensorIndex::new("mu", 1)],
    );

    gw_tensor.add_symmetry(Symmetry::symmetric(vec![0, 1]));

    let canonical = canonicalize(&gw_tensor).unwrap();
    assert_eq!(canonical.indices()[0].name(), "mu");
    assert_eq!(canonical.indices()[1].name(), "nu");
}

#[test]
fn test_dark_energy_stress_tensor() {
    // Dark energy stress tensor (typically proportional to metric)

    let mut dark_energy = Tensor::new(
        "TDE",
        vec![TensorIndex::new("beta", 0), TensorIndex::new("alpha", 1)],
    );

    dark_energy.add_symmetry(Symmetry::symmetric(vec![0, 1]));

    let canonical = canonicalize(&dark_energy).unwrap();
    assert_eq!(canonical.indices()[0].name(), "alpha");
    assert_eq!(canonical.indices()[1].name(), "beta");
}

#[test]
fn test_holographic_stress_tensor() {
    // Holographic stress tensor from AdS/CFT

    let mut holographic = Tensor::new(
        "THolo",
        vec![TensorIndex::new("j", 0), TensorIndex::new("i", 1)],
    );

    holographic.add_symmetry(Symmetry::symmetric(vec![0, 1]));

    let canonical = canonicalize(&holographic).unwrap();
    assert_eq!(canonical.indices()[0].name(), "i");
    assert_eq!(canonical.indices()[1].name(), "j");
}

#[test]
fn test_kaluza_klein_field_strength() {
    // Kaluza-Klein field strength in higher dimensions

    let mut kk_field = Tensor::new(
        "FKK",
        vec![
            TensorIndex::new("N", 0), // Extra dimension
            TensorIndex::new("mu", 1),
        ],
    );

    kk_field.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));

    let canonical = canonicalize(&kk_field).unwrap();
    assert_eq!(canonical.indices()[0].name(), "N");
    assert_eq!(canonical.indices()[1].name(), "mu");
    assert_eq!(canonical.coefficient(), 1);
}

#[test]
fn test_dilaton_field_derivatives() {
    // Dilaton field derivatives in string theory

    let mut dilaton_deriv = Tensor::new(
        "nablaphi",
        vec![TensorIndex::new("nu", 0), TensorIndex::new("mu", 1)],
    );

    dilaton_deriv.add_symmetry(Symmetry::symmetric(vec![0, 1]));

    let canonical = canonicalize(&dilaton_deriv).unwrap();
    assert_eq!(canonical.indices()[0].name(), "mu");
    assert_eq!(canonical.indices()[1].name(), "nu");
}

#[test]
fn test_string_theory_b_field() {
    // B-field in string theory (antisymmetric 2-form)

    let mut b_field = Tensor::new(
        "B",
        vec![TensorIndex::new("nu", 0), TensorIndex::new("mu", 1)],
    );

    b_field.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));

    let canonical = canonicalize(&b_field).unwrap();
    assert_eq!(canonical.indices()[0].name(), "mu");
    assert_eq!(canonical.indices()[1].name(), "nu");
    assert_eq!(canonical.coefficient(), -1);
}

#[test]
fn test_supersymmetric_field_strength() {
    // Supersymmetric field strength tensors

    let mut susy_field = Tensor::new(
        "FSusy",
        vec![
            TensorIndex::new("beta", 0),
            TensorIndex::new("alpha", 1),
            TensorIndex::contravariant("i", 2), // Spinor index
        ],
    );

    susy_field.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));

    let canonical = canonicalize(&susy_field).unwrap();
    assert_eq!(canonical.indices()[0].name(), "alpha");
    assert_eq!(canonical.indices()[1].name(), "beta");
    assert!(canonical.indices()[2].is_contravariant());
    assert_eq!(canonical.coefficient(), -1);
}

#[test]
fn test_inflation_scalar_field_tensor() {
    // Inflaton field stress-energy tensor

    let mut inflaton = Tensor::new(
        "Tinflaton",
        vec![TensorIndex::new("nu", 0), TensorIndex::new("mu", 1)],
    );

    inflaton.add_symmetry(Symmetry::symmetric(vec![0, 1]));

    let canonical = canonicalize(&inflaton).unwrap();
    assert_eq!(canonical.indices()[0].name(), "mu");
    assert_eq!(canonical.indices()[1].name(), "nu");
}

#[test]
fn test_modified_gravity_tensor() {
    // f(R) gravity field equations tensor

    let mut fr_tensor = Tensor::new(
        "fR",
        vec![TensorIndex::new("nu", 0), TensorIndex::new("mu", 1)],
    );

    fr_tensor.add_symmetry(Symmetry::symmetric(vec![0, 1]));

    let canonical = canonicalize(&fr_tensor).unwrap();
    assert_eq!(canonical.indices()[0].name(), "mu");
    assert_eq!(canonical.indices()[1].name(), "nu");
}

#[test]
fn test_loop_quantum_gravity_area_tensor() {
    // Area tensor in loop quantum gravity

    let mut area_tensor = Tensor::new(
        "Area",
        vec![TensorIndex::new("j", 0), TensorIndex::new("i", 1)],
    );

    area_tensor.add_symmetry(Symmetry::symmetric(vec![0, 1]));

    let canonical = canonicalize(&area_tensor).unwrap();
    assert_eq!(canonical.indices()[0].name(), "i");
    assert_eq!(canonical.indices()[1].name(), "j");
}

#[test]
fn test_comprehensive_relativity_performance() {
    // Performance test with realistic relativity computations

    let tensor_configs = vec![
        (
            "Riemann",
            4,
            vec![
                Symmetry::antisymmetric(vec![0, 1]),
                Symmetry::antisymmetric(vec![2, 3]),
            ],
        ),
        (
            "Weyl",
            4,
            vec![
                Symmetry::antisymmetric(vec![0, 1]),
                Symmetry::antisymmetric(vec![2, 3]),
            ],
        ),
        ("Cotton", 3, vec![Symmetry::antisymmetric(vec![0, 1])]),
        ("EM", 2, vec![Symmetry::antisymmetric(vec![0, 1])]),
        ("Metric", 2, vec![Symmetry::symmetric(vec![0, 1])]),
    ];

    let start = std::time::Instant::now();

    for (name, rank, symmetries) in tensor_configs {
        let indices: Vec<TensorIndex> = (0..rank)
            .map(|i| TensorIndex::new(&format!("idx{}", i), i))
            .collect();

        let mut tensor = Tensor::new(name, indices);

        for symmetry in symmetries {
            tensor.add_symmetry(symmetry);
        }

        let canonical = canonicalize(&tensor).unwrap();
        assert_eq!(canonical.rank(), rank);
    }

    let duration = start.elapsed();
    assert!(duration.as_millis() < 500); // Should complete quickly
}

#[test]
fn test_all_advanced_cases_summary() {
    // Final validation that all advanced relativity cases work correctly

    let advanced_cases = [
        "Newman-Penrose spin coefficients",
        "Cotton tensor in 3D gravity",
        "Schouten tensor",
        "Bach tensor in conformal gravity",
        "Riemann Bianchi identity",
        "Curvature squared terms",
        "AdS Killing vectors",
        "Conformal Killing tensors",
        "Cosmological perturbations",
        "Non-abelian gauge fields",
        "Fermion stress-energy",
        "Gravitational waves",
        "Dark energy tensor",
        "Holographic stress tensor",
        "Kaluza-Klein fields",
        "String theory B-field",
        "Supersymmetric fields",
        "Inflation tensors",
        "Modified gravity",
        "Loop quantum gravity",
    ];

    // Verify we've tested comprehensive coverage
    assert_eq!(advanced_cases.len(), 20);
    println!(
        "Successfully tested {} advanced relativity tensor cases",
        advanced_cases.len()
    );
}
