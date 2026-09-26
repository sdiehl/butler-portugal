//! Dummy index canonicalization on contractions of the Riemann tensor.

#![allow(clippy::unwrap_used)]

use butler_portugal::{canonicalize, Metric, Symmetry, Tensor, TensorIndex};

fn index(spec: &str, position: usize) -> TensorIndex {
    match spec.strip_prefix('^') {
        Some(name) => TensorIndex::contravariant(name, position),
        None => TensorIndex::covariant(spec, position),
    }
}

fn riemann_symmetries(t: &mut Tensor, offset: usize) {
    let s = |i: usize| i + offset;
    t.add_symmetry(Symmetry::antisymmetric(vec![s(0), s(1)]));
    t.add_symmetry(Symmetry::antisymmetric(vec![s(2), s(3)]));
    t.add_symmetry(Symmetry::symmetric_pairs(vec![(s(0), s(1)), (s(2), s(3))]));
}

fn riemann(specs: [&str; 4]) -> Tensor {
    let mut t = Tensor::new(
        "R",
        specs.iter().enumerate().map(|(i, s)| index(s, i)).collect(),
    );
    riemann_symmetries(&mut t, 0);
    t
}

/// `R_{lower} R^{upper}` as a product of two commuting Riemann factors.
fn riemann_squared(lower: [&str; 4], upper: [&str; 4]) -> Tensor {
    let upper = upper.map(|n| format!("^{n}"));
    Tensor::product(&[
        riemann(lower),
        riemann(upper.each_ref().map(String::as_str)),
    ])
    .unwrap()
}

fn canon(t: &Tensor) -> (i32, String) {
    let c = canonicalize(t).unwrap();
    let mut unit = c.clone();
    unit.set_coefficient(1);
    (c.coefficient(), unit.to_string())
}

#[test]
fn ricci_contractions() {
    let ricci = canon(&riemann(["^a", "b", "a", "c"]));
    assert_eq!(ricci, (1, "R_b_a_c^a".into()));
    assert_eq!(canon(&riemann(["b", "^a", "c", "a"])), ricci);
    assert_eq!(canon(&riemann(["^a", "b", "c", "a"])).0, -1);
    assert_eq!(canon(&riemann(["^a", "b", "c", "a"])).1, ricci.1);
    assert_eq!(canon(&riemann(["c", "a", "b", "^a"])), ricci);
}

#[test]
fn trace_over_antisymmetric_pair_vanishes() {
    assert_eq!(canon(&riemann(["^a", "a", "b", "c"])).0, 0);
    assert_eq!(canon(&riemann(["b", "c", "d", "^d"])).0, 0);
}

#[test]
fn antisymmetric_metric_flips_sign_on_raise_and_lower() {
    let mut up = Tensor::new("T", vec![index("^a", 0), index("a", 1)]);
    let mut down = Tensor::new("T", vec![index("a", 0), index("^a", 1)]);
    up.set_metric("", Metric::Antisymmetric);
    down.set_metric("", Metric::Antisymmetric);
    assert_eq!(canon(&up).0, -canon(&down).0);
    assert_eq!(canon(&up).1, canon(&down).1);
}

#[test]
fn kretschmann_scalar_is_rename_invariant() {
    let x = canon(&riemann_squared(["a", "b", "c", "d"], ["a", "b", "c", "d"]));
    assert_eq!(x.0, 1);
    for names in [
        ["d", "c", "b", "a"],
        ["c", "a", "d", "b"],
        ["b", "d", "a", "c"],
    ] {
        assert_eq!(canon(&riemann_squared(names, names)), x);
    }
    assert_eq!(
        canon(&riemann_squared(["b", "a", "c", "d"], ["a", "b", "c", "d"])),
        (-1, x.1)
    );
}

/// Contracting the first Bianchi identity `R^acbd + R^abdc + R^adcb = 0` with
/// `R_abcd` gives `Y - X + Y = 0`, so `R_abcd R^acbd = X / 2`.
#[test]
fn cross_contraction_is_half_kretschmann() {
    let lower = ["a", "b", "c", "d"];
    let x = canon(&riemann_squared(lower, lower));
    let y = canon(&riemann_squared(lower, ["a", "c", "b", "d"]));
    assert_eq!(y.0, 1);
    assert_ne!(y.1, x.1);
    assert_eq!(
        canon(&riemann_squared(lower, ["a", "b", "d", "c"])),
        (-1, x.1)
    );
    assert_eq!(canon(&riemann_squared(lower, ["a", "d", "c", "b"])), y);
}

#[test]
fn name_used_three_times_is_an_error() {
    let t = Tensor::new("T", vec![index("a", 0), index("^a", 1), index("a", 2)]);
    assert!(canonicalize(&t).is_err());
}

#[test]
fn identical_factors_commute() {
    let x = canon(&riemann_squared(["a", "b", "c", "d"], ["a", "b", "c", "d"]));
    assert_eq!(x, (1, "R_a_b_c_d R^a^b^c^d".into()));
    let swapped = Tensor::product(&[
        riemann(["^a", "^b", "^c", "^d"]),
        riemann(["a", "b", "c", "d"]),
    ]);
    assert_eq!(canon(&swapped.unwrap()), x);
}

#[test]
fn distinct_factors_keep_their_order() {
    let v = |name: &str, i: &str| Tensor::new(name, vec![index(i, 0)]);
    let ab = Tensor::product(&[v("A", "b"), v("B", "a")]).unwrap();
    assert_eq!(canon(&ab), (1, "A_b B_a".into()));
    let aa = Tensor::product(&[v("A", "b"), v("A", "a")]).unwrap();
    assert_eq!(canon(&aa), (1, "A_a A_b".into()));
}

#[test]
fn product_multiplies_coefficients_and_nests() {
    let mut a = Tensor::with_coefficient("A", vec![index("a", 0)], 2);
    a.set_metric("", Metric::Symmetric);
    let b = Tensor::with_coefficient("B", vec![index("^a", 0)], -3);
    let ab = Tensor::product(&[a.clone(), b.clone()]).unwrap();
    let abab = Tensor::product(&[ab.clone(), ab]).unwrap();
    assert_eq!(abab.coefficient(), 36);
    assert_eq!(abab.to_string(), "36A_a B^a A_a B^a");

    let mut anti = b;
    anti.set_metric("", Metric::Antisymmetric);
    assert!(Tensor::product(&[a, anti]).is_err());
}
