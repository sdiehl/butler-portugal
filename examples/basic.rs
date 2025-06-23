//! Basic usage examples for the Butler-Portugal library
//!
//! This example demonstrates how to use the library for common
//! tensor canonicalization tasks in theoretical physics.

use butler_portugal::*;

fn main() -> Result<()> {
    println!("Butler-Portugal Tensor Canonicalization Examples");
    println!("=================================================\n");

    // Example 1: Simple symmetric tensor
    example_symmetric_tensor()?;

    // Example 2: Antisymmetric tensor
    example_antisymmetric_tensor()?;

    // Example 3: Riemann curvature tensor
    example_riemann_tensor()?;

    // Example 4: Electromagnetic field tensor
    example_electromagnetic_tensor()?;

    // Example 5: Mixed variance tensor
    example_mixed_variance_tensor()?;

    // Example 6: Zero tensor from antisymmetry
    example_zero_tensor()?;

    Ok(())
}

fn example_symmetric_tensor() -> Result<()> {
    println!("Example 1: Symmetric Tensor (Metric tensor)");
    println!("-------------------------------------------");

    // Create metric tensor g_μν with indices in reverse order
    let mut g = Tensor::new(
        "g",
        vec![TensorIndex::new("nu", 0), TensorIndex::new("mu", 1)],
    );

    // Add symmetry: g_μν = g_νμ
    g.add_symmetry(Symmetry::symmetric(vec![0, 1]));

    println!("Original tensor: {g}");

    let canonical = canonicalize(&g)?;
    println!("Canonical form: {canonical}");
    println!("Coefficient: {}\n", canonical.coefficient());

    Ok(())
}

fn example_antisymmetric_tensor() -> Result<()> {
    println!("Example 2: Antisymmetric Tensor");
    println!("-------------------------------");

    // Create antisymmetric tensor A_μν with indices in reverse order
    let mut a = Tensor::new(
        "A",
        vec![TensorIndex::new("nu", 0), TensorIndex::new("mu", 1)],
    );

    // Add antisymmetry: A_μν = -A_νμ
    a.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));

    println!("Original tensor: {a}");

    let canonical = canonicalize(&a)?;
    println!("Canonical form: {canonical}");
    println!(
        "Coefficient: {} (sign changed due to antisymmetry)\n",
        canonical.coefficient()
    );

    Ok(())
}

fn example_riemann_tensor() -> Result<()> {
    println!("Example 3: Riemann Curvature Tensor");
    println!("-----------------------------------");

    // Create Riemann tensor R_abcd with mixed-up indices
    let mut riemann = Tensor::new(
        "R",
        vec![
            TensorIndex::new("sigma", 0),
            TensorIndex::new("rho", 1),
            TensorIndex::new("nu", 2),
            TensorIndex::new("mu", 3),
        ],
    );

    // Add Riemann tensor symmetries:
    // R_abcd = -R_bacd (antisymmetric in first pair)
    riemann.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));

    // R_abcd = -R_abdc (antisymmetric in second pair)
    riemann.add_symmetry(Symmetry::antisymmetric(vec![2, 3]));

    // R_abcd = R_cdab (symmetric exchange of pairs)
    riemann.add_symmetry(Symmetry::symmetric_pairs(vec![(0, 1), (2, 3)]));

    println!("Original tensor: {riemann}");

    let canonical = canonicalize(&riemann)?;
    println!("Canonical form: {canonical}");
    println!("Coefficient: {}\n", canonical.coefficient());

    Ok(())
}

fn example_electromagnetic_tensor() -> Result<()> {
    println!("Example 4: Electromagnetic Field Tensor");
    println!("---------------------------------------");

    // Create electromagnetic field tensor F_μν
    let mut f = Tensor::new(
        "F",
        vec![TensorIndex::new("nu", 0), TensorIndex::new("mu", 1)],
    );

    // F_μν = -F_νμ (antisymmetric)
    f.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));

    println!("Original tensor: {f}");

    let canonical = canonicalize(&f)?;
    println!("Canonical form: {canonical}");
    println!("Coefficient: {}\n", canonical.coefficient());

    Ok(())
}

fn example_mixed_variance_tensor() -> Result<()> {
    println!("Example 5: Mixed Variance Tensor");
    println!("--------------------------------");

    // Create mixed Riemann tensor R^μ_ν^ρ_σ
    let mut mixed_riemann = Tensor::new(
        "R",
        vec![
            TensorIndex::contravariant("mu", 0),
            TensorIndex::covariant("sigma", 1),
            TensorIndex::contravariant("rho", 2),
            TensorIndex::covariant("nu", 3),
        ],
    );

    // Add some symmetries
    mixed_riemann.add_symmetry(Symmetry::antisymmetric(vec![1, 3])); // Lower indices antisymmetric

    println!("Original tensor: {mixed_riemann}");

    let canonical = canonicalize(&mixed_riemann)?;
    println!("Canonical form: {canonical}");
    println!("Coefficient: {}\n", canonical.coefficient());

    Ok(())
}

fn example_zero_tensor() -> Result<()> {
    println!("Example 6: Tensor that becomes zero");
    println!("-----------------------------------");

    // Create antisymmetric tensor with repeated indices
    let mut zero_tensor = Tensor::new(
        "A",
        vec![
            TensorIndex::new("mu", 0),
            TensorIndex::new("mu", 1), // Same index repeated
        ],
    );

    // A_μν = -A_νμ, but with μ=ν this gives A_μμ = -A_μμ, so A_μμ = 0
    zero_tensor.add_symmetry(Symmetry::antisymmetric(vec![0, 1]));

    println!("Original tensor: {zero_tensor}");

    let canonical = canonicalize(&zero_tensor)?;
    println!("Canonical form: {canonical}");
    println!(
        "Coefficient: {} (zero due to antisymmetry with repeated indices)\n",
        canonical.coefficient()
    );

    Ok(())
}

// Additional utility function to demonstrate tensor operations
#[allow(dead_code)]
fn demonstrate_tensor_operations() -> Result<()> {
    println!("Tensor Operations Demo");
    println!("---------------------");

    // Create a tensor
    let mut tensor = Tensor::new(
        "T",
        vec![
            TensorIndex::new("a", 0),
            TensorIndex::new("b", 1),
            TensorIndex::new("c", 2),
        ],
    );

    println!("Original tensor: {tensor}");
    println!("Rank: {}", tensor.rank());

    // Add symmetries
    tensor.add_symmetry(Symmetry::symmetric(vec![0, 1]));
    tensor.add_symmetry(Symmetry::antisymmetric(vec![1, 2]));

    println!("Symmetries added: {}", tensor.symmetries().len());
    println!("Symmetries: {:?}", tensor.symmetries());

    // Try manual index swapping
    let sign = tensor.swap_indices(0, 1);
    println!("After swapping indices 0,1: {tensor}");
    println!("Sign change: {sign}");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_examples_run() {
        // Test that all examples run without errors
        assert!(example_symmetric_tensor().is_ok());
        assert!(example_antisymmetric_tensor().is_ok());
        assert!(example_riemann_tensor().is_ok());
        assert!(example_electromagnetic_tensor().is_ok());
        assert!(example_mixed_variance_tensor().is_ok());
        assert!(example_zero_tensor().is_ok());
    }
}
