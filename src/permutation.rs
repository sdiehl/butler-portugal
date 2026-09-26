//! Permutations in array form: `p[i]` is the image of point `i`.

/// A permutation of `0..n` stored as its image array.
pub type Permutation = Vec<usize>;

/// The identity permutation on `degree` points.
pub fn identity(degree: usize) -> Permutation {
    (0..degree).collect()
}

/// Composes two permutations of the same degree: apply `first`, then `second`.
pub fn compose(first: &[usize], second: &[usize]) -> Permutation {
    first.iter().map(|&i| second[i]).collect()
}

/// The inverse permutation.
pub fn inverse(perm: &[usize]) -> Permutation {
    let mut inv = vec![0; perm.len()];
    for (i, &p) in perm.iter().enumerate() {
        inv[p] = i;
    }
    inv
}

/// Returns true if `perm` fixes every point.
pub fn is_identity(perm: &[usize]) -> bool {
    perm.iter().enumerate().all(|(i, &p)| i == p)
}

/// The sign of a permutation: `1` if even, `-1` if odd.
pub fn parity(perm: &[usize]) -> i32 {
    let mut visited = vec![false; perm.len()];
    let mut sign = 1;
    for start in 0..perm.len() {
        if visited[start] {
            continue;
        }
        let mut len = 0;
        let mut current = start;
        while !visited[current] {
            visited[current] = true;
            current = perm[current];
            len += 1;
        }
        if len % 2 == 0 {
            sign = -sign;
        }
    }
    sign
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compose_applies_first_then_second() {
        assert_eq!(compose(&[1, 0, 2, 3], &[0, 1, 3, 2]), vec![1, 0, 3, 2]);
        assert_eq!(compose(&[1, 0, 2], &[0, 2, 1]), vec![2, 0, 1]);
    }

    #[test]
    fn inverse_round_trips() {
        let p = vec![2, 0, 3, 1];
        assert!(is_identity(&compose(&p, &inverse(&p))));
        assert!(is_identity(&compose(&inverse(&p), &p)));
    }

    #[test]
    fn parity_of_cycles() {
        assert_eq!(parity(&[0, 1, 2]), 1);
        assert_eq!(parity(&[1, 0, 2]), -1);
        assert_eq!(parity(&[2, 1, 0]), -1);
        assert_eq!(parity(&[1, 2, 0]), 1);
        assert_eq!(parity(&[1, 0, 3, 2]), 1);
    }
}
