//! Young tableaux, RSK insertion, and Young symmetrizers.

use crate::error::{ButlerPortugalError, Result};
use crate::permutation::{compose, identity, parity, Permutation};
use itertools::Itertools;
use std::fmt;

/// A Young diagram shape as a list of row lengths.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape(pub Vec<usize>);

impl Shape {
    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.0.len()
    }

    /// Number of boxes.
    pub fn size(&self) -> usize {
        self.0.iter().sum()
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.0.iter().max().copied().unwrap_or(0)
    }
}

/// A standard Young tableau: a filling with `1..=n` increasing along rows and columns.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StandardTableau {
    pub shape: Shape,
    pub entries: Vec<Vec<usize>>,
}

impl StandardTableau {
    /// Builds a tableau from row-wise entries, or `None` if it is not standard.
    pub fn new(shape: Shape, entries: Vec<Vec<usize>>) -> Option<Self> {
        if shape.0.len() != entries.len()
            || shape.0.iter().zip(&entries).any(|(&l, row)| l != row.len())
        {
            return None;
        }
        let n = shape.size();
        let mut seen = vec![false; n + 1];
        for (i, row) in entries.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                if val == 0 || val > n || seen[val] {
                    return None;
                }
                seen[val] = true;
                if j > 0 && row[j - 1] >= val {
                    return None;
                }
                if i > 0 && j < entries[i - 1].len() && entries[i - 1][j] >= val {
                    return None;
                }
            }
        }
        Some(Self { shape, entries })
    }

    /// The shape.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Number of boxes.
    pub fn size(&self) -> usize {
        self.shape.size()
    }

    /// Entries read row by row.
    pub fn row_reading_word(&self) -> Vec<usize> {
        self.entries
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect()
    }

    /// Entries read column by column.
    pub fn column_reading_word(&self) -> Vec<usize> {
        (0..self.shape.cols())
            .flat_map(|j| {
                self.entries
                    .iter()
                    .filter_map(move |row| row.get(j).copied())
            })
            .collect()
    }
}

impl fmt::Display for StandardTableau {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in &self.entries {
            for &val in row {
                write!(f, "{val:2} ")?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

/// A semistandard Young tableau: weakly increasing rows, strictly increasing columns.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SemistandardTableau {
    pub shape: Shape,
    pub entries: Vec<Vec<usize>>,
}

impl SemistandardTableau {
    /// Builds a tableau from row-wise entries, or `None` if it is not semistandard.
    pub fn new(shape: Shape, entries: Vec<Vec<usize>>) -> Option<Self> {
        if shape.0.len() != entries.len()
            || shape.0.iter().zip(&entries).any(|(&l, row)| l != row.len())
        {
            return None;
        }
        for (i, row) in entries.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                if j > 0 && row[j - 1] > val {
                    return None;
                }
                if i > 0 && j < entries[i - 1].len() && entries[i - 1][j] >= val {
                    return None;
                }
            }
        }
        Some(Self { shape, entries })
    }

    /// The shape.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
}

/// Robinson-Schensted insertion of a word, returning the insertion and recording tableaux.
pub fn rsk(word: &[usize]) -> (SemistandardTableau, StandardTableau) {
    let mut p_rows: Vec<Vec<usize>> = Vec::new();
    let mut q_rows: Vec<Vec<usize>> = Vec::new();
    for (idx, &x) in word.iter().enumerate() {
        let mut i = 0;
        let mut to_insert = x;
        loop {
            if i == p_rows.len() {
                p_rows.push(vec![to_insert]);
                q_rows.push(vec![idx + 1]);
                break;
            }
            let row = &mut p_rows[i];
            if let Some(j) = row.iter().position(|&y| y > to_insert) {
                std::mem::swap(&mut row[j], &mut to_insert);
                i += 1;
            } else {
                row.push(to_insert);
                q_rows[i].push(idx + 1);
                break;
            }
        }
    }
    let shape = Shape(p_rows.iter().map(|r| r.len()).collect());
    (
        SemistandardTableau {
            shape: shape.clone(),
            entries: p_rows,
        },
        StandardTableau {
            shape,
            entries: q_rows,
        },
    )
}

/// The signed permutations making up the Young symmetrizer of `tableau` acting
/// on `degree` slots: the product of the row symmetrizer and the column
/// antisymmetrizer, with entry `k` of the tableau standing for slot `k - 1`.
/// Fails if the tableau does not have `degree` boxes.
pub fn young_symmetrizer_permutations(
    tableau: &StandardTableau,
    degree: usize,
) -> Result<Vec<(Permutation, i32)>> {
    if tableau.size() != degree {
        return Err(ButlerPortugalError::InvalidSymmetry(format!(
            "tableau has {} boxes but the tensor has rank {degree}",
            tableau.size()
        )));
    }
    let rows: Vec<Vec<usize>> = tableau
        .entries
        .iter()
        .map(|row| row.iter().map(|&v| v - 1).collect())
        .collect();
    let cols: Vec<Vec<usize>> = (0..tableau.shape.cols())
        .map(|j| {
            tableau
                .entries
                .iter()
                .filter_map(|row| row.get(j).map(|&v| v - 1))
                .collect()
        })
        .collect();
    let columns = block_permutations(&cols, degree, true);
    Ok(block_permutations(&rows, degree, false)
        .iter()
        .flat_map(|(r, _)| columns.iter().map(|(c, s)| (compose(r, c), *s)))
        .collect())
}

/// All products of one permutation per block, with the product of parities when `signed`.
fn block_permutations(
    blocks: &[Vec<usize>],
    degree: usize,
    signed: bool,
) -> Vec<(Permutation, i32)> {
    let mut acc = vec![(identity(degree), 1)];
    for block in blocks {
        acc = acc
            .iter()
            .flat_map(|(p, s)| {
                (0..block.len())
                    .permutations(block.len())
                    .map(move |sigma| {
                        let mut q = p.clone();
                        for (i, &j) in sigma.iter().enumerate() {
                            q[block[i]] = block[j];
                        }
                        (q, if signed { s * parity(&sigma) } else { *s })
                    })
            })
            .collect();
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tableau(rows: &[&[usize]]) -> StandardTableau {
        let shape = Shape(rows.iter().map(|r| r.len()).collect());
        StandardTableau::new(shape, rows.iter().map(|r| r.to_vec()).collect()).unwrap()
    }

    #[test]
    fn standard_tableau_validation() {
        assert!(StandardTableau::new(Shape(vec![3, 2]), vec![vec![1, 2, 4], vec![3, 5]]).is_some());
        assert!(StandardTableau::new(Shape(vec![2, 2]), vec![vec![1, 2], vec![2, 3]]).is_none());
        assert!(StandardTableau::new(Shape(vec![2, 2]), vec![vec![1, 3], vec![2, 4]]).is_some());
        assert!(StandardTableau::new(Shape(vec![2, 2]), vec![vec![1, 4], vec![2, 3]]).is_none());
    }

    #[test]
    fn semistandard_tableau_validation() {
        assert!(
            SemistandardTableau::new(Shape(vec![2, 2]), vec![vec![1, 2], vec![2, 3]]).is_some()
        );
        assert!(
            SemistandardTableau::new(Shape(vec![2, 2]), vec![vec![2, 1], vec![2, 3]]).is_none()
        );
    }

    #[test]
    fn rsk_shapes_and_words() {
        let (p, q) = rsk(&[3, 1, 2, 1]);
        assert_eq!(p.entries, vec![vec![1, 1], vec![2], vec![3]]);
        assert_eq!(q.entries, vec![vec![1, 3], vec![2], vec![4]]);
        assert_eq!(q.row_reading_word(), vec![1, 3, 2, 4]);
        assert_eq!(q.column_reading_word(), vec![1, 2, 4, 3]);
    }

    #[test]
    fn symmetrizer_sizes_and_signs() {
        let row = young_symmetrizer_permutations(&tableau(&[&[1, 2, 3]]), 3).unwrap();
        assert_eq!(row.len(), 6);
        assert!(row.iter().all(|(_, s)| *s == 1));

        let column = young_symmetrizer_permutations(&tableau(&[&[1], &[2], &[3]]), 3).unwrap();
        assert_eq!(column.len(), 6);
        assert!(column.iter().all(|(p, s)| *s == parity(p)));

        assert_eq!(
            young_symmetrizer_permutations(&tableau(&[&[1, 2], &[3]]), 3)
                .unwrap()
                .len(),
            4
        );
        assert_eq!(
            young_symmetrizer_permutations(&tableau(&[&[1, 3], &[2, 4]]), 4)
                .unwrap()
                .len(),
            16
        );
        assert!(young_symmetrizer_permutations(&tableau(&[&[1, 2]]), 3).is_err());
    }
}
