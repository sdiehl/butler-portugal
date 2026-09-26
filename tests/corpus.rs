//! Checks the double coset canonicalizer against a corpus of reference
//! canonical forms and against a brute-force minimum over the full slot and
//! dummy groups.

#![allow(clippy::unwrap_used, clippy::panic)]

use std::collections::{BTreeMap, HashSet};

use butler_portugal::{canonicalize, Metric, SlotGroup, Symmetry, Tensor, TensorIndex};

struct Case {
    rank: usize,
    types: Vec<(usize, Metric)>,
    gens: Vec<(Vec<usize>, i32)>,
    g: Vec<usize>,
    expect: Option<(i32, Vec<usize>)>,
}

impl Case {
    fn free(&self) -> usize {
        self.rank - 2 * self.types.iter().map(|t| t.0).sum::<usize>()
    }

    fn pair_type(&self, pair: usize) -> usize {
        let mut seen = 0;
        self.types
            .iter()
            .position(|(k, _)| {
                seen += k;
                pair < seen
            })
            .unwrap()
    }

    fn index(&self, label: usize, position: usize) -> TensorIndex {
        let f = self.free();
        if label < f {
            return TensorIndex::covariant(&format!("f{label:02}"), position);
        }
        let pair = (label - f) / 2;
        let name = format!("d{pair:02}");
        let index = if (label - f).is_multiple_of(2) {
            TensorIndex::covariant(&name, position)
        } else {
            TensorIndex::contravariant(&name, position)
        };
        index.of_type(&format!("t{}", self.pair_type(pair)))
    }

    fn label(&self, index: &TensorIndex) -> usize {
        let number: usize = index.name()[1..].parse().unwrap();
        if index.name().starts_with('f') {
            number
        } else {
            self.free() + 2 * number + usize::from(index.is_contravariant())
        }
    }

    fn tensor(&self) -> Tensor {
        let indices = self
            .g
            .iter()
            .enumerate()
            .map(|(i, &l)| self.index(l, i))
            .collect();
        let mut t = Tensor::new("T", indices);
        if !self.gens.is_empty() {
            let (perms, signs) = self.gens.iter().cloned().unzip();
            t.add_symmetry(Symmetry::custom(perms, signs));
        }
        for (i, (_, metric)) in self.types.iter().enumerate() {
            t.set_metric(&format!("t{i}"), *metric);
        }
        t
    }

    fn solve(&self) -> Option<(i32, Vec<usize>)> {
        let c = canonicalize(&self.tensor()).unwrap();
        (c.coefficient() != 0).then(|| {
            (
                c.coefficient(),
                c.indices().iter().map(|i| self.label(i)).collect(),
            )
        })
    }

    fn dummy_elements(&self) -> Vec<(Vec<usize>, i32)> {
        let (f, n) = (self.free(), self.rank);
        let mut gens = Vec::new();
        let mut start = f;
        for (k, metric) in &self.types {
            for p in 0..*k {
                let (a, b) = (start + 2 * p, start + 2 * p + 1);
                let flip = match metric {
                    Metric::Symmetric => Some(1),
                    Metric::Antisymmetric => Some(-1),
                    Metric::Absent => None,
                };
                if let Some(sign) = flip {
                    let mut d: Vec<usize> = (0..n).collect();
                    d.swap(a, b);
                    gens.push((d, sign));
                }
                if p + 1 < *k {
                    let mut d: Vec<usize> = (0..n).collect();
                    d.swap(a, a + 2);
                    d.swap(b, b + 2);
                    gens.push((d, 1));
                }
            }
            start += 2 * k;
        }
        close(n, &gens)
    }

    fn brute_force(&self, slots: &[(Vec<usize>, i32)]) -> Option<(i32, Vec<usize>)> {
        let mut best: BTreeMap<Vec<usize>, HashSet<i32>> = BTreeMap::new();
        for (d, ds) in self.dummy_elements() {
            for (s, ss) in slots {
                let h = s.iter().map(|&i| d[self.g[i]]).collect();
                best.entry(h).or_default().insert(ds * ss);
            }
        }
        let (h, signs) = best.into_iter().next().unwrap();
        (signs.len() == 1).then(|| (*signs.iter().next().unwrap(), h))
    }
}

fn close(n: usize, gens: &[(Vec<usize>, i32)]) -> Vec<(Vec<usize>, i32)> {
    let mut seen: BTreeMap<Vec<usize>, i32> = BTreeMap::from([((0..n).collect(), 1)]);
    let mut frontier: Vec<_> = seen.clone().into_iter().collect();
    while let Some((p, s)) = frontier.pop() {
        for (q, t) in gens {
            let r: Vec<usize> = p.iter().map(|&i| q[i]).collect();
            if !seen.contains_key(&r) {
                seen.insert(r.clone(), s * t);
                frontier.push((r, s * t));
            }
        }
    }
    seen.into_iter().collect()
}

fn metric(word: &str) -> Metric {
    match word {
        "sym" => Metric::Symmetric,
        "anti" => Metric::Antisymmetric,
        "none" => Metric::Absent,
        _ => panic!("unknown metric {word}"),
    }
}

fn corpus() -> Vec<Case> {
    let text = include_str!("data/corpus.txt");
    text.split("\n\n")
        .filter(|block| block.contains("case"))
        .map(|block| {
            let mut case = Case {
                rank: 0,
                types: vec![],
                gens: vec![],
                g: vec![],
                expect: None,
            };
            for line in block.lines() {
                let mut words = line.split_whitespace();
                let key = words.next().unwrap_or("");
                let nums: Vec<i64> = words.clone().filter_map(|w| w.parse().ok()).collect();
                match key {
                    "rank" => case.rank = nums[0] as usize,
                    "pairs" => case
                        .types
                        .push((nums[0] as usize, metric(words.nth(1).unwrap()))),
                    "gen" => {
                        let (sign, perm) = nums.split_last().unwrap();
                        case.gens
                            .push((perm.iter().map(|&x| x as usize).collect(), *sign as i32));
                    }
                    "g" => case.g = nums.iter().map(|&x| x as usize).collect(),
                    "expect" if nums[0] != 0 => {
                        case.expect = Some((
                            nums[0] as i32,
                            nums[1..].iter().map(|&x| x as usize).collect(),
                        ))
                    }
                    _ => {}
                }
            }
            case
        })
        .collect()
}

#[test]
fn matches_reference_corpus() {
    let cases = corpus();
    assert_eq!(cases.len(), 1000);
    for (i, case) in cases.iter().enumerate() {
        assert_eq!(case.solve(), case.expect, "case {i}: g = {:?}", case.g);
    }
}

#[test]
fn matches_brute_force() {
    let mut checked = 0;
    for (i, case) in corpus().iter().enumerate() {
        let slots = SlotGroup::new(&case.tensor()).unwrap();
        let dummies: usize = case
            .types
            .iter()
            .map(|(k, m)| (1..=*k).product::<usize>() << if *m == Metric::Absent { 0 } else { *k })
            .product();
        if slots.is_zero() || slots.order() * dummies > 20_000 {
            continue;
        }
        let elements: Vec<_> = slots
            .elements()
            .into_iter()
            .map(|(p, s)| (p.to_vec(), s))
            .collect();
        assert_eq!(
            case.solve(),
            case.brute_force(&elements),
            "case {i}: g = {:?}",
            case.g
        );
        checked += 1;
    }
    assert!(checked > 400, "only {checked} cases small enough");
}
