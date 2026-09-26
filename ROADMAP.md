# Roadmap

## Done

- [x] Parallel Young projection behind the `parallel` feature
- [x] Early exit when the slot group contains minus identity
- [x] Group reuse via `SlotGroup::canonicalize`

## Butler-Portugal algorithm

- [ ] Classify indices as free or dummy
  - [ ] Add `IndexKind` to `TensorIndex`: free, dummy pair id
  - [ ] Pair repeated names with opposite variance automatically
  - [ ] Metric flag per index type allowing raise and lower
  - [ ] Error on a name appearing more than twice
- [ ] Build the dummy group D
  - [ ] Generators: swap two dummy pairs, both slots at once
  - [ ] With metric: swap the two slots of one pair
  - [ ] Antisymmetric metric contributes sign on pair swap
  - [ ] Reuse `schreier_sims` on the signed encoding
- [ ] Add base change to `BSGS`
  - [ ] Conjugation based base change to a given point
  - [ ] Test against orders and membership after change
- [ ] Double coset representative search
  - [ ] Port `double_coset_rep` from `xperm.c` onto `BSGS`
  - [ ] Walk S and D chains level by level
  - [ ] Prune with orbit minimum at each level
  - [ ] Keep all survivors; differing signs means zero
- [ ] Sign and zero bookkeeping
  - [ ] Combine slot sign with dummy sign
  - [ ] Zero test: minus identity in the double coset
- [ ] API
  - [ ] `canonicalize` detects dummies and uses double coset path
  - [ ] Free only tensors keep the fast slot path
  - [ ] Expose `DummyGroup` like `SlotGroup`
- [ ] Tests
  - [ ] Ricci contraction `R^a_bac` canonicalizes to `R_bc`
  - [ ] `R_abcd R^abcd` rename invariance
  - [ ] `R_abcd R^acbd` equals half of `R_abcd R^abcd`
  - [ ] Compare a corpus against xPerm output
