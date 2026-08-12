//! One corpus word as a sequence of symbol ids, and what merging a pair does to
//! the pair counts around it.

/// A word mid-training: the symbols it currently consists of.
#[derive(Debug, Clone, Default)]
pub(crate) struct Word {
    symbols: Vec<u32>,
}

impl Word {
    pub(crate) fn from_symbols(symbols: Vec<u32>) -> Self {
        Self { symbols }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn symbols(&self) -> &[u32] {
        &self.symbols
    }

    /// Every adjacent pair, for the initial count.
    pub(crate) fn pairs(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.symbols.windows(2).map(|w| (w[0], w[1]))
    }

    /// Replace every `(a, b)` with `new_id`, recording how each *other* pair's
    /// count changed.
    ///
    /// The caller subtracts the merged pair's own occurrences; what is reported
    /// here is the surrounding damage — the pairs that stop existing because
    /// their neighbour was consumed, and the pairs that come into existence
    /// against the new symbol.
    ///
    /// The preceding symbol is read from the *rewritten* sequence, not the old
    /// one. That is deliberate and self-correcting: where the previous position
    /// was itself merged, the `-1` cancels a `+1` this same pass just recorded,
    /// which is exactly right — `a b a b` under `(a,b) -> X` ends as `X X`, and
    /// the intermediate `(X,a)` it appears to gain and lose never existed.
    ///
    /// Returns how many occurrences were merged, so the caller can scale the
    /// deltas by the word's corpus frequency.
    pub(crate) fn merge(
        &mut self,
        a: u32,
        b: u32,
        new_id: u32,
        changes: &mut Vec<((u32, u32), i64)>,
    ) -> u64 {
        if self.symbols.len() < 2 {
            return 0;
        }

        let mut merged = 0u64;
        let mut out: Vec<u32> = Vec::with_capacity(self.symbols.len());
        let mut i = 0;

        while i < self.symbols.len() {
            if i + 1 < self.symbols.len() && self.symbols[i] == a && self.symbols[i + 1] == b {
                if let Some(&prev) = out.last() {
                    changes.push(((prev, a), -1));
                    changes.push(((prev, new_id), 1));
                }
                if i + 2 < self.symbols.len() {
                    let next = self.symbols[i + 2];
                    changes.push(((b, next), -1));
                    changes.push(((new_id, next), 1));
                }
                out.push(new_id);
                merged += 1;
                i += 2;
            } else {
                out.push(self.symbols[i]);
                i += 1;
            }
        }

        if merged > 0 {
            self.symbols = out;
        }
        merged
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustc_hash::FxHashMap;

    /// Count every pair in a symbol sequence, the ground truth a merge's
    /// reported deltas have to reproduce.
    fn census(symbols: &[u32]) -> FxHashMap<(u32, u32), i64> {
        let mut counts = FxHashMap::default();
        for w in symbols.windows(2) {
            *counts.entry((w[0], w[1])).or_insert(0) += 1;
        }
        counts
    }

    /// The property the merge bookkeeping exists to satisfy: applying the
    /// reported deltas (plus the merged pair's own occurrences) to the census of
    /// the old sequence yields the census of the new one. Any drift here shows
    /// up as a corrupted priority queue and a silently wrong vocabulary.
    fn check(symbols: &[u32], a: u32, b: u32, new_id: u32) -> Vec<u32> {
        let before = census(symbols);
        let mut word = Word::from_symbols(symbols.to_vec());
        let mut changes = Vec::new();
        let merged = word.merge(a, b, new_id, &mut changes);

        let mut predicted = before;
        *predicted.entry((a, b)).or_insert(0) -= merged as i64;
        for (pair, delta) in changes {
            *predicted.entry(pair).or_insert(0) += delta;
        }
        predicted.retain(|_, count| *count != 0);

        let actual = census(word.symbols());
        assert_eq!(predicted, actual, "sequence {symbols:?} merging ({a},{b})");
        word.symbols().to_vec()
    }

    #[test]
    fn merges_a_single_pair() {
        assert_eq!(check(&[1, 2, 3], 1, 2, 9), vec![9, 3]);
    }

    #[test]
    fn overlapping_runs_merge_left_to_right() {
        // "a a a" under (a,a): the first two join, the third is left alone.
        assert_eq!(check(&[1, 1, 1], 1, 1, 9), vec![9, 1]);
        assert_eq!(check(&[1, 1, 1, 1], 1, 1, 9), vec![9, 9]);
    }

    /// The case the "read the previous symbol from the rewritten sequence"
    /// rule exists for.
    #[test]
    fn adjacent_merges_do_not_invent_a_pair() {
        assert_eq!(check(&[1, 2, 1, 2], 1, 2, 9), vec![9, 9]);
    }

    #[test]
    fn a_pair_that_does_not_occur_changes_nothing() {
        let mut word = Word::from_symbols(vec![1, 2, 3]);
        let mut changes = Vec::new();
        assert_eq!(word.merge(7, 8, 9, &mut changes), 0);
        assert!(changes.is_empty());
        assert_eq!(word.symbols(), &[1, 2, 3]);
    }

    #[test]
    fn a_single_symbol_word_has_no_pairs() {
        let mut word = Word::from_symbols(vec![1]);
        let mut changes = Vec::new();
        assert_eq!(word.merge(1, 1, 9, &mut changes), 0);
        assert!(changes.is_empty());
    }

    #[test]
    fn longer_sequences_stay_consistent() {
        check(&[1, 2, 1, 2, 3, 1, 2], 1, 2, 9);
        check(&[5, 5, 5, 5, 5], 5, 5, 9);
        check(&[1, 2, 3, 1, 2, 3, 1, 2], 2, 3, 9);
    }
}
