//! The corpus as sequences of symbol ids, and what merging a pair does to the
//! pair counts around it.

/// Every word in the corpus, laid end to end in one buffer.
///
/// A `Vec<u32>` per word would be millions of separate allocations for what is
/// one contiguous run of data, and because merging only ever shortens a word,
/// each of those allocations would keep its original size for the whole run
/// while the live part of it shrank. Here the wasted space is visible as a
/// single number and can be reclaimed by sliding the words together.
#[derive(Debug, Default)]
pub(crate) struct WordSet {
    symbols: Vec<u32>,
    starts: Vec<u32>,
    lengths: Vec<u32>,
    /// Symbols still in use, as against `symbols.len()` which counts the gaps
    /// merging has opened up.
    live: usize,
}

impl WordSet {
    pub(crate) fn with_capacity(words: usize) -> Self {
        Self {
            symbols: Vec::new(),
            starts: Vec::with_capacity(words),
            lengths: Vec::with_capacity(words),
            live: 0,
        }
    }

    pub(crate) fn push(&mut self, symbols: &[u32]) {
        self.starts.push(self.symbols.len() as u32);
        self.lengths.push(symbols.len() as u32);
        self.symbols.extend_from_slice(symbols);
        self.live += symbols.len();
    }

    pub(crate) fn len(&self) -> usize {
        self.starts.len()
    }

    pub(crate) fn word(&self, index: usize) -> &[u32] {
        let start = self.starts[index] as usize;
        &self.symbols[start..start + self.lengths[index] as usize]
    }

    /// Every adjacent pair of a word, for the initial count.
    pub(crate) fn pairs(&self, index: usize) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.word(index).windows(2).map(|w| (w[0], w[1]))
    }

    /// See [`merge_symbols`]. Rewrites the word in place.
    pub(crate) fn merge(
        &mut self,
        index: usize,
        a: u32,
        b: u32,
        new_id: u32,
        changes: &mut Vec<((u32, u32), i64)>,
    ) -> u64 {
        let start = self.starts[index] as usize;
        let length = self.lengths[index] as usize;
        let (merged, kept) = merge_symbols(
            &mut self.symbols[start..start + length],
            a,
            b,
            new_id,
            changes,
        );
        if merged > 0 {
            self.lengths[index] = kept as u32;
            self.live -= length - kept;
        }
        merged
    }

    /// Whether merging has left more dead space than live symbols.
    pub(crate) fn is_sparse(&self) -> bool {
        self.symbols.len() > 2 * self.live
    }

    /// Slide every word down over the gaps and hand the tail back.
    ///
    /// Words only ever shrink, so each one moves left into space already
    /// vacated and nothing needs a second buffer to copy through.
    pub(crate) fn compact(&mut self) {
        let mut write = 0usize;
        for index in 0..self.starts.len() {
            let start = self.starts[index] as usize;
            let length = self.lengths[index] as usize;
            if write != start {
                self.symbols.copy_within(start..start + length, write);
                self.starts[index] = write as u32;
            }
            write += length;
        }
        self.symbols.truncate(write);
        self.symbols.shrink_to_fit();
    }
}

/// One word on its own, which is how [`merge_symbols`] is exercised directly by
/// the property tests below.
#[cfg(test)]
#[derive(Debug, Clone, Default)]
pub(crate) struct Word {
    symbols: Vec<u32>,
}

#[cfg(test)]
impl Word {
    pub(crate) fn from_symbols(symbols: Vec<u32>) -> Self {
        Self { symbols }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn symbols(&self) -> &[u32] {
        &self.symbols
    }

    /// See [`merge_symbols`].
    pub(crate) fn merge(
        &mut self,
        a: u32,
        b: u32,
        new_id: u32,
        changes: &mut Vec<((u32, u32), i64)>,
    ) -> u64 {
        let (merged, kept) = merge_symbols(&mut self.symbols, a, b, new_id, changes);
        if merged > 0 {
            self.symbols.truncate(kept);
        }
        merged
    }
}

/// Replace every `(a, b)` with `new_id`, recording how each *other* pair's
/// count changed.
///
/// The caller subtracts the merged pair's own occurrences; what is reported
/// here is the surrounding damage — the pairs that stop existing because their
/// neighbour was consumed, and the pairs that come into existence against the
/// new symbol.
///
/// The preceding symbol is read from the *rewritten* sequence, not the old one.
/// That is deliberate and self-correcting: where the previous position was
/// itself merged, the `-1` cancels a `+1` this same pass just recorded, which is
/// exactly right — `a b a b` under `(a,b) -> X` ends as `X X`, and the
/// intermediate `(X,a)` it appears to gain and lose never existed.
///
/// Rewrites in place and returns how many occurrences were merged — so the
/// caller can scale the deltas by the word's corpus frequency — along with how
/// many symbols the word is left with. A merge only ever shortens a word, so the
/// write cursor never overtakes the read cursor and no second buffer is needed.
/// That matters at scale rather than cosmetically: this runs once per affected
/// word per merge, so allocating a replacement here would cost hundreds of
/// millions of allocations over a large run.
pub(crate) fn merge_symbols(
    symbols: &mut [u32],
    a: u32,
    b: u32,
    new_id: u32,
    changes: &mut Vec<((u32, u32), i64)>,
) -> (u64, usize) {
    let length = symbols.len();
    if length < 2 {
        return (0, length);
    }

    let mut merged = 0u64;
    let mut write = 0usize;
    let mut i = 0;

    while i < length {
        if i + 1 < length && symbols[i] == a && symbols[i + 1] == b {
            if write > 0 {
                let prev = symbols[write - 1];
                changes.push(((prev, a), -1));
                changes.push(((prev, new_id), 1));
            }
            if i + 2 < length {
                let next = symbols[i + 2];
                changes.push(((b, next), -1));
                changes.push(((new_id, next), 1));
            }
            symbols[write] = new_id;
            merged += 1;
            i += 2;
        } else {
            symbols[write] = symbols[i];
            i += 1;
        }
        write += 1;
    }

    (merged, write)
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

    fn word_set(words: &[&[u32]]) -> WordSet {
        let mut set = WordSet::with_capacity(words.len());
        for word in words {
            set.push(word);
        }
        set
    }

    #[test]
    fn a_word_set_hands_back_what_was_put_in() {
        let set = word_set(&[&[1, 2, 3], &[4, 5], &[6]]);
        assert_eq!(set.len(), 3);
        assert_eq!(set.word(0), &[1, 2, 3]);
        assert_eq!(set.word(1), &[4, 5]);
        assert_eq!(set.word(2), &[6]);
        assert_eq!(set.pairs(0).collect::<Vec<_>>(), vec![(1, 2), (2, 3)]);
        assert!(set.pairs(2).next().is_none());
    }

    /// Compaction moves words over the gaps merging left behind. Every word has
    /// to survive it byte for byte — a wrong offset here would silently train on
    /// the neighbouring word's symbols.
    #[test]
    fn compaction_preserves_every_word() {
        let mut set = word_set(&[&[1, 2, 1, 2], &[3, 3], &[1, 2, 5], &[7]]);
        let mut changes = Vec::new();
        set.merge(0, 1, 2, 9, &mut changes);
        set.merge(2, 1, 2, 9, &mut changes);

        let before: Vec<Vec<u32>> = (0..set.len()).map(|i| set.word(i).to_vec()).collect();
        assert_eq!(before[0], vec![9, 9]);
        assert_eq!(before[2], vec![9, 5]);

        set.compact();

        let after: Vec<Vec<u32>> = (0..set.len()).map(|i| set.word(i).to_vec()).collect();
        assert_eq!(before, after);
        assert!(!set.is_sparse(), "compaction leaves no dead space");

        // And it still merges correctly against the new offsets.
        set.merge(1, 3, 3, 8, &mut changes);
        assert_eq!(set.word(1), &[8]);
        assert_eq!(set.word(2), &[9, 5]);
    }

    /// One round of merging can at best halve a word, so the buffer only counts
    /// as sparse once repeated rounds have hollowed it out.
    #[test]
    fn sparseness_is_reported_once_most_of_the_buffer_is_dead() {
        let mut set = word_set(&[&[1, 2, 1, 2, 1, 2, 1, 2, 1, 2]]);
        let mut changes = Vec::new();
        assert!(!set.is_sparse());
        set.merge(0, 1, 2, 9, &mut changes);
        assert_eq!(set.word(0), &[9, 9, 9, 9, 9]);
        assert!(!set.is_sparse(), "half dead is not yet worth a pass");

        set.merge(0, 9, 9, 8, &mut changes);
        assert_eq!(set.word(0), &[8, 8, 9]);
        assert!(set.is_sparse());
        set.compact();
        assert_eq!(set.word(0), &[8, 8, 9]);
        assert!(!set.is_sparse());
    }

    #[test]
    fn compaction_is_a_no_op_when_nothing_was_merged() {
        let mut set = word_set(&[&[1, 2], &[3, 4]]);
        assert!(!set.is_sparse());
        set.compact();
        assert_eq!(set.word(0), &[1, 2]);
        assert_eq!(set.word(1), &[3, 4]);
    }

    #[test]
    fn longer_sequences_stay_consistent() {
        check(&[1, 2, 1, 2, 3, 1, 2], 1, 2, 9);
        check(&[5, 5, 5, 5, 5], 5, 5, 9);
        check(&[1, 2, 3, 1, 2, 3, 1, 2], 2, 3, 9);
    }
}
