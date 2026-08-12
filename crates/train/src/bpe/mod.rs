//! The BPE trainer.
//!
//! Byte-pair encoding is trained by repeatedly joining the most frequent
//! adjacent pair of symbols across the corpus, each join adding one token to the
//! vocabulary, until the vocabulary is the requested size.
//!
//! Done literally that is quadratic: every merge would rescan every word to find
//! the next most frequent pair. Two structures avoid it, both taken from the
//! reference implementation because they are the reason it is tractable:
//!
//! * a **priority queue with lazy invalidation** — a merge changes the counts of
//!   pairs already in the queue, and rather than find and update them, a stale
//!   entry is detected when it reaches the top (its stored count disagrees with
//!   the live one) and re-pushed with the corrected count;
//! * **position sets** — for each pair, which words contain it, so a merge only
//!   visits the words it actually affects rather than the whole corpus.

mod word;

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::corpus::WordCounts;
use crate::error::TrainError;
use crate::vocab::{Seeding, TrainedVocab};
use word::WordSet;

/// What makes one candidate merge better than another.
///
/// The choice is not cosmetic: it decides which vocabulary you get, and the
/// right answer depends on how the vocabulary will later be *segmented with*.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Criterion {
    /// Raw corpus frequency of the pair — plain BPE.
    ///
    /// Correct when segmentation replays the merge list in order, because the
    /// thing being optimised (how often this join actually fires) is exactly
    /// what the segmenter will do.
    #[default]
    Frequency,
    /// The exact gain in corpus log-likelihood under a unigram model — the
    /// WordPiece objective.
    ///
    /// Merging `m` occurrences of `(a, b)` rewrites the token distribution:
    /// `a` and `b` each lose `m`, a new symbol gains `m`, and the corpus is `m`
    /// tokens shorter. With `f(x) = x·ln x`, the change in
    /// `Σ count·ln(count/total)` is exactly
    ///
    /// ```text
    /// f(A-m) + f(B-m) + f(m) - f(A) - f(B) - f(N-m) + f(N)
    /// ```
    ///
    /// This is **not** the `count(ab) / (count(a)·count(b))` shorthand usually
    /// quoted for WordPiece. That form is an approximation with a pointwise-
    /// mutual-information rarity bias: it rewards pairs whose halves are
    /// individually rare, so a vocabulary trained on it fills with long rare
    /// strings and drops the common subwords that do the compressive work.
    /// Measured on held-out text it needed **1.6x** the tokens of plain
    /// frequency at a 2000-piece vocabulary. The exact gain above keeps the
    /// frequency weighting the shorthand discards.
    Likelihood,
}

/// `x · ln x`, with `f(0) = 0` — the limit, and the value the derivation needs
/// wherever a symbol is fully consumed.
#[inline]
fn xlogx(x: i64) -> f64 {
    if x <= 0 {
        0.0
    } else {
        let x = x as f64;
        x * x.ln()
    }
}

impl Criterion {
    /// Score a candidate. Higher is better under both criteria.
    ///
    /// `total` is the number of symbol occurrences in the whole corpus, which a
    /// merge shrinks — the likelihood gain is defined against it, so it cannot
    /// be dropped as a constant.
    fn score(self, pair: (u32, u32), count: i64, symbols: &[i64], total: i64) -> f64 {
        match self {
            Criterion::Frequency => count as f64,
            Criterion::Likelihood => {
                let (a, b) = pair;
                let left = symbols[a as usize];
                if a == b {
                    // A pair of one symbol with itself consumes *two* of it per
                    // occurrence, so it loses `2m` rather than `m` and there is
                    // no separate second term.
                    xlogx(left - 2 * count) + xlogx(count) - xlogx(left) - xlogx(total - count)
                        + xlogx(total)
                } else {
                    let right = symbols[b as usize];
                    xlogx(left - count) + xlogx(right - count) + xlogx(count)
                        - xlogx(left)
                        - xlogx(right)
                        - xlogx(total - count)
                        + xlogx(total)
                }
            }
        }
    }
}

/// A candidate merge sitting in the priority queue.
struct Candidate {
    pair: (u32, u32),
    /// The pair's occurrence count when this entry was pushed. Kept alongside
    /// the score so staleness is decided on an exact integer rather than on a
    /// float comparison.
    count: u64,
    score: f64,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Highest score wins. On a tie the *lower* pair wins, so the comparison
        // is reversed — the tie-break has to be total and deterministic or two
        // runs over the same corpus produce different vocabularies.
        //
        // `total_cmp` rather than `partial_cmp`: scores are finite by
        // construction, but a total order is what `Ord` promises and a NaN
        // slipping in would otherwise corrupt the heap rather than sort badly.
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.pair.cmp(&self.pair))
    }
}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Configuration for [`BpeTrainer`].
#[derive(Clone)]
pub struct BpeTrainerBuilder {
    vocab_size: usize,
    min_frequency: u64,
    max_token_length: Option<usize>,
    specials: Vec<String>,
    initial_alphabet: Vec<Vec<u8>>,
    seeding: Seeding,
    continuing_subword_prefix: Option<Vec<u8>>,
    end_of_word_suffix: Option<Vec<u8>>,
    criterion: Criterion,
}

impl Default for BpeTrainerBuilder {
    fn default() -> Self {
        Self {
            vocab_size: 30_000,
            min_frequency: 0,
            max_token_length: None,
            specials: Vec::new(),
            initial_alphabet: Vec::new(),
            seeding: Seeding::Bytes,
            continuing_subword_prefix: None,
            end_of_word_suffix: None,
            criterion: Criterion::Frequency,
        }
    }
}

impl BpeTrainerBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Total pieces to train, specials excluded — they are numbered above the
    /// pieces, so asking for 32000 gives 32000 pieces however many specials are
    /// declared.
    #[must_use]
    pub fn vocab_size(mut self, size: usize) -> Self {
        self.vocab_size = size;
        self
    }

    /// A pair occurring fewer than this many times is never merged.
    #[must_use]
    pub fn min_frequency(mut self, frequency: u64) -> Self {
        self.min_frequency = frequency;
        self
    }

    /// Refuse merges that would produce a piece longer than this many bytes.
    #[must_use]
    pub fn max_token_length(mut self, bytes: usize) -> Self {
        self.max_token_length = Some(bytes);
        self
    }

    #[must_use]
    pub fn special_tokens<I, S>(mut self, tokens: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.specials = tokens.into_iter().map(Into::into).collect();
        self
    }

    /// Seed symbols to include whether or not the corpus contains them.
    ///
    /// Under [`Seeding::Bytes`] all 256 byte values are seeded regardless, so
    /// this is for the [`Seeding::Chars`] shape, where a character absent from
    /// the corpus would otherwise be unspellable.
    #[must_use]
    pub fn initial_alphabet<I, B>(mut self, symbols: I) -> Self
    where
        I: IntoIterator<Item = B>,
        B: Into<Vec<u8>>,
    {
        self.initial_alphabet = symbols.into_iter().map(Into::into).collect();
        self
    }

    #[must_use]
    pub fn seeding(mut self, seeding: Seeding) -> Self {
        self.seeding = seeding;
        self
    }

    /// Mark every non-initial symbol of a word with this prefix — WordPiece's
    /// `##`.
    ///
    /// It changes what the vocabulary *is*, not just how it prints: `a` and
    /// `##a` become separate tokens, so a word start and a continuation are
    /// distinguishable and the segmenter can tell where words begin.
    #[must_use]
    pub fn continuing_subword_prefix(mut self, prefix: impl Into<Vec<u8>>) -> Self {
        self.continuing_subword_prefix = Some(prefix.into());
        self
    }

    /// Mark the final symbol of a word with this suffix — the classic `</w>`.
    #[must_use]
    pub fn end_of_word_suffix(mut self, suffix: impl Into<Vec<u8>>) -> Self {
        self.end_of_word_suffix = Some(suffix.into());
        self
    }

    /// What makes one candidate merge better than another. See [`Criterion`];
    /// the default is plain BPE frequency.
    #[must_use]
    pub fn criterion(mut self, criterion: Criterion) -> Self {
        self.criterion = criterion;
        self
    }

    pub fn build(self) -> BpeTrainer {
        BpeTrainer {
            vocab_size: self.vocab_size,
            min_frequency: self.min_frequency,
            max_token_length: self.max_token_length,
            specials: self.specials,
            initial_alphabet: self.initial_alphabet,
            seeding: self.seeding,
            continuing_subword_prefix: self.continuing_subword_prefix,
            end_of_word_suffix: self.end_of_word_suffix,
            criterion: self.criterion,
        }
    }
}

/// Trains a byte-pair vocabulary from [`WordCounts`].
pub struct BpeTrainer {
    vocab_size: usize,
    min_frequency: u64,
    max_token_length: Option<usize>,
    specials: Vec<String>,
    initial_alphabet: Vec<Vec<u8>>,
    seeding: Seeding,
    continuing_subword_prefix: Option<Vec<u8>>,
    end_of_word_suffix: Option<Vec<u8>>,
    criterion: Criterion,
}

impl BpeTrainer {
    pub fn builder() -> BpeTrainerBuilder {
        BpeTrainerBuilder::new()
    }

    /// Train a vocabulary.
    ///
    /// # Errors
    /// [`TrainError::EmptyCorpus`] if no words were fed, and
    /// [`TrainError::VocabTooSmall`] if the requested size cannot even hold the
    /// seed alphabet — a vocabulary that cannot spell its own corpus is not a
    /// smaller vocabulary, it is a broken one.
    pub fn train(&self, counts: &WordCounts) -> Result<TrainedVocab, TrainError> {
        if counts.is_empty() {
            return Err(TrainError::EmptyCorpus);
        }

        let Seeded {
            mut pieces,
            mut piece_ids,
            alphabet_len,
            mut words,
            frequencies,
        } = self.seed(counts);

        if self.vocab_size < alphabet_len {
            return Err(TrainError::VocabTooSmall {
                requested: self.vocab_size,
                alphabet: alphabet_len,
            });
        }

        let mut pairs = count_pairs(&words, &frequencies);

        // How often each symbol occurs, which the likelihood criterion divides
        // by. Maintained through every merge rather than recomputed, since a
        // merge changes only the two symbols it consumed and the one it made.
        let mut symbol_counts: Vec<i64> = vec![0; pieces.len()];
        for (index, &frequency) in frequencies.iter().enumerate() {
            let frequency = frequency as i64;
            for &symbol in words.word(index) {
                symbol_counts[symbol as usize] += frequency;
            }
        }

        // Total symbol occurrences, which every merge shrinks by the number of
        // occurrences it rewrote. The likelihood gain is defined against it.
        let mut total: i64 = symbol_counts.iter().sum();

        let criterion = self.criterion;

        let mut queue: BinaryHeap<Candidate> = BinaryHeap::with_capacity(pairs.len());
        for (&pair, state) in pairs.iter_mut() {
            if state.count > 0 {
                let score = criterion.score(pair, state.count, &symbol_counts, total);
                queue.push(Candidate {
                    pair,
                    count: state.count as u64,
                    score,
                });
                state.queued = score;
            }
        }

        let max_len = self.max_token_length.unwrap_or(usize::MAX);
        let mut merges: Vec<(u32, u32)> = Vec::new();
        let mut changes: Vec<((u32, u32), i64)> = Vec::new();
        // Pairs one merge disturbed. Their flags live on the pair itself, so
        // this only has to remember which to revisit. Reused across merges.
        let mut touched: Vec<(u32, u32)> = Vec::new();

        while pieces.len() < self.vocab_size {
            let Some(mut top) = queue.pop() else {
                // Nothing left to merge: the corpus has no repeated pair. The
                // vocabulary is smaller than asked for, which is the honest
                // outcome rather than padding it with invented tokens.
                break;
            };

            // Lazy invalidation: an entry is stale if the corpus no longer
            // agrees with what it was pushed with, and it is then corrected and
            // allowed to compete again rather than searched for and updated in
            // place.
            //
            // Judged on the *score*, not on the pair count alone. Under
            // `Criterion::Frequency` those are the same question, but under
            // `Criterion::Likelihood` a merge somewhere else can consume
            // occurrences of `a` and change this pair's score without touching
            // this pair's count — so checking the count would let a stale
            // ordering through. Recomputing from live counts is exact and
            // deterministic: unchanged inputs give a bit-identical score.
            let live = pairs.get(&top.pair).map_or(0, |state| state.count);
            let live_score = criterion.score(top.pair, live, &symbol_counts, total);
            if live != top.count as i64 || live_score != top.score {
                if live > 0 {
                    top.count = live as u64;
                    top.score = live_score;
                    if let Some(state) = pairs.get_mut(&top.pair) {
                        state.queued = live_score;
                    }
                    queue.push(top);
                } else if let Some(state) = pairs.get_mut(&top.pair) {
                    state.queued = f64::NEG_INFINITY;
                }
                continue;
            }

            if top.count < 1 || top.count < self.min_frequency {
                break;
            }

            let (left, right) = top.pair;
            let mut piece = pieces[left as usize].clone();
            // The right operand carries the continuation prefix wherever it is
            // a non-initial symbol, and the joined piece must not keep it in the
            // middle: `un` + `##able` is `unable`, not `un##able`.
            piece.extend_from_slice(strip_prefix(
                &pieces[right as usize],
                self.continuing_subword_prefix.as_deref(),
            ));

            if piece.len() > max_len {
                // Too long to keep, and it must not be reconsidered: zero its
                // count so the lazy check cannot resurrect it. The pair still
                // occurs in the corpus though, so it is flagged rather than
                // dropped — its word list has to survive, since a zero count
                // otherwise means "occurs nowhere" and licenses reclaiming it.
                if let Some(state) = pairs.get_mut(&top.pair) {
                    state.count = 0;
                    state.queued = f64::NEG_INFINITY;
                    state.blocked = true;
                }
                continue;
            }

            let new_id = pieces.len() as u32;
            pieces.push(piece.clone());
            piece_ids.insert(piece, new_id);
            symbol_counts.push(0);
            merges.push(top.pair);

            // The merged pair is gone from every word it occurred in, and the
            // list of those words is taken rather than borrowed — the merge
            // writes new positions back into the map as it goes.
            let words_with = pairs
                .remove(&top.pair)
                .map(|state| state.words)
                .unwrap_or_default();

            // Occurrences actually rewritten, weighted by word frequency. Each
            // one consumes a `left` and a `right` and creates a `new_id`.
            //
            // This is not always `top.count`, and the gap is not a bug: a pair
            // of a symbol with itself overlaps, so `a a a` counts `(a,a)` twice
            // and only one of the two can be rewritten.
            let mut rewritten: i64 = 0;
            for &index in words_with.as_slice() {
                changes.clear();
                let merged = words.merge(index as usize, left, right, new_id, &mut changes);
                if merged == 0 {
                    continue;
                }
                let frequency = frequencies[index as usize] as i64;
                rewritten += merged as i64 * frequency;
                for &(pair, delta) in &changes {
                    let state = pairs.entry(pair).or_default();
                    state.count += delta * frequency;
                    // Gained by any word here means the pair must compete again;
                    // touched at all means its count may have reached zero.
                    if !state.touched {
                        state.touched = true;
                        touched.push(pair);
                    }
                    if delta > 0 {
                        state.gained = true;
                        state.words.note(index);
                    }
                }
            }
            drop(words_with);

            // Applied to both operands unconditionally, which is also right when
            // they are the same symbol: merging `a a` consumes two `a`s, and
            // subtracting twice from the one entry is exactly that.
            symbol_counts[left as usize] -= rewritten;
            symbol_counts[right as usize] -= rewritten;
            symbol_counts[new_id as usize] = rewritten;
            // Each rewritten occurrence turned two symbols into one.
            total -= rewritten;

            for &pair in &touched {
                let Some(state) = pairs.get_mut(&pair) else {
                    continue;
                };
                let (count, gained) = (state.count, state.gained);
                state.touched = false;
                state.gained = false;
                if count <= 0 && !state.blocked {
                    // No occurrence left anywhere, so nothing needs to remember
                    // where it used to be. Should it ever come back, every
                    // occurrence will be a new one and recorded as such.
                    pairs.remove(&pair);
                } else if count > 0 && gained {
                    let score = criterion.score(pair, count, &symbol_counts, total);
                    // Only if this beats what the pair already has waiting. A
                    // lower-scored duplicate would be popped, found stale,
                    // corrected and pushed again — it decides nothing, and
                    // pushing one per gained pair per merge is what makes the
                    // queue grow without bound over a long run.
                    if score > state.queued {
                        state.queued = score;
                        queue.push(Candidate {
                            pair,
                            count: count as u64,
                            score,
                        });
                    }
                }
            }
            touched.clear();

            // Stale entries are only discovered by popping them, so over a long
            // run the queue accumulates far more of them than there are pairs.
            // Rebuilding from the live pairs discards every one at once. What
            // comes out is what the initial fill would have produced — exactly
            // one entry per live pair at its true score — so the merge order is
            // unchanged; only the entries that would have been popped, found
            // stale and corrected are gone.
            // Merging leaves the corpus full of holes — every join frees a slot
            // that nothing reuses — and the buffer keeps its full length until
            // the words are slid together over them.
            if words.is_sparse() {
                words.compact();
            }

            if queue.len() > 2 * pairs.len().max(1024) {
                let mut fresh: BinaryHeap<Candidate> = BinaryHeap::with_capacity(pairs.len());
                for (&pair, state) in pairs.iter_mut() {
                    if state.count > 0 {
                        let score = criterion.score(pair, state.count, &symbol_counts, total);
                        fresh.push(Candidate {
                            pair,
                            count: state.count as u64,
                            score,
                        });
                        state.queued = score;
                    } else {
                        state.queued = f64::NEG_INFINITY;
                    }
                }
                queue = fresh;
            }
        }

        Ok(TrainedVocab::new(
            pieces,
            alphabet_len,
            merges,
            self.specials.clone(),
            self.seeding,
            counts.recipe().cloned(),
        ))
    }

    /// The seed pieces and the corpus expressed in them.
    ///
    /// One pass rather than two, because with a continuation prefix the
    /// alphabet is not knowable before the words are read: a symbol's spelling
    /// depends on *where in a word it sits* (`a` at the start, `##a` after it),
    /// so the decorated variants are interned as the words that need them are
    /// encoded.
    ///
    /// Under [`Seeding::Bytes`] the base alphabet is all 256 byte values, always
    /// and whether the corpus uses them or not: a byte vocabulary that omits an
    /// unseen byte cannot encode text containing it, and byte seeding exists
    /// precisely so that nothing is unspellable.
    fn seed(&self, counts: &WordCounts) -> Seeded {
        let mut pieces: Vec<Vec<u8>> = Vec::new();
        let mut ids: FxHashMap<Vec<u8>, u32> = FxHashMap::default();

        match self.seeding {
            Seeding::Bytes => {
                for byte in 0..=u8::MAX {
                    intern(vec![byte], &mut pieces, &mut ids);
                }
            }
            Seeding::Chars => {
                // Sorted, so the ids a corpus produces do not depend on hash
                // iteration order.
                let mut seen: Vec<Vec<u8>> = counts
                    .iter()
                    .flat_map(|(word, _)| {
                        self.seeding
                            .units(word)
                            .into_iter()
                            .map(<[u8]>::to_vec)
                            .collect::<Vec<_>>()
                    })
                    .collect::<FxHashSet<_>>()
                    .into_iter()
                    .collect();
                seen.sort();
                for unit in seen {
                    intern(unit, &mut pieces, &mut ids);
                }
            }
        }

        for symbol in &self.initial_alphabet {
            intern(symbol.clone(), &mut pieces, &mut ids);
        }

        // Words in a deterministic order, so the decorated variants are interned
        // in the same order on every run and the ids do not depend on hashing.
        let mut corpus: Vec<(&[u8], u64)> = counts.iter().collect();
        corpus.sort_unstable();

        let mut words = WordSet::with_capacity(corpus.len());
        let mut frequencies = Vec::with_capacity(corpus.len());
        let mut symbols: Vec<u32> = Vec::new();
        for (word, frequency) in corpus {
            let units = self.seeding.units(word);
            let last = units.len().saturating_sub(1);
            symbols.clear();
            symbols.extend(units.into_iter().enumerate().map(|(i, unit)| {
                let decorated = self.decorate(unit, i == 0, i == last);
                intern(decorated, &mut pieces, &mut ids)
            }));
            if symbols.len() < 2 {
                // Nothing to merge inside a one-symbol word, and it contributes
                // no pairs — keeping it would only cost a scan per merge.
                continue;
            }
            words.push(&symbols);
            frequencies.push(frequency);
        }

        // Everything interned so far is a seed; merges are appended after.
        let alphabet_len = pieces.len();
        Seeded {
            pieces,
            piece_ids: ids,
            alphabet_len,
            words,
            frequencies,
        }
    }

    /// One symbol's spelling given where in its word it sits.
    fn decorate(&self, unit: &[u8], is_first: bool, is_last: bool) -> Vec<u8> {
        let mut piece = Vec::with_capacity(unit.len() + 2);
        if !is_first {
            if let Some(prefix) = &self.continuing_subword_prefix {
                piece.extend_from_slice(prefix);
            }
        }
        piece.extend_from_slice(unit);
        if is_last {
            if let Some(suffix) = &self.end_of_word_suffix {
                piece.extend_from_slice(suffix);
            }
        }
        piece
    }
}

/// The seed alphabet plus the corpus expressed in it.
struct Seeded {
    pieces: Vec<Vec<u8>>,
    piece_ids: FxHashMap<Vec<u8>, u32>,
    alphabet_len: usize,
    words: WordSet,
    frequencies: Vec<u64>,
}

/// Give `piece` an id, or return the one it already has.
fn intern(piece: Vec<u8>, pieces: &mut Vec<Vec<u8>>, ids: &mut FxHashMap<Vec<u8>, u32>) -> u32 {
    match ids.get(&piece) {
        Some(&id) => id,
        None => {
            let id = pieces.len() as u32;
            ids.insert(piece.clone(), id);
            pieces.push(piece);
            id
        }
    }
}

/// `piece` without its continuation prefix, if it carries one.
fn strip_prefix<'a>(piece: &'a [u8], prefix: Option<&[u8]>) -> &'a [u8] {
    match prefix {
        Some(prefix) => piece.strip_prefix(prefix).unwrap_or(piece),
        None => piece,
    }
}

/// How many words a pair can be seen in before it needs a heap allocation.
///
/// Sized so the inline array costs nothing: a `Vec` is already three words, and
/// four indices plus a length fit inside that. Measured on a gigabyte of
/// multilingual text at a 128k vocabulary, most live pairs occur in one or two
/// words, so this keeps the great majority of them off the allocator entirely.
const INLINE_WORDS: usize = 4;

/// The words containing a pair, ascending and without repeats.
#[derive(Debug, Clone)]
enum WordList {
    Inline([u32; INLINE_WORDS], u8),
    Spilled(Vec<u32>),
}

impl Default for WordList {
    fn default() -> Self {
        WordList::Inline([0; INLINE_WORDS], 0)
    }
}

impl WordList {
    fn with_capacity(words: usize) -> Self {
        if words <= INLINE_WORDS {
            Self::default()
        } else {
            WordList::Spilled(Vec::with_capacity(words))
        }
    }

    fn as_slice(&self) -> &[u32] {
        match self {
            WordList::Inline(items, len) => &items[..*len as usize],
            WordList::Spilled(items) => items,
        }
    }

    /// Record that `index` contains the pair, keeping the list sorted and
    /// unique.
    ///
    /// Callers visit words in ascending order, so a repeat can only ever be the
    /// entry just pushed. Duplicates would be harmless — merging a word twice
    /// finds nothing the second time — but they are pure waste in the structure
    /// that dominates memory.
    fn note(&mut self, index: u32) {
        if self.as_slice().last() == Some(&index) {
            return;
        }
        match self {
            WordList::Inline(items, len) if (*len as usize) < INLINE_WORDS => {
                items[*len as usize] = index;
                *len += 1;
            }
            WordList::Inline(items, len) => {
                let mut spilled = Vec::with_capacity(INLINE_WORDS * 2);
                spilled.extend_from_slice(&items[..*len as usize]);
                spilled.push(index);
                *self = WordList::Spilled(spilled);
            }
            WordList::Spilled(items) => items.push(index),
        }
    }
}

/// Everything the merge loop knows about one pair.
///
/// One record in one map rather than a count map, a position map and a queue
/// map side by side. At a 128k vocabulary on a gigabyte of text there are tens
/// of millions of live pairs, so a separate map per attribute means paying the
/// key and the hash three times over.
#[derive(Debug, Clone)]
struct PairState {
    count: i64,
    /// The best score already waiting in the queue for this pair, which is what
    /// keeps the queue to one useful entry per pair.
    queued: f64,
    words: WordList,
    /// Set when a merge disturbed this pair, and whether any word gained it.
    touched: bool,
    gained: bool,
    /// Set when the joined piece would be longer than `max_token_length`. The
    /// pair still occurs in the corpus, so its word list must be kept even
    /// though its count is forced to zero to stop it competing.
    blocked: bool,
}

impl Default for PairState {
    fn default() -> Self {
        Self {
            count: 0,
            queued: f64::NEG_INFINITY,
            words: WordList::default(),
            touched: false,
            gained: false,
            blocked: false,
        }
    }
}

type Pairs = FxHashMap<(u32, u32), PairState>;

/// What one pass over the corpus learns about a pair before its word list is
/// sized.
#[derive(Default, Clone, Copy)]
struct PairCensus {
    count: i64,
    /// How many distinct words contain the pair.
    words: u32,
    /// The last word counted, so a pair occurring twice in one word is counted
    /// once. Stored as index plus one, leaving zero to mean "none yet".
    last: u32,
}

/// Each pair's count and the words containing it.
///
/// Two passes rather than one: the first learns how long each word list will be
/// so the second can size it exactly. Growing them by doubling instead would
/// leave up to half of a multi-gigabyte structure as slack. The census carries
/// the counts too, so this costs no more hash lookups per pair than filling the
/// lists directly would.
fn count_pairs(words: &WordSet, frequencies: &[u64]) -> Pairs {
    let mut census: FxHashMap<(u32, u32), PairCensus> = FxHashMap::default();
    for (index, &frequency) in frequencies.iter().enumerate() {
        let frequency = frequency as i64;
        let marker = index as u32 + 1;
        for pair in words.pairs(index) {
            let entry = census.entry(pair).or_default();
            entry.count += frequency;
            if entry.last != marker {
                entry.words += 1;
                entry.last = marker;
            }
        }
    }

    let mut pairs: Pairs = FxHashMap::with_capacity_and_hasher(census.len(), <_>::default());
    for (&pair, entry) in &census {
        pairs.insert(
            pair,
            PairState {
                count: entry.count,
                words: WordList::with_capacity(entry.words as usize),
                ..PairState::default()
            },
        );
    }
    drop(census);

    for index in 0..words.len() {
        for pair in words.pairs(index) {
            if let Some(state) = pairs.get_mut(&pair) {
                state.words.note(index as u32);
            }
        }
    }
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counts(words: &[(&str, u64)]) -> WordCounts {
        words
            .iter()
            .map(|(word, count)| (word.as_bytes().to_vec(), *count))
            .collect()
    }

    /// The seed alphabet is every byte, so any input is spellable.
    #[test]
    fn byte_seeding_covers_all_256_bytes() {
        let vocab = BpeTrainer::builder()
            .vocab_size(256)
            .build()
            .train(&counts(&[("ab", 1)]))
            .unwrap();
        assert_eq!(vocab.alphabet_len(), 256);
        assert!(vocab.merges().is_empty());
        for byte in 0..=u8::MAX {
            assert!(vocab.pieces().contains(&vec![byte]), "missing {byte}");
        }
    }

    /// The frequent pair is merged first, and the merged piece is the two
    /// operands concatenated.
    #[test]
    fn merges_the_most_frequent_pair_first() {
        let vocab = BpeTrainer::builder()
            .vocab_size(258)
            .build()
            .train(&counts(&[("ab", 10), ("cd", 1)]))
            .unwrap();
        assert_eq!(vocab.piece(256), Some(b"ab".as_slice()));
        assert_eq!(vocab.piece(257), Some(b"cd".as_slice()));
    }

    /// An id is its own merge rank: piece `alphabet_len + i` is what merge `i`
    /// produced. This is what makes the result loadable as a `.tiktoken`.
    #[test]
    fn ids_follow_merge_order() {
        let vocab = BpeTrainer::builder()
            .vocab_size(300)
            .build()
            .train(&counts(&[("lower", 5), ("lowest", 5), ("newer", 8)]))
            .unwrap();
        for (i, &(left, right)) in vocab.merges().iter().enumerate() {
            let id = vocab.alphabet_len() + i;
            let mut expected = vocab.piece(left).unwrap().to_vec();
            expected.extend_from_slice(vocab.piece(right).unwrap());
            assert_eq!(vocab.piece(id as u32).unwrap(), expected.as_slice());
        }
    }

    /// Every merge operand is a piece with a *lower* id than the merge result.
    /// splintr's encoder relies on it: a merge has to rank above the pieces it
    /// joins or the vocabulary cannot be rebuilt by merging upward.
    #[test]
    fn operands_always_rank_below_their_result() {
        let vocab = BpeTrainer::builder()
            .vocab_size(400)
            .build()
            .train(&counts(&[
                ("the", 20),
                ("there", 12),
                ("their", 9),
                ("then", 7),
                ("them", 3),
            ]))
            .unwrap();
        for (i, &(left, right)) in vocab.merges().iter().enumerate() {
            let id = (vocab.alphabet_len() + i) as u32;
            assert!(
                left < id && right < id,
                "merge {i} joins {left},{right} into {id}"
            );
        }
    }

    #[test]
    fn min_frequency_stops_the_merge_loop() {
        let vocab = BpeTrainer::builder()
            .vocab_size(300)
            .min_frequency(5)
            .build()
            .train(&counts(&[("ab", 10), ("cd", 1)]))
            .unwrap();
        assert_eq!(vocab.merges().len(), 1);
        assert_eq!(vocab.piece(256), Some(b"ab".as_slice()));
    }

    #[test]
    fn max_token_length_refuses_longer_pieces() {
        let vocab = BpeTrainer::builder()
            .vocab_size(300)
            .max_token_length(2)
            .build()
            .train(&counts(&[("abcd", 10)]))
            .unwrap();
        assert!(
            vocab.pieces()[256..].iter().all(|piece| piece.len() <= 2),
            "a piece longer than the limit was kept"
        );
    }

    /// A corpus with nothing left to merge ends early rather than inventing
    /// tokens to reach the requested size.
    #[test]
    fn stops_when_no_pair_repeats() {
        let vocab = BpeTrainer::builder()
            .vocab_size(5_000)
            .build()
            .train(&counts(&[("ab", 1)]))
            .unwrap();
        assert!(vocab.pieces().len() < 5_000);
    }

    #[test]
    fn rejects_a_vocabulary_below_the_alphabet() {
        let error = BpeTrainer::builder()
            .vocab_size(10)
            .build()
            .train(&counts(&[("ab", 1)]))
            .unwrap_err();
        assert!(matches!(error, TrainError::VocabTooSmall { .. }));
    }

    #[test]
    fn rejects_an_empty_corpus() {
        let error = BpeTrainer::builder()
            .build()
            .train(&WordCounts::new())
            .unwrap_err();
        assert!(matches!(error, TrainError::EmptyCorpus));
    }

    /// Two runs over the same corpus must produce byte-identical vocabularies —
    /// the tie-break is total, and nothing depends on hash iteration order.
    #[test]
    fn training_is_deterministic() {
        let corpus = counts(&[("aa", 3), ("ab", 3), ("ba", 3), ("bb", 3), ("abab", 2)]);
        let trainer = BpeTrainer::builder().vocab_size(320).build();
        let first = trainer.train(&corpus).unwrap();
        for _ in 0..8 {
            let again = trainer.train(&corpus).unwrap();
            assert_eq!(first.pieces(), again.pieces());
            assert_eq!(first.merges(), again.merges());
        }
    }

    /// Character seeding treats a multi-byte character as one symbol rather than
    /// cutting it into bytes.
    #[test]
    fn char_seeding_keeps_characters_whole() {
        let vocab = BpeTrainer::builder()
            .vocab_size(10)
            .seeding(Seeding::Chars)
            .build()
            .train(&counts(&[("Ġa", 5)]))
            .unwrap();
        assert!(vocab.pieces().contains(&"Ġ".as_bytes().to_vec()));
        assert!(!vocab.pieces().iter().any(|p| p == &vec![0xC4]));
    }

    #[test]
    fn specials_are_numbered_above_the_pieces() {
        let vocab = BpeTrainer::builder()
            .vocab_size(256)
            .special_tokens(["<pad>", "<eos>"])
            .build()
            .train(&counts(&[("ab", 1)]))
            .unwrap();
        let specials = vocab.special_encoder();
        assert_eq!(specials["<pad>"], 256);
        assert_eq!(specials["<eos>"], 257);
    }
}
