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
use word::Word;

/// A candidate merge sitting in the priority queue.
struct Candidate {
    pair: (u32, u32),
    count: u64,
    /// Indices of the words containing this pair.
    positions: FxHashSet<usize>,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.count == other.count && self.pair == other.pair
    }
}
impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Highest count wins. On a tie the *lower* pair wins, so the comparison
        // is reversed — the tie-break has to be total and deterministic or two
        // runs over the same corpus produce different vocabularies.
        self.count
            .cmp(&other.count)
            .then_with(|| other.pair.cmp(&self.pair))
    }
}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Configuration for [`BpeTrainer`].
pub struct BpeTrainerBuilder {
    vocab_size: usize,
    min_frequency: u64,
    max_token_length: Option<usize>,
    specials: Vec<String>,
    initial_alphabet: Vec<Vec<u8>>,
    seeding: Seeding,
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

    pub fn build(self) -> BpeTrainer {
        BpeTrainer {
            vocab_size: self.vocab_size,
            min_frequency: self.min_frequency,
            max_token_length: self.max_token_length,
            specials: self.specials,
            initial_alphabet: self.initial_alphabet,
            seeding: self.seeding,
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

        let (mut pieces, mut piece_ids) = self.seed_alphabet(counts);
        let alphabet_len = pieces.len();
        if self.vocab_size < alphabet_len {
            return Err(TrainError::VocabTooSmall {
                requested: self.vocab_size,
                alphabet: alphabet_len,
            });
        }

        let (mut words, frequencies) = self.encode_words(counts, &piece_ids);
        let (mut pair_counts, mut positions) = count_pairs(&words, &frequencies);

        let mut queue: BinaryHeap<Candidate> = BinaryHeap::with_capacity(pair_counts.len());
        for (pair, words_with) in positions.drain() {
            let count = pair_counts.get(&pair).copied().unwrap_or(0);
            if count > 0 {
                queue.push(Candidate {
                    pair,
                    count: count as u64,
                    positions: words_with,
                });
            }
        }

        let max_len = self.max_token_length.unwrap_or(usize::MAX);
        let mut merges: Vec<(u32, u32)> = Vec::new();
        let mut changes: Vec<((u32, u32), i64)> = Vec::new();

        while pieces.len() < self.vocab_size {
            let Some(mut top) = queue.pop() else {
                // Nothing left to merge: the corpus has no repeated pair. The
                // vocabulary is smaller than asked for, which is the honest
                // outcome rather than padding it with invented tokens.
                break;
            };

            // Lazy invalidation: an entry whose stored count no longer matches
            // the live one is stale, so correct it and let it compete again.
            let live = pair_counts.get(&top.pair).copied().unwrap_or(0);
            if live != top.count as i64 {
                if live > 0 {
                    top.count = live as u64;
                    queue.push(top);
                }
                continue;
            }

            if top.count < 1 || top.count < self.min_frequency {
                break;
            }

            let (left, right) = top.pair;
            let mut piece = pieces[left as usize].clone();
            piece.extend_from_slice(&pieces[right as usize]);

            if piece.len() > max_len {
                // Too long to keep, and it must not be reconsidered: drop its
                // count so the lazy check cannot resurrect it.
                pair_counts.remove(&top.pair);
                continue;
            }

            let new_id = pieces.len() as u32;
            pieces.push(piece.clone());
            piece_ids.insert(piece, new_id);
            merges.push(top.pair);

            // The merged pair is gone from every word it occurred in.
            pair_counts.remove(&top.pair);

            for &index in &top.positions {
                changes.clear();
                let merged = words[index].merge(left, right, new_id, &mut changes);
                if merged == 0 {
                    continue;
                }
                let frequency = frequencies[index] as i64;
                for &(pair, delta) in &changes {
                    let entry = pair_counts.entry(pair).or_insert(0);
                    *entry += delta * frequency;
                    if delta > 0 {
                        positions.entry(pair).or_default().insert(index);
                    }
                }
            }

            for (pair, words_with) in positions.drain() {
                let count = pair_counts.get(&pair).copied().unwrap_or(0);
                if count > 0 {
                    queue.push(Candidate {
                        pair,
                        count: count as u64,
                        positions: words_with,
                    });
                }
            }
        }

        Ok(TrainedVocab::new(
            pieces,
            alphabet_len,
            merges,
            self.specials.clone(),
            self.seeding,
        ))
    }

    /// The seed pieces, lowest id first.
    ///
    /// Under [`Seeding::Bytes`] that is all 256 byte values, always and whether
    /// the corpus uses them or not: a byte-level vocabulary that omits an unseen
    /// byte cannot encode text containing it, and the point of byte seeding is
    /// that no input is unspellable.
    fn seed_alphabet(&self, counts: &WordCounts) -> (Vec<Vec<u8>>, FxHashMap<Vec<u8>, u32>) {
        let mut pieces: Vec<Vec<u8>> = Vec::new();
        let mut ids: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        let push = |piece: Vec<u8>, pieces: &mut Vec<Vec<u8>>, ids: &mut FxHashMap<_, _>| {
            if !ids.contains_key(&piece) {
                ids.insert(piece.clone(), pieces.len() as u32);
                pieces.push(piece);
            }
        };

        match self.seeding {
            Seeding::Bytes => {
                for byte in 0..=u8::MAX {
                    push(vec![byte], &mut pieces, &mut ids);
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
                    push(unit, &mut pieces, &mut ids);
                }
            }
        }

        for symbol in &self.initial_alphabet {
            push(symbol.clone(), &mut pieces, &mut ids);
        }

        (pieces, ids)
    }

    /// Each distinct word as its seed symbol ids, with its corpus frequency.
    fn encode_words(
        &self,
        counts: &WordCounts,
        ids: &FxHashMap<Vec<u8>, u32>,
    ) -> (Vec<Word>, Vec<u64>) {
        let mut words = Vec::with_capacity(counts.len());
        let mut frequencies = Vec::with_capacity(counts.len());
        for (word, frequency) in counts.iter() {
            let symbols: Vec<u32> = self
                .seeding
                .units(word)
                .into_iter()
                // A unit with no id cannot appear: byte seeding covers every
                // byte, and character seeding is built from these same words.
                .filter_map(|unit| ids.get(unit).copied())
                .collect();
            if symbols.len() < 2 {
                // Nothing to merge inside a one-symbol word, and it contributes
                // no pairs — keeping it would only cost a scan per merge.
                continue;
            }
            words.push(Word::from_symbols(symbols));
            frequencies.push(frequency);
        }
        (words, frequencies)
    }
}

/// How often each pair occurs across the corpus, weighted by word frequency.
type PairCounts = FxHashMap<(u32, u32), i64>;

/// For each pair, the indices of the words containing it — so a merge visits
/// only the words it affects.
type PairPositions = FxHashMap<(u32, u32), FxHashSet<usize>>;

/// Initial pair counts and, for each pair, which words contain it.
fn count_pairs(words: &[Word], frequencies: &[u64]) -> (PairCounts, PairPositions) {
    let mut counts: PairCounts = FxHashMap::default();
    let mut positions: PairPositions = FxHashMap::default();
    for (index, word) in words.iter().enumerate() {
        let frequency = frequencies[index] as i64;
        for pair in word.pairs() {
            *counts.entry(pair).or_insert(0) += frequency;
            positions.entry(pair).or_default().insert(index);
        }
    }
    (counts, positions)
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
