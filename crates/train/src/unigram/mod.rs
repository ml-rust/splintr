//! The Unigram trainer.
//!
//! Unigram works the opposite way round from BPE. Rather than starting from
//! characters and building up, it starts from a large pool of candidate pieces
//! and *removes* the ones the corpus can most afford to lose, until the target
//! size is reached. Each surviving piece carries a log-probability, and
//! segmentation maximises their sum.
//!
//! The loop is:
//!
//! 1. **Seed** a candidate pool of frequent substrings.
//! 2. **EM** — the E-step computes each piece's expected count by
//!    forward-backward over every segmentation of every word; the M-step turns
//!    those into probabilities.
//! 3. **Prune** — score each piece by what removing it would cost, drop the
//!    cheapest, and repeat from 2 until the vocabulary is small enough.
//!
//! # Seeding without a suffix array
//!
//! SentencePiece builds a suffix array here, because it enumerates substrings
//! over the raw concatenated corpus and needs the text plus the array resident —
//! which is why it subsamples sentences on large inputs, and why its trainer
//! carries a C++ dependency.
//!
//! This trainer does not, because it does not start from raw text. [`WordCounts`]
//! is already aggregated to distinct words with frequencies, and distinct word
//! *types* are orders of magnitude fewer than corpus positions. Candidates are
//! counted one length at a time over those types: peak memory is a single length
//! bucket rather than the whole corpus, `min_frequency` prunes during the pass
//! instead of after it, and the counts are exact and frequency-weighted rather
//! than sampled.

mod lattice;

use rustc_hash::FxHashMap;

use crate::corpus::WordCounts;
use crate::error::TrainError;
use lattice::Lattice;

/// A trained Unigram vocabulary: pieces and their log-probabilities.
///
/// This is exactly what splintr's `SentencePieceTokenizer::new` takes — a token
/// list and a score list of the same length.
#[derive(Debug, Clone)]
pub struct UnigramVocab {
    tokens: Vec<String>,
    scores: Vec<f64>,
    special_count: usize,
}

impl UnigramVocab {
    /// Every token in id order, specials first.
    pub fn tokens(&self) -> &[String] {
        &self.tokens
    }

    /// The log-probability of each token, aligned with [`tokens`](Self::tokens).
    ///
    /// Specials score zero: they are not produced by segmentation, and a
    /// probability for them would be a claim the training data never made.
    pub fn scores(&self) -> &[f64] {
        &self.scores
    }

    /// The token and score lists, consuming the vocabulary.
    pub fn into_parts(self) -> (Vec<String>, Vec<f64>) {
        (self.tokens, self.scores)
    }

    pub fn special_count(&self) -> usize {
        self.special_count
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn id(&self, token: &str) -> Option<u32> {
        self.tokens
            .iter()
            .position(|t| t == token)
            .map(|i| i as u32)
    }
}

/// Configuration for [`UnigramTrainer`].
#[derive(Clone)]
pub struct UnigramTrainerBuilder {
    vocab_size: usize,
    seed_size: usize,
    max_piece_chars: usize,
    min_frequency: u64,
    shrink_factor: f64,
    em_iterations: usize,
    specials: Vec<String>,
}

impl Default for UnigramTrainerBuilder {
    fn default() -> Self {
        Self {
            vocab_size: 8_000,
            // Large enough that pruning has real choices to make, small enough
            // that the EM passes stay affordable.
            seed_size: 1_000_000,
            max_piece_chars: 16,
            min_frequency: 2,
            // Each round keeps this fraction, so the pool reaches the target in
            // a logarithmic number of rounds rather than one brutal cut.
            shrink_factor: 0.75,
            em_iterations: 2,
            specials: Vec::new(),
        }
    }
}

impl UnigramTrainerBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Total tokens to produce, specials included.
    #[must_use]
    pub fn vocab_size(mut self, size: usize) -> Self {
        self.vocab_size = size;
        self
    }

    /// How many candidate pieces to start from.
    #[must_use]
    pub fn seed_size(mut self, size: usize) -> Self {
        self.seed_size = size;
        self
    }

    /// Longest candidate piece, in characters.
    #[must_use]
    pub fn max_piece_chars(mut self, chars: usize) -> Self {
        self.max_piece_chars = chars;
        self
    }

    /// Candidates occurring fewer than this many times are never seeded.
    #[must_use]
    pub fn min_frequency(mut self, frequency: u64) -> Self {
        self.min_frequency = frequency;
        self
    }

    /// Fraction of the pool kept by each pruning round.
    #[must_use]
    pub fn shrink_factor(mut self, factor: f64) -> Self {
        self.shrink_factor = factor;
        self
    }

    /// EM iterations run between pruning rounds.
    #[must_use]
    pub fn em_iterations(mut self, iterations: usize) -> Self {
        self.em_iterations = iterations;
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

    pub fn build(self) -> UnigramTrainer {
        UnigramTrainer { config: self }
    }
}

/// Trains a Unigram vocabulary from [`WordCounts`].
pub struct UnigramTrainer {
    config: UnigramTrainerBuilder,
}

impl UnigramTrainer {
    pub fn builder() -> UnigramTrainerBuilder {
        UnigramTrainerBuilder::new()
    }

    /// Train a vocabulary.
    ///
    /// # Errors
    /// [`TrainError::EmptyCorpus`] if no words were fed, and
    /// [`TrainError::VocabTooSmall`] if the target cannot hold the characters
    /// the corpus requires — those are never pruned, since dropping one makes
    /// some word unsegmentable.
    pub fn train(&self, counts: &WordCounts) -> Result<UnigramVocab, TrainError> {
        if counts.is_empty() {
            return Err(TrainError::EmptyCorpus);
        }

        // Only text can be segmented into characters; a word that is not UTF-8
        // has no character lattice.
        let words: Vec<(&str, u64)> = counts
            .iter()
            .filter_map(|(word, frequency)| {
                std::str::from_utf8(word).ok().map(|text| (text, frequency))
            })
            .collect();
        if words.is_empty() {
            return Err(TrainError::EmptyCorpus);
        }

        let target = self
            .config
            .vocab_size
            .saturating_sub(self.config.specials.len());
        let (mut pieces, mut scores, required) = self.seed(&words);
        if target < required {
            return Err(TrainError::VocabTooSmall {
                requested: self.config.vocab_size,
                alphabet: required + self.config.specials.len(),
            });
        }

        let mut lattice = Lattice::default();
        loop {
            for _ in 0..self.config.em_iterations.max(1) {
                self.expectation_maximization(&words, &pieces, &mut scores, &mut lattice);
            }
            if pieces.len() <= target {
                break;
            }
            let keep = ((pieces.len() as f64 * self.config.shrink_factor) as usize).max(target);
            let losses = self.losses(&words, &pieces, &scores, &mut lattice);
            let before = pieces.len();
            self.prune(&mut pieces, &mut scores, &losses, required, keep);
            if pieces.len() == before {
                // Nothing could be dropped — every remaining piece is required.
                break;
            }
        }

        // A last EM pass so the scores describe the vocabulary actually shipped
        // rather than the one before the final cut.
        self.expectation_maximization(&words, &pieces, &mut scores, &mut lattice);

        let mut tokens = self.config.specials.clone();
        let mut out_scores = vec![0.0; self.config.specials.len()];
        tokens.extend(pieces.into_iter().map(|(text, _)| text));
        out_scores.extend(scores);

        Ok(UnigramVocab {
            tokens,
            scores: out_scores,
            special_count: self.config.specials.len(),
        })
    }

    /// The candidate pool: every character the corpus uses, plus the most
    /// promising longer substrings.
    ///
    /// Returns the pieces (with their seed counts), their initial scores, and
    /// how many leading entries are *required* — the single characters, which
    /// pruning may never touch because dropping one makes some word
    /// unsegmentable.
    #[allow(clippy::type_complexity)]
    fn seed(&self, words: &[(&str, u64)]) -> (Vec<(String, u64)>, Vec<f64>, usize) {
        // Characters first, so their ids are the low ones and "required" is a
        // prefix of the piece list rather than a scattered set.
        let mut char_counts: FxHashMap<String, u64> = FxHashMap::default();
        for (word, frequency) in words {
            for ch in word.chars() {
                *char_counts.entry(ch.to_string()).or_insert(0) += frequency;
            }
        }
        let mut characters: Vec<(String, u64)> = char_counts.into_iter().collect();
        characters.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let required = characters.len();

        // Longer candidates, counted one length at a time so peak memory is a
        // single bucket rather than every substring of the corpus at once.
        let mut candidates: Vec<(String, u64)> = Vec::new();
        for length in 2..=self.config.max_piece_chars {
            let mut bucket: FxHashMap<&str, u64> = FxHashMap::default();
            for (word, frequency) in words {
                let offsets: Vec<usize> = word
                    .char_indices()
                    .map(|(i, _)| i)
                    .chain(std::iter::once(word.len()))
                    .collect();
                if offsets.len() <= length {
                    continue;
                }
                for start in 0..(offsets.len() - 1 - length + 1) {
                    let text = &word[offsets[start]..offsets[start + length]];
                    *bucket.entry(text).or_insert(0) += frequency;
                }
            }
            candidates.extend(
                bucket
                    .into_iter()
                    .filter(|(_, count)| *count >= self.config.min_frequency)
                    .map(|(text, count)| (text.to_string(), count)),
            );
        }

        // Ranked by count times length: what a piece is worth is how much text
        // it covers, not how often it appears. Ties broken by spelling so the
        // pool does not depend on hash order.
        candidates.sort_by(|a, b| {
            let left = a.1 * a.0.chars().count() as u64;
            let right = b.1 * b.0.chars().count() as u64;
            right.cmp(&left).then_with(|| a.0.cmp(&b.0))
        });
        candidates.truncate(self.config.seed_size.saturating_sub(required));

        let mut pieces = characters;
        pieces.extend(candidates);

        let total: f64 = pieces.iter().map(|(_, count)| *count as f64).sum();
        let scores = pieces
            .iter()
            .map(|(_, count)| (*count as f64 / total).ln())
            .collect();
        (pieces, scores, required)
    }

    /// One EM iteration: expected counts from every segmentation, then
    /// renormalised into log-probabilities.
    fn expectation_maximization(
        &self,
        words: &[(&str, u64)],
        pieces: &[(String, u64)],
        scores: &mut [f64],
        lattice: &mut Lattice,
    ) {
        let index: FxHashMap<&str, u32> = pieces
            .iter()
            .enumerate()
            .map(|(id, (text, _))| (text.as_str(), id as u32))
            .collect();

        let mut expected = vec![0.0f64; pieces.len()];
        for (word, frequency) in words {
            let n = lattice.build(word, self.config.max_piece_chars, |s| index.get(s).copied());
            if n == 0 || !lattice.is_connected(n) {
                continue;
            }
            let weight = *frequency as f64;
            lattice.expectations(n, scores, |piece, expectation| {
                expected[piece as usize] += expectation * weight;
            });
        }

        // A piece the corpus never expects would take a score of negative
        // infinity and poison every lattice it appears in, so it keeps a floor
        // instead; the pruning pass is what removes it.
        let total: f64 = expected.iter().sum();
        if total <= 0.0 {
            return;
        }
        for (score, count) in scores.iter_mut().zip(&expected) {
            *score = if *count > 0.0 {
                (*count / total).ln()
            } else {
                f64::MIN_EXP as f64
            };
        }
    }

    /// What removing each piece would cost the corpus.
    ///
    /// A piece is worth the log-probability its uses contribute above the best
    /// segmentation available without it. Measured on the Viterbi path, since
    /// that is what inference will actually take.
    fn losses(
        &self,
        words: &[(&str, u64)],
        pieces: &[(String, u64)],
        scores: &[f64],
        lattice: &mut Lattice,
    ) -> Vec<f64> {
        let index: FxHashMap<&str, u32> = pieces
            .iter()
            .enumerate()
            .map(|(id, (text, _))| (text.as_str(), id as u32))
            .collect();

        // How often each piece is actually chosen.
        let mut uses = vec![0.0f64; pieces.len()];
        let mut path = Vec::new();
        for (word, frequency) in words {
            let n = lattice.build(word, self.config.max_piece_chars, |s| index.get(s).copied());
            if n == 0 || !lattice.is_connected(n) {
                continue;
            }
            path.clear();
            lattice.viterbi(n, scores, &mut path);
            for &piece in &path {
                uses[piece as usize] += *frequency as f64;
            }
        }

        // What each use would cost if the piece were gone: its own score against
        // the best way to spell it out of other pieces.
        let mut losses = vec![0.0f64; pieces.len()];
        for (id, (text, _)) in pieces.iter().enumerate() {
            if uses[id] == 0.0 {
                continue;
            }
            let n = lattice.build(text, self.config.max_piece_chars, |s| index.get(s).copied());
            let alternative = lattice.viterbi_excluding(n, scores, id as u32);
            let penalty = if alternative.is_finite() {
                scores[id] - alternative
            } else {
                // Nothing else spells it, so losing it would make its own text
                // unsegmentable. Never a candidate for pruning.
                f64::INFINITY
            };
            losses[id] = uses[id] * penalty.max(0.0);
        }
        losses
    }

    /// Keep the required characters and the `keep - required` costliest pieces.
    fn prune(
        &self,
        pieces: &mut Vec<(String, u64)>,
        scores: &mut Vec<f64>,
        losses: &[f64],
        required: usize,
        keep: usize,
    ) {
        if pieces.len() <= keep {
            return;
        }
        let mut ranked: Vec<usize> = (required..pieces.len()).collect();
        // Costliest first; ties by id so the choice is total and reproducible.
        ranked.sort_by(|&a, &b| losses[b].total_cmp(&losses[a]).then_with(|| a.cmp(&b)));
        ranked.truncate(keep.saturating_sub(required));

        let mut survives = vec![false; pieces.len()];
        survives[..required].fill(true);
        for id in ranked {
            survives[id] = true;
        }

        let mut id = 0;
        pieces.retain(|_| {
            let keep = survives[id];
            id += 1;
            keep
        });
        let mut id = 0;
        scores.retain(|_| {
            let keep = survives[id];
            id += 1;
            keep
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus() -> WordCounts {
        let stems = [
            "play", "run", "walk", "talk", "jump", "read", "write", "build", "learn", "teach",
        ];
        let suffixes = ["", "s", "ed", "ing", "er", "ers"];
        let mut words = Vec::new();
        for (i, stem) in stems.iter().enumerate() {
            for (j, suffix) in suffixes.iter().enumerate() {
                words.push((
                    format!("{stem}{suffix}").into_bytes(),
                    ((i + 2) * (suffixes.len() - j)) as u64,
                ));
            }
        }
        words.into_iter().collect()
    }

    fn train(size: usize) -> UnigramVocab {
        UnigramTrainer::builder()
            .vocab_size(size)
            .special_tokens(["<unk>", "</s>"])
            .min_frequency(1)
            .build()
            .train(&corpus())
            .expect("training succeeds")
    }

    #[test]
    fn produces_the_requested_size() {
        let vocab = train(200);
        assert_eq!(vocab.len(), 200);
        assert_eq!(vocab.tokens().len(), vocab.scores().len());
    }

    #[test]
    fn specials_lead_and_score_zero() {
        let vocab = train(200);
        assert_eq!(&vocab.tokens()[..2], &["<unk>", "</s>"]);
        assert_eq!(&vocab.scores()[..2], &[0.0, 0.0]);
        assert_eq!(vocab.special_count(), 2);
    }

    /// Every character the corpus uses survives, whatever its loss — otherwise
    /// some word could not be segmented at all.
    #[test]
    fn every_corpus_character_survives_pruning() {
        let corpus = corpus();
        let vocab = train(120);
        for (word, _) in corpus.iter() {
            for ch in std::str::from_utf8(word).unwrap().chars() {
                assert!(
                    vocab.id(&ch.to_string()).is_some(),
                    "character {ch:?} was pruned away"
                );
            }
        }
    }

    /// Scores are log-probabilities, so they are negative and the surviving
    /// pieces carry real mass.
    #[test]
    fn scores_are_log_probabilities() {
        let vocab = train(200);
        for (token, score) in vocab.tokens()[vocab.special_count()..]
            .iter()
            .zip(&vocab.scores()[vocab.special_count()..])
        {
            assert!(*score <= 0.0, "{token:?} scored {score}, above ln(1)");
            assert!(score.is_finite(), "{token:?} scored {score}");
        }
    }

    /// The whole point: the vocabulary drives splintr's own Unigram segmenter
    /// and reproduces the corpus.
    #[test]
    fn the_vocabulary_segments_and_round_trips_its_corpus() {
        let corpus = corpus();
        let vocab = train(200);
        let (tokens, scores) = vocab.clone().into_parts();
        let tokenizer = splintr::SentencePieceTokenizer::new(tokens, scores, None, 1)
            .expect("the trained vocabulary builds a tokenizer");

        for (word, _) in corpus.iter() {
            let text = std::str::from_utf8(word).unwrap();
            let ids = tokenizer.encode(text);
            assert!(!ids.is_empty(), "{text} produced nothing");
            let decoded = tokenizer.decode(&ids).expect("decodes");
            assert_eq!(decoded, text, "round trip failed for {text}");
        }
    }

    /// A larger vocabulary must not segment the corpus into more tokens than a
    /// smaller one.
    #[test]
    fn a_larger_vocabulary_does_not_segment_worse() {
        let corpus = corpus();
        let mut lengths = Vec::new();
        for size in [80usize, 150, 250] {
            let (tokens, scores) = train(size).into_parts();
            let tokenizer = splintr::SentencePieceTokenizer::new(tokens, scores, None, 1).unwrap();
            let total: usize = corpus
                .iter()
                .map(|(word, _)| tokenizer.encode(std::str::from_utf8(word).unwrap()).len())
                .sum();
            lengths.push((size, total));
        }
        for pair in lengths.windows(2) {
            assert!(
                pair[1].1 <= pair[0].1,
                "vocab {} needed {} tokens where vocab {} needed {}",
                pair[1].0,
                pair[1].1,
                pair[0].0,
                pair[0].1
            );
        }
    }

    #[test]
    fn training_is_deterministic() {
        let corpus = corpus();
        let trainer = UnigramTrainer::builder()
            .vocab_size(150)
            .special_tokens(["<unk>"])
            .min_frequency(1)
            .build();
        let first = trainer.train(&corpus).unwrap();
        for _ in 0..3 {
            let again = trainer.train(&corpus).unwrap();
            assert_eq!(again.tokens(), first.tokens());
            assert_eq!(again.scores(), first.scores());
        }
    }

    #[test]
    fn rejects_a_target_below_the_character_set() {
        let error = UnigramTrainer::builder()
            .vocab_size(3)
            .build()
            .train(&corpus())
            .unwrap_err();
        assert!(matches!(error, TrainError::VocabTooSmall { .. }));
    }

    #[test]
    fn rejects_an_empty_corpus() {
        let error = UnigramTrainer::builder()
            .build()
            .train(&WordCounts::new())
            .unwrap_err();
        assert!(matches!(error, TrainError::EmptyCorpus));
    }
}
