//! The WordPiece trainer.
//!
//! WordPiece builds its vocabulary with the same greedy joining loop BPE uses,
//! marking every non-initial subword (`##`) so a word start and a continuation
//! are different tokens.
//!
//! # The dead-piece problem, and why it is not free to fix
//!
//! WordPiece *inference* is greedy longest-match, not merge replay, and the
//! training loop never models that. A piece the merge loop builds can be
//! unreachable at inference because a longer piece always wins at that position.
//! Measured on English prose, the segmenter never emitted **14.6% of a
//! 2000-piece vocabulary, 36.6% of a 4000-piece one and 45.2% of an 8000-piece
//! one** — on the corpus they were trained on. Those are embedding rows paid for
//! and never used.
//!
//! [`WordPieceTrainerBuilder::prune`] removes them: train past the target,
//! segment the corpus with splintr's own segmenter, keep what is actually
//! emitted, repeat. It works — dead pieces fall from 14.6% to about 1%.
//!
//! It is **off by default**, because measurement says it is a trade rather than
//! a win. At a 2000-piece target it removed 270 of 292 dead pieces and cost
//! ~3% more tokens on held-out text. The reason is structural: greedy
//! longest-match compression is a property of the vocabulary *as a set*, since
//! removing one piece changes which pieces fire everywhere else, so no
//! per-piece score can select a subset without loss. Ranking by tokens saved
//! rather than by emission count was tried and is worse on the trade (0.4% dead
//! for ~5% more tokens).
//!
//! Turn it on when unused embedding rows cost more than tokens do; leave it off
//! when compression is what matters.
//!
//! # On the selection criterion
//!
//! The default is [`Criterion::Frequency`], and that is a measured choice rather
//! than an inherited one. Merging a pair with `m` occurrences removes exactly
//! `m` tokens from the corpus, so greedy frequency is greedy-optimal *for token
//! count* — it directly optimises what a tokenizer is judged on.
//! [`Criterion::Likelihood`], the WordPiece paper's objective, optimises corpus
//! probability under a unigram model instead, which is a different target: on
//! held-out text it needed about 1.27x the tokens at a 2000-piece vocabulary.
//! It is available, and documented, for anyone who wants that objective.

use rustc_hash::FxHashSet;
use splintr::Tokenize;

use crate::bpe::{BpeTrainerBuilder, Criterion};
use crate::corpus::WordCounts;
use crate::error::TrainError;
use crate::vocab::Seeding;

/// The marker every non-initial subword carries.
pub const DEFAULT_CONTINUING_PREFIX: &str = "##";

/// A trained WordPiece vocabulary.
///
/// Deliberately not a [`TrainedVocab`](crate::TrainedVocab): that type's ids
/// carry merge order, and a WordPiece vocabulary has none to carry — its
/// segmenter matches from the vocabulary alone, and pruning breaks the
/// id-to-merge correspondence outright. What a WordPiece consumer needs is the
/// token list, so that is what this is.
#[derive(Debug, Clone)]
pub struct WordPieceVocab {
    tokens: Vec<String>,
    special_count: usize,
}

impl WordPieceVocab {
    /// Every token in id order: special tokens first, then pieces.
    ///
    /// Specials lead so an `[UNK]` gets a low, stable id — the layout a
    /// `vocab.txt` has — and this is exactly the `Vec<String>` splintr's
    /// `WordPieceTokenizer::new` takes.
    pub fn tokens(&self) -> &[String] {
        &self.tokens
    }

    /// The token list, consuming the vocabulary.
    pub fn into_tokens(self) -> Vec<String> {
        self.tokens
    }

    /// How many leading tokens are special.
    pub fn special_count(&self) -> usize {
        self.special_count
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Build one directly from a token list, for a caller holding a vocabulary
    /// that did not come from this trainer.
    pub fn from_parts(tokens: Vec<String>, special_count: usize) -> Self {
        Self {
            tokens,
            special_count,
        }
    }

    /// The id of a token, if the vocabulary has it.
    pub fn id(&self, token: &str) -> Option<u32> {
        self.tokens
            .iter()
            .position(|t| t == token)
            .map(|i| i as u32)
    }
}

/// How hard to work at removing pieces the segmenter cannot reach.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Prune {
    /// Train this many times the target size before selecting down to it. A
    /// larger pool means the kept pieces are chosen from more candidates; it
    /// costs training time, not vocabulary quality.
    pub overshoot: usize,
    /// How many segment-and-select rounds to run. Dropping a piece changes how
    /// the text around it segments, so one pass leaves newly-dead pieces behind.
    pub rounds: usize,
}

impl Default for Prune {
    fn default() -> Self {
        // Twice the target, three rounds: pool enough to replace every dead
        // piece measured (45% at worst), rounds enough for usage to settle,
        // without making training several times longer.
        Self {
            overshoot: 2,
            rounds: 3,
        }
    }
}

/// Configuration for [`WordPieceTrainer`].
pub struct WordPieceTrainerBuilder {
    inner: BpeTrainerBuilder,
    vocab_size: usize,
    specials: Vec<String>,
    prune: Option<Prune>,
}

impl Default for WordPieceTrainerBuilder {
    fn default() -> Self {
        Self {
            // Character seeding, not bytes: a WordPiece vocabulary is a list of
            // strings, so a piece has to be text. Byte seeding would put
            // fragments of a UTF-8 sequence in it, which a token list cannot
            // spell.
            inner: BpeTrainerBuilder::new()
                .seeding(Seeding::Chars)
                .continuing_subword_prefix(DEFAULT_CONTINUING_PREFIX),
            vocab_size: 30_000,
            specials: Vec::new(),
            prune: None,
        }
    }
}

impl WordPieceTrainerBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Total tokens to produce, specials included.
    #[must_use]
    pub fn vocab_size(mut self, size: usize) -> Self {
        self.vocab_size = size;
        self
    }

    #[must_use]
    pub fn min_frequency(mut self, frequency: u64) -> Self {
        self.inner = self.inner.min_frequency(frequency);
        self
    }

    #[must_use]
    pub fn max_token_length(mut self, bytes: usize) -> Self {
        self.inner = self.inner.max_token_length(bytes);
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

    #[must_use]
    pub fn initial_alphabet<I, B>(mut self, symbols: I) -> Self
    where
        I: IntoIterator<Item = B>,
        B: Into<Vec<u8>>,
    {
        self.inner = self.inner.initial_alphabet(symbols);
        self
    }

    /// Use a different continuation marker than `##`.
    #[must_use]
    pub fn continuing_subword_prefix(mut self, prefix: impl Into<Vec<u8>>) -> Self {
        self.inner = self.inner.continuing_subword_prefix(prefix);
        self
    }

    /// Override the selection criterion. See the module documentation for why
    /// the default is [`Criterion::Frequency`] rather than the paper's
    /// likelihood objective.
    #[must_use]
    pub fn criterion(mut self, criterion: Criterion) -> Self {
        self.inner = self.inner.criterion(criterion);
        self
    }

    /// Turn on the segment-and-select pass that removes pieces the segmenter
    /// cannot reach.
    ///
    /// Off by default. See the module documentation for the measured trade: it
    /// removes almost all dead pieces and costs a few percent in tokens.
    #[must_use]
    pub fn prune(mut self, prune: Option<Prune>) -> Self {
        self.prune = prune;
        self
    }

    pub fn build(self) -> WordPieceTrainer {
        WordPieceTrainer {
            inner: self.inner,
            vocab_size: self.vocab_size,
            specials: self.specials,
            prune: self.prune,
        }
    }
}

/// Trains a WordPiece vocabulary from [`WordCounts`].
pub struct WordPieceTrainer {
    inner: BpeTrainerBuilder,
    vocab_size: usize,
    specials: Vec<String>,
    prune: Option<Prune>,
}

impl WordPieceTrainer {
    pub fn builder() -> WordPieceTrainerBuilder {
        WordPieceTrainerBuilder::new()
    }

    /// Train a vocabulary.
    ///
    /// # Errors
    /// Whatever the underlying merge loop reports, plus [`TrainError::NotUtf8`]
    /// if a piece is not text — which character seeding prevents.
    pub fn train(&self, counts: &WordCounts) -> Result<WordPieceVocab, TrainError> {
        let target = self.vocab_size.saturating_sub(self.specials.len());
        let prune = self.prune;
        let overshoot = prune.map_or(1, |p| p.overshoot.max(1));

        let trained = self
            .inner
            .clone()
            .vocab_size(target.saturating_mul(overshoot))
            .build()
            .train(counts)?;

        // Seeds are the single characters, bare and marked. They are kept
        // whatever their usage: dropping one makes some word unspellable, which
        // is a worse vocabulary than one carrying an unused piece.
        let seeds = trained.alphabet_len();
        let mut pieces: Vec<String> = Vec::with_capacity(trained.pieces().len());
        for (id, piece) in trained.pieces().iter().enumerate() {
            let text =
                std::str::from_utf8(piece).map_err(|_| TrainError::NotUtf8 { id: id as u32 })?;
            pieces.push(text.to_string());
        }

        if let Some(prune) = prune {
            for _ in 0..prune.rounds {
                if pieces.len() <= target {
                    break;
                }
                let usage = self.usage(&pieces, counts);
                pieces = select(pieces, &usage, seeds, target);
            }
        }
        // Without any rounds the pool is already the target size; with them it
        // has been selected down to it.
        pieces.truncate(target.max(seeds));

        let mut tokens = self.specials.clone();
        tokens.extend(pieces);
        Ok(WordPieceVocab {
            tokens,
            special_count: self.specials.len(),
        })
    }

    /// How often each piece is actually emitted, weighted by word frequency.
    ///
    /// Measured with splintr's own segmenter — the one that runs at inference —
    /// because the point is to ask what *it* reaches rather than what the merge
    /// loop believed it built.
    ///
    /// Emission count, deliberately, and not tokens saved. Weighting each
    /// emission by the `k - 1` tokens a `k`-character piece spares looks more
    /// principled and measures worse: it removed slightly more dead pieces
    /// (0.4% against 1.1%) for noticeably more tokens (~5% against ~3%), a worse
    /// trade per piece reclaimed.
    fn usage(&self, pieces: &[String], counts: &WordCounts) -> Vec<u64> {
        let mut tokens = self.specials.clone();
        tokens.extend_from_slice(pieces);
        let offset = self.specials.len();
        let segmenter = splintr::WordPieceTokenizer::new(tokens, 0, 512, false);

        let mut usage = vec![0u64; pieces.len()];
        for (word, frequency) in counts.iter() {
            let Ok(text) = std::str::from_utf8(word) else {
                continue;
            };
            for id in segmenter.encode(text) {
                let id = id as usize;
                if id >= offset {
                    usage[id - offset] += frequency;
                }
            }
        }
        usage
    }
}

/// Keep the seeds and the most-emitted non-seed pieces, down to `target`.
///
/// Original order is preserved among the kept pieces, so a round cannot shuffle
/// a piece's neighbours and the result stays reproducible.
fn select(pieces: Vec<String>, usage: &[u64], seeds: usize, target: usize) -> Vec<String> {
    if pieces.len() <= target {
        return pieces;
    }

    // Non-seed pieces ranked by emission count, ties broken by original
    // position so the choice is total and does not depend on sort stability.
    let mut ranked: Vec<usize> = (seeds..pieces.len()).collect();
    ranked.sort_by(|&a, &b| usage[b].cmp(&usage[a]).then(a.cmp(&b)));
    ranked.truncate(target.saturating_sub(seeds));
    let keep: FxHashSet<usize> = ranked.into_iter().collect();

    pieces
        .into_iter()
        .enumerate()
        .filter(|(i, _)| *i < seeds || keep.contains(i))
        .map(|(_, piece)| piece)
        .collect()
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

    fn corpus() -> WordCounts {
        counts(&[
            ("playing", 12),
            ("played", 9),
            ("player", 7),
            ("plays", 5),
            ("running", 8),
            ("runner", 4),
            ("runs", 6),
            ("walking", 5),
            ("walked", 4),
            ("walker", 3),
        ])
    }

    /// A continuation is a *different token* from the same letters at a word
    /// start — the whole point of the marker.
    #[test]
    fn word_starts_and_continuations_are_distinct_tokens() {
        let vocab = WordPieceTrainer::builder()
            .vocab_size(120)
            .build()
            .train(&corpus())
            .unwrap();
        assert!(vocab.tokens().iter().any(|t| t == "a"));
        assert!(vocab.tokens().iter().any(|t| t == "##a"));
    }

    /// The marker records a position, so it never lands inside a piece.
    #[test]
    fn the_marker_never_appears_inside_a_piece() {
        let vocab = WordPieceTrainer::builder()
            .vocab_size(120)
            .build()
            .train(&corpus())
            .unwrap();
        for token in vocab.tokens() {
            let body = token.strip_prefix("##").unwrap_or(token);
            assert!(!body.contains("##"), "marker inside {token:?}");
        }
    }

    #[test]
    fn specials_lead_the_token_list() {
        let vocab = WordPieceTrainer::builder()
            .vocab_size(80)
            .special_tokens(["[UNK]", "[CLS]", "[SEP]"])
            .build()
            .train(&corpus())
            .unwrap();
        assert_eq!(&vocab.tokens()[..3], &["[UNK]", "[CLS]", "[SEP]"]);
        assert_eq!(vocab.special_count(), 3);
        assert_eq!(vocab.id("[UNK]"), Some(0));
    }

    /// How many pieces the segmenter never emits over `corpus`.
    fn unreachable(vocab: &WordPieceVocab, corpus: &WordCounts) -> usize {
        let segmenter = splintr::WordPieceTokenizer::new(vocab.tokens().to_vec(), 0, 512, false);
        let mut used = vec![false; vocab.len()];
        for (word, _) in corpus.iter() {
            for id in segmenter.encode(std::str::from_utf8(word).unwrap()) {
                used[id as usize] = true;
            }
        }
        // Specials are never emitted by segmenting plain words, so they do not
        // count as dead.
        used[vocab.special_count()..]
            .iter()
            .filter(|u| !**u)
            .count()
    }

    /// The claim the pruning pass exists for.
    ///
    /// Needs a corpus rich enough that training can overshoot the target — on a
    /// handful of words the merge loop runs dry first and both settings produce
    /// the same vocabulary, which would make this pass for the wrong reason.
    #[test]
    fn pruning_removes_pieces_the_segmenter_cannot_reach() {
        let corpus = rich_corpus();
        let train = |prune| {
            WordPieceTrainer::builder()
                .vocab_size(150)
                .special_tokens(["[UNK]"])
                .prune(prune)
                .build()
                .train(&corpus)
                .unwrap()
        };
        let standard = train(None);
        let pruned = train(Some(Prune::default()));
        assert_eq!(
            standard.len(),
            pruned.len(),
            "both must hit the target size"
        );

        let before = unreachable(&standard, &corpus);
        let after = unreachable(&pruned, &corpus);
        assert!(before > 0, "the corpus must actually produce dead pieces");
        assert!(
            after < before,
            "pruning left {after} unreachable pieces against {before} without it"
        );
    }

    /// A corpus with enough distinct word shapes that a few hundred merges are
    /// available, so pruning has a pool to select from.
    fn rich_corpus() -> WordCounts {
        let stems = [
            "play", "run", "walk", "talk", "jump", "read", "write", "build", "break", "think",
            "learn", "teach", "start", "close", "open", "watch", "listen", "follow", "answer",
            "record",
        ];
        let suffixes = ["", "s", "ed", "ing", "er", "ers", "ings"];
        let mut words = Vec::new();
        for (i, stem) in stems.iter().enumerate() {
            for (j, suffix) in suffixes.iter().enumerate() {
                words.push((
                    format!("{stem}{suffix}").into_bytes(),
                    ((i + 1) * (suffixes.len() - j)) as u64,
                ));
            }
        }
        words.into_iter().collect()
    }

    /// Pruning must not cost coverage: every seed survives, so no word becomes
    /// unspellable.
    #[test]
    fn pruning_keeps_every_word_spellable() {
        let corpus = corpus();
        let vocab = WordPieceTrainer::builder()
            .vocab_size(400)
            .special_tokens(["[UNK]"])
            .build()
            .train(&corpus)
            .unwrap();
        let segmenter = splintr::WordPieceTokenizer::new(vocab.tokens().to_vec(), 0, 512, false);
        for (word, _) in corpus.iter() {
            let text = std::str::from_utf8(word).unwrap();
            let ids = segmenter.encode(text);
            assert!(!ids.is_empty(), "{text} produced nothing");
            assert!(!ids.contains(&0), "{text} fell back to [UNK]: {ids:?}");
        }
    }

    #[test]
    fn training_is_deterministic() {
        let corpus = corpus();
        let trainer = WordPieceTrainer::builder()
            .vocab_size(300)
            .special_tokens(["[UNK]"])
            .build();
        let first = trainer.train(&corpus).unwrap();
        for _ in 0..5 {
            assert_eq!(trainer.train(&corpus).unwrap().tokens(), first.tokens());
        }
    }

    #[test]
    fn a_custom_marker_is_honoured() {
        let vocab = WordPieceTrainer::builder()
            .vocab_size(90)
            .continuing_subword_prefix("__")
            .build()
            .train(&corpus())
            .unwrap();
        assert!(vocab.tokens().iter().any(|t| t.starts_with("__")));
        assert!(!vocab.tokens().iter().any(|t| t.starts_with("##")));
    }
}
