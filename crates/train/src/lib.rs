//! Tokenizer training for splintr.
//!
//! Three trainers, each producing a vocabulary splintr loads back:
//!
//! | trainer | output | writers |
//! |---|---|---|
//! | [`BpeTrainer`] | [`TrainedVocab`] | `.tiktoken`, `tokenizer.json` |
//! | [`WordPieceTrainer`] | [`WordPieceVocab`] | `vocab.txt`, `tokenizer.json` |
//! | [`UnigramTrainer`] | [`UnigramVocab`] | `.spm`, `tokenizer.json` |
//!
//! The writers live in [`mod@write`], and every one of them is net-new: the main
//! crate reads these formats and writes none of them.
//!
//! # Why this is a separate crate
//!
//! Nothing here is needed to use a tokenizer, and `splintr` is deliberately
//! small. Training lives beside it rather than inside it, so encoding carries no
//! coupling to a corpus reader or a merge loop.
//!
//! # Choosing a trainer
//!
//! [`BpeTrainer`] for a tiktoken- or GPT-style vocabulary, segmented by
//! replaying merges. [`WordPieceTrainer`] for BERT-style, segmented by greedy
//! longest match. [`UnigramTrainer`] for SentencePiece-style, segmented by
//! maximising a sum of log-probabilities — measurably the best of the three at
//! small vocabulary sizes, and the one to reach for when compression matters
//! more than matching an existing model family.
//!
//! Anything a SentencePiece-style segmenter will load must be trained through
//! [`Corpus::with_metaspace`]: those segmenters prepend `▁` before matching, so
//! a vocabulary trained without it cannot spell any word's first character.
//!
//! # The shape of a run
//!
//! ```no_run
//! use splintr::{PreTokStage, PreTokenizer, SplitBehavior, SplitPattern};
//! use splintr_train::{BpeTrainer, Corpus, write};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let pre_tokenizer = PreTokenizer::new(vec![PreTokStage::Split {
//!     pattern: SplitPattern::Regex(r"\s+".into()),
//!     behavior: SplitBehavior::Removed,
//!     invert: false,
//! }])?;
//!
//! let mut corpus = Corpus::new().with_pre_tokenizer(pre_tokenizer);
//! corpus.feed_file("corpus.txt")?;
//!
//! let vocab = BpeTrainer::builder()
//!     .vocab_size(32_000)
//!     .special_tokens(["<|endoftext|>"])
//!     .build()
//!     .train(corpus.counts())?;
//!
//! write::tiktoken_file(&vocab, "my.tiktoken")?;
//! # Ok(())
//! # }
//! ```
//!
//! The pre-tokenizer is splintr's own, so the boundaries the vocabulary is
//! trained on are the boundaries it will later be encoded against — which is the
//! reason to train here rather than to import someone else's output.

pub(crate) mod bpe;
pub(crate) mod corpus;
mod error;
pub(crate) mod unigram;
mod vocab;
pub(crate) mod wordpiece;
pub mod write;

pub use bpe::{BpeTrainer, BpeTrainerBuilder, Criterion};
pub use corpus::{Corpus, PreTok, WordCounts, METASPACE};
pub use error::TrainError;
pub use unigram::{UnigramTrainer, UnigramTrainerBuilder, UnigramVocab};
pub use vocab::{Seeding, TrainedVocab};
pub use wordpiece::{
    Prune, WordPieceTrainer, WordPieceTrainerBuilder, WordPieceVocab, DEFAULT_CONTINUING_PREFIX,
};
