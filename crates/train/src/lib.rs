//! Tokenizer training for splintr.
//!
//! Produces vocabularies that splintr loads: a trainer emits a
//! [`TrainedVocab`], and the writers in [`write`] state it as a `.tiktoken` rank
//! file or a HuggingFace `tokenizer.json`.
//!
//! # Why this is a separate crate
//!
//! Nothing here is needed to use a tokenizer, and `splintr` is deliberately
//! small. Training lives beside it rather than inside it, so encoding carries no
//! coupling to a corpus reader or a merge loop.
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
//! trained on are the boundaries it will later be encoded against.

mod bpe;
mod corpus;
mod error;
mod vocab;
pub mod write;

pub use bpe::{BpeTrainer, BpeTrainerBuilder};
pub use corpus::{Corpus, WordCounts};
pub use error::TrainError;
pub use vocab::{Seeding, TrainedVocab};
