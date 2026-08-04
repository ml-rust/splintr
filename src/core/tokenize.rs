//! Unified tokenizer trait for all splintr backends.
//!
//! The `Tokenize` trait provides a common interface across BPE, SentencePiece,
//! and WordPiece tokenizers, enabling generic code that works with any backend.

use super::policy::{PolicyError, SpecialMode};

/// Common interface for all tokenizer backends.
///
/// Implemented by [`Tokenizer`](super::Tokenizer) (BPE),
/// [`SentencePieceTokenizer`](super::SentencePieceTokenizer) (unigram), and
/// [`WordPieceTokenizer`](super::WordPieceTokenizer) (WordPiece).
pub trait Tokenize: Send + Sync {
    /// Encode text into token IDs.
    fn encode(&self, text: &str) -> Vec<u32>;

    /// Encode text into token IDs under an explicit [`SpecialMode`], governing
    /// whether special/control tokens spelled out in `text` are matched.
    ///
    /// Deliberately not defaulted: every implementor of this trait lives in
    /// this crate, and a default body that ignored `mode` would make the
    /// allow-list/deny-all guarantee silently inert for any future backend
    /// that forgot to override it.
    fn encode_with(&self, text: &str, mode: &SpecialMode<'_>) -> Result<Vec<u32>, PolicyError>;

    /// Decode token IDs back to text.
    ///
    /// Returns an error if any token ID is invalid.
    fn decode(&self, ids: &[u32]) -> Result<String, TokenizeError>;

    /// Return the vocabulary size (number of distinct tokens).
    fn vocab_size(&self) -> usize;
}

/// Error type for the [`Tokenize`] trait's decode method.
#[derive(Debug, thiserror::Error)]
pub enum TokenizeError {
    #[error("Decoding error: invalid UTF-8")]
    Utf8Error,
    #[error("Decoding error: token ID {0} out of range")]
    InvalidTokenId(u32),
    /// The `tokenizer.json` declares a `decoder` pipeline whose named step
    /// cannot be evaluated one chunk at a time, so no streaming decoder can
    /// reproduce [`decode`](Tokenize::decode) for it. Refused rather than
    /// silently answered with the backend's own decode, which renders the raw
    /// pieces the declared pipeline exists to turn into text.
    #[error(
        "the declared decoder pipeline cannot be streamed: its `{0}` step is not incrementally computable — decode the whole sequence instead"
    )]
    UnstreamableDecoder(&'static str),
    #[error("{0}")]
    Other(String),
}
