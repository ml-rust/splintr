//! Unified tokenizer trait for all splintr backends.
//!
//! The `Tokenize` trait provides a common interface across BPE, SentencePiece,
//! and WordPiece tokenizers, enabling generic code that works with any backend.

/// Common interface for all tokenizer backends.
///
/// Implemented by [`Tokenizer`](super::Tokenizer) (BPE),
/// [`SentencePieceTokenizer`](super::SentencePieceTokenizer) (unigram), and
/// [`WordPieceTokenizer`](super::WordPieceTokenizer) (WordPiece).
pub trait Tokenize: Send + Sync {
    /// Encode text into token IDs.
    fn encode(&self, text: &str) -> Vec<u32>;

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
    #[error("{0}")]
    Other(String),
}
