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

    /// The `([CLS], [SEP])` pair a BERT-family model expects to wrap every
    /// sequence, or `None` for tokenizers with no such convention.
    ///
    /// [`Tokenize::encode`] deliberately does NOT insert these — a caller
    /// building a multi-segment input (a reranker's `[CLS] q [SEP] d [SEP]`)
    /// must place them itself. That makes it every encoder caller's job to add
    /// them in the single-segment case too, and one that forgets feeds the
    /// model a sequence unlike anything it saw in training. This accessor lets
    /// such a caller do it generically, without downcasting to a concrete
    /// tokenizer type.
    ///
    /// Defaults to `None`, so non-BERT backends are unaffected.
    fn cls_sep_ids(&self) -> Option<(u32, u32)> {
        None
    }
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
