//! Errors from loading a HuggingFace `tokenizer.json`.

use thiserror::Error;

use super::super::policy::PolicyError;
use super::super::sentencepiece::SentencePieceError;
use super::super::tokenizer::TokenizerError;

/// Errors from loading a HuggingFace `tokenizer.json`.
#[derive(Debug, Error)]
pub enum HfJsonError {
    #[error("failed to parse tokenizer.json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("tokenizer.json missing field: {0}")]
    MissingField(&'static str),
    #[error("unsupported model.type `{0}` (expected BPE, Unigram, or WordPiece)")]
    UnsupportedModelType(String),
    #[error(
        "unsupported normalizer type(s) `{0}` — refusing to load rather than silently drop them"
    )]
    UnsupportedNormalizer(String),
    #[error("normalizer Replace pattern `{0}` failed to compile as a regex")]
    InvalidNormalizerRegex(String),
    #[error("unsupported pre_tokenizer type(s) `{0}` and no recognized split — refusing to guess the split pattern")]
    UnsupportedPreTokenizer(String),
    #[error("vocab entry `{0}` is not valid byte-level encoding")]
    InvalidByteLevel(String),
    #[error("could not determine the {0} token id from the tokenizer.json")]
    MissingSpecial(&'static str),
    #[error(transparent)]
    Policy(#[from] PolicyError),
    #[error(transparent)]
    Tokenizer(#[from] TokenizerError),
    #[error(transparent)]
    SentencePiece(#[from] SentencePieceError),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
