//! Errors from building a tokenizer out of a [`GgufVocab`](super::GgufVocab).

use thiserror::Error;

use super::super::sentencepiece::SentencePieceError;
use super::super::spm::SpmError;
use super::super::tokenizer::TokenizerError;
use super::super::wordpiece::WordPieceError;

/// Errors from [`from_gguf_vocab`](super::from_gguf_vocab).
///
/// Separate from the HuggingFace loader's error: no json is involved here, and
/// every failure below is about the *vocabulary metadata* the caller extracted,
/// so a shared type would only offer variants that can never occur.
#[derive(Debug, Error)]
pub enum GgufVocabError {
    #[error(
        "unsupported tokenizer.ggml.model `{0}`. Supported: bert (WordPiece), t5 (Unigram), \
         llama (SentencePiece BPE), gpt2 (byte-level BPE)"
    )]
    UnsupportedModel(String),
    #[error("the GGUF vocabulary is empty — tokenizer.ggml.tokens carries no tokens")]
    EmptyVocab,
    #[error(
        "GGUF declares tokenizer.ggml.model = gpt2 but carries no tokenizer.ggml.merges; \
         byte-level BPE is defined by that list and cannot be reconstructed from the \
         vocabulary alone"
    )]
    MissingMerges,
    #[error(
        "unsupported tokenizer.ggml.pre `{0}` for a byte-level BPE vocabulary. The \
         pre-tokenizer decides where text is split before merging, so guessing one would \
         silently produce different token ids. Supported: qwen2, llama-bpe, and the GPT-2 \
         family (default, gpt-2, phi-2, roberta-bpe, jina-v*)"
    )]
    UnsupportedPreTokenizer(String),
    #[error(transparent)]
    Tokenizer(#[from] TokenizerError),
    #[error(transparent)]
    SentencePiece(#[from] SentencePieceError),
    #[error(transparent)]
    Spm(#[from] SpmError),
    #[error(transparent)]
    WordPiece(#[from] WordPieceError),
}
