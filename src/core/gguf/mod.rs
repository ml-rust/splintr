//! Tokenizers from a GGUF file's embedded vocabulary.
//!
//! splintr does not read GGUF containers — the caller parses the file and fills
//! a [`GgufVocab`] with the `tokenizer.ggml.*` metadata, and
//! [`from_gguf_vocab`] turns it into an [`AnyTokenizer`](super::AnyTokenizer).
//! What lives here is the dialect knowledge: which algorithm each
//! `tokenizer.ggml.model` names, how the surrounding flags have to be honoured,
//! and which of them are refused rather than guessed.

mod error;
mod loader;
mod vocab;

pub use error::GgufVocabError;
pub use loader::from_gguf_vocab;
pub use vocab::GgufVocab;

#[cfg(test)]
mod tests;
