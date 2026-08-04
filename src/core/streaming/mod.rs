//! UTF-8 safe streaming decoder for token-by-token LLM output.
//!
//! This module provides a stateful decoder that buffers incomplete UTF-8 sequences
//! and only emits complete, valid UTF-8 characters. This is critical for streaming
//! LLM output where token boundaries may not align with character boundaries.
//!
//! # One decoder, configured by the tokenizer
//!
//! There is a single [`StreamingDecoder`](crate::StreamingDecoder), obtained
//! only from a tokenizer's own factory —
//! [`Tokenizer::streaming_decoder`](crate::Tokenizer::streaming_decoder) or
//! [`SpmTokenizer::streaming_decoder`](crate::SpmTokenizer::streaming_decoder).
//! Everything that used to be the caller's choice — ByteLevel unmapping (GPT-2,
//! Llama, DeepSeek V3), the `special=true` ids to drop, the metaspace ▁
//! substitution, `<0xNN>` byte fallback, the SentencePiece dummy-prefix strip —
//! is taken from the tokenizer's own configuration, so a stream agrees with
//! whole-sequence decoding by construction.

mod decoder;
mod render;
mod state;
// Crate-internal: the Python bindings drive their own decoders off the same
// `Utf8Buffer`. Never re-exported, so it stays out of the public API.
pub(crate) mod utf8;

pub use decoder::StreamingDecoder;
pub(crate) use render::{ByteFallbackRule, Lead, RenderRules, Rendered, Surfaces};
pub(crate) use state::{DecodePost, DecodeState};
