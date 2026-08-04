//! UTF-8 safe streaming decoder for token-by-token LLM output.
//!
//! This module provides a stateful decoder that buffers incomplete UTF-8 sequences
//! and only emits complete, valid UTF-8 characters. This is critical for streaming
//! LLM output where token boundaries may not align with character boundaries.
//!
//! # ByteLevel Support
//!
//! For tokenizers using ByteLevel encoding (GPT-2, Llama, DeepSeek V3), the
//! [`ByteLevelStreamingDecoder`](crate::ByteLevelStreamingDecoder) handles the
//! ByteLevel-to-bytes conversion before UTF-8 assembly.

mod decoder;
// Crate-internal: the Python bindings drive their own decoders off the same
// `Utf8Buffer`. Never re-exported, so it stays out of the public API.
pub(crate) mod utf8;

pub use decoder::{ByteLevelStreamingDecoder, StreamingDecoder};
