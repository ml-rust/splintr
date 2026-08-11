//! OpenAI Whisper (multilingual).
//!
//! Extracted from `openai/whisper-large-v3` on HuggingFace. The vocabulary is upstream's and keeps
//! upstream's licence — see this crate's `README.md` and `LICENSE`.
//!
//! Data only. [`splintr`](https://docs.rs/splintr) re-exports these constants
//! under its `vocab-whisper` feature, which is the way to reach them.

#![no_std]

/// Whisper base BPE vocabulary (GPT-2 byte-level, 50,257 tokens).
///
/// Shared by every multilingual variant (v1/v2/v3) — they differ only in the
/// programmatically-generated special tokens. The English-only checkpoints use
/// a different base BPE and are not bundled.
pub const WHISPER_VOCAB_PACKED: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/whisper.splv"));
