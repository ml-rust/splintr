//! Qwen 2 and 3 — and Baichuan-M2, which ships the same file.
//!
//! Extracted from `Qwen/Qwen3-8B` on HuggingFace. The vocabulary is upstream's and keeps
//! upstream's licence — see this crate's `README.md` and `LICENSE`.
//!
//! Data only. [`splintr`](https://docs.rs/splintr) re-exports these constants
//! under its `vocab-qwen` feature, which is the way to reach them.

#![no_std]

/// Qwen 2/3 vocabulary (151,643 tokens, byte-level BPE stored as raw bytes).
///
/// Also Baichuan-M2's: that checkpoint ships Qwen's tokenizer verbatim, all
/// 151,643 ids identical, so it is served by this file rather than a copy.
pub const QWEN3_VOCAB_PACKED: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/qwen3.splv"));
