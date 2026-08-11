//! DeepSeek V3 and R1.
//!
//! Extracted from `deepseek-ai/DeepSeek-V3` on HuggingFace. The vocabulary is upstream's and keeps
//! upstream's licence — see this crate's `README.md` and `LICENSE`.
//!
//! Data only. [`splintr`](https://docs.rs/splintr) re-exports these constants
//! under its `vocab-deepseek` feature, which is the way to reach them.

#![no_std]

/// DeepSeek V3/R1 vocabulary (128,000 merge ranks, byte-level BPE).
pub const DEEPSEEK_V3_VOCAB_PACKED: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/deepseek_v3.splv"));
