//! Meta Llama 3, 3.1, 3.2 and 3.3.
//!
//! Extracted from `meta-llama/Llama-3.1` on HuggingFace. The vocabulary is upstream's and keeps
//! upstream's licence — see this crate's `README.md` and `LICENSE`.
//!
//! Data only. [`splintr`](https://docs.rs/splintr) re-exports these constants
//! under its `vocab-llama3` feature, which is the way to reach them.

#![no_std]

/// Llama 3 vocabulary (128,000 merge ranks, byte-level BPE).
pub const LLAMA3_VOCAB_PACKED: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/llama3.splv"));
