//! GLM-4 and GLM-4.5.
//!
//! Extracted from `zai-org/GLM-4.5` on HuggingFace. The vocabulary is upstream's and keeps
//! upstream's licence — see this crate's `README.md`.
//!
//! Data only. [`splintr`](https://docs.rs/splintr) re-exports these constants
//! under its `vocab-glm` feature, which is the way to reach them.

#![no_std]

/// GLM-4/4.5 vocabulary (151,329 tokens, byte-level BPE stored as raw bytes).
pub const GLM4_VOCAB_PACKED: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/glm4.splv"));
