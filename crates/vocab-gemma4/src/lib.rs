//! Gemma 4.
//!
//! Extracted from `google/gemma-4-12B-it` on HuggingFace and converted into the
//! plain-text `.mbpe` format this crate ships, which packs to byte-identical
//! binaries. The vocabulary is upstream's and keeps upstream's licence —
//! Apache-2.0, see this crate's `README.md` and `LICENSE`.
//!
//! Data only. [`splintr`](https://docs.rs/splintr) re-exports these constants
//! under its `vocab-gemma` feature, which is the way to reach them.

#![no_std]

/// Gemma 4 vocabulary (262,144 pieces, SentencePiece-spelled: `▁` word
/// boundaries and `<0xNN>` byte fallback).
pub const GEMMA4_VOCAB_PACKED: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemma4.splv"));

/// Gemma 4 merge order — the token ids it merges in, highest priority first.
///
/// Shipped apart from the vocabulary because Gemma 4 needs both and they are
/// not the same order. Its ids and its merge priority disagree in 465 places,
/// and its 514,906 merges collapse onto 236,339 distinct tokens, so a single
/// rank per token — what a `.tiktoken` carries — cannot express it. Ranking by
/// id instead mistokenizes 8.1% of real documents.
pub const GEMMA4_MERGES_PACKED: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemma4.splm"));
