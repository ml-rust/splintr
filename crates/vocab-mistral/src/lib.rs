//! Mistral V1, V2 and V3/Tekken.
//!
//! Extracted from `mistralai/Mistral-7B-Instruct-v0.3` (V1/V2 SentencePiece) and `mistralai/Mistral-Nemo-Instruct-2407` (V3/Tekken) on HuggingFace. The vocabulary is upstream's and keeps
//! upstream's licence — see this crate's `README.md` and `LICENSE`.
//!
//! Data only. [`splintr`](https://docs.rs/splintr) re-exports these constants
//! under its `vocab-mistral` feature, which is the way to reach them.

#![no_std]

/// Mistral V1 SentencePiece vocabulary (32,000 pieces with their scores).
///
/// Extracted straight from `tokenizer.model`, so pieces keep their
/// SentencePiece spelling (`<0x41>`, `▁▁`) and every score survives —
/// including the `-1e9` "never merge" sentinel on the 15 whitespace runs,
/// which the `.tiktoken` form of this vocabulary silently inverted into a
/// *preferred* merge.
pub const MISTRAL_SPM_VOCAB: &[u8] = include_bytes!("../vocabs/mistral.spm");

/// Mistral V2 SentencePiece vocabulary (32,768 pieces with their scores).
pub const MISTRAL_V2_SPM_VOCAB: &[u8] = include_bytes!("../vocabs/mistral_v2.spm");

/// Mistral V3/Tekken vocabulary (tiktoken-based, ~131k tokens).
pub const MISTRAL_V3_VOCAB_PACKED: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/mistral_v3_tekken.splv"));
