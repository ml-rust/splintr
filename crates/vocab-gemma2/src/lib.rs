//! Gemma 2.
//!
//! Converted from Google's Gemma 2 `tokenizer.model` (MD5 `f9e2445870ec741aa6346bbd75531bb4`), id for id.
//! The vocabulary is upstream's and keeps upstream's licence — the Gemma Terms
//! of Use, see this crate's `LICENSE`, `NOTICE` and `README.md`.
//!
//! Data only. [`splintr`](https://docs.rs/splintr) re-exports this constant
//! under its `vocab-gemma2` feature, which is the way to reach it.

#![no_std]

/// Gemma 2 SentencePiece vocabulary (256,000 pieces with their scores and
/// piece types).
///
/// Carries the piece type as well as the score, which a `.tiktoken` cannot:
/// 245 of these pieces are `USER_DEFINED`, matched verbatim and never merged,
/// and a `USER_DEFINED` piece is indistinguishable from a `CONTROL` one by
/// score or spelling alone.
pub const GEMMA2_SPM_VOCAB: &[u8] = include_bytes!("../vocabs/gemma2.spm");
