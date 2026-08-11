//! Gemma 3.
//!
//! Converted from Google's Gemma 3 `tokenizer.model` (MD5 `00d2276cbec4474f6cf3df98fbc18cbb`), id for id.
//! The vocabulary is upstream's and keeps upstream's licence — the Gemma Terms
//! of Use, see this crate's `LICENSE`, `NOTICE` and `README.md`.
//!
//! Data only. [`splintr`](https://docs.rs/splintr) re-exports this constant
//! under its `vocab-gemma3` feature, which is the way to reach it.

#![no_std]

/// Gemma 3 SentencePiece vocabulary (262,144 pieces with their scores and
/// piece types).
///
/// Carries the piece type as well as the score, which a `.tiktoken` cannot:
/// 6,410 of these pieces are `USER_DEFINED`, matched verbatim and never merged,
/// and a `USER_DEFINED` piece is indistinguishable from a `CONTROL` one by
/// score or spelling alone.
///
/// EmbeddingGemma ships this vocabulary byte-identically — all 262,144
/// pieces *and* scores — so this file serves it too.
pub const GEMMA3_SPM_VOCAB: &[u8] = include_bytes!("../vocabs/gemma3.spm");
