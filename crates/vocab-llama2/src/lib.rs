//! Llama 2 and Code Llama.
//!
//! Extracted from `codellama/CodeLlama-7b-hf` on HuggingFace. The vocabulary is
//! upstream's and keeps upstream's licence — see this crate's `README.md` and
//! `LICENSE`.
//!
//! Data only. [`splintr`](https://docs.rs/splintr) re-exports these constants
//! under its `vocab-llama2` feature, which is the way to reach them.

#![no_std]

/// Llama 2 SentencePiece vocabulary (32,000 pieces with their scores).
///
/// The vocabulary of Llama 2 itself and of everything that adopted it whole —
/// TinyLlama, Vicuna, WizardLM, Alpaca and the rest of that generation.
///
/// Derived at build time from the first 32,000 lines of
/// [`CODELLAMA_SPM_VOCAB`], which are Llama 2's pieces and Llama 2's scores
/// unchanged. Meta's own `tokenizer.model` is reachable only through a gated
/// repository; Code Llama's is not, and carries the identical prefix.
pub const LLAMA2_SPM_VOCAB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/llama2.spm"));

/// Code Llama SentencePiece vocabulary (32,016 pieces with their scores).
///
/// Llama 2's 32,000 followed by 16 infill pieces — `▁<PRE>`, `▁<MID>`,
/// `▁<SUF>`, `▁<EOT>` and the fragments they merge from. Those 16 carry
/// scores far below every genuine merge, so they never form except where the
/// text spells them out.
pub const CODELLAMA_SPM_VOCAB: &[u8] = include_bytes!("../vocabs/codellama.spm");
