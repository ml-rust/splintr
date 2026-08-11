//! Moonshot Kimi K2 and K3.
//!
//! Extracted from `moonshotai/Kimi-K2-Instruct` on HuggingFace. The vocabulary is upstream's and keeps
//! upstream's licence — see this crate's `README.md` and `LICENSE`.
//!
//! Data only. [`splintr`](https://docs.rs/splintr) re-exports these constants
//! under its `vocab-kimi` feature, which is the way to reach them.

#![no_std]

/// Kimi vocabulary (163,584 merge ranks, byte-level BPE stored as raw bytes).
///
/// Moonshot's own `tiktoken.model`, unmodified — it is already in this format.
/// Byte-identical across Kimi K2, K2.5, K2.6, K2.7, K3, Kimi-Linear and Kimi-VL,
/// so one payload serves the whole family. Only the 256-slot special block above
/// it differs between K2 and K3, which is why those are separate variants.
pub const KIMI_VOCAB_PACKED: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kimi.splv"));
