//! ModernBERT.
//!
//! Extracted from `answerdotai/ModernBERT-base` on HuggingFace. The vocabulary
//! is upstream's and keeps upstream's licence — see this crate's `README.md`
//! and `LICENSE`.
//!
//! Data only. [`splintr`](https://docs.rs/splintr) re-exports these constants
//! under its `vocab-modernbert` feature, which is the way to reach them.

#![no_std]

/// ModernBERT vocabulary (50,254 merge ranks, byte-level BPE stored as raw
/// bytes).
///
/// This is the merged part only. ModernBERT's file also declares 114 added
/// tokens at ids 50,254-50,367 — 26 runs of 2 to 27 literal spaces, then
/// `[UNK]`/`[CLS]`/`[SEP]`/`[PAD]`/`[MASK]` and 83 `[unusedN]` slots — which
/// are matched whole and never merged, so they belong in splintr's special
/// table rather than in the rank file. The space runs are spelled literally
/// upstream rather than in the byte-level alphabet everything below 50,254
/// uses, which is the other reason they cannot live here.
pub const MODERNBERT_VOCAB_PACKED: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/modernbert.splv"));
