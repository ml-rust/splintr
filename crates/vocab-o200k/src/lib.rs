//! OpenAI o200k_base — GPT-4o, and the ranks gpt-oss layers its harmony specials over.
//!
//! Extracted from OpenAI, `https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken`. The vocabulary is upstream's and keeps
//! upstream's licence — see this crate's `README.md` and `LICENSE`.
//!
//! Data only. [`splintr`](https://docs.rs/splintr) re-exports these constants
//! under its `vocab-o200k` feature, which is the way to reach them.

#![no_std]

/// OpenAI o200k_base (199,998 merge ranks, byte-level BPE).
pub const O200K_BASE_VOCAB_PACKED: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/o200k_base.splv"));
