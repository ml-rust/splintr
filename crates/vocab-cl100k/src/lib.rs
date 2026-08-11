//! OpenAI cl100k_base — GPT-4 and GPT-3.5-turbo.
//!
//! Extracted from OpenAI, `https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken`. The vocabulary is upstream's and keeps
//! upstream's licence — see this crate's `README.md` and `LICENSE`.
//!
//! Data only. [`splintr`](https://docs.rs/splintr) re-exports these constants
//! under its `vocab-cl100k` feature, which is the way to reach them.

#![no_std]

/// OpenAI cl100k_base (100,256 merge ranks, byte-level BPE).
pub const CL100K_BASE_VOCAB_PACKED: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/cl100k_base.splv"));
