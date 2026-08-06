//! The encode pipeline, in the order text moves through it.
//!
//! Every module here contributes inherent methods to the same [`Tokenizer`],
//! so the split is by *stage*, not by type:
//!
//! - [`prepare`] — prefix space, normalizer, pre-token splitting. Turns input
//!   text into the spans the rest of the pipeline works over.
//! - [`merge`] — byte-pair merging of one piece, plus the byte-fallback
//!   resolution that renders what the vocabulary cannot represent.
//! - [`chunk`] — one pre-token chunk to ids: ByteLevel encoding, the chunk
//!   cache, and the sequential/parallel map over a text's chunks.
//! - [`content`] — the whole-content pipeline that drives the fork between the
//!   pre-tokenizer engine, the metaspace fold and the plain chunk map.
//! - [`entry`] — the public `encode*` surface, and the special-token handling
//!   that wraps the content pipeline.
//!
//! [`Tokenizer`]: super::types::Tokenizer

mod chunk;
mod content;
mod entry;
mod merge;
mod prepare;
