//! HuggingFace pre-tokenizer pipeline.
//!
//! HF pre-tokenizers form an ordered pipeline: each stage takes the current list
//! of string pieces and splits each one further. A single regex can't model that
//! (e.g. Falcon = `Punctuation → ByteLevel → Digits → Split`), so this engine
//! applies the stages in order and returns the final pre-token pieces.
//!
//! When a `ByteLevel` stage is present the pieces come out byte-level-encoded
//! (each byte mapped to a printable code point), ready for BPE against a
//! byte-level vocab — so the consumer must NOT byte-level-encode again.
//!
//! The pipeline is split by role: `spec` is the public description a caller
//! builds, `stage` is its compiled counterpart, `split` holds the splitters
//! every stage is made of, `pipeline` runs them in order, and `parse` reads
//! a HuggingFace `pre_tokenizer` JSON node into a spec. Those modules are
//! private, so they are named here rather than linked.

mod parse;
mod pipeline;
mod spec;
mod split;
mod stage;
#[cfg(test)]
mod tests;

pub(crate) use parse::parse;
pub use pipeline::PreTokenizer;
pub use spec::{PreTokStage, SplitBehavior, SplitPattern};
