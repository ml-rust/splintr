//! Stating a trained vocabulary as a file splintr can load back.
//!
//! Two formats, because they say different things:
//!
//! * [`tiktoken`] writes the ranks and nothing else. It is splintr's native
//!   shape — the engine ranks merges by the token a merge produces, which is
//!   exactly what a rank file lists — and it round-trips through
//!   `Tokenizer::from_file` with no new loader. It cannot carry the
//!   pre-tokenizer or the special tokens; the caller supplies those, as they do
//!   for every other `.tiktoken`.
//! * [`hf_json`] writes a full `tokenizer.json`: the vocabulary, the ordered
//!   merge list, the pre-tokenizer and decoder chain, and the added tokens. More
//!   is stated, and it is what other tooling reads.

mod hf_json;
mod tiktoken;

pub use hf_json::{hf_json, hf_json_file, HfJsonOptions};
pub use tiktoken::{tiktoken, tiktoken_file};
