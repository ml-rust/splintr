//! Stating a trained vocabulary as a file splintr can load back.
//!
//! One writer per format, because the formats say different things:
//!
//! * [`tiktoken`] — BPE ranks and nothing else. splintr's native shape: the
//!   engine ranks merges by the token a merge produces, which is exactly what a
//!   rank file lists, so it round-trips through `Tokenizer::from_file` with no
//!   new loader. It cannot carry the pre-tokenizer or the special tokens; the
//!   caller supplies those, as they do for every other `.tiktoken`.
//! * [`bpe_json`] — a full `tokenizer.json` for a BPE vocabulary: vocabulary,
//!   ordered merges, pre-tokenizer and decoder chain, added tokens.
//! * [`vocab_txt`] — a WordPiece token list, the BERT-family format.
//! * [`wordpiece_json`] / [`unigram_json`] — `tokenizer.json` for the other two
//!   models.
//! * [`spm`] — splintr's SentencePiece text format, `base64(piece) score type`.
//!
//! Every writer is net-new: the main crate reads these formats and writes none
//! of them.

mod bpe_json;
mod model_json;
mod spm;
mod tiktoken;
mod vocab_txt;

pub use bpe_json::{bpe_json, bpe_json_file, BpeJsonOptions};
pub use model_json::{
    unigram_json, unigram_json_file, wordpiece_json, wordpiece_json_file, UnigramJsonOptions,
    WordPieceJsonOptions,
};
pub use spm::{spm, spm_file};
pub use tiktoken::{recipe_json, recipe_json_file, tiktoken, tiktoken_file};
pub use vocab_txt::{vocab_txt, vocab_txt_file};
