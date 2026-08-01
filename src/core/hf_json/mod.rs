//! Generic HuggingFace `tokenizer.json` loader.
//!
//! [`from_json_path`] / [`from_json_bytes`] read any HF `tokenizer.json` and
//! dispatch on `model.type` to the matching splintr backend:
//!
//! | `model.type` | Backend                       | Example models                     |
//! |--------------|-------------------------------|------------------------------------|
//! | `BPE`        | [`Tokenizer`](super::tokenizer::Tokenizer) (byte-level/raw)| GPT-2, Whisper, Llama 3, Qwen      |
//! | `Unigram`    | [`SentencePieceTokenizer`](super::sentencepiece::SentencePieceTokenizer)    | T5, Gemma, Albert, XLNet           |
//! | `WordPiece`  | [`WordPieceTokenizer`](super::wordpiece::WordPieceTokenizer)         | BERT, DistilBERT, Electra          |
//!
//! Everything needed is read from the file itself — "you supply the json, you
//! supply the tokens": the split regex, byte-level flag, BPE **merge order**
//! (independent of token ids, so RoBERTa-style vocabs work), the normalizer
//! (including SentencePiece's exact `Precompiled` charsmap), and special tokens.
//! Output is verified id-for-id against HuggingFace `tokenizers` across all three
//! families (GPT-2/RoBERTa/Qwen/Whisper; T5/Albert/XLNet; BERT/DistilBERT).
//!
//! For the bundled, zero-config vocabularies (including Whisper multilingual),
//! prefer [`crate::pretrained::from_pretrained`].

pub(super) mod components;
mod error;
mod loader;

pub use error::HfJsonError;
pub use loader::{from_json_bytes, from_json_path};

#[cfg(test)]
mod tests;
