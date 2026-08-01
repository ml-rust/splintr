//! The vocabulary a GGUF file carries, as plain data.
//!
//! splintr never opens a GGUF container: parsing the header, the metadata
//! key-value block and the tensor table is the model runtime's job, and pulling
//! a GGUF parser into a tokenizer crate would make every consumer pay for it.
//! What splintr owns is the part that is pure tokenizer knowledge — which
//! algorithm `tokenizer.ggml.model` names, and how the surrounding flags have to
//! be honoured to reproduce llama.cpp's ids.
//!
//! So the caller reads the metadata block it has already parsed into this
//! struct, one field per `tokenizer.ggml.*` key, and hands it to
//! [`from_gguf_vocab`](super::from_gguf_vocab). Every field except `tokens`
//! is optional exactly as the GGUF key is, and `None` means "the file does not
//! say" — never "false" or "zero", because the defaults differ per dialect and
//! the loader is the one that knows them.

/// The `tokenizer.ggml.*` metadata of a GGUF file.
///
/// Field names mirror the GGUF keys with the `tokenizer.ggml.` prefix dropped.
#[derive(Debug, Clone, Default)]
pub struct GgufVocab {
    /// `tokenizer.ggml.model`: which tokenization *algorithm* the vocabulary was
    /// built with — `"bert"`, `"t5"`, `"llama"` or `"gpt2"`. llama.cpp treats an
    /// absent key as `"llama"`, so a caller whose file omits it passes
    /// `"llama"`.
    pub model: String,
    /// `tokenizer.ggml.tokens`: the vocabulary, indexed by token id.
    pub tokens: Vec<String>,
    /// `tokenizer.ggml.scores`: per-token score. Log-probabilities for `t5`
    /// (Unigram), merge ranks for `llama` (SentencePiece BPE), absent otherwise.
    pub scores: Option<Vec<f32>>,
    /// `tokenizer.ggml.merges`: `"a b"` pairs in priority order. Required by
    /// `gpt2`, which is *defined* by this list.
    pub merges: Option<Vec<String>>,
    /// `tokenizer.ggml.token_type`: the GGUF token-type enum per id, where
    /// `3` == CONTROL. Every dialect uses it to find the special tokens: they
    /// must be matched verbatim in the input and reachable by name.
    pub token_type: Option<Vec<u32>>,
    /// `tokenizer.ggml.add_space_prefix` (SentencePiece `add_dummy_prefix`),
    /// defaulting to true.
    pub add_space_prefix: Option<bool>,
    /// `tokenizer.ggml.remove_extra_whitespaces`, defaulting to false. Read only
    /// by the `t5` path, where it decides both whether the first word is marked
    /// (see `unigram_prefix_space` in the loader) and whether a run of spaces
    /// collapses to a single `▁` piece.
    pub remove_extra_whitespaces: Option<bool>,
    /// `tokenizer.ggml.add_bos_token`.
    pub add_bos_token: Option<bool>,
    /// `tokenizer.ggml.add_eos_token`.
    pub add_eos_token: Option<bool>,
    /// `tokenizer.ggml.bos_token_id`.
    pub bos_token_id: Option<u32>,
    /// `tokenizer.ggml.eos_token_id`.
    pub eos_token_id: Option<u32>,
    /// `tokenizer.ggml.unknown_token_id`.
    pub unknown_token_id: Option<u32>,
    /// `tokenizer.ggml.padding_token_id`.
    pub padding_token_id: Option<u32>,
    /// `tokenizer.ggml.cls_token_id`.
    pub cls_token_id: Option<u32>,
    /// `tokenizer.ggml.sep_token_id`.
    pub sep_token_id: Option<u32>,
    /// `tokenizer.ggml.pre`: which pre-tokenizer a `gpt2` vocabulary was built
    /// with. llama.cpp treats an absent key as `"default"`.
    pub pre: Option<String>,
    /// `tokenizer.ggml.precompiled_charsmap`: SentencePiece's normalization
    /// table, verbatim — a darts-clone trie mapping input byte sequences to
    /// their normalized form. GGUF stores it as a `UINT8` array, so a caller
    /// passes those bytes unchanged.
    ///
    /// Read by the `t5` (Unigram) path, which is where llama.cpp applies it
    /// (`llm_tokenizer_ugm::normalize`). Without it the characters the table
    /// folds — tab, newline, NBSP, ZWJ, fullwidth punctuation — reach Viterbi
    /// as themselves, and a Unigram vocabulary that only ever saw their
    /// normalized forms has no piece for them, so each one becomes `<unk>`.
    ///
    /// `None` means the file declares no table, which is the common case: only
    /// SentencePiece-derived vocabularies carry one.
    pub precompiled_charsmap: Option<Vec<u8>>,
}
