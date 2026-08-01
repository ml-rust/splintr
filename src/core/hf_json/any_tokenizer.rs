//! The tagged-union tokenizer handle returned by the HF json loader.

use super::super::sentencepiece::SentencePieceTokenizer;
use super::super::spm::SpmTokenizer;
use super::super::tokenize::{Tokenize, TokenizeError};
use super::super::tokenizer::Tokenizer;
use super::super::wordpiece::WordPieceTokenizer;

use super::policy::SpecialPolicy;
use super::HfJsonError;

/// A tokenizer loaded from a `tokenizer.json`, tagged by its backend family.
///
/// Implements [`Tokenize`] so callers can encode/decode generically, or match
/// on the variant for backend-specific APIs.
pub enum Backend {
    /// Byte-pair encoding (byte-level or raw).
    Bpe(Tokenizer),
    /// Unigram / SentencePiece.
    Unigram(SentencePieceTokenizer),
    /// WordPiece (BERT family).
    WordPiece(WordPieceTokenizer),
    /// SentencePiece **BPE** (llama.cpp `SPM` vocabularies).
    Spm(SpmTokenizer),
}

impl Backend {
    /// The raw surface string of a token id, used to feed a declared decoder
    /// pipeline.
    fn token_surface(&self, id: u32) -> Option<String> {
        match self {
            Backend::Bpe(t) => t.token_surface(id),
            Backend::Unigram(t) => t.token_surface(id),
            Backend::WordPiece(t) => t.token_surface(id),
            Backend::Spm(t) => t.token_surface(id),
        }
    }
}

/// A loaded tokenizer: a backend family plus the special-token policy parsed
/// from the same file.
///
/// The policy — not the caller and not the backend — owns boundary tokens.
/// [`encode`](AnyTokenizer::encode) applies the single-sequence template (HF's
/// default `add_special_tokens=True`), [`encode_pair`](AnyTokenizer::encode_pair)
/// the pair template, and [`encode_raw`](AnyTokenizer::encode_raw) gives the
/// bare backend output for callers assembling their own sequence.
pub struct AnyTokenizer {
    pub(super) backend: Backend,
    pub(super) policy: SpecialPolicy,
    /// The `decoder` pipeline declared in the json. When present it drives
    /// decoding (config-driven); when absent the backend's built-in decode runs.
    pub(super) decoder: Option<super::super::decoder::Decoder>,
    /// Ids of `special=true` added tokens, skipped before the decoder pipeline.
    pub(super) special_decode: rustc_hash::FxHashSet<u32>,
}

impl AnyTokenizer {
    /// Pair a backend with a special-token policy.
    ///
    /// Decoding uses the backend's own; callers loading a `tokenizer.json` get
    /// the declared `decoder` pipeline through [`from_json_bytes`](super::from_json_bytes) instead.
    pub fn new(backend: Backend, policy: SpecialPolicy) -> Self {
        Self {
            backend,
            policy,
            decoder: None,
            special_decode: rustc_hash::FxHashSet::default(),
        }
    }

    /// The `model.type` family name this was built from.
    pub fn family(&self) -> &'static str {
        match &self.backend {
            Backend::Bpe(_) => "BPE",
            Backend::Unigram(_) => "Unigram",
            Backend::WordPiece(_) => "WordPiece",
            Backend::Spm(_) => "Spm",
        }
    }

    /// Borrow the backend tokenizer (to reach backend-specific APIs).
    pub fn backend(&self) -> &Backend {
        &self.backend
    }

    /// Consume into the backend tokenizer.
    pub fn into_backend(self) -> Backend {
        self.backend
    }

    /// The special-token policy parsed from the json.
    pub fn policy(&self) -> &SpecialPolicy {
        &self.policy
    }

    /// Encode one sequence and apply the policy's single-sequence template.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.policy.apply_single(self.encode_raw(text))
    }

    /// Encode without applying the policy — the backend's content tokens alone
    /// (HF's `add_special_tokens=False`).
    pub fn encode_raw(&self, text: &str) -> Vec<u32> {
        match &self.backend {
            Backend::Bpe(t) => Tokenize::encode(t, text),
            Backend::Unigram(t) => Tokenize::encode(t, text),
            Backend::WordPiece(t) => Tokenize::encode(t, text),
            Backend::Spm(t) => Tokenize::encode(t, text),
        }
    }

    /// Encode two sequences into one input using the policy's pair template
    /// (a reranker's `[CLS] query [SEP] document [SEP]`).
    ///
    /// Errors when the tokenizer defines no pair template rather than
    /// concatenating the two halves without a separator.
    pub fn encode_pair(&self, a: &str, b: &str) -> Result<Vec<u32>, HfJsonError> {
        self.policy
            .apply_pair(&self.encode_raw(a), &self.encode_raw(b))
    }

    /// Whether `id` is the end-of-sequence token.
    pub fn is_eos(&self, id: u32) -> bool {
        self.policy.is_eos(id)
    }

    /// The end-of-sequence token id, when the json names one.
    pub fn eos_token_id(&self) -> Option<u32> {
        self.policy.eos_token_id()
    }

    /// The id of an added token by its content (e.g. `"[CLS]"`).
    pub fn special_token_id(&self, name: &str) -> Option<u32> {
        self.policy.special_token_id(name)
    }
}

impl Tokenize for AnyTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        AnyTokenizer::encode(self, text)
    }

    fn decode(&self, ids: &[u32]) -> Result<String, TokenizeError> {
        // When the json declares a `decoder`, drive decoding from it: collect the
        // surface strings (skipping special-flagged added tokens, matching HF's
        // default `skip_special_tokens=true`) and run the configured pipeline.
        if let Some(decoder) = &self.decoder {
            let surfaces: Vec<String> = ids
                .iter()
                .filter(|id| !self.special_decode.contains(id))
                .filter_map(|&id| self.backend.token_surface(id))
                .collect();
            return Ok(decoder.decode(surfaces));
        }
        match &self.backend {
            Backend::Bpe(t) => Tokenize::decode(t, ids),
            Backend::Unigram(t) => Tokenize::decode(t, ids),
            Backend::WordPiece(t) => Tokenize::decode(t, ids),
            Backend::Spm(t) => Tokenize::decode(t, ids),
        }
    }

    fn vocab_size(&self) -> usize {
        match &self.backend {
            Backend::Bpe(t) => Tokenize::vocab_size(t),
            Backend::Unigram(t) => Tokenize::vocab_size(t),
            Backend::WordPiece(t) => Tokenize::vocab_size(t),
            Backend::Spm(t) => Tokenize::vocab_size(t),
        }
    }
}
