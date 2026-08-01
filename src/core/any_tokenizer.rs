//! The tagged-union tokenizer handle returned by the HF json loader.

use super::policy::{PolicyError, SpecialMode, SpecialPolicy};
use super::sentencepiece::SentencePieceTokenizer;
use super::spm::SpmTokenizer;
use super::tokenize::{Tokenize, TokenizeError};
use super::tokenizer::Tokenizer;
use super::wordpiece::WordPieceTokenizer;

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
    pub(super) decoder: Option<super::decoder::Decoder>,
    /// Ids of `special=true` added tokens, skipped before the decoder pipeline.
    pub(super) special_decode: rustc_hash::FxHashSet<u32>,
}

impl AnyTokenizer {
    /// Pair a backend with a special-token policy.
    ///
    /// Decoding uses the backend's own; callers loading a `tokenizer.json` get
    /// the declared `decoder` pipeline through [`from_json_bytes`](super::hf_json::from_json_bytes) instead.
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

    /// Encode one sequence under an explicit [`SpecialMode`], then apply the
    /// policy's single-sequence template — the mode-aware sibling of
    /// [`encode`](Self::encode).
    ///
    /// Boundary tokens (BOS/EOS/CLS/SEP) come from the policy template, NOT
    /// from matching text against the vocabulary, so they are applied under
    /// EVERY mode including [`SpecialMode::Ordinary`]: refusing to match a
    /// special token spelled out in user-supplied text says nothing about
    /// whether this tokenizer itself wraps content in its own boundary
    /// tokens — those two concerns are independent, and conflating them would
    /// mean a caller who locks down special-token matching for safety
    /// unexpectedly also loses the boundary tokens the model was trained
    /// with.
    pub fn encode_with(&self, text: &str, mode: &SpecialMode<'_>) -> Result<Vec<u32>, PolicyError> {
        let raw = match &self.backend {
            Backend::Bpe(t) => Tokenize::encode_with(t, text, mode),
            Backend::Unigram(t) => Tokenize::encode_with(t, text, mode),
            Backend::WordPiece(t) => Tokenize::encode_with(t, text, mode),
            Backend::Spm(t) => Tokenize::encode_with(t, text, mode),
        }?;
        Ok(self.policy.apply_single(raw))
    }

    /// Encode many sequences, applying the policy's single-sequence template to
    /// each — the batch form of [`encode`](Self::encode).
    ///
    /// Parallel across texts when the `rayon` feature is on, mirroring
    /// [`Tokenizer::encode_batch`](super::tokenizer::Tokenizer::encode_batch);
    /// the parallelism lives here rather than in one backend so every family
    /// gets it.
    pub fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<u32>> {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            texts.par_iter().map(|&text| self.encode(text)).collect()
        }
        #[cfg(not(feature = "rayon"))]
        {
            texts.iter().map(|&text| self.encode(text)).collect()
        }
    }

    /// Encode two sequences into one input using the policy's pair template
    /// (a reranker's `[CLS] query [SEP] document [SEP]`).
    ///
    /// Errors when the tokenizer defines no pair template rather than
    /// concatenating the two halves without a separator.
    pub fn encode_pair(&self, a: &str, b: &str) -> Result<Vec<u32>, PolicyError> {
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

    fn encode_with(&self, text: &str, mode: &SpecialMode<'_>) -> Result<Vec<u32>, PolicyError> {
        AnyTokenizer::encode_with(self, text, mode)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The policy's boundary template (BOS here) must still be applied under
    /// `SpecialMode::Ordinary` — that mode only turns off matching a special
    /// token's literal spelling *in the content*, it says nothing about the
    /// boundary tokens the loaded tokenizer always wraps a sequence in.
    #[test]
    fn boundary_template_still_applies_under_ordinary_mode() {
        let mut encoder = rustc_hash::FxHashMap::default();
        for b in 32u8..=126 {
            encoder.insert(vec![b], b as u32);
        }
        let mut special_tokens = rustc_hash::FxHashMap::default();
        special_tokens.insert("<s>".to_string(), 1000);

        let tokenizer = Tokenizer::new(encoder, special_tokens.clone(), r"\S+|\s+")
            .unwrap()
            .with_added_token_matching(true);

        let policy = SpecialPolicy::boundary(Some(1000), None, None, special_tokens);
        let any = AnyTokenizer::new(Backend::Bpe(tokenizer), policy);

        let ids = any
            .encode_with("hi", &SpecialMode::Ordinary)
            .expect("ordinary mode never refuses on this input");

        assert_eq!(ids.first(), Some(&1000), "BOS from the policy template");
        // Content tokens should be exactly "hi" encoded byte-by-byte, unmodified.
        assert_eq!(&ids[1..], &[b'h' as u32, b'i' as u32]);
    }

    /// `AnyTokenizer::decode` must run the declared `decoder` pipeline and drop
    /// the `special_decode` ids first — those two fields are the whole reason
    /// decoding is config-driven rather than inferred from the backend, and a
    /// handle that carries them but ignores them decodes to raw pieces
    /// (`▁hello▁world`) instead of text.
    ///
    /// The pipeline here is Mistral's, verbatim from its `tokenizer.json`.
    #[test]
    fn decode_applies_declared_pipeline_and_skips_special_ids() {
        let mut encoder = rustc_hash::FxHashMap::default();
        encoder.insert("\u{2581}hello".as_bytes().to_vec(), 10);
        encoder.insert("\u{2581}world".as_bytes().to_vec(), 11);
        let mut special_tokens = rustc_hash::FxHashMap::default();
        special_tokens.insert("<s>".to_string(), 1);

        let tokenizer = Tokenizer::new(encoder, special_tokens.clone(), r"\S+|\s+").unwrap();
        let policy = SpecialPolicy::boundary(Some(1), None, None, special_tokens);

        let declared = serde_json::json!({
            "type": "Sequence",
            "decoders": [
                {"type": "Replace", "pattern": {"String": "\u{2581}"}, "content": " "},
                {"type": "ByteFallback"},
                {"type": "Fuse"},
                {"type": "Strip", "content": " ", "start": 1, "stop": 0}
            ]
        });

        let ids = [1, 10, 11];

        // Without the pipeline the backend's own decode renders the pieces raw —
        // this is exactly the wrong output the pipeline exists to prevent.
        let bare = AnyTokenizer::new(Backend::Bpe(tokenizer.clone()), policy.clone());
        assert_eq!(
            Tokenize::decode(&bare, &ids).unwrap(),
            "<s>\u{2581}hello\u{2581}world"
        );

        let configured = AnyTokenizer {
            backend: Backend::Bpe(tokenizer),
            policy,
            decoder: super::super::decoder::parse(Some(&declared)),
            special_decode: [1].into_iter().collect(),
        };
        assert!(
            configured.decoder.is_some(),
            "the declared Sequence decoder must parse"
        );
        assert_eq!(
            Tokenize::decode(&configured, &ids).unwrap(),
            "hello world",
            "declared decoder pipeline + special_decode must both apply"
        );
    }
}
