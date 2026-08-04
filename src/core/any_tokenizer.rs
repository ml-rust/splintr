//! The tagged-union tokenizer handle returned by the HF json loader.

use super::policy::{PolicyError, SpecialMode, SpecialPolicy};
use super::sentencepiece::SentencePieceTokenizer;
use super::spm::SpmTokenizer;
use super::streaming::{DecodeState, StreamingDecoder, Surfaces};
use super::tokenize::{Tokenize, TokenizeError};
use super::tokenizer::{Tokenizer, TokenizerError};
use super::wordpiece::WordPieceTokenizer;
use std::sync::Arc;

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

    /// The backend's own streaming decoder, configured from its own vocabulary.
    ///
    /// Only ever reached when the json declared no `decoder` pipeline — the same
    /// condition under which `AnyTokenizer::decode_inner` delegates to that
    /// backend's whole-sequence decode.
    fn streaming_decoder(&self) -> StreamingDecoder {
        match self {
            Backend::Bpe(t) => t.streaming_decoder(),
            Backend::Unigram(t) => t.streaming_decoder(),
            Backend::WordPiece(t) => t.streaming_decoder(),
            Backend::Spm(t) => t.streaming_decoder(),
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

    /// Whether the source declared a `decoder` pipeline that
    /// [`decode`](Self::decode) drives.
    ///
    /// A caller reaching *past* this handle for a backend's own decode (that
    /// backend's `streaming_decoder`, `decode_bytes`) must consult this first:
    /// when a pipeline is declared, those paths skip it and render the
    /// backend's pieces (`▁hello▁world`) instead of text. Streaming through
    /// this handle's own [`streaming_decoder`](Self::streaming_decoder) does
    /// run the declared pipeline — except for the shapes that cannot be
    /// evaluated incrementally at all, which it refuses rather than answers
    /// wrongly.
    pub fn declares_decoder(&self) -> bool {
        self.decoder.is_some()
    }

    /// Switch the BPE backend's regex engine in place — see
    /// [`Tokenizer::pcre2`].
    ///
    /// Configures the handle rather than returning a new one, so the policy,
    /// the declared `decoder` pipeline and the `special=true` id set travel
    /// with it untouched; rebuilding the handle around a reconfigured backend
    /// would have to re-derive all three.
    ///
    /// # Errors
    /// [`TokenizerError::NotBpeBackend`] for any other backend family: the
    /// option configures a regex pre-tokenizer, and Unigram/WordPiece/SPM have
    /// none to configure.
    pub fn set_pcre2(&mut self, use_pcre2: bool) -> Result<(), TokenizerError> {
        self.reconfigure_bpe(|bpe| bpe.pcre2(use_pcre2))
    }

    /// Enable or disable JIT compilation for the BPE backend's regex engine in
    /// place — see [`Tokenizer::jit`]. Errors as [`Self::set_pcre2`] does.
    pub fn set_jit(&mut self, use_jit: bool) -> Result<(), TokenizerError> {
        self.reconfigure_bpe(|bpe| bpe.jit(use_jit))
    }

    /// Apply one of [`Tokenizer`]'s consuming builder steps to the BPE backend
    /// held here, leaving every other field of this handle alone.
    ///
    /// The clone is cheap: [`Tokenizer`]'s `Clone` shares the compiled regex
    /// and the later pre-tokenizer passes through their `Arc`s.
    fn reconfigure_bpe<F>(&mut self, step: F) -> Result<(), TokenizerError>
    where
        F: FnOnce(Tokenizer) -> Result<Tokenizer, TokenizerError>,
    {
        // Read the family name up front: it is `&'static str`, so the borrow it
        // needs ends here rather than fighting the `&mut self.backend` below.
        let family = self.family();
        match &mut self.backend {
            Backend::Bpe(bpe) => {
                *bpe = step(bpe.clone())?;
                Ok(())
            }
            _ => Err(TokenizerError::NotBpeBackend(family)),
        }
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

    /// Encode many sequences under an explicit [`SpecialMode`], applying the
    /// policy's single-sequence template to each — the batch form of
    /// [`encode_with`](Self::encode_with), as [`encode_batch`](Self::encode_batch)
    /// is the batch form of [`encode`](Self::encode).
    ///
    /// Fails as a whole if any one text violates the mode's allow-list, rather
    /// than returning a partly-encoded batch with the offending entry silently
    /// dropped.
    pub fn encode_batch_with(
        &self,
        texts: &[&str],
        mode: &SpecialMode<'_>,
    ) -> Result<Vec<Vec<u32>>, PolicyError> {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            texts
                .par_iter()
                .map(|&text| self.encode_with(text, mode))
                .collect()
        }
        #[cfg(not(feature = "rayon"))]
        {
            texts
                .iter()
                .map(|&text| self.encode_with(text, mode))
                .collect()
        }
    }

    /// [`encode`](Self::encode) with the work parallelized *within* the single
    /// text, where the backend supports it.
    ///
    /// Same semantics and same ids as [`encode`](Self::encode) — only the
    /// execution strategy differs, and it pays off only for very large inputs
    /// (typically >1MB) where the split work outweighs the thread-pool
    /// coordination. Backends with no intra-text parallel path simply run
    /// [`encode`](Self::encode), so the result never depends on which one this
    /// handle holds.
    pub fn encode_rayon(&self, text: &str) -> Vec<u32> {
        match &self.backend {
            Backend::Bpe(t) => self.policy.apply_single(t.encode_rayon(text)),
            // Unigram, WordPiece and SPM merge sequentially; there is no
            // intra-text split to parallelize, so this *is* their fast path.
            _ => self.encode(text),
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

    /// Decode ids back to text, running whatever decode pipeline the source
    /// declared and dropping the ids marked `special = true`.
    ///
    /// Inherent so the universal handle is usable without importing
    /// [`Tokenize`] — every other entry point on this type already is, and
    /// `decode` being trait-only made it the one method a caller had to reach
    /// for a trait to spell. The trait impl delegates here, so the two can
    /// never disagree.
    pub fn decode(&self, ids: &[u32]) -> Result<String, TokenizeError> {
        self.decode_inner(ids)
    }

    /// The one decode implementation, shared by the inherent [`Self::decode`]
    /// and the [`Tokenize`] impl so neither can drift from the other.
    fn decode_inner(&self, ids: &[u32]) -> Result<String, TokenizeError> {
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

    /// Decode many id lists — the batch form of [`decode`](Self::decode),
    /// running the same declared pipeline and the same `special = true` skip.
    ///
    /// Parallel across lists when the `rayon` feature is on.
    pub fn decode_batch(&self, token_lists: &[Vec<u32>]) -> Result<Vec<String>, TokenizeError> {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            token_lists
                .par_iter()
                .map(|ids| self.decode_inner(ids))
                .collect()
        }
        #[cfg(not(feature = "rayon"))]
        {
            token_lists
                .iter()
                .map(|ids| self.decode_inner(ids))
                .collect()
        }
    }

    /// A [`StreamingDecoder`] that reproduces this handle's
    /// [`decode`](Self::decode).
    ///
    /// The decision mirrors [`decode`](Self::decode) exactly, because the two
    /// must never disagree about what a sequence of ids says:
    ///
    /// * A declared `decoder` pipeline drives the stream, over this handle's
    ///   token surfaces and with the same `special = true` ids dropped.
    /// * With no pipeline declared, the backend's own factory answers — the
    ///   same delegation whole-sequence decoding does.
    ///
    /// # Errors
    /// [`TokenizeError::UnstreamableDecoder`], naming the step, when a pipeline
    /// *is* declared but one of its ops cannot be evaluated one chunk at a time
    /// (a `BPEDecoder`, a trailing `Strip`, a `Replace` over the fused text —
    /// see `Decoder::lower`). Falling back to the backend's own decode would
    /// answer with the raw pieces (`▁hello▁world`) the declared pipeline exists
    /// to turn into text, so this refuses instead. Whole-sequence
    /// [`decode`](Self::decode) still handles those pipelines.
    ///
    /// Unlike a backend's own factory, this one materializes the surface table
    /// it renders through — a pipeline is declared over surface *strings*, and
    /// only the whole vocabulary as strings can be rendered that way — so it
    /// costs one pass over the vocabulary. Build the decoder once per stream,
    /// not once per token.
    ///
    /// One id is treated differently from whole-sequence decoding, and in the
    /// direction every other stream in this crate already takes: an id outside
    /// the vocabulary entirely, which [`decode`](Self::decode) drops silently,
    /// is reported by [`StreamingDecoder::add_token`] and skipped by
    /// [`add_token_lossy`](StreamingDecoder::add_token_lossy) — the same strict
    /// and lossy pair every backend's stream offers.
    pub fn streaming_decoder(&self) -> Result<StreamingDecoder, TokenizeError> {
        let Some(decoder) = &self.decoder else {
            return Ok(self.backend.streaming_decoder());
        };
        let Some((rules, post)) = decoder.lower() else {
            // The two are exact complements of one lowering pass, so the
            // fallback spelling below is unreachable; it exists only because the
            // type system cannot say so.
            return Err(TokenizeError::UnstreamableDecoder(
                decoder.unstreamable_op().unwrap_or("declared"),
            ));
        };

        // The surfaces the declared pipeline runs over, exactly as
        // `decode_inner` collects them: `token_surface` per id, `special_decode`
        // dropped ahead of it. An id with no surface at all is dropped too —
        // which is a *skip* here, since a rendering rule reads a dense table and
        // an empty slot would otherwise render as an empty surface (and, on a
        // WordPiece pipeline, carry a word separator with it).
        let vocab_size = Tokenize::vocab_size(self);
        let mut surfaces = Vec::with_capacity(vocab_size);
        let mut skip = self.special_decode.clone();
        for id in 0..vocab_size {
            let id = id as u32;
            match self.backend.token_surface(id) {
                Some(surface) => surfaces.push(surface),
                None => {
                    skip.insert(id);
                    surfaces.push(String::new());
                }
            }
        }

        let rules = rules.with_vocabulary(
            Surfaces::ByIndex(Arc::new(surfaces)),
            // No separate special-token table: `token_surface` already answers
            // for the special ids the backend knows, so every id the declared
            // pipeline can see has a slot above.
            Arc::new(rustc_hash::FxHashMap::default()),
            Arc::new(skip),
        );
        Ok(StreamingDecoder::new(Arc::new(DecodeState::new(
            rules, post,
        ))))
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
        self.decode_inner(ids)
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

    /// Reconfiguring the regex engine must keep the handle intact — same ids,
    /// same policy — rather than quietly returning a bare backend.
    #[test]
    fn set_jit_reconfigures_the_bpe_backend_in_place() {
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
        let mut any = AnyTokenizer::new(Backend::Bpe(tokenizer), policy);

        let before = any.encode("hi there");
        any.set_jit(false).expect("BPE backend accepts the option");

        assert_eq!(any.encode("hi there"), before, "ids must not depend on JIT");
        assert_eq!(any.family(), "BPE");
        assert_eq!(
            any.encode("hi").first(),
            Some(&1000),
            "the policy's BOS template must survive the reconfiguration"
        );
    }

    /// A backend with no regex pre-tokenizer must refuse the option rather than
    /// report a switch that did not happen — a caller told "done" would believe
    /// it had changed engines and never find out otherwise.
    #[test]
    fn set_pcre2_refuses_on_a_non_bpe_backend() {
        let vocab = vec!["[UNK]".to_string(), "hello".to_string()];
        let mut any = AnyTokenizer::new(
            Backend::WordPiece(WordPieceTokenizer::new(vocab, 0, 100, false)),
            SpecialPolicy::default(),
        );

        let err = any.set_pcre2(true).expect_err("WordPiece has no regex");
        assert!(
            matches!(err, TokenizerError::NotBpeBackend("WordPiece")),
            "unexpected error: {err}"
        );
        assert!(any.set_jit(false).is_err());
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
