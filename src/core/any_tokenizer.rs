//! The tagged-union tokenizer handle returned by the HF json loader.

use super::policy::{PolicyError, SpecialDecode, SpecialMode, SpecialPolicy};
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

    /// The backend's own streaming decoder, configured from its own vocabulary
    /// and from an explicit [`SpecialDecode`].
    ///
    /// Only ever reached when the json declared no `decoder` pipeline — the same
    /// condition under which `AnyTokenizer::decode_inner` delegates to that
    /// backend's whole-sequence decode.
    fn streaming_decoder_with(&self, specials: SpecialDecode) -> StreamingDecoder {
        match self {
            Backend::Bpe(t) => t.streaming_decoder_with(specials),
            Backend::Unigram(t) => t.streaming_decoder_with(specials),
            Backend::WordPiece(t) => t.streaming_decoder_with(specials),
            Backend::Spm(t) => t.streaming_decoder_with(specials),
        }
    }

    /// The backend's own strict whole-sequence decode under an explicit
    /// [`SpecialDecode`]. Reached under exactly the condition
    /// [`streaming_decoder_with`](Self::streaming_decoder_with) is.
    fn decode_with(&self, ids: &[u32], specials: SpecialDecode) -> Result<String, TokenizeError> {
        match self {
            Backend::Bpe(t) => Tokenize::decode_with(t, ids, specials),
            Backend::Unigram(t) => Tokenize::decode_with(t, ids, specials),
            Backend::WordPiece(t) => Tokenize::decode_with(t, ids, specials),
            Backend::Spm(t) => Tokenize::decode_with(t, ids, specials),
        }
    }

    /// The backend's own lossy whole-sequence decode. Reached under exactly the
    /// condition [`streaming_decoder_with`](Self::streaming_decoder_with) is.
    fn decode_lossy(&self, ids: &[u32]) -> String {
        match self {
            Backend::Bpe(t) => Tokenize::decode_lossy(t, ids),
            Backend::Unigram(t) => Tokenize::decode_lossy(t, ids),
            Backend::WordPiece(t) => Tokenize::decode_lossy(t, ids),
            Backend::Spm(t) => Tokenize::decode_lossy(t, ids),
        }
    }

    /// The backend's own per-id rendering. Reached under exactly the condition
    /// [`streaming_decoder_with`](Self::streaming_decoder_with) is.
    fn decode_token_bytes(&self, id: u32) -> Result<Vec<u8>, TokenizeError> {
        match self {
            Backend::Bpe(t) => Tokenize::decode_token_bytes(t, id),
            Backend::Unigram(t) => Tokenize::decode_token_bytes(t, id),
            Backend::WordPiece(t) => Tokenize::decode_token_bytes(t, id),
            Backend::Spm(t) => Tokenize::decode_token_bytes(t, id),
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

    /// The pre-token pieces this handle's encode path splits `text` into, or
    /// `None` when this backend has no such stage to report.
    ///
    /// `Some` only for [`Backend::Bpe`], where pre-tokenization is a distinct
    /// stage — see [`Tokenizer::pre_tokenize`] for what `text` must already be
    /// (normalized, unprefixed) and which byte space the pieces come back in.
    /// [`normalize`](Self::normalize) is the stage that produces it.
    ///
    /// `None` is the honest answer rather than a fabricated one for the rest.
    /// [`Backend::Unigram`] and [`Backend::Spm`] are SentencePiece: they have no
    /// pre-tokenizer split at all, the vocabulary itself carries the word
    /// boundary as `▁`. [`Backend::WordPiece`]'s whitespace/punctuation
    /// splitting is real but fused into its own encode rather than exposed as a
    /// stage, and reconstructing it here would pin a copy instead of the thing
    /// that runs.
    pub fn pre_tokenize(&self, text: &str) -> Option<Vec<String>> {
        match &self.backend {
            Backend::Bpe(t) => Some(t.pre_tokenize(text)),
            Backend::Unigram(_) | Backend::WordPiece(_) | Backend::Spm(_) => None,
        }
    }

    /// The input as this handle's normalization stage leaves it, or `None` when
    /// this backend has no such stage to report.
    ///
    /// This is the text every later stage is driven with, and therefore what
    /// [`pre_tokenize`](Self::pre_tokenize) — which sits *after* it and
    /// deliberately does not normalize — must be handed instead of the raw input.
    ///
    /// Where exactly the line between "normalization" and "the model" falls is
    /// each backend's reference implementation's to draw, and each variant
    /// reports the stage its own reference exposes:
    ///
    /// - [`Backend::Bpe`] and [`Backend::Unigram`] — the declared HF `normalizer`
    ///   pipeline's output, matching `tokenizers`' `normalizer.normalize_str`;
    ///   the input unchanged when no normalizer is declared, which is the case
    ///   for every vocabulary in [`crate::pretrained`]. `add_prefix_space` and
    ///   the metaspace escaping are *not* included: HuggingFace hangs both off
    ///   pre-tokenizer nodes.
    /// - [`Backend::Spm`] — the metaspace escaping *and* the dummy prefix,
    ///   matching `sentencepiece`'s own `SentencePieceProcessor.normalize`. This
    ///   backend has nothing between that and the merge loop, so here the string
    ///   really is what the model sees. It is also the only coverage that stage
    ///   can have: SentencePiece has no pre-tokenizer split, so `pre_tokenize`
    ///   reports `None`.
    /// - [`Backend::WordPiece`] — `None`. Its BertNormalizer is real but fused
    ///   into its own encode as flags rather than exposed as a stage, exactly as
    ///   its splitting is, and reconstructing it here would pin a copy instead of
    ///   the thing that runs.
    ///
    /// Added-token extraction runs upstream of all of this, on the raw input;
    /// what is reported is what one content gap becomes.
    pub fn normalize(&self, text: &str) -> Option<String> {
        match &self.backend {
            Backend::Bpe(t) => Some(t.normalize(text)),
            Backend::Unigram(t) => Some(t.normalize(text)),
            Backend::Spm(t) => Some(t.normalize(text)),
            Backend::WordPiece(_) => None,
        }
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
        self.decode_inner(ids, SpecialDecode::Skip)
    }

    /// Decode ids back to text under an explicit [`SpecialDecode`] — see
    /// [`Tokenize::decode_with`], whose contract this is.
    ///
    /// Inherent for the same reason [`decode`](Self::decode) is: the universal
    /// handle should not be the one type whose decode needs a trait imported to
    /// spell. Branches on `self.decoder` exactly as [`decode`](Self::decode)
    /// does, so a declared pipeline answers in both modes and the backend's own
    /// rules answer in both.
    pub fn decode_with(
        &self,
        ids: &[u32],
        specials: SpecialDecode,
    ) -> Result<String, TokenizeError> {
        self.decode_inner(ids, specials)
    }

    /// The one decode implementation, shared by the inherent [`Self::decode`],
    /// [`Self::decode_with`] and the [`Tokenize`] impl so none of them can drift
    /// from the others.
    fn decode_inner(&self, ids: &[u32], specials: SpecialDecode) -> Result<String, TokenizeError> {
        // When the json declares a `decoder`, drive decoding from it.
        if let Some(decoder) = &self.decoder {
            return Ok(self.decode_declared(decoder, ids, specials));
        }
        self.backend.decode_with(ids, specials)
    }

    /// The declared `decoder` pipeline's own decode: collect the surface strings
    /// (skipping special-flagged added tokens, matching HF's default
    /// `skip_special_tokens=true`) and run the configured pipeline.
    ///
    /// Infallible, and that is the pipeline's own property rather than a
    /// simplification here: an id with no surface at all is dropped by the
    /// `filter_map`, and the chain's own `ByteFallback` step substitutes U+FFFD
    /// for a byte run it cannot decode. So the declared path has no separate
    /// lossy form to write — strict and lossy decoding through it are the same
    /// call, which is why [`decode_lossy`](Self::decode_lossy) shares this
    /// function rather than inventing one.
    fn decode_declared(
        &self,
        decoder: &super::decoder::Decoder,
        ids: &[u32],
        specials: SpecialDecode,
    ) -> String {
        let surfaces: Vec<String> = ids
            .iter()
            // The one thing [`SpecialDecode`] changes on this path: under
            // `Render` the declared-special ids keep their surface and go
            // through the pipeline like any other token. An id with no surface
            // at all still drops, in both modes — `filter_map` below, and there
            // is nothing to render for it either way.
            .filter(|id| match specials {
                SpecialDecode::Skip => !self.special_decode.contains(id),
                SpecialDecode::Render => true,
            })
            .filter_map(|&id| self.backend.token_surface(id))
            .collect();
        decoder.decode(surfaces)
    }

    /// Decode ids back to text, surviving what [`decode`](Self::decode) would
    /// report: an id the backend does not know is skipped and bytes that cannot
    /// be valid UTF-8 become U+FFFD.
    ///
    /// Branches exactly as [`decode`](Self::decode) does, on the same
    /// `self.decoder`, so the two can never disagree about which machinery
    /// answers. On the declared-pipeline branch they are literally the same
    /// call — see `decode_declared`, which is already total — and only the
    /// backend branch has a distinct lenient drive to reach.
    pub fn decode_lossy(&self, ids: &[u32]) -> String {
        if let Some(decoder) = &self.decoder {
            return self.decode_declared(decoder, ids, SpecialDecode::Skip);
        }
        self.backend.decode_lossy(ids)
    }

    /// The bytes one id contributes to this handle's decoded output — see
    /// [`Tokenize::decode_token_bytes`], whose contract this is.
    ///
    /// Branches as [`decode`](Self::decode) does: a declared `decoder` pipeline
    /// renders the id, and with none declared the backend's own rules do. A
    /// caller reaching past this handle for a backend's per-id rendering while a
    /// pipeline is declared would get the raw piece (`▁hello`) the pipeline
    /// exists to turn into text — the hazard [`declares_decoder`](Self::declares_decoder)
    /// warns about — so this method exists to be the one that does not.
    ///
    /// # Errors
    /// [`TokenizeError::InvalidTokenId`] for an id outside the vocabulary
    /// entirely, and — only on the declared branch, and for exactly the pipelines
    /// [`streaming_decoder`](Self::streaming_decoder) refuses —
    /// [`TokenizeError::UnstreamableDecoder`]: a pipeline whose steps cannot be
    /// evaluated one chunk at a time cannot be evaluated one *token* at a time
    /// either, and whole-sequence [`decode`](Self::decode) remains the way to
    /// read those.
    pub fn decode_token_bytes(&self, id: u32) -> Result<Vec<u8>, TokenizeError> {
        let Some(decoder) = &self.decoder else {
            return self.backend.decode_token_bytes(id);
        };
        self.declared_token_bytes(decoder, id)
    }

    /// [`decode_token_bytes`](Self::decode_token_bytes) as text — see
    /// [`Tokenize::decode_token`], which documents why
    /// [`TokenizeError::Utf8Error`] here is the ordinary signal to stream
    /// instead.
    pub fn decode_token(&self, id: u32) -> Result<String, TokenizeError> {
        super::tokenize::token_text_of(self.decode_token_bytes(id)?)
    }

    /// One id rendered through the *declared* pipeline's lowered rules — the
    /// same lowering [`streaming_decoder`](Self::streaming_decoder) streams
    /// through, so the two cannot disagree about what an id stands for.
    ///
    /// Unlike that method this does not materialize the whole surface table: a
    /// single id needs a single surface, so the rules are given a one-slot
    /// vocabulary and asked about slot 0. The skip and the missing-surface
    /// decisions are therefore made here rather than folded into the rules'
    /// tables, and they are made to match what
    /// [`streaming_decoder`](Self::streaming_decoder) folds into *its* tables
    /// over the whole vocabulary — because
    /// [`Tokenize::decode_token_bytes`] is documented as the bytes an id
    /// contributes to the decoded stream, so the two must give the same answer
    /// for the same id:
    ///
    /// * a skipped special contributes nothing;
    /// * an id **inside** the vocabulary with no surface contributes nothing —
    ///   the streaming decoder puts exactly these ids in its skip set, and
    ///   whole-sequence [`decode`](Self::decode) drops them in `decode_declared`'s
    ///   `filter_map`. This method used to report them invalid, which made two of
    ///   this handle's own APIs disagree about the same id;
    /// * an id **outside** the vocabulary is invalid — the one id the stream also
    ///   refuses (its surface table has no slot to read), so the disagreement
    ///   does not reappear at the other end.
    ///
    /// A `WordPiece` pipeline's declared per-token `cleanup` does not run here:
    /// the unit it cleans is the token *plus* the separator it carries, and the
    /// separator is a fact about the sequence that this method deliberately
    /// drops — so the space that cleanup exists to remove is not present to
    /// remove.
    fn declared_token_bytes(
        &self,
        decoder: &super::decoder::Decoder,
        id: u32,
    ) -> Result<Vec<u8>, TokenizeError> {
        // The refusal is decided first, so it does not depend on which id was
        // asked about: a pipeline this handle cannot render through renders no
        // id, not merely the ones that are neither skipped nor missing.
        let Some((rules, _post)) = decoder.lower() else {
            // The post-ops are dropped on purpose, not overlooked: they are the
            // sequence-level half of decoding, and this method reports what one
            // id contributes before any of it.
            return Err(TokenizeError::UnstreamableDecoder(
                decoder.unstreamable_op().unwrap_or("declared"),
            ));
        };
        if self.special_decode.contains(&id) {
            return Ok(Vec::new());
        }
        let Some(surface) = self.backend.token_surface(id) else {
            // The same partition `streaming_decoder` makes over `0..vocab_size`:
            // a surface-less id inside that range goes in its skip set (so it
            // contributes nothing), and only an id past the end has no slot at
            // all.
            return if (id as usize) < Tokenize::vocab_size(self) {
                Ok(Vec::new())
            } else {
                Err(TokenizeError::InvalidTokenId(id))
            };
        };
        let rules = rules.with_vocabulary(
            Surfaces::ByIndex(Arc::new(vec![surface])),
            Arc::new(rustc_hash::FxHashMap::default()),
            Arc::new(rustc_hash::FxHashSet::default()),
        );
        // Slot 0 is the surface just placed there, so it is never unknown; the
        // mapping restores the caller's id for the spelling the type system
        // cannot prove away.
        super::tokenize::token_bytes_of(&rules, 0).map_err(|_| TokenizeError::InvalidTokenId(id))
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
                .map(|ids| self.decode_inner(ids, SpecialDecode::Skip))
                .collect()
        }
        #[cfg(not(feature = "rayon"))]
        {
            token_lists
                .iter()
                .map(|ids| self.decode_inner(ids, SpecialDecode::Skip))
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
        self.streaming_decoder_with(SpecialDecode::Skip)
    }

    /// A [`StreamingDecoder`] that reproduces this handle's
    /// [`decode_with`](Self::decode_with) under the same [`SpecialDecode`] —
    /// see [`Tokenize::streaming_decoder_with`], whose contract this is.
    ///
    /// The whole of [`streaming_decoder`](Self::streaming_decoder)'s body, which
    /// is now this method under [`SpecialDecode::Skip`], so the stream and the
    /// whole-sequence decode agree in both modes and refuse on exactly the same
    /// declared pipelines.
    pub fn streaming_decoder_with(
        &self,
        specials: SpecialDecode,
    ) -> Result<StreamingDecoder, TokenizeError> {
        let Some(decoder) = &self.decoder else {
            return Ok(self.backend.streaming_decoder_with(specials));
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
        // `decode_inner` collects them: `token_surface` per id, and
        // `special_decode` dropped ahead of it under the same `specials` that
        // path reads. An id with no surface at all is dropped too — which is a
        // *skip* here, since a rendering rule reads a dense table and an empty
        // slot would otherwise render as an empty surface (and, on a WordPiece
        // pipeline, carry a word separator with it).
        //
        // Those surface-less ids stay skipped in *both* modes, which is why this
        // path composes its own set rather than calling
        // `DecodeState::with_special_decode` — that empties the skip set whole,
        // which is right only where nothing but declared specials is in it.
        let vocab_size = Tokenize::vocab_size(self);
        let mut surfaces = Vec::with_capacity(vocab_size);
        let mut skip = match specials {
            SpecialDecode::Skip => self.special_decode.clone(),
            SpecialDecode::Render => rustc_hash::FxHashSet::default(),
        };
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

    /// Every named special token this tokenizer knows, content to id.
    ///
    /// Works for every loader — a bundled vocabulary, a `tokenizer.json`, a
    /// GGUF — so "what markers does this thing have?" has one answer that does
    /// not depend on where the vocabulary came from.
    pub fn special_tokens(&self) -> &rustc_hash::FxHashMap<String, u32> {
        self.policy.special_tokens()
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
        self.decode_inner(ids, SpecialDecode::Skip)
    }

    /// The inherent [`decode_with`](AnyTokenizer::decode_with), which
    /// [`decode`](AnyTokenizer::decode) is itself a mode of.
    fn decode_with(&self, ids: &[u32], specials: SpecialDecode) -> Result<String, TokenizeError> {
        self.decode_inner(ids, specials)
    }

    fn decode_lossy(&self, ids: &[u32]) -> String {
        AnyTokenizer::decode_lossy(self, ids)
    }

    /// The one implementor that can refuse — see the inherent
    /// [`streaming_decoder`](AnyTokenizer::streaming_decoder), which has the
    /// same signature, so the trait method is a plain delegation rather than a
    /// widening.
    fn streaming_decoder(&self) -> Result<StreamingDecoder, TokenizeError> {
        AnyTokenizer::streaming_decoder(self)
    }

    /// The inherent
    /// [`streaming_decoder_with`](AnyTokenizer::streaming_decoder_with), which
    /// refuses on exactly the pipelines its default-mode sibling refuses on.
    fn streaming_decoder_with(
        &self,
        specials: SpecialDecode,
    ) -> Result<StreamingDecoder, TokenizeError> {
        AnyTokenizer::streaming_decoder_with(self, specials)
    }

    fn decode_token_bytes(&self, id: u32) -> Result<Vec<u8>, TokenizeError> {
        AnyTokenizer::decode_token_bytes(self, id)
    }

    fn decode_token(&self, id: u32) -> Result<String, TokenizeError> {
        AnyTokenizer::decode_token(self, id)
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

    // =========================================================================
    // Per-id decoding and lossy decoding through the universal handle
    // =========================================================================

    /// The Mistral-shaped handle the declared-pipeline tests share: a BPE
    /// backend whose surfaces are raw SentencePiece pieces, one `special = true`
    /// id, one `<0xNN>` piece for the declared `ByteFallback` step to resolve,
    /// and the `tokenizer.json` `decoder` chain verbatim.
    fn declared_handle() -> AnyTokenizer {
        let mut encoder = rustc_hash::FxHashMap::default();
        encoder.insert("\u{2581}hello".as_bytes().to_vec(), 10);
        encoder.insert("\u{2581}world".as_bytes().to_vec(), 11);
        encoder.insert("<0xF0>".as_bytes().to_vec(), 12);
        let mut special_tokens = rustc_hash::FxHashMap::default();
        special_tokens.insert("<s>".to_string(), 1);

        let tokenizer = Tokenizer::new(encoder, special_tokens.clone(), r"\S+|\s+")
            .expect("the test pattern compiles");
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

        AnyTokenizer {
            backend: Backend::Bpe(tokenizer),
            policy,
            decoder: super::super::decoder::parse(Some(&declared)),
            special_decode: [1].into_iter().collect(),
        }
    }

    /// Per-id decoding on a declared pipeline renders through that pipeline's
    /// own lowered rules, not the backend's: `▁hello` is ` hello`, never the raw
    /// piece. The three answers are the ones the trait states — content bytes, an
    /// empty contribution for a skipped special, an error for an id in no table.
    #[test]
    fn decode_token_bytes_runs_the_declared_pipeline_per_id() {
        let any = declared_handle();
        assert!(any.declares_decoder());

        assert_eq!(any.decode_token_bytes(10).unwrap(), b" hello".to_vec());
        // The declared `Strip` is a *sequence* post-op, so the leading space the
        // `Replace` produced is still here — this is what "no sequence-level
        // post-processing" means.
        assert_eq!(any.decode_token(10).unwrap(), " hello");

        assert_eq!(any.decode_token_bytes(1).unwrap(), Vec::<u8>::new());
        assert_eq!(any.decode_token(1).unwrap(), "");

        assert!(matches!(
            any.decode_token_bytes(999),
            Err(TokenizeError::InvalidTokenId(999))
        ));
    }

    /// An id inside the vocabulary that carries no surface must give the SAME
    /// answer from `decode_token_bytes` and from the stream, because
    /// `decode_token_bytes` is documented as the bytes an id contributes to the
    /// decoded stream. `streaming_decoder` puts such an id in its skip set and
    /// whole-sequence `decode` drops it, so nothing is the right answer;
    /// `decode_token_bytes` used to report `InvalidTokenId` for it, which is two
    /// of this handle's own APIs contradicting each other about one id.
    ///
    /// `declared_handle`'s surfaces are ids 1, 10, 11, 12, so its `vocab_size` is
    /// 13 and id 5 is a hole inside it — while 999 is past the end, the one id
    /// the stream also has no slot for and therefore still an error.
    #[test]
    fn a_surface_less_id_inside_the_vocabulary_contributes_nothing_to_both_apis() {
        let any = declared_handle();
        assert_eq!(Tokenize::vocab_size(&any), 13);

        assert_eq!(any.decode_token_bytes(5).unwrap(), Vec::<u8>::new());
        assert_eq!(any.decode_token(5).unwrap(), "");

        // The stream agrees: the hole is skipped, not rejected, and the ids
        // around it decode exactly as they do without it.
        let mut streamed = any.streaming_decoder().expect("this pipeline lowers");
        let mut out = streamed
            .add_tokens(&[10, 5, 11])
            .expect("the hole is skipped, not reported")
            .unwrap_or_default();
        out.push_str(&streamed.flush());
        assert_eq!(out, "hello world");

        // …and so does whole-sequence decoding, on both its strict and lossy forms.
        assert_eq!(Tokenize::decode(&any, &[10, 5, 11]).unwrap(), "hello world");
        assert_eq!(Tokenize::decode_lossy(&any, &[10, 5, 11]), "hello world");

        // Past the end of the vocabulary is still an error, so the reconciliation
        // did not simply make every unknown id silent.
        assert!(matches!(
            any.decode_token_bytes(999),
            Err(TokenizeError::InvalidTokenId(999))
        ));
    }

    /// The declared `ByteFallback` step's `<0xNN>` piece is one byte, so it has
    /// bytes but is not text on its own — the case that motivates the pair.
    #[test]
    fn a_declared_byte_fallback_id_has_bytes_but_no_text_of_its_own() {
        let any = declared_handle();

        assert_eq!(any.decode_token_bytes(12).unwrap(), vec![0xF0]);
        assert!(matches!(
            any.decode_token(12),
            Err(TokenizeError::Utf8Error)
        ));
    }

    /// Agreement on the declared branch, stated exactly. Concatenating the per-id
    /// bytes gives the stream *before* its post-ops; the streaming decoder then
    /// applies the declared `Strip`, and the difference between the two is
    /// precisely the one leading space that strip removes.
    #[test]
    fn concatenated_token_bytes_equal_the_stream_before_post_processing() {
        let any = declared_handle();
        let ids = [1, 10, 11];

        let joined: Vec<u8> = ids
            .iter()
            .flat_map(|&id| any.decode_token_bytes(id).expect("every id is known"))
            .collect();
        assert_eq!(String::from_utf8(joined).unwrap(), " hello world");

        let mut streamed = any.streaming_decoder().expect("this pipeline lowers");
        let mut out = streamed.add_tokens_lossy(&ids).unwrap_or_default();
        out.push_str(&streamed.flush());
        assert_eq!(out, "hello world");
    }

    /// With no pipeline declared, per-id decoding delegates to the backend — the
    /// same branch `decode` takes.
    #[test]
    fn decode_token_bytes_delegates_to_the_backend_without_a_declared_pipeline() {
        let mut encoder = rustc_hash::FxHashMap::default();
        encoder.insert("\u{2581}hello".as_bytes().to_vec(), 10);
        let tokenizer = Tokenizer::new(encoder, rustc_hash::FxHashMap::default(), r"\S+|\s+")
            .expect("the test pattern compiles");
        let any = AnyTokenizer::new(Backend::Bpe(tokenizer), SpecialPolicy::default());

        assert!(!any.declares_decoder());
        // The backend's own rules: the raw piece, ▁ and all.
        assert_eq!(any.decode_token(10).unwrap(), "\u{2581}hello");
        assert!(matches!(
            any.decode_token_bytes(999),
            Err(TokenizeError::InvalidTokenId(999))
        ));
    }

    /// `decode_lossy` must take the same branch `decode` takes, or the two
    /// disagree about what a sequence says. On a declared pipeline they are the
    /// same call — that pipeline is already total — and on the backend branch
    /// they differ only in surviving an unknown id.
    #[test]
    fn decode_lossy_mirrors_decode_on_both_branches() {
        let any = declared_handle();
        let ids = [1, 10, 11];

        assert_eq!(Tokenize::decode(&any, &ids).unwrap(), "hello world");
        assert_eq!(Tokenize::decode_lossy(&any, &ids), "hello world");
        // An id the backend does not know is dropped by the declared pipeline's
        // own surface collection, on both halves alike.
        assert_eq!(
            Tokenize::decode_lossy(&any, &[1, 10, 999, 11]),
            "hello world"
        );

        let mut encoder = rustc_hash::FxHashMap::default();
        encoder.insert(b"hello".to_vec(), 10);
        encoder.insert(b" world".to_vec(), 11);
        let tokenizer = Tokenizer::new(encoder, rustc_hash::FxHashMap::default(), r"\S+|\s+")
            .expect("the test pattern compiles");
        let bare = AnyTokenizer::new(Backend::Bpe(tokenizer), SpecialPolicy::default());

        assert!(Tokenize::decode(&bare, &[10, 999, 11]).is_err());
        assert_eq!(Tokenize::decode_lossy(&bare, &[10, 999, 11]), "hello world");
    }
}
