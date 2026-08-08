//! Python bindings for the splintr tokenizer.
//!
//! This module provides PyO3 wrappers around the core Rust tokenizer,
//! exposing a Python-friendly API while maintaining Rust performance.
//!
//! # Bundled Vocabularies
//!
//! The module includes pre-loaded vocabularies for:
//! - `cl100k_base`: GPT-4, GPT-3.5-turbo (~100k tokens)
//! - `o200k_base`: GPT-4o (~200k tokens)
//!
//! # Thread Safety
//!
//! The tokenizer is thread-safe and can be shared across Python threads.
//! Batch operations use Rayon for true parallelism, bypassing the GIL
//! during Rust computation.
//!
//! # Example
//!
//! ```python
//! from splintr import Tokenizer
//!
//! # Load pretrained model
//! tokenizer = Tokenizer.from_pretrained("cl100k_base")
//!
//! # Encode/decode
//! tokens = tokenizer.encode("Hello, world!")
//! text = tokenizer.decode(tokens)
//!
//! # Streaming decode for LLM output
//! decoder = tokenizer.streaming_decoder()
//! for token_id in token_stream:
//!     if text := decoder.add_token(token_id):
//!         print(text, end="", flush=True)
//! ```

use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::core::SentencePieceTokenizer;

use crate::core::hf_json::{
    from_json_bytes as core_from_json_bytes, from_json_path as core_from_json_path,
};
use crate::core::spm::SpmTokenizer;
use crate::core::wordpiece::WordPieceTokenizer;
use crate::core::{AnyTokenizer, Backend, SpecialDecode, SpecialMode, SpecialPolicy};
use crate::core::{StreamingDecoder, Tokenize, Tokenizer};

// Special tokens are defined in crate::core::pretrained module.
// See that module for the full token documentation and implementations.

/// Python wrapper for the Rust Tokenizer.
#[pyclass(name = "Tokenizer")]
pub struct PyTokenizer {
    inner: Tokenizer,
    policy: SpecialPolicy,
}

#[pymethods]
impl PyTokenizer {
    /// Create a new tokenizer from a vocabulary file.
    ///
    /// Args:
    ///     vocab_path: Path to a tiktoken-format vocabulary file
    ///     pattern: PCRE2 regex pattern for tokenization
    ///     special_tokens: Optional dict of special tokens to IDs
    #[new]
    #[pyo3(signature = (vocab_path, pattern, special_tokens=None))]
    fn new(
        vocab_path: &str,
        pattern: &str,
        special_tokens: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let special = parse_special_tokens(special_tokens)?;

        let inner = Tokenizer::from_file(vocab_path, pattern, special)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;

        Ok(Self {
            inner,
            policy: SpecialPolicy::default(),
        })
    }

    /// Create a tokenizer from a pretrained model name.
    ///
    /// Currently supported:
    /// - "cl100k_base" (GPT-4, GPT-3.5-turbo)
    /// - "o200k_base" (GPT-4o)
    /// - "llama3" / "llama3.1" / "llama3.2" / "llama3.3" (Meta Llama 3 family)
    /// - "deepseek_v3" / "deepseek-v3" (DeepSeek V3)
    /// - "mistral" / "mistral_v1" (Mistral V1: 32k SentencePiece)
    /// - "mistral_v2" (Mistral V2: 32k + control tokens)
    /// - "mistral_v3" (Mistral V3/Tekken: 131k)
    /// - "whisper" / "whisper_v1" / "whisper_v2" / "whisper_v3" (OpenAI Whisper
    ///   multilingual, ~51k; bare "whisper" → v2). English-only checkpoints use a
    ///   different base BPE and are not bundled — load those with `splintr.from_json`.
    ///
    /// For any model not bundled here, load its HuggingFace `tokenizer.json`
    /// directly with the module-level `splintr.from_json(path)`.
    ///
    /// Args:
    ///     name: Model name (e.g., "cl100k_base", "o200k_base", "llama3", "mistral_v3")
    ///
    /// Returns:
    ///     An `AnyTokenizer` — the same universal loaded-tokenizer handle
    ///     `splintr.from_json` returns, for **every** bundled vocabulary. It
    ///     carries the vocabulary's special-token policy and decode pipeline
    ///     with it, and `.family` reports the backend it dispatched to ("BPE"
    ///     for the byte-level vocabularies, "Spm" for the SentencePiece ones).
    ///
    ///     Because this delegates to the same core loader `splintr.pretrained`
    ///     uses in Rust, a name produces the same ids on both sides of the
    ///     binding. In particular `encode` matches special tokens spelled out
    ///     in the text — `encode("<|begin_of_text|>hi")` is `[128000, 6151]`,
    ///     not the marker shattered into ordinary tokens. Use
    ///     `encode_ordinary` to refuse those matches, or
    ///     `encode_allowed_special` to permit a named subset.
    #[staticmethod]
    fn from_pretrained(py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        // Delegate: the per-vocabulary construction (which file, which
        // pre-tokenizer passes, which specials, byte-level or SPM, added-token
        // matching, the policy) lives in `core::pretrained` and is not repeated
        // here. The hand-rolled copy this replaced had drifted — it built every
        // vocabulary with added-token matching off and a default policy, so the
        // same name gave different ids in Python than in Rust.
        let loaded = crate::core::pretrained::from_pretrained(name)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        any_tokenizer_to_py(py, loaded)
    }

    /// Create a tokenizer from raw vocabulary bytes.
    ///
    /// Args:
    ///     vocab_data: Raw bytes of tiktoken-format vocabulary
    ///     pattern: PCRE2 regex pattern for tokenization
    ///     special_tokens: Optional dict of special tokens to IDs
    #[staticmethod]
    #[pyo3(signature = (vocab_data, pattern, special_tokens=None))]
    fn from_bytes(
        vocab_data: &[u8],
        pattern: &str,
        special_tokens: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let special = parse_special_tokens(special_tokens)?;

        let inner = Tokenizer::from_bytes(vocab_data, pattern, special)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(Self {
            inner,
            policy: SpecialPolicy::default(),
        })
    }

    /// Switch between regex backends.
    ///
    /// The tokenizer supports two regex backends:
    /// - regexr (default): Custom pure-Rust regex engine with JIT and SIMD
    /// - PCRE2: Industry-standard regex library (requires `pcre2` feature)
    ///
    /// Args:
    ///     use_pcre2: If True, switch to PCRE2 backend. If False, switch to regexr (default: True)
    ///
    /// Returns:
    ///     New Tokenizer instance with the specified backend
    ///
    /// Raises:
    ///     ValueError: If use_pcre2=True and pcre2 feature is not enabled
    ///
    /// Example:
    ///     tokenizer = Tokenizer("vocab.tiktoken", CL100K_BASE_PATTERN).pcre2(True)
    ///     tokenizer = tokenizer.pcre2(False)
    #[pyo3(signature = (use_pcre2=true))]
    fn pcre2(&self, use_pcre2: bool) -> PyResult<Self> {
        let new_inner = self.inner.clone();
        let result = new_inner
            .pcre2(use_pcre2)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: result,
            policy: self.policy.clone(),
        })
    }

    /// Enable or disable JIT compilation for the regex backend.
    ///
    /// JIT (Just-In-Time) compilation can significantly improve regex matching
    /// performance. JIT availability depends on:
    /// - Platform support (e.g., x86-64)
    /// - Crate feature flags (regexr jit, pcre2 jit)
    ///
    /// When enabled, JIT will be used if available on the current platform.
    /// JIT is enabled by default.
    ///
    /// Args:
    ///     use_jit: Whether to try using JIT compilation (default: True)
    ///
    /// Returns:
    ///     New Tokenizer instance with the specified JIT preference
    ///
    /// Example:
    ///     tokenizer = Tokenizer("vocab.tiktoken", CL100K_BASE_PATTERN).jit(False)
    ///     tokenizer = tokenizer.pcre2(True).jit(True)
    #[pyo3(signature = (use_jit=true))]
    fn jit(&self, use_jit: bool) -> PyResult<Self> {
        let new_inner = self.inner.clone();
        let result = new_inner
            .jit(use_jit)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: result,
            policy: self.policy.clone(),
        })
    }

    /// Encode text to token IDs — model-ready.
    ///
    /// Applies this tokenizer's boundary template (`post_processor`: `[CLS]…[SEP]`,
    /// `<s>…</s>`), so the result is what the model was trained to receive. This is
    /// HuggingFace's `tokenizer.encode(text)` with its default
    /// `add_special_tokens=True`. Use `encode_raw` for the untemplated form.
    /// The two are identical for a tokenizer that declares no template — which is
    /// every tokenizer this class's own constructors build, since they take a
    /// vocabulary and a pattern and no template.
    ///
    /// Whether a special token *spelled out inside* `text` is matched is a separate
    /// question, governed by `encode_ordinary` / `encode_with_special` /
    /// `encode_allowed_special`; `encode` uses this tokenizer's configured default,
    /// which for a directly-constructed `Tokenizer` is "do not match" — nothing has
    /// told it which added tokens to look for (see `encode_with_special` to opt in).
    /// The loaders (`Tokenizer.from_pretrained`, `splintr.from_json`) return an
    /// `AnyTokenizer` with matching **on**.
    ///
    /// Sequential encoding, optimal for texts under ~1MB.
    ///
    /// Args:
    ///     text: Input text to encode
    ///
    /// Returns:
    ///     List of token IDs
    fn encode(&self, text: &str) -> Vec<u32> {
        self.policy.apply_single(self.inner.encode(text))
    }

    /// Encode text to token IDs — content tokens only, no boundary template.
    ///
    /// The backend's own output with **no** `post_processor` template applied:
    /// HuggingFace's `tokenizer.encode(text, add_special_tokens=False)`. Use it
    /// when you assemble the sequence yourself (a chat template, a reranker pair)
    /// and place the boundary tokens by hand.
    ///
    /// `encode` is exactly this call plus the template; any leading/trailing ids
    /// `encode` adds and this does not *are* the template.
    ///
    /// Args:
    ///     text: Input text to encode
    ///
    /// Returns:
    ///     List of token IDs
    fn encode_raw(&self, text: &str) -> Vec<u32> {
        self.inner.encode(text)
    }

    /// `encode` using Rayon to parallelize within a single text.
    ///
    /// Same semantics and same result as `encode` (boundary template applied,
    /// HF's `add_special_tokens=True`) — only the execution strategy differs.
    /// It has higher overhead than `encode()` due to thread pool coordination,
    /// but can be faster for very large texts (typically >1MB) where the
    /// parallelization benefit outweighs the overhead.
    ///
    /// For most use cases, prefer `encode()` (sequential) or `encode_batch()`
    /// (parallel across multiple texts).
    ///
    /// Args:
    ///     text: Input text to encode
    ///
    /// Returns:
    ///     List of token IDs
    fn encode_rayon(&self, text: &str) -> Vec<u32> {
        self.policy.apply_single(self.inner.encode_rayon(text))
    }

    /// Encode text, matching **every** configured special token spelled out in it.
    ///
    /// `<|endoftext|>` typed in `text` becomes that control token's real id rather
    /// than ordinary bytes. This is HuggingFace's `add_special_tokens=True` applied
    /// to *added tokens found in the text* — tiktoken's `allowed_special="all"` —
    /// and it is what `encode` does only when the tokenizer was built with
    /// added-token matching on (a directly-constructed `Tokenizer` is not, so this
    /// method is how you opt in).
    ///
    /// The boundary template is applied too, as in `encode`.
    ///
    /// Args:
    ///     text: Input text to encode
    ///
    /// Returns:
    ///     List of token IDs
    fn encode_with_special(&self, text: &str) -> Vec<u32> {
        self.policy
            .apply_single(self.inner.encode_with_special(text))
    }

    /// Encode text, never matching a special token spelled out in it.
    ///
    /// A special token written literally in the input (e.g. `<|endoftext|>`)
    /// is encoded as ordinary text instead of being promoted to its control-token
    /// id — tiktoken's `disallowed_special=()`, `allowed_special=set()`. Use this
    /// when tokenizing untrusted text where the caller must not be able to forge
    /// control tokens.
    ///
    /// The boundary template is still applied (as in `encode`): boundary tokens
    /// come from the template, not from matching text against the vocabulary, so
    /// locking down matching in the *content* must not strip the boundary tokens
    /// the model was trained with. Use `encode_raw` for the untemplated form.
    ///
    /// Args:
    ///     text: Input text to encode
    ///
    /// Returns:
    ///     List of token IDs
    fn encode_ordinary(&self, text: &str) -> Vec<u32> {
        self.policy.apply_single(self.inner.encode_ordinary(text))
    }

    /// Encode text, matching only the named special tokens spelled out in it.
    ///
    /// tiktoken's `allowed_special={...}`: any *other* configured special token
    /// written literally in the text raises `ValueError` instead of being silently
    /// promoted to its control-token id — use this to accept a known, bounded set
    /// of special tokens (e.g. a chat template's own markers) from otherwise
    /// untrusted text. The boundary template is applied, as in `encode`.
    ///
    /// Args:
    ///     text: Input text to encode
    ///     allowed_special: Special token strings permitted to match in `text`
    ///
    /// Returns:
    ///     List of token IDs
    ///
    /// Raises:
    ///     ValueError: If `text` spells out a configured special token that is
    ///         not in `allowed_special`
    fn encode_allowed_special(
        &self,
        text: &str,
        allowed_special: Vec<String>,
    ) -> PyResult<Vec<u32>> {
        let allowed: FxHashSet<String> = allowed_special.into_iter().collect();
        let ids = self
            .inner
            .encode_with(text, &SpecialMode::Allow(&allowed))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(self.policy.apply_single(ids))
    }

    /// Decode token IDs to a string.
    ///
    /// Args:
    ///     tokens: List of token IDs
    ///
    /// Returns:
    ///     Decoded string
    ///
    /// Raises:
    ///     ValueError: If decoded bytes are not valid UTF-8
    fn decode(&self, tokens: Vec<u32>) -> PyResult<String> {
        self.inner
            .decode(&tokens)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Decode token IDs to a string, rendering the vocabulary's special tokens
    /// instead of dropping them.
    ///
    /// `decode` implements HuggingFace's default `skip_special_tokens=True`, so
    /// a control marker (`[INST]`, `<|eot_id|>`, `<s>`) renders as nothing.
    /// This is its `skip_special_tokens=False`: every declared special id comes
    /// back as its own spelling, which is what a chat-template round trip, an
    /// inspection of raw model output, or a transcript that must keep its
    /// markers needs.
    ///
    /// Args:
    ///     tokens: List of token IDs
    ///
    /// Returns:
    ///     Decoded string, control markers included
    ///
    /// Raises:
    ///     ValueError: Exactly as `decode` does
    ///
    /// Example:
    ///     tok = Tokenizer.from_pretrained("mistral_v2")
    ///     ids = tok.encode_with_special("[INST]Hi[/INST]")
    ///     tok.decode(ids)               # "Hi"
    ///     tok.decode_with_special(ids)  # "[INST]Hi[/INST]"
    fn decode_with_special(&self, tokens: Vec<u32>) -> PyResult<String> {
        self.inner
            .decode_with(&tokens, SpecialDecode::Render)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Decode token IDs to bytes.
    ///
    /// Args:
    ///     tokens: List of token IDs
    ///
    /// Returns:
    ///     Decoded bytes
    ///
    /// Raises:
    ///     ValueError: If `tokens` contains an id not in the vocabulary
    fn decode_bytes(&self, tokens: Vec<u32>) -> PyResult<Vec<u8>> {
        self.inner
            .decode_bytes(&tokens)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Decode token IDs to string, replacing invalid UTF-8.
    ///
    /// Args:
    ///     tokens: List of token IDs
    ///
    /// Returns:
    ///     Decoded string with replacement characters for invalid UTF-8
    fn decode_lossy(&self, tokens: Vec<u32>) -> String {
        self.inner.decode_lossy(&tokens)
    }

    /// The bytes `id` contributes to the decoded stream — the ByteLevel
    /// alphabet unmapped and `<0xNN>` byte fallback resolved, and nothing
    /// else: no leading-space strip, no first-token rule, no word separator.
    ///
    /// An id in the vocabulary that carries no surface (a special this
    /// tokenizer's decode drops, say) returns **empty** bytes, not an error.
    /// Because of that, concatenating this over a sequence of ids is *not*
    /// what `decode` returns for the same sequence — `decode` layers
    /// post-processing this method deliberately skips. Use
    /// `streaming_decoder` to render a sequence.
    ///
    /// Args:
    ///     id: Token ID
    ///
    /// Returns:
    ///     The id's raw bytes, or `b""` if it carries no surface
    ///
    /// Raises:
    ///     ValueError: If `id` is outside the vocabulary
    fn decode_token_bytes(&self, id: u32) -> PyResult<Vec<u8>> {
        self.inner
            .decode_token_bytes(id)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// `decode_token_bytes` as text.
    ///
    /// Raises far more often than `decode` does: a single `<0xNN>`
    /// byte-fallback id, or a token holding one byte of a multi-byte
    /// character, is not valid UTF-8 standing alone — that is the expected
    /// signal to stop decoding id-at-a-time and use `streaming_decoder`,
    /// which buffers exactly those partial sequences across tokens.
    ///
    /// Args:
    ///     id: Token ID
    ///
    /// Returns:
    ///     The id's text, or `""` if it carries no surface
    ///
    /// Raises:
    ///     ValueError: If `id` is outside the vocabulary, or if its bytes are
    ///         not valid UTF-8 on their own
    fn decode_token(&self, id: u32) -> PyResult<String> {
        self.inner
            .decode_token(id)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Batch form of `encode` — model-ready ids for each text.
    ///
    /// Uses Rayon to parallelize encoding across texts. Each result carries the
    /// boundary template, exactly as `encode` does (HF `add_special_tokens=True`).
    ///
    /// Args:
    ///     texts: List of texts to encode
    ///
    /// Returns:
    ///     List of token ID lists
    fn encode_batch(&self, texts: Vec<String>) -> Vec<Vec<u32>> {
        self.inner
            .encode_batch(&texts)
            .into_iter()
            .map(|ids| self.policy.apply_single(ids))
            .collect()
    }

    /// Batch form of `encode_with_special` — every special token spelled out in
    /// each text is matched, and the boundary template applied.
    ///
    /// Args:
    ///     texts: List of texts to encode
    ///
    /// Returns:
    ///     List of token ID lists
    fn encode_batch_with_special(&self, texts: Vec<String>) -> Vec<Vec<u32>> {
        self.inner
            .encode_batch_with_special(&texts)
            .into_iter()
            .map(|ids| self.policy.apply_single(ids))
            .collect()
    }

    /// Batch decode multiple token lists in parallel.
    ///
    /// Uses Rayon to parallelize decoding across token lists.
    ///
    /// Args:
    ///     token_lists: List of token ID lists
    ///
    /// Returns:
    ///     List of decoded strings
    ///
    /// Raises:
    ///     ValueError: If any decoded bytes are not valid UTF-8
    fn decode_batch(&self, token_lists: Vec<Vec<u32>>) -> PyResult<Vec<String>> {
        self.inner
            .decode_batch(&token_lists)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Batch decode multiple token lists, replacing invalid UTF-8.
    ///
    /// Args:
    ///     token_lists: List of token ID lists
    ///
    /// Returns:
    ///     List of decoded strings with replacement characters for invalid UTF-8
    fn decode_batch_lossy(&self, token_lists: Vec<Vec<u32>>) -> Vec<String> {
        self.inner.decode_batch_lossy(&token_lists)
    }

    /// Get the vocabulary size (including special tokens).
    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    /// Create a streaming decoder for UTF-8 safe token-by-token decoding.
    ///
    /// Useful for streaming LLM output where token boundaries may not
    /// align with UTF-8 character boundaries.
    ///
    /// The decoder takes every spelling rule (ByteLevel alphabet, `<0xNN>` byte
    /// fallback, metaspace, the specials `decode` drops) from **this**
    /// tokenizer's own configuration, so `"".join(chunks) + flush()` reproduces
    /// `decode` for any ids — there is no second decoder class to pick between
    /// and therefore no way to pick the wrong one.
    ///
    /// Returns:
    ///     StreamingDecoder instance
    ///
    /// Example:
    ///     decoder = tokenizer.streaming_decoder()
    ///     for token_id in token_stream:
    ///         if text := decoder.add_token(token_id):
    ///             print(text, end="", flush=True)
    ///     print(decoder.flush())
    fn streaming_decoder(&self) -> PyStreamingDecoder {
        PyStreamingDecoder::new(self.inner.streaming_decoder())
    }

    /// A streaming decoder that renders special tokens instead of dropping them.
    ///
    /// `streaming_decoder` reproduces `decode`; this one reproduces
    /// `decode_with_special`, so a generation loop can watch the control markers
    /// go past. `"".join(chunks) + flush()` equals `decode_with_special(ids)`.
    ///
    /// Returns:
    ///     StreamingDecoder instance
    fn streaming_decoder_with_special(&self) -> PyStreamingDecoder {
        PyStreamingDecoder::new(self.inner.streaming_decoder_with(SpecialDecode::Render))
    }

    /// Clear the encoding cache.
    fn clear_cache(&self) {
        self.inner.clear_cache();
    }

    /// Get the number of entries in the cache.
    #[getter]
    fn cache_len(&self) -> usize {
        self.inner.cache_len()
    }

    /// String representation.
    fn __repr__(&self) -> String {
        format!("Tokenizer(vocab_size={})", self.inner.vocab_size())
    }
}

/// Python wrapper for the SentencePiece unigram tokenizer.
///
/// For models using SentencePiece unigram tokenization (e.g., loaded from GGUF).
/// Uses Viterbi maximum-score segmentation (true SentencePiece Unigram, not
/// greedy) with byte fallback.
#[pyclass(name = "SentencePieceTokenizer")]
pub struct PySentencePieceTokenizer {
    inner: SentencePieceTokenizer,
    policy: SpecialPolicy,
    /// BOS id to prepend on `encode`, applied here rather than in the backend
    /// (which is policy-free). `None` when constructed via a loader path,
    /// where `AnyTokenizer`'s `SpecialPolicy` already owns boundary tokens.
    bos_token_id: Option<u32>,
}

#[pymethods]
impl PySentencePieceTokenizer {
    /// Create a new SentencePiece unigram tokenizer.
    ///
    /// Args:
    ///     tokens: List of token strings, indexed by token ID
    ///     scores: Per-token Unigram scores (log-probs) that Viterbi maximizes (empty list defaults to all zeros / uniform)
    ///     bos_token_id: Optional beginning-of-sequence token ID
    ///     eos_token_id: End-of-sequence token ID
    #[new]
    #[pyo3(signature = (tokens, scores, eos_token_id, bos_token_id=None))]
    fn new(
        tokens: Vec<String>,
        scores: Vec<f64>,
        eos_token_id: u32,
        bos_token_id: Option<u32>,
    ) -> PyResult<Self> {
        let inner = SentencePieceTokenizer::new(tokens, scores, bos_token_id, eos_token_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner,
            policy: SpecialPolicy::default(),
            bos_token_id,
        })
    }

    /// Encode text to token IDs — model-ready.
    ///
    /// Viterbi maximum-score Unigram segmentation, with spaces replaced by ▁
    /// (U+2581) following the SentencePiece convention. Adds the boundary tokens
    /// this tokenizer owns — the configured BOS, plus any `post_processor`
    /// template — so the result is HuggingFace's `tokenizer.encode(text)` with its
    /// default `add_special_tokens=True`. Use `encode_raw` for the untemplated form.
    ///
    /// Special tokens spelled out inside `text` are matched; `encode_ordinary` and
    /// `encode_allowed_special` constrain that.
    ///
    /// Args:
    ///     text: Input text to encode
    ///
    /// Returns:
    ///     List of token IDs
    fn encode(&self, text: &str) -> Vec<u32> {
        self.policy
            .apply_single(self.with_bos(self.inner.encode(text)))
    }

    /// Encode text to token IDs — content tokens only, no boundary tokens.
    ///
    /// HuggingFace's `tokenizer.encode(text, add_special_tokens=False)`: neither
    /// the configured BOS nor any `post_processor` template is added. Use it when
    /// you assemble the sequence yourself and place the boundary tokens by hand.
    ///
    /// `encode` is exactly this call plus those boundary tokens.
    ///
    /// Args:
    ///     text: Input text to encode
    ///
    /// Returns:
    ///     List of token IDs
    fn encode_raw(&self, text: &str) -> Vec<u32> {
        self.inner.encode(text)
    }

    /// Encode text, matching **every** configured special token spelled out in it.
    ///
    /// tiktoken's `allowed_special="all"`. This backend always matches its added
    /// tokens, so this is what `encode` already does — the name exists so the
    /// method means the same thing on every splintr tokenizer class. Boundary
    /// tokens are added, as in `encode`.
    ///
    /// Args:
    ///     text: Input text to encode
    ///
    /// Returns:
    ///     List of token IDs
    fn encode_with_special(&self, text: &str) -> Vec<u32> {
        self.encode(text)
    }

    /// Encode text, never matching a special token spelled out in it.
    ///
    /// A special token written literally in the input is encoded as ordinary text
    /// instead of being promoted to its control-token id — tiktoken's
    /// `allowed_special=set()`. Use this when tokenizing untrusted text where the
    /// caller must not be able to forge control tokens.
    ///
    /// Boundary tokens (the configured BOS, the template) are still added, as in
    /// `encode`: they come from this tokenizer's own configuration, not from
    /// matching text against the vocabulary, so locking down matching in the
    /// *content* must not strip them. Use `encode_raw` for the untemplated form.
    ///
    /// Args:
    ///     text: Input text to encode
    ///
    /// Returns:
    ///     List of token IDs
    fn encode_ordinary(&self, text: &str) -> Vec<u32> {
        self.policy
            .apply_single(self.with_bos(self.inner.encode_ordinary(text)))
    }

    /// Encode text, matching only the named special tokens spelled out in it.
    ///
    /// tiktoken's `allowed_special={...}`: any *other* configured special token
    /// written literally in the text raises `ValueError` instead of being silently
    /// promoted to its control-token id — use this to accept a known, bounded set
    /// of special tokens from otherwise untrusted text. Boundary tokens are added,
    /// as in `encode`.
    ///
    /// Args:
    ///     text: Input text to encode
    ///     allowed_special: Special token strings permitted to match in `text`
    ///
    /// Returns:
    ///     List of token IDs
    ///
    /// Raises:
    ///     ValueError: If `text` spells out a configured special token that is
    ///         not in `allowed_special`
    fn encode_allowed_special(
        &self,
        text: &str,
        allowed_special: Vec<String>,
    ) -> PyResult<Vec<u32>> {
        let allowed: FxHashSet<String> = allowed_special.into_iter().collect();
        let ids = self
            .inner
            .encode_with(text, &SpecialMode::Allow(&allowed))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(self.policy.apply_single(self.with_bos(ids)))
    }

    /// Batch form of `encode` — model-ready ids for each text.
    ///
    /// Uses Rayon to parallelize encoding across texts.
    ///
    /// Args:
    ///     texts: List of texts to encode
    ///
    /// Returns:
    ///     List of token ID lists
    fn encode_batch(&self, texts: Vec<String>) -> Vec<Vec<u32>> {
        #[cfg(feature = "rayon")]
        {
            texts.par_iter().map(|text| self.encode(text)).collect()
        }
        #[cfg(not(feature = "rayon"))]
        {
            texts.iter().map(|text| self.encode(text)).collect()
        }
    }

    /// Decode token IDs to text.
    ///
    /// Skips BOS/EOS tokens and converts ▁ back to spaces.
    ///
    /// Args:
    ///     ids: List of token IDs
    ///
    /// Returns:
    ///     Decoded string
    ///
    /// Raises:
    ///     ValueError: If a token ID is out of range
    fn decode(&self, ids: Vec<u32>) -> PyResult<String> {
        self.inner
            .decode(&ids)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Decode token IDs to a string, rendering the vocabulary's special tokens
    /// instead of dropping them.
    ///
    /// `decode` implements HuggingFace's default `skip_special_tokens=True`, so
    /// a control marker (`[INST]`, `<|eot_id|>`, `<s>`) renders as nothing.
    /// This is its `skip_special_tokens=False`: every declared special id comes
    /// back as its own spelling, which is what a chat-template round trip, an
    /// inspection of raw model output, or a transcript that must keep its
    /// markers needs.
    ///
    /// Args:
    ///     ids: List of token IDs
    ///
    /// Returns:
    ///     Decoded string, control markers included
    ///
    /// Raises:
    ///     ValueError: Exactly as `decode` does
    ///
    /// Example:
    ///     tok = Tokenizer.from_pretrained("mistral_v2")
    ///     ids = tok.encode_with_special("[INST]Hi[/INST]")
    ///     tok.decode(ids)               # "Hi"
    ///     tok.decode_with_special(ids)  # "[INST]Hi[/INST]"
    fn decode_with_special(&self, ids: Vec<u32>) -> PyResult<String> {
        self.inner
            .decode_with(&ids, SpecialDecode::Render)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Decode token IDs to text, skipping invalid IDs.
    ///
    /// Args:
    ///     ids: List of token IDs
    ///
    /// Returns:
    ///     Decoded string (invalid token IDs are silently skipped)
    fn decode_lossy(&self, ids: Vec<u32>) -> String {
        self.inner.decode_lossy(&ids)
    }

    /// The bytes `id` contributes to the decoded stream — `▁` unmapped and
    /// `<0xNN>` byte fallback resolved, and nothing else: no leading-space
    /// strip, no first-token rule, no word separator.
    ///
    /// An id in the vocabulary that carries no surface (a special this
    /// tokenizer's decode drops, say) returns **empty** bytes, not an error.
    /// Because of that, concatenating this over a sequence of ids is *not*
    /// what `decode` returns for the same sequence — `decode` layers
    /// post-processing this method deliberately skips. Use
    /// `streaming_decoder` to render a sequence.
    ///
    /// Args:
    ///     id: Token ID
    ///
    /// Returns:
    ///     The id's raw bytes, or `b""` if it carries no surface
    ///
    /// Raises:
    ///     ValueError: If `id` is outside the vocabulary
    fn decode_token_bytes(&self, id: u32) -> PyResult<Vec<u8>> {
        self.inner
            .decode_token_bytes(id)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// `decode_token_bytes` as text.
    ///
    /// Raises far more often than `decode` does: a single `<0xNN>`
    /// byte-fallback id, or a token holding one byte of a multi-byte
    /// character, is not valid UTF-8 standing alone — that is the expected
    /// signal to stop decoding id-at-a-time and use `streaming_decoder`,
    /// which buffers exactly those partial sequences across tokens.
    ///
    /// Args:
    ///     id: Token ID
    ///
    /// Returns:
    ///     The id's text, or `""` if it carries no surface
    ///
    /// Raises:
    ///     ValueError: If `id` is outside the vocabulary, or if its bytes are
    ///         not valid UTF-8 on their own
    fn decode_token(&self, id: u32) -> PyResult<String> {
        self.inner
            .decode_token(id)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Get vocabulary size.
    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    /// Check if a token is the EOS token.
    fn is_eos(&self, token_id: u32) -> bool {
        self.inner.is_eos(token_id)
    }

    /// Get the EOS token ID.
    #[getter]
    fn eos_token_id(&self) -> u32 {
        self.inner.eos_token_id()
    }

    /// Get the BOS token ID (if configured).
    #[getter]
    fn bos_token_id(&self) -> Option<u32> {
        self.inner.bos_token_id()
    }

    /// Create a streaming decoder for UTF-8 safe token-by-token decoding.
    ///
    /// Built from this tokenizer's own decode configuration — `▁` word
    /// separator, `<0xNN>` byte fallback and all — so `"".join(chunks) +
    /// flush()` reproduces `decode`.
    ///
    /// Returns:
    ///     StreamingDecoder instance
    fn streaming_decoder(&self) -> PyStreamingDecoder {
        PyStreamingDecoder::new(self.inner.streaming_decoder())
    }

    /// A streaming decoder that renders special tokens instead of dropping them.
    ///
    /// `streaming_decoder` reproduces `decode`; this one reproduces
    /// `decode_with_special`, so a generation loop can watch the control markers
    /// go past. `"".join(chunks) + flush()` equals `decode_with_special(ids)`.
    ///
    /// Returns:
    ///     StreamingDecoder instance
    fn streaming_decoder_with_special(&self) -> PyStreamingDecoder {
        PyStreamingDecoder::new(self.inner.streaming_decoder_with(SpecialDecode::Render))
    }

    fn __repr__(&self) -> String {
        format!(
            "SentencePieceTokenizer(vocab_size={})",
            self.inner.vocab_size()
        )
    }
}

impl PySentencePieceTokenizer {
    /// Prepend the constructor-supplied BOS id, if any. The backend itself is
    /// policy-free (`SentencePieceTokenizer::encode` never inserts boundary
    /// tokens), so this binding owns the BOS-prepending behaviour it has
    /// always exposed to Python.
    fn with_bos(&self, mut ids: Vec<u32>) -> Vec<u32> {
        if let Some(bos) = self.bos_token_id {
            ids.insert(0, bos);
        }
        ids
    }
}

/// Python wrapper for the SentencePiece **BPE** tokenizer.
///
/// This is the llama.cpp `SPM` algorithm (`tokenizer.ggml.model = "llama"`), not
/// the Unigram tokenizer that `SentencePieceTokenizer` implements. It merges the
/// best-scoring adjacent *pieces* repeatedly, which is what keeps the `▁`
/// word-boundary marker (U+2581) intact: a byte-level merger cannot build it,
/// because its middle byte pair `E2 96` is not a piece any SentencePiece
/// vocabulary was trained on.
///
/// This is the backend behind `Tokenizer.from_pretrained("mistral" |
/// "mistral_v1" | "mistral_v2")`, which returns it wrapped in an `AnyTokenizer`
/// (`.family == "Spm"`) so the vocabulary's policy travels with it. Construct
/// this class directly only when you hold the pieces and scores yourself.
#[pyclass(name = "SpmTokenizer")]
pub struct PySpmTokenizer {
    inner: SpmTokenizer,
    policy: SpecialPolicy,
}

#[pymethods]
impl PySpmTokenizer {
    /// Create a SentencePiece BPE tokenizer.
    ///
    /// Args:
    ///     tokens: List of token strings, indexed by token ID (`▁` marks word
    ///         boundaries, `<0xNN>` are the byte-fallback tokens)
    ///     scores: Per-token merge ranks, higher merging earlier (empty list
    ///         falls back to token-ID order, the convention these vocabularies follow)
    ///     bos_token_id: Optional beginning-of-sequence token ID
    ///     eos_token_id: Optional end-of-sequence token ID
    #[new]
    #[pyo3(signature = (tokens, scores, bos_token_id=None, eos_token_id=None))]
    fn new(
        tokens: Vec<String>,
        scores: Vec<f32>,
        bos_token_id: Option<u32>,
        eos_token_id: Option<u32>,
    ) -> PyResult<Self> {
        let inner = SpmTokenizer::new(tokens, scores, bos_token_id, eos_token_id)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner,
            policy: SpecialPolicy::default(),
        })
    }

    /// Encode text to token IDs — model-ready.
    ///
    /// Applies this tokenizer's boundary template, so the result is HuggingFace's
    /// `tokenizer.encode(text)` with its default `add_special_tokens=True`. Use
    /// `encode_raw` for the untemplated form; the two are identical when the
    /// vocabulary declares no template.
    ///
    /// Control tokens present in the text (`[INST]`, `<s>`, chat markers) are
    /// recognized as single IDs — SentencePiece merging would otherwise shred
    /// them into ordinary pieces. `encode_ordinary` and `encode_allowed_special`
    /// constrain that.
    fn encode(&self, text: &str) -> Vec<u32> {
        self.policy
            .apply_single(Tokenize::encode(&self.inner, text))
    }

    /// Encode text to token IDs — content tokens only, no boundary template.
    ///
    /// HuggingFace's `tokenizer.encode(text, add_special_tokens=False)`. Use it
    /// when you assemble the sequence yourself and place the boundary tokens by
    /// hand. `encode` is exactly this call plus the template.
    ///
    /// Args:
    ///     text: Input text to encode
    ///
    /// Returns:
    ///     List of token IDs
    fn encode_raw(&self, text: &str) -> Vec<u32> {
        Tokenize::encode(&self.inner, text)
    }

    /// Encode text, matching **every** control token spelled out in it.
    ///
    /// tiktoken's `allowed_special="all"`. Unlike the byte-level BPE `Tokenizer`,
    /// whose `encode` treats special tokens as ordinary text unless the vocabulary
    /// was built with added-token matching on, this SPM-BPE backend's `encode`
    /// already recognizes control tokens — so this is what `encode` does. The name
    /// exists so the method means the same thing on every splintr tokenizer class.
    /// The boundary template is applied, as in `encode`.
    ///
    /// Args:
    ///     text: Input text to encode
    ///
    /// Returns:
    ///     List of token IDs
    fn encode_with_special(&self, text: &str) -> Vec<u32> {
        self.encode(text)
    }

    /// Batch form of `encode` — model-ready ids for each text.
    ///
    /// Uses Rayon to parallelize encoding across texts, mirroring
    /// `Tokenizer::encode_batch`.
    ///
    /// Args:
    ///     texts: List of texts to encode
    ///
    /// Returns:
    ///     List of token ID lists
    fn encode_batch(&self, texts: Vec<String>) -> Vec<Vec<u32>> {
        #[cfg(feature = "rayon")]
        {
            texts.par_iter().map(|text| self.encode(text)).collect()
        }
        #[cfg(not(feature = "rayon"))]
        {
            texts.iter().map(|text| self.encode(text)).collect()
        }
    }

    /// Encode text, never matching a control token spelled out in it.
    ///
    /// A control token written literally in the input (`[INST]`, `<s>`, chat
    /// markers) is encoded as ordinary text instead of being recognized as a
    /// single id — tiktoken's `allowed_special=set()`. Use this when tokenizing
    /// untrusted text where the caller must not be able to forge control tokens.
    ///
    /// The boundary template is still applied, as in `encode`: it comes from this
    /// tokenizer's own configuration, not from matching text against the
    /// vocabulary. Use `encode_raw` for the untemplated form.
    ///
    /// Args:
    ///     text: Input text to encode
    ///
    /// Returns:
    ///     List of token IDs
    fn encode_ordinary(&self, text: &str) -> Vec<u32> {
        self.policy.apply_single(self.inner.encode_ordinary(text))
    }

    /// Encode text, matching only the named control tokens spelled out in it.
    ///
    /// tiktoken's `allowed_special={...}`: any *other* configured control token
    /// written literally in the text raises `ValueError` instead of being silently
    /// recognized as a single id — use this to accept a known, bounded set of
    /// control tokens from otherwise untrusted text. The boundary template is
    /// applied, as in `encode`.
    ///
    /// Args:
    ///     text: Input text to encode
    ///     allowed_special: Control token strings permitted to match in `text`
    ///
    /// Returns:
    ///     List of token IDs
    ///
    /// Raises:
    ///     ValueError: If `text` spells out a configured control token that is
    ///         not in `allowed_special`
    fn encode_allowed_special(
        &self,
        text: &str,
        allowed_special: Vec<String>,
    ) -> PyResult<Vec<u32>> {
        let allowed: FxHashSet<String> = allowed_special.into_iter().collect();
        let ids = self
            .inner
            .encode_with(text, &SpecialMode::Allow(&allowed))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(self.policy.apply_single(ids))
    }

    /// Decode token IDs to text, converting `▁` back to spaces, reassembling
    /// `<0xNN>` byte-fallback runs, and dropping the single leading space that
    /// `add_dummy_prefix` introduced on encode (matching `sp.decode`).
    ///
    /// Raises:
    ///     ValueError: If a token ID is out of range or the result is not UTF-8
    fn decode(&self, ids: Vec<u32>) -> PyResult<String> {
        Tokenize::decode(&self.inner, &ids).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Decode token IDs to a string, rendering the vocabulary's special tokens
    /// instead of dropping them.
    ///
    /// `decode` implements HuggingFace's default `skip_special_tokens=True`, so
    /// a control marker (`[INST]`, `<|eot_id|>`, `<s>`) renders as nothing.
    /// This is its `skip_special_tokens=False`: every declared special id comes
    /// back as its own spelling, which is what a chat-template round trip, an
    /// inspection of raw model output, or a transcript that must keep its
    /// markers needs.
    ///
    /// Args:
    ///     ids: List of token IDs
    ///
    /// Returns:
    ///     Decoded string, control markers included
    ///
    /// Raises:
    ///     ValueError: Exactly as `decode` does
    ///
    /// Example:
    ///     tok = Tokenizer.from_pretrained("mistral_v2")
    ///     ids = tok.encode_with_special("[INST]Hi[/INST]")
    ///     tok.decode(ids)               # "Hi"
    ///     tok.decode_with_special(ids)  # "[INST]Hi[/INST]"
    fn decode_with_special(&self, ids: Vec<u32>) -> PyResult<String> {
        Tokenize::decode_with(&self.inner, &ids, SpecialDecode::Render)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// The bytes `id` contributes to the decoded stream — `▁` unmapped and
    /// `<0xNN>` byte fallback resolved, and nothing else: no leading-space
    /// strip, no first-token rule, no word separator.
    ///
    /// An id in the vocabulary that carries no surface (a special this
    /// tokenizer's decode drops, say) returns **empty** bytes, not an error.
    /// Because of that, concatenating this over a sequence of ids is *not*
    /// what `decode` returns for the same sequence — `decode` layers
    /// post-processing this method deliberately skips. Use
    /// `streaming_decoder` to render a sequence.
    ///
    /// Args:
    ///     id: Token ID
    ///
    /// Returns:
    ///     The id's raw bytes, or `b""` if it carries no surface
    ///
    /// Raises:
    ///     ValueError: If `id` is outside the vocabulary
    fn decode_token_bytes(&self, id: u32) -> PyResult<Vec<u8>> {
        Tokenize::decode_token_bytes(&self.inner, id)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// `decode_token_bytes` as text.
    ///
    /// Raises far more often than `decode` does: a single `<0xNN>`
    /// byte-fallback id, or a token holding one byte of a multi-byte
    /// character, is not valid UTF-8 standing alone — that is the expected
    /// signal to stop decoding id-at-a-time and use `streaming_decoder`,
    /// which buffers exactly those partial sequences across tokens.
    ///
    /// Args:
    ///     id: Token ID
    ///
    /// Returns:
    ///     The id's text, or `""` if it carries no surface
    ///
    /// Raises:
    ///     ValueError: If `id` is outside the vocabulary, or if its bytes are
    ///         not valid UTF-8 on their own
    fn decode_token(&self, id: u32) -> PyResult<String> {
        Tokenize::decode_token(&self.inner, id).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Get vocabulary size.
    #[getter]
    fn vocab_size(&self) -> usize {
        Tokenize::vocab_size(&self.inner)
    }

    /// Get the EOS token ID, if the vocabulary defines one.
    #[getter]
    fn eos_token_id(&self) -> Option<u32> {
        self.inner.eos_token_id()
    }

    /// Get the BOS token ID, if the vocabulary defines one.
    #[getter]
    fn bos_token_id(&self) -> Option<u32> {
        self.inner.bos_token_id()
    }

    /// Create a streaming decoder for UTF-8 safe token-by-token decoding.
    ///
    /// Built from this tokenizer's own decode configuration — `▁` word
    /// separator, `<0xNN>` byte fallback and all — so `"".join(chunks) +
    /// flush()` reproduces `decode`.
    ///
    /// Returns:
    ///     StreamingDecoder instance
    fn streaming_decoder(&self) -> PyStreamingDecoder {
        PyStreamingDecoder::new(self.inner.streaming_decoder())
    }

    /// A streaming decoder that renders special tokens instead of dropping them.
    ///
    /// `streaming_decoder` reproduces `decode`; this one reproduces
    /// `decode_with_special`, so a generation loop can watch the control markers
    /// go past. `"".join(chunks) + flush()` equals `decode_with_special(ids)`.
    ///
    /// Returns:
    ///     StreamingDecoder instance
    fn streaming_decoder_with_special(&self) -> PyStreamingDecoder {
        PyStreamingDecoder::new(self.inner.streaming_decoder_with(SpecialDecode::Render))
    }

    fn __repr__(&self) -> String {
        format!(
            "SpmTokenizer(vocab_size={})",
            Tokenize::vocab_size(&self.inner)
        )
    }
}

/// WordPiece tokenizer (BERT family). Construct via [`from_json`] or directly.
#[pyclass(name = "WordPieceTokenizer")]
pub struct PyWordPieceTokenizer {
    inner: WordPieceTokenizer,
    policy: SpecialPolicy,
}

#[pymethods]
impl PyWordPieceTokenizer {
    /// Create a WordPiece tokenizer.
    ///
    /// Args:
    ///     vocab: List of token strings, indexed by token ID (`##` marks continuations)
    ///     unk_token_id: ID of the unknown token (e.g. `[UNK]`)
    ///     max_word_len: Max characters per word before it maps to unk (default 100)
    ///     do_lower_case: Lowercase input before tokenizing (BERT-uncased style)
    ///     strip_accents: Strip accents, independently of casing. `None` (the
    ///         default) follows `do_lower_case`, matching HuggingFace's rule for
    ///         a `BertNormalizer` whose `strip_accents` is absent/`null`; pass
    ///         `True`/`False` to set it on its own (a cased multilingual BERT
    ///         vocabulary distinguishing `café` from `cafe` needs `False`).
    #[new]
    #[pyo3(signature = (vocab, unk_token_id, max_word_len=100, do_lower_case=false, strip_accents=None))]
    fn new(
        vocab: Vec<String>,
        unk_token_id: u32,
        max_word_len: usize,
        do_lower_case: bool,
        strip_accents: Option<bool>,
    ) -> Self {
        let inner = WordPieceTokenizer::new(vocab, unk_token_id, max_word_len, do_lower_case);
        Self {
            inner: match strip_accents {
                Some(strip) => inner.with_strip_accents(strip),
                None => inner,
            },
            policy: SpecialPolicy::default(),
        }
    }

    /// Encode text to token IDs — model-ready.
    ///
    /// Applies this tokenizer's `post_processor` template (BERT's `[CLS]…[SEP]`),
    /// so the result is HuggingFace's `tokenizer.encode(text)` with its default
    /// `add_special_tokens=True`. Use `encode_raw` for the untemplated form; the
    /// two are identical when no post-processor is declared — which is the case
    /// for this class's direct constructor, since it takes a vocabulary and no
    /// template.
    ///
    /// Special tokens spelled out inside `text` are matched; `encode_ordinary` and
    /// `encode_allowed_special` constrain that.
    fn encode(&self, text: &str) -> Vec<u32> {
        self.policy
            .apply_single(Tokenize::encode(&self.inner, text))
    }

    /// Encode text to token IDs — content tokens only, no `post_processor`.
    ///
    /// HuggingFace's `tokenizer.encode(text, add_special_tokens=False)`: no
    /// `[CLS]`/`[SEP]` wrapping. Use it when you assemble the sequence yourself
    /// (a reranker pair, a custom template). `encode` is exactly this call plus
    /// the template.
    ///
    /// Args:
    ///     text: Input text to encode
    ///
    /// Returns:
    ///     List of token IDs
    fn encode_raw(&self, text: &str) -> Vec<u32> {
        Tokenize::encode(&self.inner, text)
    }

    /// Encode text, matching **every** configured special token spelled out in it.
    ///
    /// tiktoken's `allowed_special="all"`. This backend always matches its added
    /// tokens, so this is what `encode` already does — the name exists so the
    /// method means the same thing on every splintr tokenizer class. The
    /// `post_processor` template is applied, as in `encode`.
    ///
    /// Args:
    ///     text: Input text to encode
    ///
    /// Returns:
    ///     List of token IDs
    fn encode_with_special(&self, text: &str) -> Vec<u32> {
        self.encode(text)
    }

    /// Encode text, never matching a special token spelled out in it.
    ///
    /// A special token written literally in the input (e.g. `[CLS]`, `[SEP]`) is
    /// encoded as ordinary text instead of being recognized as a single id —
    /// tiktoken's `allowed_special=set()`. Use this when tokenizing untrusted text
    /// where the caller must not be able to forge control tokens.
    ///
    /// The `post_processor` template is still applied, as in `encode`: its
    /// `[CLS]`/`[SEP]` come from the template, not from matching text against the
    /// vocabulary. Use `encode_raw` for the untemplated form.
    ///
    /// Args:
    ///     text: Input text to encode
    ///
    /// Returns:
    ///     List of token IDs
    fn encode_ordinary(&self, text: &str) -> Vec<u32> {
        self.policy.apply_single(self.inner.encode_ordinary(text))
    }

    /// Encode text, matching only the named special tokens spelled out in it.
    ///
    /// tiktoken's `allowed_special={...}`: any *other* configured special token
    /// written literally in the text raises `ValueError` instead of being silently
    /// recognized as a single id — use this to accept a known, bounded set of
    /// special tokens from otherwise untrusted text. The `post_processor` template
    /// is applied, as in `encode`.
    ///
    /// Args:
    ///     text: Input text to encode
    ///     allowed_special: Special token strings permitted to match in `text`
    ///
    /// Returns:
    ///     List of token IDs
    ///
    /// Raises:
    ///     ValueError: If `text` spells out a configured special token that is
    ///         not in `allowed_special`
    fn encode_allowed_special(
        &self,
        text: &str,
        allowed_special: Vec<String>,
    ) -> PyResult<Vec<u32>> {
        let allowed: FxHashSet<String> = allowed_special.into_iter().collect();
        let ids = self
            .inner
            .encode_with(text, &SpecialMode::Allow(&allowed))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(self.policy.apply_single(ids))
    }

    /// Batch form of `encode` — model-ready ids for each text.
    ///
    /// Uses Rayon to parallelize encoding across texts.
    ///
    /// Args:
    ///     texts: List of texts to encode
    ///
    /// Returns:
    ///     List of token ID lists
    fn encode_batch(&self, texts: Vec<String>) -> Vec<Vec<u32>> {
        #[cfg(feature = "rayon")]
        {
            texts.par_iter().map(|text| self.encode(text)).collect()
        }
        #[cfg(not(feature = "rayon"))]
        {
            texts.iter().map(|text| self.encode(text)).collect()
        }
    }

    /// Decode token IDs to text.
    fn decode(&self, ids: Vec<u32>) -> PyResult<String> {
        Tokenize::decode(&self.inner, &ids).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Decode token IDs to a string, rendering the vocabulary's special tokens
    /// instead of dropping them.
    ///
    /// `decode` implements HuggingFace's default `skip_special_tokens=True`, so
    /// a control marker (`[INST]`, `<|eot_id|>`, `<s>`) renders as nothing.
    /// This is its `skip_special_tokens=False`: every declared special id comes
    /// back as its own spelling, which is what a chat-template round trip, an
    /// inspection of raw model output, or a transcript that must keep its
    /// markers needs.
    ///
    /// Args:
    ///     ids: List of token IDs
    ///
    /// Returns:
    ///     Decoded string, control markers included
    ///
    /// Raises:
    ///     ValueError: Exactly as `decode` does
    ///
    /// Example:
    ///     tok = Tokenizer.from_pretrained("mistral_v2")
    ///     ids = tok.encode_with_special("[INST]Hi[/INST]")
    ///     tok.decode(ids)               # "Hi"
    ///     tok.decode_with_special(ids)  # "[INST]Hi[/INST]"
    fn decode_with_special(&self, ids: Vec<u32>) -> PyResult<String> {
        Tokenize::decode_with(&self.inner, &ids, SpecialDecode::Render)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// The token's own bytes, with any `##` continuation marker removed and
    /// **without** the word separator a word-starting token carries: that
    /// separator sits between two tokens, so it belongs to the sequence and
    /// not to this id.
    ///
    /// An id in the vocabulary that carries no surface (a special this
    /// tokenizer's decode drops, say) returns **empty** bytes, not an error.
    /// Because of that, concatenating this over a sequence of ids is *not*
    /// what `decode` returns for the same sequence — `decode` layers
    /// post-processing this method deliberately skips. Use
    /// `streaming_decoder` to render a sequence.
    ///
    /// Args:
    ///     id: Token ID
    ///
    /// Returns:
    ///     The id's raw bytes, or `b""` if it carries no surface
    ///
    /// Raises:
    ///     ValueError: If `id` is outside the vocabulary
    fn decode_token_bytes(&self, id: u32) -> PyResult<Vec<u8>> {
        Tokenize::decode_token_bytes(&self.inner, id)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// `decode_token_bytes` as text.
    ///
    /// Raises far more often than `decode` does: a token holding one byte of
    /// a multi-byte character is not valid UTF-8 standing alone — that is the
    /// expected signal to stop decoding id-at-a-time and use
    /// `streaming_decoder`, which buffers exactly those partial sequences
    /// across tokens.
    ///
    /// Args:
    ///     id: Token ID
    ///
    /// Returns:
    ///     The id's text, or `""` if it carries no surface
    ///
    /// Raises:
    ///     ValueError: If `id` is outside the vocabulary, or if its bytes are
    ///         not valid UTF-8 on their own
    fn decode_token(&self, id: u32) -> PyResult<String> {
        Tokenize::decode_token(&self.inner, id).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Vocabulary size.
    fn vocab_size(&self) -> usize {
        Tokenize::vocab_size(&self.inner)
    }

    #[getter]
    fn unk_token_id(&self) -> u32 {
        self.inner.unk_token_id()
    }
    #[getter]
    fn cls_token_id(&self) -> Option<u32> {
        self.inner.cls_token_id()
    }
    #[getter]
    fn sep_token_id(&self) -> Option<u32> {
        self.inner.sep_token_id()
    }
    #[getter]
    fn pad_token_id(&self) -> Option<u32> {
        self.inner.pad_token_id()
    }

    /// Create a streaming decoder for UTF-8 safe token-by-token decoding.
    ///
    /// Built from this tokenizer's own decode configuration — the `##`
    /// continuation prefix and the word separator included — so
    /// `"".join(chunks) + flush()` reproduces `decode`.
    ///
    /// Returns:
    ///     StreamingDecoder instance
    fn streaming_decoder(&self) -> PyStreamingDecoder {
        PyStreamingDecoder::new(self.inner.streaming_decoder())
    }

    /// A streaming decoder that renders special tokens instead of dropping them.
    ///
    /// `streaming_decoder` reproduces `decode`; this one reproduces
    /// `decode_with_special`, so a generation loop can watch the control markers
    /// go past. `"".join(chunks) + flush()` equals `decode_with_special(ids)`.
    ///
    /// Returns:
    ///     StreamingDecoder instance
    fn streaming_decoder_with_special(&self) -> PyStreamingDecoder {
        PyStreamingDecoder::new(self.inner.streaming_decoder_with(SpecialDecode::Render))
    }
}

/// Python wrapper for a loaded [`AnyTokenizer`] — the universal handle every
/// loader returns.
///
/// This holds the `AnyTokenizer` **whole** rather than unpacking it into one of
/// the family-specific wrappers above. That matters because an `AnyTokenizer`
/// is more than its backend: it also carries the special-token policy, the
/// `decoder` pipeline declared in the `tokenizer.json`, and the set of
/// `special=true` ids to drop on decode. Unpacking kept only the backend and
/// the policy, so a file whose decoding *is* its declared pipeline (Mistral's
/// `Replace ▁→" "` → `ByteFallback` → `Fuse` → `Strip`) decoded to raw pieces
/// like `▁hello▁world` the moment it crossed into Python. Carrying the handle
/// whole means a field added to `AnyTokenizer` cannot go missing here.
///
/// Every method delegates to `AnyTokenizer`'s own inherent method, so Python
/// and Rust cannot drift apart in what `encode`/`decode` mean.
#[pyclass(name = "AnyTokenizer")]
pub struct PyAnyTokenizer {
    inner: AnyTokenizer,
}

#[pymethods]
impl PyAnyTokenizer {
    /// Encode text to token IDs — model-ready.
    ///
    /// Applies the model's boundary template (`post_processor`: `[CLS]…[SEP]`,
    /// `<s>…</s>`), matching HuggingFace's `tokenizer.encode(text)` with its
    /// default `add_special_tokens=True`. Use `encode_raw` for the untemplated
    /// form.
    ///
    /// Special tokens spelled out literally in `text` are matched (every loader
    /// turns added-token matching on); use `encode_ordinary` or
    /// `encode_allowed_special` to constrain that.
    ///
    /// Args:
    ///     text: Input text to encode
    ///
    /// Returns:
    ///     List of token IDs
    fn encode(&self, text: &str) -> Vec<u32> {
        self.inner.encode(text)
    }

    /// Encode text to token IDs — content tokens only, no boundary template.
    ///
    /// The backend's output alone, matching HuggingFace's
    /// `tokenizer.encode(text, add_special_tokens=False)`. Use this when
    /// assembling your own sequence (a chat template, a reranker pair) and
    /// placing the boundary tokens yourself. `encode` is exactly this call plus
    /// the template.
    ///
    /// Args:
    ///     text: Input text to encode
    ///
    /// Returns:
    ///     List of token IDs
    fn encode_raw(&self, text: &str) -> Vec<u32> {
        self.inner.encode_raw(text)
    }

    /// Encode text, matching **every** configured special token spelled out in it.
    ///
    /// tiktoken's `allowed_special="all"`. This is what `encode` already does —
    /// the name exists so the method means the same thing on every splintr
    /// tokenizer class. Distinct from `encode_ordinary` (matches none) and
    /// `encode_allowed_special` (matches a named subset). The boundary template is
    /// applied, as in `encode`.
    ///
    /// Args:
    ///     text: Input text to encode
    ///
    /// Returns:
    ///     List of token IDs
    fn encode_with_special(&self, text: &str) -> PyResult<Vec<u32>> {
        self.inner
            .encode_with(text, &SpecialMode::All)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Encode text, never matching a special token spelled out in it.
    ///
    /// A special token written literally in the input (e.g. `<|endoftext|>`,
    /// `[INST]`) is encoded as ordinary text instead of being promoted to its
    /// control-token id — tiktoken's `allowed_special=set()`. Use this when
    /// tokenizing untrusted text where the caller must not be able to forge
    /// control tokens.
    ///
    /// The model's boundary template is still applied, exactly as in
    /// `AnyTokenizer::encode_with`: boundary tokens come from the template, not
    /// from matching text against the vocabulary, so locking down special-token
    /// matching in the *content* must not also strip the boundary tokens the
    /// model was trained with. Use `encode_raw` for the untemplated form.
    ///
    /// Args:
    ///     text: Input text to encode
    ///
    /// Returns:
    ///     List of token IDs
    fn encode_ordinary(&self, text: &str) -> PyResult<Vec<u32>> {
        self.inner
            .encode_with(text, &SpecialMode::Ordinary)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Encode text, matching only the named special tokens spelled out in it.
    ///
    /// tiktoken's `allowed_special={...}`: any *other* configured special token
    /// written literally in the text raises `ValueError` instead of being
    /// silently promoted to its control-token id — use this to accept a known,
    /// bounded set of special tokens (e.g. a chat template's own markers) from
    /// otherwise untrusted text. The boundary template is applied, as in
    /// `encode_ordinary`.
    ///
    /// Args:
    ///     text: Input text to encode
    ///     allowed_special: Special token strings permitted to match in `text`
    ///
    /// Returns:
    ///     List of token IDs
    ///
    /// Raises:
    ///     ValueError: If `text` spells out a configured special token that is
    ///         not in `allowed_special`
    fn encode_allowed_special(
        &self,
        text: &str,
        allowed_special: Vec<String>,
    ) -> PyResult<Vec<u32>> {
        let allowed: FxHashSet<String> = allowed_special.into_iter().collect();
        self.inner
            .encode_with(text, &SpecialMode::Allow(&allowed))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Batch encode multiple texts, applying the boundary template to each —
    /// the batch form of `encode`.
    ///
    /// Uses Rayon to parallelize across texts.
    ///
    /// Args:
    ///     texts: List of texts to encode
    ///
    /// Returns:
    ///     List of token ID lists
    fn encode_batch(&self, texts: Vec<String>) -> Vec<Vec<u32>> {
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        self.inner.encode_batch(&refs)
    }

    /// Batch form of `encode_with_special` — every special token spelled out in
    /// each text is matched, and the boundary template applied.
    ///
    /// Uses Rayon to parallelize across texts.
    ///
    /// Args:
    ///     texts: List of texts to encode
    ///
    /// Returns:
    ///     List of token ID lists
    fn encode_batch_with_special(&self, texts: Vec<String>) -> PyResult<Vec<Vec<u32>>> {
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        self.inner
            .encode_batch_with(&refs, &SpecialMode::All)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// `encode` with the work parallelized *within* the single text.
    ///
    /// Same semantics and same ids as `encode` (boundary template applied,
    /// special tokens in the text matched) — only the execution strategy
    /// differs. It carries thread-pool overhead `encode` does not, so it pays
    /// off only for very large texts (typically >1MB). For most uses prefer
    /// `encode` (sequential) or `encode_batch` (parallel across texts).
    ///
    /// A backend with no intra-text parallel path simply runs `encode`, so the
    /// ids never depend on which one this handle holds.
    ///
    /// Args:
    ///     text: Input text to encode
    ///
    /// Returns:
    ///     List of token IDs
    fn encode_rayon(&self, text: &str) -> Vec<u32> {
        self.inner.encode_rayon(text)
    }

    /// Decode token IDs to a string.
    ///
    /// Runs the `decoder` pipeline declared in the source `tokenizer.json` when
    /// there is one — dropping `special=true` ids first, matching HuggingFace's
    /// default `skip_special_tokens=True` — and the backend's built-in decode
    /// otherwise.
    ///
    /// Args:
    ///     ids: List of token IDs
    ///
    /// Returns:
    ///     Decoded string
    ///
    /// Raises:
    ///     ValueError: If a token ID is out of range or the bytes are not UTF-8
    fn decode(&self, ids: Vec<u32>) -> PyResult<String> {
        Tokenize::decode(&self.inner, &ids).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Decode token IDs to a string, rendering the vocabulary's special tokens
    /// instead of dropping them.
    ///
    /// `decode` implements HuggingFace's default `skip_special_tokens=True`, so
    /// a control marker (`[INST]`, `<|eot_id|>`, `<s>`) renders as nothing.
    /// This is its `skip_special_tokens=False`: every declared special id comes
    /// back as its own spelling, which is what a chat-template round trip, an
    /// inspection of raw model output, or a transcript that must keep its
    /// markers needs.
    ///
    /// Args:
    ///     ids: List of token IDs
    ///
    /// Returns:
    ///     Decoded string, control markers included
    ///
    /// Raises:
    ///     ValueError: Exactly as `decode` does
    ///
    /// Example:
    ///     tok = Tokenizer.from_pretrained("mistral_v2")
    ///     ids = tok.encode_with_special("[INST]Hi[/INST]")
    ///     tok.decode(ids)               # "Hi"
    ///     tok.decode_with_special(ids)  # "[INST]Hi[/INST]"
    fn decode_with_special(&self, ids: Vec<u32>) -> PyResult<String> {
        Tokenize::decode_with(&self.inner, &ids, SpecialDecode::Render)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Batch decode multiple token lists — the batch form of `decode`, running
    /// the same declared `decoder` pipeline and the same `special=true` skip.
    ///
    /// Uses Rayon to parallelize across token lists.
    ///
    /// Args:
    ///     token_lists: List of token ID lists
    ///
    /// Returns:
    ///     List of decoded strings
    ///
    /// Raises:
    ///     ValueError: If any token ID is out of range or the bytes are not UTF-8
    fn decode_batch(&self, token_lists: Vec<Vec<u32>>) -> PyResult<Vec<String>> {
        self.inner
            .decode_batch(&token_lists)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Decode token IDs to raw bytes.
    ///
    /// The byte-level BPE backend's own decode, without the UTF-8 validation
    /// `decode` performs — use it when the ids may end mid-character.
    ///
    /// Args:
    ///     tokens: List of token IDs
    ///
    /// Returns:
    ///     Decoded bytes
    ///
    /// Raises:
    ///     ValueError: If this tokenizer's backend is not byte-level BPE, or if
    ///         its source declared a `decoder` pipeline (see `decode`, which
    ///         runs it — bytes taken from under it would render the backend's
    ///         raw pieces instead of text)
    fn decode_bytes(&self, tokens: Vec<u32>) -> PyResult<Vec<u8>> {
        self.bpe_raw()?
            .decode_bytes(&tokens)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Decode token IDs to a string, replacing invalid UTF-8.
    ///
    /// Args:
    ///     tokens: List of token IDs
    ///
    /// Returns:
    ///     Decoded string with replacement characters for invalid UTF-8
    ///
    /// Raises:
    ///     ValueError: Under the same conditions as `decode_bytes`
    fn decode_lossy(&self, tokens: Vec<u32>) -> PyResult<String> {
        Ok(self.bpe_raw()?.decode_lossy(&tokens))
    }

    /// Batch form of `decode_lossy`, parallelized across token lists.
    ///
    /// Args:
    ///     token_lists: List of token ID lists
    ///
    /// Returns:
    ///     List of decoded strings with replacement characters for invalid UTF-8
    ///
    /// Raises:
    ///     ValueError: Under the same conditions as `decode_bytes`
    fn decode_batch_lossy(&self, token_lists: Vec<Vec<u32>>) -> PyResult<Vec<String>> {
        Ok(self.bpe_raw()?.decode_batch_lossy(&token_lists))
    }

    /// The bytes `id` contributes to the decoded stream — running the
    /// declared `decoder` pipeline when there is one, and the backend's own
    /// rules otherwise, exactly as `decode` branches. No sequence-level
    /// post-processing runs: no leading-space strip, no first-token rule, no
    /// word separator.
    ///
    /// An id in the vocabulary that carries no surface (a special this
    /// tokenizer's decode drops, say) returns **empty** bytes, not an error.
    /// Because of that, concatenating this over a sequence of ids is *not*
    /// what `decode` returns for the same sequence — `decode` layers
    /// post-processing this method deliberately skips. Use
    /// `streaming_decoder` to render a sequence.
    ///
    /// Args:
    ///     id: Token ID
    ///
    /// Returns:
    ///     The id's raw bytes, or `b""` if it carries no surface
    ///
    /// Raises:
    ///     ValueError: If `id` is outside the vocabulary entirely, or — only
    ///         when this tokenizer declares a `decoder` pipeline holding a
    ///         step that cannot be evaluated one token at a time — under the
    ///         same conditions as `streaming_decoder`
    fn decode_token_bytes(&self, id: u32) -> PyResult<Vec<u8>> {
        self.inner
            .decode_token_bytes(id)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// `decode_token_bytes` as text.
    ///
    /// Raises far more often than `decode` does: a single `<0xNN>`
    /// byte-fallback id, or a token holding one byte of a multi-byte
    /// character, is not valid UTF-8 standing alone — that is the expected
    /// signal to stop decoding id-at-a-time and use `streaming_decoder`,
    /// which buffers exactly those partial sequences across tokens.
    ///
    /// Args:
    ///     id: Token ID
    ///
    /// Returns:
    ///     The id's text, or `""` if it carries no surface
    ///
    /// Raises:
    ///     ValueError: Under the same conditions as `decode_token_bytes`, and
    ///         also if the id's bytes are not valid UTF-8 on their own
    fn decode_token(&self, id: u32) -> PyResult<String> {
        self.inner
            .decode_token(id)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Switch this tokenizer's regex backend between regexr and PCRE2.
    ///
    /// Only the byte-level BPE backend has a regex pre-tokenizer to configure;
    /// on any other family this raises rather than silently reporting success.
    ///
    /// Unlike `Tokenizer.pcre2`, this configures **this** handle and returns it
    /// (so calls still chain) rather than handing back a second tokenizer: the
    /// policy, the declared `decoder` pipeline and the special-id set stay
    /// attached instead of having to be re-derived.
    ///
    /// Args:
    ///     use_pcre2: If True, switch to PCRE2. If False, switch to regexr (default: True)
    ///
    /// Returns:
    ///     This same tokenizer, reconfigured
    ///
    /// Raises:
    ///     ValueError: If use_pcre2=True and the `pcre2` feature is not enabled,
    ///         or if this tokenizer's backend is not byte-level BPE
    ///
    /// Example:
    ///     tokenizer = Tokenizer.from_pretrained("cl100k_base").pcre2(True)
    #[pyo3(signature = (use_pcre2=true))]
    fn pcre2<'py>(mut slf: PyRefMut<'py, Self>, use_pcre2: bool) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner
            .set_pcre2(use_pcre2)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(slf)
    }

    /// Enable or disable JIT compilation for the regex backend.
    ///
    /// JIT availability depends on platform support (e.g. x86-64) and on the
    /// crate features (regexr `jit`, pcre2 `jit`). It is enabled by default.
    /// As with `pcre2`, this configures and returns **this** handle.
    ///
    /// Args:
    ///     use_jit: Whether to try using JIT compilation (default: True)
    ///
    /// Returns:
    ///     This same tokenizer, reconfigured
    ///
    /// Raises:
    ///     ValueError: If this tokenizer's backend is not byte-level BPE
    ///
    /// Example:
    ///     tokenizer = Tokenizer.from_pretrained("cl100k_base").jit(False)
    #[pyo3(signature = (use_jit=true))]
    fn jit<'py>(mut slf: PyRefMut<'py, Self>, use_jit: bool) -> PyResult<PyRefMut<'py, Self>> {
        slf.inner
            .set_jit(use_jit)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(slf)
    }

    /// Create a streaming decoder for UTF-8 safe token-by-token decoding.
    ///
    /// Useful for streaming LLM output where token boundaries may not align
    /// with UTF-8 character boundaries.
    ///
    /// The decision mirrors `decode` exactly, so the two can never disagree
    /// about what a sequence of ids says: a declared `decoder` pipeline drives
    /// the stream, and with none declared the backend's own decoder does. Every
    /// family is served, not just byte-level BPE.
    ///
    /// Returns:
    ///     StreamingDecoder instance
    ///
    /// Raises:
    ///     ValueError: If this tokenizer declares a `decoder` pipeline holding a
    ///         step that cannot be evaluated one chunk at a time (a
    ///         `BPEDecoder`, a trailing `Strip`, a `Replace` over the fused
    ///         text). The message names the step. Falling back to the backend's
    ///         own decode would render the raw pieces (`▁hello▁world`) the
    ///         pipeline exists to turn into text, so this refuses instead —
    ///         whole-sequence `decode` still handles those files.
    ///
    /// Example:
    ///     decoder = tokenizer.streaming_decoder()
    ///     for token_id in token_stream:
    ///         if text := decoder.add_token(token_id):
    ///             print(text, end="", flush=True)
    ///     print(decoder.flush())
    fn streaming_decoder(&self) -> PyResult<PyStreamingDecoder> {
        self.inner
            .streaming_decoder()
            .map(PyStreamingDecoder::new)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// A streaming decoder that renders special tokens instead of dropping them.
    ///
    /// `streaming_decoder` reproduces `decode`; this one reproduces
    /// `decode_with_special`, so a generation loop can watch the control markers
    /// go past. `"".join(chunks) + flush()` equals `decode_with_special(ids)`.
    ///
    /// Returns:
    ///     StreamingDecoder instance
    ///
    /// Raises:
    ///     ValueError: Exactly as `streaming_decoder` does
    fn streaming_decoder_with_special(&self) -> PyResult<PyStreamingDecoder> {
        self.inner
            .streaming_decoder_with(SpecialDecode::Render)
            .map(PyStreamingDecoder::new)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Clear the encoding cache.
    ///
    /// Raises:
    ///     ValueError: If this tokenizer's backend is not byte-level BPE — it is
    ///         the only one that caches encoded chunks
    fn clear_cache(&self) -> PyResult<()> {
        self.bpe()?.clear_cache();
        Ok(())
    }

    /// Get the number of entries in the encoding cache.
    ///
    /// Raises:
    ///     ValueError: Under the same conditions as `clear_cache`
    #[getter]
    fn cache_len(&self) -> PyResult<usize> {
        Ok(self.bpe()?.cache_len())
    }

    /// Get the vocabulary size (including special tokens).
    #[getter]
    fn vocab_size(&self) -> usize {
        Tokenize::vocab_size(&self.inner)
    }

    /// Check if a token is the EOS token.
    fn is_eos(&self, token_id: u32) -> bool {
        self.inner.is_eos(token_id)
    }

    /// Get the EOS token ID, if the source names one.
    #[getter]
    fn eos_token_id(&self) -> Option<u32> {
        self.inner.eos_token_id()
    }

    /// Get the id of an added token by its content (e.g. `"[CLS]"`,
    /// `"<|im_end|>"`), or None if this tokenizer does not define it.
    fn special_token_id(&self, name: &str) -> Option<u32> {
        self.inner.special_token_id(name)
    }

    /// Every named special token this tokenizer knows, as `{content: id}`.
    ///
    /// The enumerating counterpart to `special_token_id`, which can only answer
    /// about a name you already have. Works for every loader, so listing a
    /// vocabulary's markers never means looking them up in a table that may
    /// have drifted:
    ///
    /// ```python
    /// tok = Tokenizer.from_pretrained("qwen3")
    /// base = base_vocab_size("qwen3")
    /// for name, tid in sorted(tok.special_tokens().items(), key=lambda kv: kv[1]):
    ///     print(f"{tid:>7}  {name}  {'(model)' if tid < base else '(splintr)'}")
    /// ```
    fn special_tokens(&self) -> std::collections::HashMap<String, u32> {
        self.inner
            .special_tokens()
            .iter()
            .map(|(name, id)| (name.clone(), *id))
            .collect()
    }

    /// The backend family this handle holds ("BPE", "Unigram", "WordPiece", "Spm").
    #[getter]
    fn family(&self) -> &'static str {
        self.inner.family()
    }

    fn __repr__(&self) -> String {
        format!(
            "AnyTokenizer(family={}, vocab_size={})",
            self.inner.family(),
            Tokenize::vocab_size(&self.inner)
        )
    }
}

impl PyAnyTokenizer {
    /// Borrow the byte-level BPE backend, for the surfaces only it defines
    /// (the chunk cache and the raw byte decodes).
    ///
    /// Delegating to the backend keeps those methods reachable from the
    /// universal handle without a second construction path; a family that does
    /// not define them says so rather than answering with a plausible-looking
    /// substitute.
    fn bpe(&self) -> PyResult<&Tokenizer> {
        match self.inner.backend() {
            Backend::Bpe(bpe) => Ok(bpe),
            _ => Err(PyValueError::new_err(format!(
                "this method needs the byte-level BPE backend; this tokenizer is {}",
                self.inner.family()
            ))),
        }
    }

    /// As [`Self::bpe`], and additionally refuses when the source declared a
    /// `decoder` pipeline.
    ///
    /// The raw byte surfaces read token bytes straight out of the backend,
    /// which skips that pipeline — a Mistral-style `Replace ▁→" "` → `ByteFallback`
    /// → `Fuse` → `Strip` chain would silently render `▁hello▁world` instead of
    /// `hello world`. Bundled vocabularies declare no pipeline, so this is a
    /// guard on json-loaded handles, not a restriction on `from_pretrained`.
    fn bpe_raw(&self) -> PyResult<&Tokenizer> {
        if self.inner.declares_decoder() {
            return Err(PyValueError::new_err(
                "this tokenizer declares a `decoder` pipeline, which raw byte-level \
                 decoding would bypass; use `decode` / `decode_batch` instead",
            ));
        }
        self.bpe()
    }
}

/// Wrap a loaded [`AnyTokenizer`] for Python, carrying the handle whole.
fn any_tokenizer_to_py(py: Python<'_>, any: AnyTokenizer) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, PyAnyTokenizer { inner: any })?.into_any())
}

/// Load any HuggingFace `tokenizer.json` from a file path.
///
/// Args:
///     path: Path to a `tokenizer.json` file
///
/// Returns:
///     An `AnyTokenizer` — the universal loaded-tokenizer handle. It keeps the
///     file's special-token policy AND its declared `decoder` pipeline, so
///     `decode` reproduces HuggingFace's output for files (Mistral, Llama,
///     Gemma) whose decoding is defined by that pipeline. Query `.family` for
///     the backend it dispatched to ("BPE", "Unigram", "WordPiece", "Spm").
#[pyfunction]
pub fn from_json(py: Python<'_>, path: &str) -> PyResult<Py<PyAny>> {
    let any = core_from_json_path(path).map_err(|e| PyValueError::new_err(e.to_string()))?;
    any_tokenizer_to_py(py, any)
}

/// Load any HuggingFace `tokenizer.json` from raw bytes. See [`from_json`].
#[pyfunction]
pub fn from_json_bytes(py: Python<'_>, data: &[u8]) -> PyResult<Py<PyAny>> {
    let any = core_from_json_bytes(data).map_err(|e| PyValueError::new_err(e.to_string()))?;
    any_tokenizer_to_py(py, any)
}

/// Base vocabulary size for a bundled pretrained tokenizer — the size the
/// upstream reference implementation reports, *without* splintr's 54 added
/// agent tokens.
///
/// Use this to size a model's embedding or logit layer, or to identify which
/// vocabulary a checkpoint uses from the shape of its token-embedding
/// tensor — both must match the checkpoint's own vocabulary, not splintr's
/// extended one. `Tokenizer.vocab_size` / `AnyTokenizer.vocab_size` report
/// the *extended* size (base + agent tokens); this reports the base alone.
/// Agent tokens are always appended above every id the base vocabulary uses,
/// so this is also exactly the id at which splintr's additions start.
///
/// Args:
///     name: Vocabulary name, same names accepted by `Tokenizer.from_pretrained`
///         (e.g. "cl100k_base", "o200k_base", "llama3", "mistral_v3", "whisper_v3")
///
/// Returns:
///     The base vocabulary size.
///
/// Raises:
///     ValueError: If `name` is not a known pretrained vocabulary.
#[pyfunction]
pub fn base_vocab_size(name: &str) -> PyResult<u32> {
    crate::core::pretrained::base_vocab_size_by_name(name)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Parse special tokens from Python dict to FxHashMap.
fn parse_special_tokens(
    special_tokens: Option<&Bound<'_, PyDict>>,
) -> PyResult<FxHashMap<String, u32>> {
    let mut result = FxHashMap::default();

    if let Some(dict) = special_tokens {
        for (key, value) in dict.iter() {
            let k: String = key.extract()?;
            let v: u32 = value.extract()?;
            result.insert(k, v);
        }
    }

    Ok(result)
}

/// Python wrapper for the Rust [`StreamingDecoder`].
///
/// Handles UTF-8 safe streaming decode for token-by-token LLM output: it
/// buffers incomplete UTF-8 sequences and only emits complete characters, and
/// it renders each id through exactly the code whole-sequence `decode` renders
/// it through — ByteLevel alphabet unmapped, `<0xNN>` byte fallback resolved,
/// metaspace substituted, `special=true` ids dropped.
///
/// There is one such class, and no constructor: a decoder comes only from a
/// tokenizer's own `streaming_decoder()`, which takes every one of those rules
/// from that tokenizer's configuration. Pairing a decoder with the wrong kind
/// of vocabulary — the mistake that used to turn a byte-level stream into
/// mojibake — is therefore not expressible from Python either.
#[pyclass(name = "StreamingDecoder")]
pub struct PyStreamingDecoder {
    inner: StreamingDecoder,
}

#[pymethods]
impl PyStreamingDecoder {
    /// Add a token and return any complete UTF-8 characters.
    ///
    /// An id in no table at all is skipped rather than raised, matching
    /// `decode_lossy` — a stream is fed by a running model and must survive one
    /// stray id rather than abort the generation.
    ///
    /// Args:
    ///     token_id: The token ID to decode
    ///
    /// Returns:
    ///     String of complete characters, or None if still buffering
    fn add_token(&mut self, token_id: u32) -> Option<String> {
        self.inner.add_token_lossy(token_id)
    }

    /// Add multiple tokens at once and return complete UTF-8 characters.
    ///
    /// Feeding ids in groups is indistinguishable from feeding them one by one:
    /// only the emission points differ, never the concatenated text.
    ///
    /// Args:
    ///     token_ids: List of token IDs to decode
    ///
    /// Returns:
    ///     String of complete characters, or None if still buffering
    fn add_tokens(&mut self, token_ids: Vec<u32>) -> Option<String> {
        self.inner.add_tokens_lossy(&token_ids)
    }

    /// Flush any remaining buffered bytes.
    ///
    /// If there are incomplete UTF-8 sequences in the buffer, they will be
    /// replaced with the Unicode replacement character (U+FFFD).
    ///
    /// Returns:
    ///     Any remaining buffered content
    fn flush(&mut self) -> String {
        self.inner.flush()
    }

    /// Reset the decoder state, discarding any buffered bytes.
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Check if there are buffered bytes waiting for completion.
    #[getter]
    fn has_pending(&self) -> bool {
        self.inner.has_pending()
    }

    /// Get the number of pending bytes in the buffer.
    #[getter]
    fn pending_bytes(&self) -> usize {
        self.inner.pending_bytes()
    }

    fn __repr__(&self) -> String {
        format!(
            "StreamingDecoder(pending_bytes={})",
            self.inner.pending_bytes()
        )
    }
}

impl PyStreamingDecoder {
    /// Wrap a decoder a tokenizer's own factory built. Not exposed to Python:
    /// the pyclass has no `__init__`, for the reason the type-level docs give.
    fn new(inner: StreamingDecoder) -> Self {
        Self { inner }
    }
}

// =============================================================================
// Agent Token Constants for Python
// =============================================================================
// Auto-generated from scripts/generate_agent_tokens.py
// To regenerate: python scripts/generate_agent_tokens.py > src/python/agent_tokens_generated.rs

include!("agent_tokens_generated.rs");
