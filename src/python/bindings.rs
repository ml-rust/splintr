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
use crate::core::pretrained::{
    cl100k_base_special_tokens, deepseek_v3_special_tokens, llama3_special_tokens,
    mistral_v3_special_tokens, o200k_base_special_tokens, patterns as pretrained_patterns,
    CL100K_BASE_VOCAB, DEEPSEEK_V3_VOCAB, LLAMA3_VOCAB, MISTRAL_V3_VOCAB, O200K_BASE_VOCAB,
};
use crate::core::spm::SpmTokenizer;
use crate::core::wordpiece::WordPieceTokenizer;
use crate::core::{byte_level_decode_bytes, Tokenize, Tokenizer};
use crate::core::{AnyTokenizer, Backend, PretrainedVocab, SpecialMode, SpecialPolicy};

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
    ///     A `Tokenizer` for byte-level BPE vocabularies, or an `SpmTokenizer`
    ///     for the SentencePiece ones ("mistral" / "mistral_v1" / "mistral_v2"),
    ///     which are merged as pieces rather than as bytes.
    #[staticmethod]
    fn from_pretrained(py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        let bpe = |inner: Tokenizer| -> PyResult<Py<PyAny>> {
            Ok(Py::new(
                py,
                Self {
                    inner,
                    policy: SpecialPolicy::default(),
                },
            )?
            .into_any())
        };
        match name {
            "cl100k_base" => {
                let special = cl100k_base_special_tokens();
                bpe(Tokenizer::from_bytes_chain(
                    CL100K_BASE_VOCAB,
                    pretrained_patterns(PretrainedVocab::Cl100kBase),
                    special,
                )
                .map_err(|e| PyValueError::new_err(e.to_string()))?)
            }
            "o200k_base" => {
                let special = o200k_base_special_tokens();
                bpe(Tokenizer::from_bytes_chain(
                    O200K_BASE_VOCAB,
                    pretrained_patterns(PretrainedVocab::O200kBase),
                    special,
                )
                .map_err(|e| PyValueError::new_err(e.to_string()))?)
            }
            "llama3" | "llama3.1" | "llama3.2" | "llama3.3" => {
                let special = llama3_special_tokens();
                bpe(Tokenizer::from_bytes_chain(
                    LLAMA3_VOCAB,
                    pretrained_patterns(PretrainedVocab::Llama3),
                    special,
                )
                .map_err(|e| PyValueError::new_err(e.to_string()))?)
            }
            "deepseek_v3" | "deepseek-v3" => {
                let special = deepseek_v3_special_tokens();
                // DeepSeek uses ByteLevel BPE encoding. The pre-tokenizer comes
                // from `pretrained::patterns`, which returns DeepSeek's own
                // three-pass expression list — not the o200k or Llama 3 split,
                // neither of which produces DeepSeek's ids.
                bpe(Tokenizer::from_bytes_byte_level_chain(
                    DEEPSEEK_V3_VOCAB,
                    pretrained_patterns(PretrainedVocab::DeepseekV3),
                    special,
                )
                .map_err(|e| PyValueError::new_err(e.to_string()))?)
            }
            // Mistral V1/V2 are SentencePiece: their pieces are merged by the
            // SPM-BPE backend, never by byte-level BPE, which cannot build the
            // `▁` word-boundary marker (U+2581 = E2 96 81) because `E2 96` is
            // not a piece any SentencePiece vocabulary was trained on. Built
            // through the core loader so Rust and Python get the same ids.
            "mistral" | "mistral_v1" | "mistral_v2" => {
                let loaded = crate::core::pretrained::from_pretrained(name)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                any_tokenizer_to_py(py, loaded)
            }
            // Mistral V3: ByteLevel BPE (like DeepSeek/GPT-2) - Ġ represents space
            // Uses its own pattern (no contractions, single-digit numbers)
            "mistral_v3" => {
                let special = mistral_v3_special_tokens();
                bpe(Tokenizer::from_bytes_byte_level_chain(
                    MISTRAL_V3_VOCAB,
                    pretrained_patterns(PretrainedVocab::MistralV3),
                    special,
                )
                .map_err(|e| PyValueError::new_err(e.to_string()))?)
            }
            // Whisper multilingual (v1/v2/v3). Base BPE is bundled; specials are
            // generated per variant. Delegates to the core name→variant mapping.
            name if name.starts_with("whisper") => {
                let loaded = crate::core::pretrained::from_pretrained(name)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let Backend::Bpe(inner) = loaded.into_backend() else {
                    return Err(PyValueError::new_err(
                        "whisper vocabularies must load as a BPE tokenizer",
                    ));
                };
                // Every other BPE arm here builds its tokenizer with added-token
                // matching OFF: on the Python surface `encode` treats specials
                // as ordinary text and `encode_with_special` opts in. Leaving it
                // on for whisper alone would make Python inconsistent with
                // itself, so it is turned back off; the Python surface is
                // aligned with the Rust `encode` semantics in a later change.
                bpe(inner.with_added_token_matching(false))
            }
            _ => Err(PyValueError::new_err(format!(
                "Unknown pretrained model: {}. See from_pretrained docstring for supported models.",
                name
            ))),
        }
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
    ///     tokenizer = Tokenizer.from_pretrained("cl100k_base").pcre2(True)
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
    ///     tokenizer = Tokenizer.from_pretrained("cl100k_base").jit(False)
    ///     tokenizer = Tokenizer.from_pretrained("cl100k_base").pcre2(True).jit(True)
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

    /// Encode text to token IDs.
    ///
    /// Special tokens in the input are treated as regular text.
    /// This method uses sequential encoding which is optimal for most use cases.
    ///
    /// Args:
    ///     text: Input text to encode
    ///
    /// Returns:
    ///     List of token IDs
    fn encode(&self, text: &str) -> Vec<u32> {
        self.inner.encode(text)
    }

    /// Encode and apply the model's `post_processor` template (e.g. `[CLS]…[SEP]`,
    /// `<s>…</s>`), matching HuggingFace's default `encode` (add_special_tokens=True).
    /// Equals `encode` for models without a post-processor (e.g. `from_pretrained`).
    /// Distinct from `encode_with_special` (which recognizes special-token *strings*
    /// embedded in the input text).
    fn encode_with_special_tokens(&self, text: &str) -> Vec<u32> {
        self.policy.apply_single(self.inner.encode(text))
    }

    /// Encode text to token IDs using Rayon parallel processing.
    ///
    /// This method parallelizes the BPE encoding of individual chunks using Rayon.
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
        self.inner.encode_rayon(text)
    }

    /// Encode text with special token handling.
    ///
    /// Special tokens in the input are encoded directly without BPE.
    ///
    /// Args:
    ///     text: Input text to encode
    ///
    /// Returns:
    ///     List of token IDs
    fn encode_with_special(&self, text: &str) -> Vec<u32> {
        self.inner.encode_with_special(text)
    }

    /// Encode text to token IDs, never matching special tokens.
    ///
    /// A special token spelled out literally in the input (e.g. `<|endoftext|>`)
    /// is encoded as ordinary text instead of being promoted to its control-token
    /// id. Use this when tokenizing untrusted text where the caller must not be
    /// able to forge control tokens.
    ///
    /// Args:
    ///     text: Input text to encode
    ///
    /// Returns:
    ///     List of token IDs
    fn encode_ordinary(&self, text: &str) -> Vec<u32> {
        self.inner.encode_ordinary(text)
    }

    /// Encode text to token IDs, matching only the named special tokens.
    ///
    /// Any other configured special token spelled out literally in the text
    /// raises `ValueError` instead of being silently promoted to its
    /// control-token id — use this to accept a known, bounded set of special
    /// tokens (e.g. a chat template's own markers) from otherwise untrusted text.
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

    /// Decode token IDs to bytes.
    ///
    /// Args:
    ///     tokens: List of token IDs
    ///
    /// Returns:
    ///     Decoded bytes
    fn decode_bytes(&self, tokens: Vec<u32>) -> Vec<u8> {
        self.inner.decode_bytes(&tokens)
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

    /// Batch encode multiple texts in parallel.
    ///
    /// Uses Rayon to parallelize encoding across texts.
    ///
    /// Args:
    ///     texts: List of texts to encode
    ///
    /// Returns:
    ///     List of token ID lists
    fn encode_batch(&self, texts: Vec<String>) -> Vec<Vec<u32>> {
        self.inner.encode_batch(&texts)
    }

    /// Batch encode multiple texts with special token handling.
    ///
    /// Args:
    ///     texts: List of texts to encode
    ///
    /// Returns:
    ///     List of token ID lists
    fn encode_batch_with_special(&self, texts: Vec<String>) -> Vec<Vec<u32>> {
        self.inner.encode_batch_with_special(&texts)
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
        PyStreamingDecoder::new(
            self.inner.decoder().clone(),
            self.inner.special_tokens_decoder().clone(),
        )
    }

    /// Create a ByteLevel streaming decoder for UTF-8 safe token-by-token decoding.
    ///
    /// This decoder is designed for tokenizers using ByteLevel BPE encoding
    /// (GPT-2, Llama, DeepSeek V3) where tokens represent ByteLevel-encoded
    /// characters that need to be decoded back to raw bytes before UTF-8 assembly.
    ///
    /// Returns:
    ///     ByteLevelStreamingDecoder instance
    ///
    /// Example:
    ///     tokenizer = Tokenizer.from_pretrained("deepseek_v3")
    ///     decoder = tokenizer.byte_level_streaming_decoder()
    ///     for token_id in token_stream:
    ///         if text := decoder.add_token(token_id):
    ///             print(text, end="", flush=True)
    ///     print(decoder.flush())
    fn byte_level_streaming_decoder(&self) -> PyByteLevelStreamingDecoder {
        PyByteLevelStreamingDecoder::new(
            self.inner.decoder().clone(),
            self.inner.special_tokens_decoder().clone(),
        )
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
        scores: Vec<f32>,
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

    /// Encode text to token IDs using Viterbi maximum-score Unigram segmentation.
    ///
    /// Prepends BOS token if configured. Replaces spaces with ▁ (U+2581)
    /// following the SentencePiece convention.
    ///
    /// Args:
    ///     text: Input text to encode
    ///
    /// Returns:
    ///     List of token IDs
    fn encode(&self, text: &str) -> Vec<u32> {
        self.with_bos(self.inner.encode(text))
    }

    /// Encode and apply the model's `post_processor` template, matching
    /// HuggingFace's default `encode`. Equals `encode` when there is none.
    fn encode_with_special_tokens(&self, text: &str) -> Vec<u32> {
        self.policy
            .apply_single(self.with_bos(self.inner.encode(text)))
    }

    /// Encode text to token IDs, never matching special tokens.
    ///
    /// A special token spelled out literally in the input is encoded as
    /// ordinary text instead of being promoted to its control-token id. BOS is
    /// still prepended if configured — that is a boundary token this tokenizer
    /// always adds, independent of whether special-token matching in the
    /// content is on. Use this when tokenizing untrusted text where the caller
    /// must not be able to forge control tokens.
    ///
    /// Args:
    ///     text: Input text to encode
    ///
    /// Returns:
    ///     List of token IDs
    fn encode_ordinary(&self, text: &str) -> Vec<u32> {
        self.with_bos(self.inner.encode_ordinary(text))
    }

    /// Encode text to token IDs, matching only the named special tokens.
    ///
    /// Any other configured special token spelled out literally in the text
    /// raises `ValueError` instead of being silently promoted to its
    /// control-token id — use this to accept a known, bounded set of special
    /// tokens from otherwise untrusted text. BOS is still prepended if configured.
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
        Ok(self.with_bos(ids))
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
/// Returned by `Tokenizer.from_pretrained("mistral" | "mistral_v1" | "mistral_v2")`.
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

    /// Encode text to token IDs.
    ///
    /// Control tokens present in the text (`[INST]`, `<s>`, chat markers) are
    /// recognized as single IDs — SentencePiece merging would otherwise shred
    /// them into ordinary pieces. Boundary tokens are not added; use
    /// `encode_with_special_tokens` for the model's template.
    fn encode(&self, text: &str) -> Vec<u32> {
        Tokenize::encode(&self.inner, text)
    }

    /// Encode and apply the model's boundary template, matching HuggingFace's
    /// default `encode`. Equals `encode` when there is none.
    fn encode_with_special_tokens(&self, text: &str) -> Vec<u32> {
        self.policy
            .apply_single(Tokenize::encode(&self.inner, text))
    }

    /// Encode text with control-token handling.
    ///
    /// Unlike the byte-level BPE `Tokenizer`, whose `encode` treats special
    /// tokens as ordinary text unless the vocabulary was built with
    /// added-token matching on, this SPM-BPE backend's `encode` already
    /// recognizes control tokens (`[INST]`, `<s>`, chat markers) — see
    /// `encode`'s docs. This method is that same behavior under the name the
    /// BPE surface uses, so callers migrating between the two wrapper types
    /// do not need to special-case SPM.
    ///
    /// Args:
    ///     text: Input text to encode
    ///
    /// Returns:
    ///     List of token IDs
    fn encode_with_special(&self, text: &str) -> Vec<u32> {
        self.encode(text)
    }

    /// Batch encode multiple texts in parallel.
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

    /// Encode text to token IDs, never matching control tokens.
    ///
    /// A control token spelled out literally in the input (`[INST]`, `<s>`,
    /// chat markers) is encoded as ordinary text instead of being recognized as
    /// a single id. Use this when tokenizing untrusted text where the caller
    /// must not be able to forge control tokens.
    ///
    /// Args:
    ///     text: Input text to encode
    ///
    /// Returns:
    ///     List of token IDs
    fn encode_ordinary(&self, text: &str) -> Vec<u32> {
        self.inner.encode_ordinary(text)
    }

    /// Encode text to token IDs, matching only the named control tokens.
    ///
    /// Any other configured control token spelled out literally in the text
    /// raises `ValueError` instead of being silently recognized as a single
    /// id — use this to accept a known, bounded set of control tokens from
    /// otherwise untrusted text.
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
        self.inner
            .encode_with(text, &SpecialMode::Allow(&allowed))
            .map_err(|e| PyValueError::new_err(e.to_string()))
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
    #[new]
    #[pyo3(signature = (vocab, unk_token_id, max_word_len=100, do_lower_case=false))]
    fn new(
        vocab: Vec<String>,
        unk_token_id: u32,
        max_word_len: usize,
        do_lower_case: bool,
    ) -> Self {
        Self {
            inner: WordPieceTokenizer::new(vocab, unk_token_id, max_word_len, do_lower_case),
            policy: SpecialPolicy::default(),
        }
    }

    /// Encode text to token IDs.
    fn encode(&self, text: &str) -> Vec<u32> {
        Tokenize::encode(&self.inner, text)
    }

    /// Encode and apply the model's `post_processor` template (e.g. `[CLS]…[SEP]`),
    /// matching HuggingFace's default `encode`. Equals `encode` when there is no
    /// post-processor.
    fn encode_with_special_tokens(&self, text: &str) -> Vec<u32> {
        self.policy
            .apply_single(Tokenize::encode(&self.inner, text))
    }

    /// Encode text to token IDs, never matching special tokens.
    ///
    /// A special token spelled out literally in the input (e.g. `[CLS]`,
    /// `[SEP]`) is encoded as ordinary text instead of being recognized as a
    /// single id. Use this when tokenizing untrusted text where the caller
    /// must not be able to forge control tokens.
    ///
    /// Args:
    ///     text: Input text to encode
    ///
    /// Returns:
    ///     List of token IDs
    fn encode_ordinary(&self, text: &str) -> Vec<u32> {
        self.inner.encode_ordinary(text)
    }

    /// Encode text to token IDs, matching only the named special tokens.
    ///
    /// Any other configured special token spelled out literally in the text
    /// raises `ValueError` instead of being silently recognized as a single
    /// id — use this to accept a known, bounded set of special tokens from
    /// otherwise untrusted text.
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

    /// Decode token IDs to text.
    fn decode(&self, ids: Vec<u32>) -> PyResult<String> {
        Tokenize::decode(&self.inner, &ids).map_err(|e| PyValueError::new_err(e.to_string()))
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
}

/// Wrap a loaded [`AnyTokenizer`] in the matching Python class, carrying its
/// special-token policy.
fn any_tokenizer_to_py(py: Python<'_>, any: AnyTokenizer) -> PyResult<Py<PyAny>> {
    let policy = any.policy().clone();
    Ok(match any.into_backend() {
        Backend::Bpe(t) => Py::new(py, PyTokenizer { inner: t, policy })?.into_any(),
        Backend::Unigram(t) => Py::new(
            py,
            PySentencePieceTokenizer {
                inner: t,
                policy,
                bos_token_id: None,
            },
        )?
        .into_any(),
        Backend::WordPiece(t) => Py::new(py, PyWordPieceTokenizer { inner: t, policy })?.into_any(),
        Backend::Spm(t) => Py::new(py, PySpmTokenizer { inner: t, policy })?.into_any(),
    })
}

/// Load any HuggingFace `tokenizer.json` from a file path.
///
/// Dispatches on the model family and returns the matching tokenizer object:
/// `Tokenizer` (BPE), `SentencePieceTokenizer` (Unigram), or `WordPieceTokenizer`.
///
/// Args:
///     path: Path to a `tokenizer.json` file
///
/// Returns:
///     A Tokenizer / SentencePieceTokenizer / WordPieceTokenizer instance
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

/// Python wrapper for streaming decoder.
///
/// Handles UTF-8 safe streaming decode for token-by-token LLM output.
/// Buffers incomplete UTF-8 sequences and only emits complete characters.
#[pyclass(name = "StreamingDecoder")]
pub struct PyStreamingDecoder {
    decoder: FxHashMap<u32, Vec<u8>>,
    special_decoder: FxHashMap<u32, String>,
    buffer: Vec<u8>,
}

#[pymethods]
impl PyStreamingDecoder {
    /// Add a token and return any complete UTF-8 characters.
    ///
    /// Args:
    ///     token_id: The token ID to decode
    ///
    /// Returns:
    ///     String of complete characters, or None if still buffering
    fn add_token(&mut self, token_id: u32) -> Option<String> {
        // Get bytes for this token
        let bytes = match self.decoder.get(&token_id) {
            Some(b) => b.as_slice(),
            // An id in neither table is unknown; buffer nothing and yield nothing.
            None => self.special_decoder.get(&token_id)?.as_bytes(),
        };

        // Add to buffer
        self.buffer.extend_from_slice(bytes);

        // Try to extract complete UTF-8 characters
        self.extract_complete_utf8()
    }

    /// Add multiple tokens at once and return complete UTF-8 characters.
    ///
    /// Args:
    ///     token_ids: List of token IDs to decode
    ///
    /// Returns:
    ///     String of complete characters, or None if still buffering
    fn add_tokens(&mut self, token_ids: Vec<u32>) -> Option<String> {
        for token_id in token_ids {
            let bytes = if let Some(b) = self.decoder.get(&token_id) {
                b.as_slice()
            } else if let Some(s) = self.special_decoder.get(&token_id) {
                s.as_bytes()
            } else {
                continue;
            };

            self.buffer.extend_from_slice(bytes);
        }

        self.extract_complete_utf8()
    }

    /// Flush any remaining buffered bytes.
    ///
    /// If there are incomplete UTF-8 sequences in the buffer, they will be
    /// replaced with the Unicode replacement character (U+FFFD).
    ///
    /// Returns:
    ///     Any remaining buffered content
    fn flush(&mut self) -> String {
        if self.buffer.is_empty() {
            return String::new();
        }

        let result = String::from_utf8_lossy(&self.buffer).into_owned();
        self.buffer.clear();
        result
    }

    /// Reset the decoder state, discarding any buffered bytes.
    fn reset(&mut self) {
        self.buffer.clear();
    }

    /// Check if there are buffered bytes waiting for completion.
    #[getter]
    fn has_pending(&self) -> bool {
        !self.buffer.is_empty()
    }

    /// Get the number of pending bytes in the buffer.
    #[getter]
    fn pending_bytes(&self) -> usize {
        self.buffer.len()
    }

    fn __repr__(&self) -> String {
        format!("StreamingDecoder(pending_bytes={})", self.buffer.len())
    }
}

impl PyStreamingDecoder {
    fn new(decoder: FxHashMap<u32, Vec<u8>>, special_decoder: FxHashMap<u32, String>) -> Self {
        Self {
            decoder,
            special_decoder,
            buffer: Vec::with_capacity(16),
        }
    }

    fn extract_complete_utf8(&mut self) -> Option<String> {
        if self.buffer.is_empty() {
            return None;
        }

        let valid_len = self.find_valid_utf8_len();

        if valid_len == 0 {
            return None;
        }

        let valid_bytes: Vec<u8> = self.buffer.drain(..valid_len).collect();
        // SAFETY: We've verified this is valid UTF-8
        let result = unsafe { String::from_utf8_unchecked(valid_bytes) };

        Some(result)
    }

    fn find_valid_utf8_len(&self) -> usize {
        let bytes = &self.buffer;
        let len = bytes.len();

        if len == 0 {
            return 0;
        }

        // First, try to validate the entire buffer
        if std::str::from_utf8(bytes).is_ok() {
            return len;
        }

        // Find how many bytes at the end might be an incomplete sequence
        for incomplete_len in 1..=3.min(len) {
            let check_len = len - incomplete_len;
            if check_len == 0 {
                continue;
            }

            if std::str::from_utf8(&bytes[..check_len]).is_ok()
                && Self::could_be_incomplete_sequence(&bytes[check_len..])
            {
                return check_len;
            }
        }

        // If nothing works, find the last position that's valid
        for i in (0..len).rev() {
            if std::str::from_utf8(&bytes[..=i]).is_ok() {
                return i + 1;
            }
        }

        0
    }

    fn could_be_incomplete_sequence(bytes: &[u8]) -> bool {
        if bytes.is_empty() {
            return false;
        }

        let first = bytes[0];

        match first {
            // 2-byte sequence: 110xxxxx
            0xC0..=0xDF => bytes.len() < 2,
            // 3-byte sequence: 1110xxxx
            0xE0..=0xEF => bytes.len() < 3,
            // 4-byte sequence: 11110xxx
            0xF0..=0xF7 => bytes.len() < 4,
            // Continuation byte or invalid
            _ => false,
        }
    }
}

/// Python wrapper for ByteLevel streaming decoder.
///
/// Handles UTF-8 safe streaming decode for token-by-token LLM output from
/// ByteLevel-encoded tokenizers (GPT-2, Llama, DeepSeek V3). First decodes
/// ByteLevel encoding to raw bytes, then assembles into valid UTF-8 strings.
#[pyclass(name = "ByteLevelStreamingDecoder")]
pub struct PyByteLevelStreamingDecoder {
    decoder: FxHashMap<u32, Vec<u8>>,
    special_decoder: FxHashMap<u32, String>,
    buffer: Vec<u8>,
}

#[pymethods]
impl PyByteLevelStreamingDecoder {
    /// Add a token and return any complete UTF-8 characters.
    ///
    /// The token's ByteLevel-encoded bytes are first decoded to raw bytes,
    /// then assembled into valid UTF-8 strings.
    ///
    /// Args:
    ///     token_id: The token ID to decode
    ///
    /// Returns:
    ///     String of complete characters, or None if still buffering
    fn add_token(&mut self, token_id: u32) -> Option<String> {
        match self.decoder.get(&token_id) {
            Some(encoded_bytes) => {
                // Decode ByteLevel encoding to raw bytes
                if let Some(raw_bytes) = byte_level_decode_bytes(encoded_bytes) {
                    self.buffer.extend_from_slice(&raw_bytes);
                } else {
                    // Fallback: treat as raw bytes if ByteLevel decode fails
                    self.buffer.extend_from_slice(encoded_bytes);
                }
            }
            // Special tokens are NOT ByteLevel-encoded, add directly. An id in
            // neither table is unknown; buffer nothing and yield nothing.
            None => self
                .buffer
                .extend_from_slice(self.special_decoder.get(&token_id)?.as_bytes()),
        }

        self.extract_complete_utf8()
    }

    /// Add multiple tokens at once and return complete UTF-8 characters.
    ///
    /// Args:
    ///     token_ids: List of token IDs to decode
    ///
    /// Returns:
    ///     String of complete characters, or None if still buffering
    fn add_tokens(&mut self, token_ids: Vec<u32>) -> Option<String> {
        for token_id in token_ids {
            if let Some(encoded_bytes) = self.decoder.get(&token_id) {
                if let Some(raw_bytes) = byte_level_decode_bytes(encoded_bytes) {
                    self.buffer.extend_from_slice(&raw_bytes);
                } else {
                    self.buffer.extend_from_slice(encoded_bytes);
                }
            } else if let Some(special) = self.special_decoder.get(&token_id) {
                self.buffer.extend_from_slice(special.as_bytes());
            }
        }

        self.extract_complete_utf8()
    }

    /// Flush any remaining buffered bytes.
    ///
    /// If there are incomplete UTF-8 sequences in the buffer, they will be
    /// replaced with the Unicode replacement character (U+FFFD).
    ///
    /// Returns:
    ///     Any remaining buffered content
    fn flush(&mut self) -> String {
        if self.buffer.is_empty() {
            return String::new();
        }

        let result = String::from_utf8_lossy(&self.buffer).into_owned();
        self.buffer.clear();
        result
    }

    /// Reset the decoder state, discarding any buffered bytes.
    fn reset(&mut self) {
        self.buffer.clear();
    }

    /// Check if there are buffered bytes waiting for completion.
    #[getter]
    fn has_pending(&self) -> bool {
        !self.buffer.is_empty()
    }

    /// Get the number of pending bytes in the buffer.
    #[getter]
    fn pending_bytes(&self) -> usize {
        self.buffer.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "ByteLevelStreamingDecoder(pending_bytes={})",
            self.buffer.len()
        )
    }
}

impl PyByteLevelStreamingDecoder {
    fn new(decoder: FxHashMap<u32, Vec<u8>>, special_decoder: FxHashMap<u32, String>) -> Self {
        Self {
            decoder,
            special_decoder,
            buffer: Vec::with_capacity(16),
        }
    }

    fn extract_complete_utf8(&mut self) -> Option<String> {
        if self.buffer.is_empty() {
            return None;
        }

        let valid_len = self.find_valid_utf8_len();

        if valid_len == 0 {
            return None;
        }

        let valid_bytes: Vec<u8> = self.buffer.drain(..valid_len).collect();
        // SAFETY: We've verified this is valid UTF-8
        let result = unsafe { String::from_utf8_unchecked(valid_bytes) };

        Some(result)
    }

    fn find_valid_utf8_len(&self) -> usize {
        let bytes = &self.buffer;
        let len = bytes.len();

        if len == 0 {
            return 0;
        }

        // First, try to validate the entire buffer
        if std::str::from_utf8(bytes).is_ok() {
            return len;
        }

        // Find how many bytes at the end might be an incomplete sequence
        for incomplete_len in 1..=3.min(len) {
            let check_len = len - incomplete_len;
            if check_len == 0 {
                continue;
            }

            if std::str::from_utf8(&bytes[..check_len]).is_ok()
                && Self::could_be_incomplete_sequence(&bytes[check_len..])
            {
                return check_len;
            }
        }

        // If nothing works, find the last position that's valid
        for i in (0..len).rev() {
            if std::str::from_utf8(&bytes[..=i]).is_ok() {
                return i + 1;
            }
        }

        0
    }

    fn could_be_incomplete_sequence(bytes: &[u8]) -> bool {
        if bytes.is_empty() {
            return false;
        }

        let first = bytes[0];

        match first {
            0xC0..=0xDF => bytes.len() < 2,
            0xE0..=0xEF => bytes.len() < 3,
            0xF0..=0xF7 => bytes.len() < 4,
            _ => false,
        }
    }
}

// =============================================================================
// Agent Token Constants for Python
// =============================================================================
// Auto-generated from scripts/generate_agent_tokens.py
// To regenerate: python scripts/generate_agent_tokens.py > src/python/agent_tokens_generated.rs

include!("agent_tokens_generated.rs");
