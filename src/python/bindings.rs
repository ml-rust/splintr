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
use rustc_hash::FxHashMap;

use crate::core::pretrained::{
    cl100k_base_special_tokens, deepseek_v3_special_tokens, llama3_special_tokens,
    mistral_v1_special_tokens, mistral_v2_special_tokens, o200k_base_special_tokens,
    CL100K_BASE_VOCAB, DEEPSEEK_V3_VOCAB, LLAMA3_VOCAB, MISTRAL_V2_VOCAB, MISTRAL_VOCAB,
    O200K_BASE_VOCAB,
};
use crate::core::{
    byte_level_decode_bytes, Tokenizer, CL100K_BASE_PATTERN, LLAMA3_PATTERN, O200K_BASE_PATTERN,
    SENTENCEPIECE_PATTERN,
};

// Special tokens are defined in crate::core::pretrained module.
// See that module for the full token documentation and implementations.

/// Python wrapper for the Rust Tokenizer.
#[pyclass(name = "Tokenizer")]
pub struct PyTokenizer {
    inner: Tokenizer,
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

        Ok(Self { inner })
    }

    /// Create a tokenizer from a pretrained model name.
    ///
    /// Currently supported:
    /// - "cl100k_base" (GPT-4, GPT-3.5-turbo)
    /// - "o200k_base" (GPT-4o)
    /// - "llama3" / "llama3.1" / "llama3.2" / "llama3.3" (Meta Llama 3 family)
    /// - "deepseek_v3" / "deepseek-v3" (DeepSeek V3)
    /// - "mistral_v1" / "mistral" / "mistral-7b" (Mistral V1: 32k SentencePiece)
    /// - "mistral_v2" / "mistral-7b-v0.3" / "codestral" (Mistral V2: 32k + control tokens)
    /// - "mistral_v3" / "mistral-nemo" / "mistral-large" (Mistral V3: Tekken 131k)
    ///
    /// Args:
    ///     name: Model name (e.g., "cl100k_base", "o200k_base", "llama3", "mistral_v2")
    ///
    /// Returns:
    ///     Tokenizer instance
    #[staticmethod]
    fn from_pretrained(name: &str) -> PyResult<Self> {
        match name {
            "cl100k_base" => {
                let special = cl100k_base_special_tokens();
                let inner = Tokenizer::from_bytes(CL100K_BASE_VOCAB, CL100K_BASE_PATTERN, special)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(Self { inner })
            }
            "o200k_base" => {
                let special = o200k_base_special_tokens();
                let inner = Tokenizer::from_bytes(O200K_BASE_VOCAB, O200K_BASE_PATTERN, special)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(Self { inner })
            }
            "llama3" | "llama3.1" | "llama3.2" | "llama3.3" => {
                let special = llama3_special_tokens();
                let inner = Tokenizer::from_bytes(LLAMA3_VOCAB, LLAMA3_PATTERN, special)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(Self { inner })
            }
            "deepseek_v3" | "deepseek-v3" => {
                let special = deepseek_v3_special_tokens();
                // DeepSeek uses ByteLevel BPE encoding
                let inner =
                    Tokenizer::from_bytes_byte_level(DEEPSEEK_V3_VOCAB, LLAMA3_PATTERN, special)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(Self { inner })
            }
            // Mistral V1: Default "mistral" → V1
            "mistral" | "mistral_v1" => {
                let special = mistral_v1_special_tokens();
                let inner = Tokenizer::from_bytes_sentencepiece(
                    MISTRAL_VOCAB,
                    SENTENCEPIECE_PATTERN,
                    special,
                )
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(Self { inner })
            }
            // Mistral V2: All 32,768 tokens in vocab file, only agent tokens are special
            "mistral_v2" => {
                let special = mistral_v2_special_tokens();
                let inner = Tokenizer::from_bytes_sentencepiece(
                    MISTRAL_V2_VOCAB,
                    SENTENCEPIECE_PATTERN,
                    special,
                )
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(Self { inner })
            }
            // Mistral V3: Tekken vocabulary (will be added in Phase 3)
            "mistral_v3" => Err(PyValueError::new_err(format!(
                "Mistral V3/Tekken support (model: {}) is not yet implemented. Coming in Phase 3!",
                name
            ))),
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

        Ok(Self { inner })
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
        Ok(Self { inner: result })
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
        Ok(Self { inner: result })
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
        let bytes = if let Some(b) = self.decoder.get(&token_id) {
            b.as_slice()
        } else if let Some(s) = self.special_decoder.get(&token_id) {
            s.as_bytes()
        } else {
            return None;
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
        if let Some(encoded_bytes) = self.decoder.get(&token_id) {
            // Decode ByteLevel encoding to raw bytes
            if let Some(raw_bytes) = byte_level_decode_bytes(encoded_bytes) {
                self.buffer.extend_from_slice(&raw_bytes);
            } else {
                // Fallback: treat as raw bytes if ByteLevel decode fails
                self.buffer.extend_from_slice(encoded_bytes);
            }
        } else if let Some(special) = self.special_decoder.get(&token_id) {
            // Special tokens are NOT ByteLevel-encoded, add directly
            self.buffer.extend_from_slice(special.as_bytes());
        } else {
            return None;
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
