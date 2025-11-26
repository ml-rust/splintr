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

use crate::core::{Tokenizer, CL100K_BASE_PATTERN, O200K_BASE_PATTERN};

/// Bundled cl100k_base vocabulary (GPT-4, GPT-3.5-turbo)
const CL100K_BASE_VOCAB: &[u8] = include_bytes!("../../python/splintr/vocabs/cl100k_base.tiktoken");

/// Bundled o200k_base vocabulary (GPT-4o)
const O200K_BASE_VOCAB: &[u8] = include_bytes!("../../python/splintr/vocabs/o200k_base.tiktoken");

/// Get the standard special tokens for cl100k_base encoding.
///
/// Returns a map of special token strings to their token IDs for GPT-4
/// and GPT-3.5-turbo models. Includes:
/// - `<|endoftext|>`: End of text marker
/// - `<|fim_prefix|>`: Fill-in-the-middle prefix
/// - `<|fim_middle|>`: Fill-in-the-middle middle section
/// - `<|fim_suffix|>`: Fill-in-the-middle suffix
/// - `<|endofprompt|>`: End of prompt marker
fn cl100k_base_special_tokens() -> FxHashMap<String, u32> {
    let mut special = FxHashMap::default();
    special.insert("<|endoftext|>".to_string(), 100257);
    special.insert("<|fim_prefix|>".to_string(), 100258);
    special.insert("<|fim_middle|>".to_string(), 100259);
    special.insert("<|fim_suffix|>".to_string(), 100260);
    special.insert("<|endofprompt|>".to_string(), 100276);
    special
}

/// Get the standard special tokens for o200k_base encoding (GPT-4o).
///
/// Returns a simplified set of special tokens for GPT-4o models:
/// - `<|endoftext|>`: End of text marker
/// - `<|endofprompt|>`: End of prompt marker
fn o200k_base_special_tokens() -> FxHashMap<String, u32> {
    let mut special = FxHashMap::default();
    special.insert("<|endoftext|>".to_string(), 199999);
    special.insert("<|endofprompt|>".to_string(), 200018);
    special
}

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
    ///
    /// Args:
    ///     name: Model name (e.g., "cl100k_base", "o200k_base")
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
            _ => Err(PyValueError::new_err(format!(
                "Unknown pretrained model: {}. Supported: cl100k_base, o200k_base",
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

    /// Encode text to token IDs.
    ///
    /// Special tokens in the input are treated as regular text.
    ///
    /// Args:
    ///     text: Input text to encode
    ///
    /// Returns:
    ///     List of token IDs
    fn encode(&self, text: &str) -> Vec<u32> {
        self.inner.encode(text)
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
