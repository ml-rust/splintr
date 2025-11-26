use aho_corasick::AhoCorasick;
use lru::LruCache;
use pcre2::bytes::Regex;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use rustc_hash::FxHasher;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::Mutex;
use thiserror::Error;

use super::bpe::byte_pair_encode;
use super::vocab::{build_decoder, load_tiktoken_bpe, load_tiktoken_bpe_file, VocabError};

#[derive(Error, Debug)]
pub enum TokenizerError {
    #[error("Regex compilation error: {0}")]
    RegexError(#[from] pcre2::Error),
    #[error("Vocabulary error: {0}")]
    VocabError(#[from] VocabError),
    #[error("Decoding error: invalid UTF-8")]
    Utf8Error,
    #[error("Aho-Corasick build error: {0}")]
    AhoCorasickError(#[from] aho_corasick::BuildError),
}

/// Default regex pattern for cl100k_base (GPT-4, GPT-3.5-turbo)
pub const CL100K_BASE_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Default regex pattern for o200k_base (GPT-4o)
pub const O200K_BASE_PATTERN: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Default cache size for encoded chunks
const DEFAULT_CACHE_SIZE: usize = 4096;

/// High-performance tokenizer using PCRE2 with JIT and Rayon parallelism.
///
/// Key optimizations:
/// - PCRE2 with JIT compilation (2-4x faster than fancy-regex)
/// - Rayon parallelism for encoding multiple chunks
/// - Linked-list BPE algorithm (avoids O(N²) on pathological inputs)
/// - FxHashMap for fast lookups
/// - Aho-Corasick for fast multi-pattern special token matching
/// - LRU cache for frequently encoded chunks
pub struct Tokenizer {
    encoder: FxHashMap<Vec<u8>, u32>,
    decoder: FxHashMap<u32, Vec<u8>>,
    special_tokens: FxHashMap<String, u32>,
    special_tokens_decoder: FxHashMap<u32, String>,
    special_token_strings: Vec<String>, // Ordered list for Aho-Corasick pattern indices
    regex: Regex,
    special_matcher: Option<AhoCorasick>,
    chunk_cache: Mutex<LruCache<u64, Vec<u32>>>,
}

impl Tokenizer {
    /// Create a new tokenizer from encoder map, special tokens, and regex pattern.
    ///
    /// # Arguments
    /// * `encoder` - Map of byte sequences to token IDs
    /// * `special_tokens` - Map of special token strings to token IDs
    /// * `pattern` - PCRE2 regex pattern for tokenization
    pub fn new(
        encoder: FxHashMap<Vec<u8>, u32>,
        special_tokens: FxHashMap<String, u32>,
        pattern: &str,
    ) -> Result<Self, TokenizerError> {
        Self::with_cache_size(encoder, special_tokens, pattern, DEFAULT_CACHE_SIZE)
    }

    /// Create a new tokenizer with custom cache size.
    pub fn with_cache_size(
        encoder: FxHashMap<Vec<u8>, u32>,
        special_tokens: FxHashMap<String, u32>,
        pattern: &str,
        cache_size: usize,
    ) -> Result<Self, TokenizerError> {
        // Build decoder maps
        let decoder = build_decoder(&encoder);
        let special_tokens_decoder: FxHashMap<u32, String> = special_tokens
            .iter()
            .map(|(k, v)| (*v, k.clone()))
            .collect();

        // Compile main regex with JIT
        let mut regex_builder = pcre2::bytes::RegexBuilder::new();
        regex_builder.jit_if_available(true);
        regex_builder.utf(true);
        regex_builder.ucp(true); // Unicode property support
        let regex = regex_builder.build(pattern)?;

        // Build Aho-Corasick automaton for special tokens (much faster than regex alternation)
        let special_token_strings: Vec<String> = special_tokens.keys().cloned().collect();
        let special_matcher = if special_token_strings.is_empty() {
            None
        } else {
            Some(AhoCorasick::new(&special_token_strings)?)
        };

        // Initialize LRU cache
        let cache_size = NonZeroUsize::new(cache_size.max(1)).unwrap();
        let chunk_cache = Mutex::new(LruCache::new(cache_size));

        Ok(Self {
            encoder,
            decoder,
            special_tokens,
            special_tokens_decoder,
            special_token_strings,
            regex,
            special_matcher,
            chunk_cache,
        })
    }

    /// Create a tokenizer from a tiktoken vocabulary file.
    pub fn from_file(
        vocab_path: &str,
        pattern: &str,
        special_tokens: FxHashMap<String, u32>,
    ) -> Result<Self, TokenizerError> {
        let encoder = load_tiktoken_bpe_file(vocab_path)?;
        Self::new(encoder, special_tokens, pattern)
    }

    /// Create a tokenizer from raw vocabulary bytes.
    pub fn from_bytes(
        vocab_data: &[u8],
        pattern: &str,
        special_tokens: FxHashMap<String, u32>,
    ) -> Result<Self, TokenizerError> {
        let encoder = load_tiktoken_bpe(vocab_data)?;
        Self::new(encoder, special_tokens, pattern)
    }

    /// Compute a fast hash for a byte slice to use as an LRU cache key.
    ///
    /// Uses FxHasher which is significantly faster than the default SipHash
    /// for small keys like text chunks, with acceptable collision rates for
    /// caching purposes.
    #[inline]
    fn hash_slice(slice: &[u8]) -> u64 {
        let mut hasher = FxHasher::default();
        slice.hash(&mut hasher);
        hasher.finish()
    }

    /// Encode a single text chunk with LRU caching.
    ///
    /// This method implements a multi-tier encoding strategy:
    /// 1. **Direct lookup**: Check if the entire chunk is a known token (O(1))
    /// 2. **Cache hit**: Return cached BPE result if available (O(1))
    /// 3. **BPE encode**: Perform full BPE encoding and cache the result
    ///
    /// The cache dramatically improves performance for:
    /// - Repeated encoding of the same text
    /// - Common substrings across different inputs
    /// - Text with repetitive patterns (e.g., log files, structured data)
    fn encode_chunk(&self, slice: &[u8]) -> Vec<u32> {
        // Fast path: check if entire chunk is a known token
        if let Some(&rank) = self.encoder.get(slice) {
            return vec![rank];
        }

        // Check cache
        let hash = Self::hash_slice(slice);
        if let Ok(mut cache) = self.chunk_cache.lock() {
            if let Some(cached) = cache.get(&hash) {
                return cached.clone();
            }
        }

        // Perform BPE encoding
        let result = byte_pair_encode(slice, &self.encoder);

        // Store in cache
        if let Ok(mut cache) = self.chunk_cache.lock() {
            cache.put(hash, result.clone());
        }

        result
    }

    /// Encode text to token IDs (ignores special tokens in input).
    ///
    /// Uses Rayon to parallelize BPE encoding across regex-matched chunks.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let text_bytes = text.as_bytes();

        // Collect regex matches (chunks to encode)
        let chunks: Vec<(usize, usize)> = self
            .regex
            .find_iter(text_bytes)
            .filter_map(|m| m.ok())
            .map(|m| (m.start(), m.end()))
            .collect();

        if chunks.is_empty() {
            return vec![];
        }

        // Parallel BPE encoding using Rayon
        let results: Vec<Vec<u32>> = chunks
            .par_iter()
            .map(|&(start, end)| {
                let slice = &text_bytes[start..end];
                self.encode_chunk(slice)
            })
            .collect();

        // Flatten results
        results.into_iter().flatten().collect()
    }

    /// Encode text with special token handling.
    ///
    /// Special tokens in the input are encoded directly without BPE.
    /// Uses Aho-Corasick for fast multi-pattern matching.
    pub fn encode_with_special(&self, text: &str) -> Vec<u32> {
        let Some(ref special_matcher) = self.special_matcher else {
            return self.encode(text);
        };

        let text_bytes = text.as_bytes();
        let mut result = Vec::new();
        let mut last_end = 0;

        // Find all special tokens using Aho-Corasick (much faster than regex alternation)
        for m in special_matcher.find_iter(text_bytes) {
            let start = m.start();
            let end = m.end();

            // Encode text before the special token
            if start > last_end {
                let slice = &text[last_end..start];
                result.extend(self.encode(slice));
            }

            // Add the special token directly using the pattern index
            let pattern_idx = m.pattern().as_usize();
            let token_str = &self.special_token_strings[pattern_idx];
            if let Some(&rank) = self.special_tokens.get(token_str) {
                result.push(rank);
            }

            last_end = end;
        }

        // Encode remaining text after last special token
        if last_end < text.len() {
            result.extend(self.encode(&text[last_end..]));
        }

        result
    }

    /// Decode token IDs back to bytes.
    pub fn decode_bytes(&self, tokens: &[u32]) -> Vec<u8> {
        let mut result = Vec::with_capacity(tokens.len() * 4);

        for &token in tokens {
            if let Some(bytes) = self.decoder.get(&token) {
                result.extend_from_slice(bytes);
            } else if let Some(special) = self.special_tokens_decoder.get(&token) {
                result.extend_from_slice(special.as_bytes());
            }
        }

        result
    }

    /// Decode token IDs to a string.
    ///
    /// Returns an error if the decoded bytes are not valid UTF-8.
    pub fn decode(&self, tokens: &[u32]) -> Result<String, TokenizerError> {
        let bytes = self.decode_bytes(tokens);
        String::from_utf8(bytes).map_err(|_| TokenizerError::Utf8Error)
    }

    /// Decode token IDs to a string, replacing invalid UTF-8 with replacement character.
    pub fn decode_lossy(&self, tokens: &[u32]) -> String {
        let bytes = self.decode_bytes(tokens);
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Batch encode multiple texts in parallel.
    ///
    /// Uses Rayon to parallelize across texts AND within each text's BPE encoding.
    pub fn encode_batch(&self, texts: &[String]) -> Vec<Vec<u32>> {
        texts.par_iter().map(|text| self.encode(text)).collect()
    }

    /// Batch encode multiple texts with special token handling.
    pub fn encode_batch_with_special(&self, texts: &[String]) -> Vec<Vec<u32>> {
        texts
            .par_iter()
            .map(|text| self.encode_with_special(text))
            .collect()
    }

    /// Get the vocabulary size (number of tokens).
    pub fn vocab_size(&self) -> usize {
        self.encoder.len() + self.special_tokens.len()
    }

    /// Get the encoder map (token bytes -> ID).
    pub fn encoder(&self) -> &FxHashMap<Vec<u8>, u32> {
        &self.encoder
    }

    /// Get the decoder map (token ID -> bytes).
    pub fn decoder(&self) -> &FxHashMap<u32, Vec<u8>> {
        &self.decoder
    }

    /// Get the special tokens map.
    pub fn special_tokens(&self) -> &FxHashMap<String, u32> {
        &self.special_tokens
    }

    /// Get the special tokens decoder map.
    pub fn special_tokens_decoder(&self) -> &FxHashMap<u32, String> {
        &self.special_tokens_decoder
    }

    /// Clear the encoding cache.
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.chunk_cache.lock() {
            cache.clear();
        }
    }

    /// Get cache statistics (hits would require additional tracking).
    pub fn cache_len(&self) -> usize {
        self.chunk_cache.lock().map(|c| c.len()).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_tokenizer() -> Tokenizer {
        let mut encoder = FxHashMap::default();
        // Single bytes (ASCII printable range)
        for b in 32u8..=126 {
            encoder.insert(vec![b], b as u32);
        }
        // Some merged tokens
        encoder.insert(b"Hello".to_vec(), 200);
        encoder.insert(b"World".to_vec(), 201);
        encoder.insert(b" World".to_vec(), 202);

        let mut special_tokens = FxHashMap::default();
        special_tokens.insert("<|endoftext|>".to_string(), 50256);

        // Simple pattern that matches words and spaces
        let pattern = r"\S+|\s+";

        Tokenizer::new(encoder, special_tokens, pattern).unwrap()
    }

    #[test]
    fn test_encode_decode() {
        let tokenizer = make_test_tokenizer();

        let text = "Hello World";
        let tokens = tokenizer.encode(text);
        let decoded = tokenizer.decode(&tokens).unwrap();

        assert_eq!(decoded, text);
    }

    #[test]
    fn test_encode_with_special() {
        let tokenizer = make_test_tokenizer();

        let text = "Hello<|endoftext|>World";
        let tokens = tokenizer.encode_with_special(text);

        // Should contain the special token
        assert!(tokens.contains(&50256));
    }

    #[test]
    fn test_batch_encode() {
        let tokenizer = make_test_tokenizer();

        let texts = vec!["Hello".to_string(), "World".to_string()];
        let batch_tokens = tokenizer.encode_batch(&texts);

        assert_eq!(batch_tokens.len(), 2);
    }

    #[test]
    fn test_vocab_size() {
        let tokenizer = make_test_tokenizer();
        assert!(tokenizer.vocab_size() > 0);
    }

    #[test]
    fn test_cache_works() {
        let tokenizer = make_test_tokenizer();

        // Use text that isn't a direct token match to trigger BPE and caching
        // "HelloWorld" isn't in the encoder, so it will go through BPE
        let text = "HelloWorld";
        let tokens1 = tokenizer.encode(text);
        let tokens2 = tokenizer.encode(text);

        // Results should be identical
        assert_eq!(tokens1, tokens2);

        // Cache should have entries (BPE result was cached)
        assert!(tokenizer.cache_len() > 0);
    }

    #[test]
    fn test_clear_cache() {
        let tokenizer = make_test_tokenizer();

        // Use text that triggers BPE encoding
        tokenizer.encode("HelloWorld");
        assert!(tokenizer.cache_len() > 0);

        tokenizer.clear_cache();
        assert_eq!(tokenizer.cache_len(), 0);
    }
}
