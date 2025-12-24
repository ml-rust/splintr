use aho_corasick::AhoCorasick;
use lru::LruCache;
use rayon::prelude::*;
use regexr::{Regex as RegexrRegex, RegexBuilder};
use rustc_hash::FxHashMap;
use rustc_hash::FxHasher;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::Mutex;
use thiserror::Error;

#[cfg(feature = "pcre2")]
use pcre2::bytes::Regex as Pcre2Regex;

use super::bpe::byte_pair_encode;
use super::byte_level::{byte_level_decode_bytes, byte_level_encode};
use super::vocab::{build_decoder, load_tiktoken_bpe, load_tiktoken_bpe_file, VocabError};

#[derive(Error, Debug)]
pub enum TokenizerError {
    #[error("Regex compilation error (regexr): {0}")]
    RegexrError(#[from] regexr::Error),
    #[cfg(feature = "pcre2")]
    #[error("Regex compilation error (PCRE2): {0}")]
    Pcre2Error(#[from] pcre2::Error),
    #[error("Vocabulary error: {0}")]
    VocabError(#[from] VocabError),
    #[error("Decoding error: invalid UTF-8")]
    Utf8Error,
    #[error("Aho-Corasick build error: {0}")]
    AhoCorasickError(#[from] aho_corasick::BuildError),
    #[error("PCRE2 feature not enabled. Compile with --features pcre2")]
    Pcre2NotEnabled,
    #[error("Unknown pretrained model: {0}")]
    UnknownPretrained(String),
}

/// Default regex pattern for cl100k_base (GPT-4, GPT-3.5-turbo)
pub const CL100K_BASE_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Default regex pattern for o200k_base (GPT-4o)
pub const O200K_BASE_PATTERN: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Default regex pattern for Llama 3/3.1/3.2/3.3 (same as o200k_base)
pub const LLAMA3_PATTERN: &str = O200K_BASE_PATTERN;

/// Regex pattern for SentencePiece-based tokenizers (Mistral V1/V2, Llama 2, Gemma).
///
/// SentencePiece tokenizers use a simple word boundary approach:
/// - `[^\s]+` = Match one or more non-whitespace characters (words)
/// - `|\s+` = OR match one or more whitespace characters
///
/// This differs from GPT-style tokenizers which use complex patterns for contractions,
/// unicode categories, and punctuation handling. SentencePiece relies on the BPE
/// vocabulary itself to handle these cases during encoding.
pub const SENTENCEPIECE_PATTERN: &str = r"[^\s]+|\s+";

// =============================================================================
// Agent Token Constants (cl100k_base: 100277+, o200k_base: 200019+)
// =============================================================================
// These tokens extend the vocabulary for agent/chat applications without
// conflicting with OpenAI's reserved special token ranges.

/// Agent tokens for cl100k_base (GPT-4, GPT-3.5-turbo).
///
/// These special tokens extend the cl100k_base vocabulary for building chat models,
/// reasoning systems, and autonomous agents. Token IDs start at 100277 to avoid
/// conflicts with OpenAI's reserved range (100257-100276).
///
/// # Token Categories
///
/// ## Conversation Structure (100277-100281)
/// Standard ChatML-style tokens for multi-turn conversations:
/// - `<|system|>`: Marks system instructions that define assistant behavior
/// - `<|user|>`: Marks user input/queries
/// - `<|assistant|>`: Marks assistant responses
/// - `<|im_start|>`: Generic message start delimiter (ChatML format)
/// - `<|im_end|>`: Generic message end delimiter (ChatML format)
///
/// ## Reasoning/Thinking (100282-100283)
/// Chain-of-Thought (CoT) tokens for System 2 reasoning
///
/// ## ReAct Agent Loop (100284-100291)
/// Tokens for ReAct (Reason + Act) agent architectures
///
/// ## Tool/Function Calling (100292-100297)
/// Structured tool use with explicit success/error handling
///
/// ## Code Execution (100298-100303)
/// Jupyter notebook-style code interpreter flow
///
/// ## RAG/Citations (100304-100311)
/// Retrieval-Augmented Generation with source attribution
///
/// ## Memory/State (100312-100315)
/// Long-term memory and state persistence
///
/// ## Control Tokens (100316-100318)
/// Sequence control and formatting
///
/// ## Multimodal (100319-100324)
/// Placeholders for non-text content
///
/// ## Document Structure (100325-100330)
/// Semantic layout tokens for parsing structured documents
pub mod cl100k_agent_tokens {
    pub const SYSTEM: u32 = 100277;
    pub const USER: u32 = 100278;
    pub const ASSISTANT: u32 = 100279;
    pub const IM_START: u32 = 100280;
    pub const IM_END: u32 = 100281;
    pub const THINK: u32 = 100282;
    pub const THINK_END: u32 = 100283;
    pub const PLAN: u32 = 100284;
    pub const PLAN_END: u32 = 100285;
    pub const STEP: u32 = 100286;
    pub const STEP_END: u32 = 100287;
    pub const ACT: u32 = 100288;
    pub const ACT_END: u32 = 100289;
    pub const OBSERVE: u32 = 100290;
    pub const OBSERVE_END: u32 = 100291;
    pub const FUNCTION: u32 = 100292;
    pub const FUNCTION_END: u32 = 100293;
    pub const RESULT: u32 = 100294;
    pub const RESULT_END: u32 = 100295;
    pub const ERROR: u32 = 100296;
    pub const ERROR_END: u32 = 100297;
    pub const CODE: u32 = 100298;
    pub const CODE_END: u32 = 100299;
    pub const OUTPUT: u32 = 100300;
    pub const OUTPUT_END: u32 = 100301;
    pub const LANG: u32 = 100302;
    pub const LANG_END: u32 = 100303;
    pub const CONTEXT: u32 = 100304;
    pub const CONTEXT_END: u32 = 100305;
    pub const QUOTE: u32 = 100306;
    pub const QUOTE_END: u32 = 100307;
    pub const CITE: u32 = 100308;
    pub const CITE_END: u32 = 100309;
    pub const SOURCE: u32 = 100310;
    pub const SOURCE_END: u32 = 100311;
    pub const MEMORY: u32 = 100312;
    pub const MEMORY_END: u32 = 100313;
    pub const RECALL: u32 = 100314;
    pub const RECALL_END: u32 = 100315;
    pub const PAD: u32 = 100316;
    pub const STOP: u32 = 100317;
    pub const SEP: u32 = 100318;
    pub const IMAGE: u32 = 100319;
    pub const IMAGE_END: u32 = 100320;
    pub const AUDIO: u32 = 100321;
    pub const AUDIO_END: u32 = 100322;
    pub const VIDEO: u32 = 100323;
    pub const VIDEO_END: u32 = 100324;
    pub const TITLE: u32 = 100325;
    pub const TITLE_END: u32 = 100326;
    pub const SECTION: u32 = 100327;
    pub const SECTION_END: u32 = 100328;
    pub const SUMMARY: u32 = 100329;
    pub const SUMMARY_END: u32 = 100330;
}

/// Agent tokens for o200k_base (GPT-4o).
///
/// See [`cl100k_agent_tokens`] for detailed documentation on each token category.
/// The token semantics are identical; only the IDs differ.
pub mod o200k_agent_tokens {
    pub const SYSTEM: u32 = 200019;
    pub const USER: u32 = 200020;
    pub const ASSISTANT: u32 = 200021;
    pub const IM_START: u32 = 200022;
    pub const IM_END: u32 = 200023;
    pub const THINK: u32 = 200024;
    pub const THINK_END: u32 = 200025;
    pub const PLAN: u32 = 200026;
    pub const PLAN_END: u32 = 200027;
    pub const STEP: u32 = 200028;
    pub const STEP_END: u32 = 200029;
    pub const ACT: u32 = 200030;
    pub const ACT_END: u32 = 200031;
    pub const OBSERVE: u32 = 200032;
    pub const OBSERVE_END: u32 = 200033;
    pub const FUNCTION: u32 = 200034;
    pub const FUNCTION_END: u32 = 200035;
    pub const RESULT: u32 = 200036;
    pub const RESULT_END: u32 = 200037;
    pub const ERROR: u32 = 200038;
    pub const ERROR_END: u32 = 200039;
    pub const CODE: u32 = 200040;
    pub const CODE_END: u32 = 200041;
    pub const OUTPUT: u32 = 200042;
    pub const OUTPUT_END: u32 = 200043;
    pub const LANG: u32 = 200044;
    pub const LANG_END: u32 = 200045;
    pub const CONTEXT: u32 = 200046;
    pub const CONTEXT_END: u32 = 200047;
    pub const QUOTE: u32 = 200048;
    pub const QUOTE_END: u32 = 200049;
    pub const CITE: u32 = 200050;
    pub const CITE_END: u32 = 200051;
    pub const SOURCE: u32 = 200052;
    pub const SOURCE_END: u32 = 200053;
    pub const MEMORY: u32 = 200054;
    pub const MEMORY_END: u32 = 200055;
    pub const RECALL: u32 = 200056;
    pub const RECALL_END: u32 = 200057;
    pub const PAD: u32 = 200058;
    pub const STOP: u32 = 200059;
    pub const SEP: u32 = 200060;
    pub const IMAGE: u32 = 200061;
    pub const IMAGE_END: u32 = 200062;
    pub const AUDIO: u32 = 200063;
    pub const AUDIO_END: u32 = 200064;
    pub const VIDEO: u32 = 200065;
    pub const VIDEO_END: u32 = 200066;
    pub const TITLE: u32 = 200067;
    pub const TITLE_END: u32 = 200068;
    pub const SECTION: u32 = 200069;
    pub const SECTION_END: u32 = 200070;
    pub const SUMMARY: u32 = 200071;
    pub const SUMMARY_END: u32 = 200072;
}

/// Default cache size for encoded chunks
const DEFAULT_CACHE_SIZE: usize = 4096;

/// Regex backend enum for switching between regexr (default) and PCRE2 (optional)
enum RegexBackend {
    Regexr(Box<RegexrRegex>),
    #[cfg(feature = "pcre2")]
    Pcre2(Pcre2Regex),
}

impl RegexBackend {
    /// Find all matches in the given text, returning (start, end) byte offsets
    fn find_iter<'a>(&'a self, text: &'a str) -> Vec<(usize, usize)> {
        match self {
            RegexBackend::Regexr(regex) => regex
                .find_iter(text)
                .map(|m| (m.start(), m.end()))
                .collect(),
            #[cfg(feature = "pcre2")]
            RegexBackend::Pcre2(regex) => regex
                .find_iter(text.as_bytes())
                .filter_map(|m| m.ok())
                .map(|m| (m.start(), m.end()))
                .collect(),
        }
    }
}

/// High-performance BPE tokenizer with regexr backend (default) or PCRE2 (optional).
///
/// # Performance Characteristics
///
/// This tokenizer is optimized for high throughput across different workloads:
///
/// - **Single text encoding**: Uses sequential processing via [`encode`].
///   Benchmarks show sequential is faster for texts up to ~1MB due to Rayon
///   thread pool overhead. Sequential achieves ~50 MB/s consistently.
///
/// - **Batch encoding**: Uses Rayon parallelism via [`encode_batch`].
///   Parallelizes across texts (not within a single text), achieving ~110 MB/s
///   on batch workloads - approximately 10-12x faster than tiktoken.
///
/// - **Very large single texts (>1MB)**: Use [`encode_rayon`] for texts larger
///   than ~1MB where Rayon parallelization within the text becomes beneficial.
///
/// # Regex Backend
///
/// By default, uses the `regexr` backend (pure Rust with JIT and SIMD support).
/// To use PCRE2 instead, enable the `pcre2` feature and call `.pcre2(true)`:
///
/// ```ignore
/// // Default (regexr)
/// let tokenizer = Tokenizer::from_pretrained("cl100k_base")?;
///
/// // With PCRE2 (requires --features pcre2)
/// let tokenizer = Tokenizer::from_pretrained("cl100k_base")?.pcre2(true)?;
/// ```
///
/// # Key Optimizations
///
/// - Regexr with JIT compilation and SIMD acceleration (default)
/// - Optional PCRE2 with JIT (2-4x faster than fancy-regex)
/// - Rayon parallelism for batch encoding (across texts, not within)
/// - Linked-list BPE algorithm (avoids O(N²) on pathological inputs)
/// - FxHashMap for fast lookups
/// - Aho-Corasick for fast multi-pattern special token matching
/// - LRU cache for frequently encoded chunks
/// - Optional ByteLevel encoding for GPT-2/Llama/DeepSeek style tokenizers
/// - Optional SentencePiece mode for Mistral/Gemma style tokenizers (▁ → space)
pub struct Tokenizer {
    encoder: FxHashMap<Vec<u8>, u32>,
    decoder: FxHashMap<u32, Vec<u8>>,
    special_tokens: FxHashMap<String, u32>,
    special_tokens_decoder: FxHashMap<u32, String>,
    special_token_strings: Vec<String>,
    regex: RegexBackend,
    pattern: String,
    special_matcher: Option<AhoCorasick>,
    chunk_cache: Mutex<LruCache<u64, Vec<u32>>>,
    use_byte_level: bool,
    use_sentencepiece: bool,
    cache_size: usize,
    use_jit: bool,
    use_pcre2: bool,
}

impl Tokenizer {
    /// Create a new tokenizer from encoder map, special tokens, and regex pattern.
    ///
    /// Uses regexr as the default regex backend.
    ///
    /// # Arguments
    /// * `encoder` - Map of byte sequences to token IDs
    /// * `special_tokens` - Map of special token strings to token IDs
    /// * `pattern` - Regex pattern for tokenization
    pub fn new(
        encoder: FxHashMap<Vec<u8>, u32>,
        special_tokens: FxHashMap<String, u32>,
        pattern: &str,
    ) -> Result<Self, TokenizerError> {
        Self::with_options(encoder, special_tokens, pattern, DEFAULT_CACHE_SIZE, false)
    }

    /// Create a new tokenizer with ByteLevel encoding enabled.
    ///
    /// ByteLevel encoding is required for GPT-2, Llama, DeepSeek, and similar tokenizers
    /// that use a byte-to-unicode mapping for handling arbitrary byte sequences.
    pub fn new_byte_level(
        encoder: FxHashMap<Vec<u8>, u32>,
        special_tokens: FxHashMap<String, u32>,
        pattern: &str,
    ) -> Result<Self, TokenizerError> {
        Self::with_options(encoder, special_tokens, pattern, DEFAULT_CACHE_SIZE, true)
    }

    /// Create a new tokenizer with SentencePiece mode enabled.
    ///
    /// SentencePiece mode is required for Mistral, Gemma, and similar tokenizers
    /// that use ▁ (U+2581) as word boundary marker. During decoding, ▁ is converted to space.
    pub fn new_sentencepiece(
        encoder: FxHashMap<Vec<u8>, u32>,
        special_tokens: FxHashMap<String, u32>,
        pattern: &str,
    ) -> Result<Self, TokenizerError> {
        Self::with_full_options(
            encoder,
            special_tokens,
            pattern,
            DEFAULT_CACHE_SIZE,
            false,
            true,
        )
    }

    /// Create a new tokenizer with custom cache size.
    pub fn with_cache_size(
        encoder: FxHashMap<Vec<u8>, u32>,
        special_tokens: FxHashMap<String, u32>,
        pattern: &str,
        cache_size: usize,
    ) -> Result<Self, TokenizerError> {
        Self::with_options(encoder, special_tokens, pattern, cache_size, false)
    }

    /// Create a new tokenizer with full configuration options.
    ///
    /// # Arguments
    /// * `encoder` - Map of byte sequences to token IDs
    /// * `special_tokens` - Map of special token strings to token IDs
    /// * `pattern` - Regex pattern for tokenization
    /// * `cache_size` - Size of the LRU cache for encoded chunks
    /// * `use_byte_level` - Enable ByteLevel encoding for GPT-2/Llama/DeepSeek style tokenizers
    pub fn with_options(
        encoder: FxHashMap<Vec<u8>, u32>,
        special_tokens: FxHashMap<String, u32>,
        pattern: &str,
        cache_size: usize,
        use_byte_level: bool,
    ) -> Result<Self, TokenizerError> {
        Self::with_full_options(
            encoder,
            special_tokens,
            pattern,
            cache_size,
            use_byte_level,
            false,
        )
    }

    /// Create a new tokenizer with all configuration options including SentencePiece mode.
    ///
    /// # Arguments
    /// * `encoder` - Map of byte sequences to token IDs
    /// * `special_tokens` - Map of special token strings to token IDs
    /// * `pattern` - Regex pattern for tokenization
    /// * `cache_size` - Size of the LRU cache for encoded chunks
    /// * `use_byte_level` - Enable ByteLevel encoding for GPT-2/Llama/DeepSeek style tokenizers
    /// * `use_sentencepiece` - Enable SentencePiece mode (▁ → space during decode)
    pub fn with_full_options(
        encoder: FxHashMap<Vec<u8>, u32>,
        special_tokens: FxHashMap<String, u32>,
        pattern: &str,
        cache_size: usize,
        use_byte_level: bool,
        use_sentencepiece: bool,
    ) -> Result<Self, TokenizerError> {
        // Build decoder maps
        let decoder = build_decoder(&encoder);
        let special_tokens_decoder: FxHashMap<u32, String> = special_tokens
            .iter()
            .map(|(k, v)| (*v, k.clone()))
            .collect();

        // Compile regex with regexr (default backend)
        let regex = RegexBuilder::new(pattern).jit(true).build()?;

        // Build Aho-Corasick automaton for special tokens
        let special_token_strings: Vec<String> = special_tokens.keys().cloned().collect();
        let special_matcher = if special_token_strings.is_empty() {
            None
        } else {
            Some(AhoCorasick::new(&special_token_strings)?)
        };

        // Initialize LRU cache
        let cache_size_nz = NonZeroUsize::new(cache_size.max(1)).unwrap();
        let chunk_cache = Mutex::new(LruCache::new(cache_size_nz));

        Ok(Self {
            encoder,
            decoder,
            special_tokens,
            special_tokens_decoder,
            special_token_strings,
            regex: RegexBackend::Regexr(Box::new(regex)),
            pattern: pattern.to_string(),
            special_matcher,
            chunk_cache,
            use_byte_level,
            use_sentencepiece,
            cache_size,
            use_jit: true,
            use_pcre2: false,
        })
    }

    /// Switch to PCRE2 regex backend.
    ///
    /// PCRE2 is an alternative regex backend. Requires the `pcre2` feature
    /// to be enabled at compile time.
    ///
    /// # Example
    /// ```ignore
    /// let tokenizer = Tokenizer::from_pretrained("cl100k_base")?.pcre2(true)?;
    /// ```
    ///
    /// # Errors
    /// Returns an error if `pcre2` feature is not enabled or regex compilation fails.
    #[cfg(feature = "pcre2")]
    pub fn pcre2(mut self, use_pcre2: bool) -> Result<Self, TokenizerError> {
        self.use_pcre2 = use_pcre2;
        if use_pcre2 {
            let mut regex_builder = pcre2::bytes::RegexBuilder::new();
            if self.use_jit {
                regex_builder.jit_if_available(true);
            }
            regex_builder.utf(true);
            regex_builder.ucp(true);
            let regex = regex_builder.build(&self.pattern)?;
            self.regex = RegexBackend::Pcre2(regex);
        } else {
            // Switch back to regexr backend
            let regex = RegexBuilder::new(&self.pattern).jit(self.use_jit).build()?;
            self.regex = RegexBackend::Regexr(Box::new(regex));
        }
        Ok(self)
    }

    /// Switch to PCRE2 regex backend (stub when feature not enabled).
    #[cfg(not(feature = "pcre2"))]
    pub fn pcre2(self, use_pcre2: bool) -> Result<Self, TokenizerError> {
        if use_pcre2 {
            Err(TokenizerError::Pcre2NotEnabled)
        } else {
            Ok(self)
        }
    }

    /// Enable or disable JIT compilation for the regex backend.
    ///
    /// JIT (Just-In-Time) compilation can significantly improve regex matching
    /// performance. JIT availability depends on platform support (e.g., x86-64)
    /// and crate feature flags. When enabled, JIT will be used if available.
    ///
    /// # Arguments
    /// * `use_jit` - Whether to try using JIT compilation
    ///
    /// # Example
    /// ```ignore
    /// let tokenizer = Tokenizer::from_pretrained("cl100k_base")?.jit(false)?;
    /// ```
    #[cfg(feature = "pcre2")]
    pub fn jit(mut self, use_jit: bool) -> Result<Self, TokenizerError> {
        self.use_jit = use_jit;
        if self.use_pcre2 {
            let mut regex_builder = pcre2::bytes::RegexBuilder::new();
            if use_jit {
                regex_builder.jit_if_available(true);
            }
            regex_builder.utf(true);
            regex_builder.ucp(true);
            let regex = regex_builder.build(&self.pattern)?;
            self.regex = RegexBackend::Pcre2(regex);
        } else {
            let regex = RegexBuilder::new(&self.pattern).jit(use_jit).build()?;
            self.regex = RegexBackend::Regexr(Box::new(regex));
        }
        Ok(self)
    }

    /// Enable or disable JIT compilation (non-pcre2 version).
    #[cfg(not(feature = "pcre2"))]
    pub fn jit(mut self, use_jit: bool) -> Result<Self, TokenizerError> {
        self.use_jit = use_jit;
        let regex = RegexBuilder::new(&self.pattern).jit(use_jit).build()?;
        self.regex = RegexBackend::Regexr(Box::new(regex));
        Ok(self)
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

    /// Create a tokenizer from raw vocabulary bytes with ByteLevel encoding.
    pub fn from_bytes_byte_level(
        vocab_data: &[u8],
        pattern: &str,
        special_tokens: FxHashMap<String, u32>,
    ) -> Result<Self, TokenizerError> {
        let encoder = load_tiktoken_bpe(vocab_data)?;
        Self::new_byte_level(encoder, special_tokens, pattern)
    }

    /// Create a tokenizer from raw vocabulary bytes with SentencePiece mode.
    ///
    /// SentencePiece mode converts ▁ (U+2581) to space during decoding.
    /// Used for Mistral, Gemma, and similar tokenizers.
    pub fn from_bytes_sentencepiece(
        vocab_data: &[u8],
        pattern: &str,
        special_tokens: FxHashMap<String, u32>,
    ) -> Result<Self, TokenizerError> {
        let encoder = load_tiktoken_bpe(vocab_data)?;
        Self::new_sentencepiece(encoder, special_tokens, pattern)
    }

    /// Create a SentencePiece tokenizer with explicit decoder to preserve all token IDs.
    ///
    /// This is used for vocabs with duplicate byte sequences (like Mistral V2 where byte fallback
    /// tokens may duplicate BPE merges). The decoder preserves ALL token IDs, while the encoder
    /// only keeps the lowest ID for each byte sequence.
    pub fn from_bytes_sentencepiece_with_decoder(
        vocab_data: &[u8],
        pattern: &str,
        special_tokens: FxHashMap<String, u32>,
    ) -> Result<Self, TokenizerError> {
        use crate::core::vocab::load_tiktoken_bpe_with_decoder;
        let (encoder, mut decoder) = load_tiktoken_bpe_with_decoder(vocab_data)?;

        // Add special tokens to decoder
        for (token_str, id) in &special_tokens {
            decoder.insert(*id, token_str.as_bytes().to_vec());
        }

        // Build the tokenizer manually with explicit decoder
        let special_tokens_decoder: FxHashMap<u32, String> = special_tokens
            .iter()
            .map(|(k, v)| (*v, k.clone()))
            .collect();

        // Compile regex
        let regex = RegexBuilder::new(pattern).jit(true).build()?;

        // Build Aho-Corasick automaton for special tokens
        let special_token_strings: Vec<String> = special_tokens.keys().cloned().collect();
        let special_matcher = if special_token_strings.is_empty() {
            None
        } else {
            Some(AhoCorasick::new(&special_token_strings)?)
        };

        // Initialize LRU cache
        let cache_size_nz = NonZeroUsize::new(DEFAULT_CACHE_SIZE.max(1)).unwrap();
        let chunk_cache = Mutex::new(LruCache::new(cache_size_nz));

        Ok(Self {
            encoder,
            decoder,
            special_tokens,
            special_tokens_decoder,
            special_token_strings,
            regex: RegexBackend::Regexr(Box::new(regex)),
            pattern: pattern.to_string(),
            special_matcher,
            chunk_cache,
            use_byte_level: false,
            use_sentencepiece: true,
            cache_size: DEFAULT_CACHE_SIZE,
            use_jit: true,
            use_pcre2: false,
        })
    }

    /// Compute a fast hash for a byte slice to use as an LRU cache key.
    #[inline]
    fn hash_slice(slice: &[u8]) -> u64 {
        let mut hasher = FxHasher::default();
        slice.hash(&mut hasher);
        hasher.finish()
    }

    /// Encode a single text chunk with LRU caching.
    ///
    /// For SentencePiece mode, `add_prefix` controls whether to prepend ▁.
    #[allow(dead_code)]
    fn encode_chunk_sentencepiece(&self, slice: &[u8], add_prefix: bool) -> Vec<u32> {
        let bytes_to_encode: std::borrow::Cow<[u8]> = if add_prefix {
            let mut with_prefix = Vec::with_capacity(slice.len() + 3);
            with_prefix.extend_from_slice("▁".as_bytes()); // U+2581 is 3 bytes in UTF-8
            with_prefix.extend_from_slice(slice);
            std::borrow::Cow::Owned(with_prefix)
        } else {
            std::borrow::Cow::Borrowed(slice)
        };

        self.encode_bytes_with_cache(bytes_to_encode.as_ref())
    }

    /// Encode bytes with BPE and caching.
    fn encode_bytes_with_cache(&self, bytes: &[u8]) -> Vec<u32> {
        // Fast path: check if entire chunk is a known token
        if let Some(&rank) = self.encoder.get(bytes) {
            return vec![rank];
        }

        // Check cache
        let hash = Self::hash_slice(bytes);
        if let Ok(mut cache) = self.chunk_cache.lock() {
            if let Some(cached) = cache.get(&hash) {
                return cached.clone();
            }
        }

        // Perform BPE encoding
        let result = byte_pair_encode(bytes, &self.encoder);

        // Store in cache
        if let Ok(mut cache) = self.chunk_cache.lock() {
            cache.put(hash, result.clone());
        }

        result
    }

    /// Encode a single text chunk with LRU caching and position tracking.
    fn encode_chunk_with_position(&self, slice: &[u8], _position: usize) -> Vec<u32> {
        // Apply ByteLevel preprocessing if enabled
        let bytes_to_encode: std::borrow::Cow<[u8]> = if self.use_byte_level {
            let byte_level_str = byte_level_encode(slice);
            std::borrow::Cow::Owned(byte_level_str.into_bytes())
        } else {
            std::borrow::Cow::Borrowed(slice)
        };

        // Fast path: check if entire chunk is a known token
        if let Some(&rank) = self.encoder.get(bytes_to_encode.as_ref()) {
            return vec![rank];
        }

        // Check cache
        let hash = Self::hash_slice(bytes_to_encode.as_ref());
        if let Ok(mut cache) = self.chunk_cache.lock() {
            if let Some(cached) = cache.get(&hash) {
                return cached.clone();
            }
        }

        // Perform BPE encoding
        let result = byte_pair_encode(bytes_to_encode.as_ref(), &self.encoder);

        // Store in cache
        if let Ok(mut cache) = self.chunk_cache.lock() {
            cache.put(hash, result.clone());
        }

        result
    }

    /// Encode text to token IDs (ignores special tokens in input).
    ///
    /// Uses sequential processing, which is faster than parallel for texts up to ~1MB.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let text_bytes = text.as_bytes();
        let chunks = self.regex.find_iter(text);

        if chunks.is_empty() {
            return vec![];
        }

        if self.use_sentencepiece {
            // SentencePiece mode: convert spaces to ▁, encode newlines as bytes
            // Rules:
            // - Spaces → ▁ characters (may merge with following word or form ▁▁▁ tokens)
            // - Newlines/tabs → encoded as byte tokens
            // - Words after spaces → get ▁ prefix (as part of BPE, not explicit)
            let mut results = Vec::new();
            let mut pending_underscores = 0usize; // Count of ▁ to prepend to next word

            for &(start, end) in chunks.iter() {
                let slice = &text_bytes[start..end];

                if slice.is_empty() {
                    continue;
                }

                if slice[0].is_ascii_whitespace() {
                    // Whitespace chunk - process each character
                    for &b in slice {
                        if b == b' ' {
                            // Space → accumulate ▁ for next word
                            pending_underscores += 1;
                        } else {
                            // Non-space whitespace (newline, tab, etc.)
                            // First, emit any accumulated ▁ characters
                            if pending_underscores > 0 {
                                let underscores = "▁".repeat(pending_underscores);
                                results
                                    .extend(self.encode_bytes_with_cache(underscores.as_bytes()));
                                pending_underscores = 0;
                            }
                            // Encode the non-space whitespace as a byte
                            results.extend(self.encode_bytes_with_cache(&[b]));
                        }
                    }
                } else {
                    // Word chunk - prepend accumulated ▁ characters and encode together
                    if pending_underscores > 0 {
                        let mut with_prefix =
                            Vec::with_capacity(pending_underscores * 3 + slice.len());
                        for _ in 0..pending_underscores {
                            with_prefix.extend_from_slice("▁".as_bytes());
                        }
                        with_prefix.extend_from_slice(slice);
                        results.extend(self.encode_bytes_with_cache(&with_prefix));
                        pending_underscores = 0;
                    } else {
                        results.extend(self.encode_bytes_with_cache(slice));
                    }
                }
            }

            // Handle trailing underscores (spaces at end of text)
            if pending_underscores > 0 {
                let underscores = "▁".repeat(pending_underscores);
                results.extend(self.encode_bytes_with_cache(underscores.as_bytes()));
            }

            results
        } else {
            // Non-SentencePiece mode: use original logic
            let results: Vec<Vec<u32>> = chunks
                .iter()
                .map(|&(start, end)| {
                    let slice = &text_bytes[start..end];
                    self.encode_chunk_with_position(slice, start)
                })
                .collect();

            results.into_iter().flatten().collect()
        }
    }

    /// Encode text to token IDs using Rayon parallel processing.
    ///
    /// Only beneficial for very large texts (>1MB).
    /// Note: For SentencePiece tokenizers, this falls back to sequential encoding
    /// because SentencePiece requires tracking state between chunks.
    pub fn encode_rayon(&self, text: &str) -> Vec<u32> {
        if self.use_sentencepiece {
            // SentencePiece requires sequential state tracking for ▁ prefix logic
            return self.encode(text);
        }

        let text_bytes = text.as_bytes();
        let chunks = self.regex.find_iter(text);

        if chunks.is_empty() {
            return vec![];
        }

        let results: Vec<Vec<u32>> = chunks
            .par_iter()
            .map(|&(start, end)| {
                let slice = &text_bytes[start..end];
                self.encode_chunk_with_position(slice, start)
            })
            .collect();

        results.into_iter().flatten().collect()
    }

    /// Encode text with special token handling.
    ///
    /// Special tokens in the input are encoded directly without BPE.
    pub fn encode_with_special(&self, text: &str) -> Vec<u32> {
        let Some(ref special_matcher) = self.special_matcher else {
            return self.encode(text);
        };

        let text_bytes = text.as_bytes();
        let mut result = Vec::new();
        let mut last_end = 0;

        for m in special_matcher.find_iter(text_bytes) {
            let start = m.start();
            let end = m.end();

            if start > last_end {
                let slice = &text[last_end..start];
                result.extend(self.encode(slice));
            }

            let pattern_idx = m.pattern().as_usize();
            let token_str = &self.special_token_strings[pattern_idx];
            if let Some(&rank) = self.special_tokens.get(token_str) {
                result.push(rank);
            }

            last_end = end;
        }

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
                if self.use_byte_level {
                    if let Some(decoded) = byte_level_decode_bytes(bytes) {
                        result.extend_from_slice(&decoded);
                    } else {
                        result.extend_from_slice(bytes);
                    }
                } else {
                    result.extend_from_slice(bytes);
                }
            } else if let Some(special) = self.special_tokens_decoder.get(&token) {
                result.extend_from_slice(special.as_bytes());
            }
        }

        result
    }

    /// Decode token IDs to a string.
    pub fn decode(&self, tokens: &[u32]) -> Result<String, TokenizerError> {
        let bytes = self.decode_bytes(tokens);
        let text = String::from_utf8(bytes).map_err(|_| TokenizerError::Utf8Error)?;
        Ok(self.postprocess_decode(text))
    }

    /// Decode token IDs to a string, replacing invalid UTF-8 with replacement character.
    pub fn decode_lossy(&self, tokens: &[u32]) -> String {
        let bytes = self.decode_bytes(tokens);
        let text = String::from_utf8_lossy(&bytes).into_owned();
        self.postprocess_decode(text)
    }

    /// Post-process decoded text for SentencePiece tokenizers.
    ///
    /// Converts ▁ (U+2581, lower one eighth block) to space.
    ///
    /// Note: Unlike some tokenizer implementations, we do NOT strip leading spaces.
    /// The ▁ character represents a word boundary and should become a space.
    /// If you need to strip leading space from the very first token in a sequence,
    /// handle that at a higher level (e.g., in your generation loop).
    #[inline]
    fn postprocess_decode(&self, text: String) -> String {
        if self.use_sentencepiece {
            // Replace ▁ with space - this preserves word boundaries
            text.replace('\u{2581}', " ")
        } else {
            text
        }
    }

    /// Batch encode multiple texts in parallel.
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

    /// Batch decode multiple token lists in parallel.
    pub fn decode_batch(&self, token_lists: &[Vec<u32>]) -> Result<Vec<String>, TokenizerError> {
        token_lists
            .par_iter()
            .map(|tokens| self.decode(tokens))
            .collect()
    }

    /// Batch decode multiple token lists in parallel, replacing invalid UTF-8.
    pub fn decode_batch_lossy(&self, token_lists: &[Vec<u32>]) -> Vec<String> {
        token_lists
            .par_iter()
            .map(|tokens| self.decode_lossy(tokens))
            .collect()
    }

    /// Get the vocabulary size (number of tokens).
    ///
    /// Returns the vocabulary size (total number of token IDs, including special tokens).
    /// This returns max_token_id + 1, representing the full vocabulary range.
    pub fn vocab_size(&self) -> usize {
        // Find the maximum token ID across both decoder and special tokens
        let max_decoder_id = self.decoder.keys().max().copied().unwrap_or(0);
        let max_special_id = self.special_tokens.values().max().copied().unwrap_or(0);
        let max_id = max_decoder_id.max(max_special_id);

        // vocab_size is max_id + 1 (total slots from 0 to max_id inclusive)
        (max_id + 1) as usize
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

    /// Get the current cache size.
    pub fn cache_len(&self) -> usize {
        self.chunk_cache.lock().map(|c| c.len()).unwrap_or(0)
    }
}

impl Clone for Tokenizer {
    fn clone(&self) -> Self {
        // Clone the regex backend with the same JIT setting
        let regex = match &self.regex {
            RegexBackend::Regexr(_) => {
                let regex = RegexBuilder::new(&self.pattern)
                    .jit(self.use_jit)
                    .build()
                    .unwrap();
                RegexBackend::Regexr(Box::new(regex))
            }
            #[cfg(feature = "pcre2")]
            RegexBackend::Pcre2(_) => {
                let mut regex_builder = pcre2::bytes::RegexBuilder::new();
                if self.use_jit {
                    regex_builder.jit_if_available(true);
                }
                regex_builder.utf(true);
                regex_builder.ucp(true);
                let regex = regex_builder.build(&self.pattern).unwrap();
                RegexBackend::Pcre2(regex)
            }
        };

        // Create a new empty cache (caches are not shared)
        let cache_size_nz = NonZeroUsize::new(self.cache_size.max(1)).unwrap();
        let chunk_cache = Mutex::new(LruCache::new(cache_size_nz));

        // Rebuild special matcher
        let special_matcher = if self.special_token_strings.is_empty() {
            None
        } else {
            Some(AhoCorasick::new(&self.special_token_strings).unwrap())
        };

        Self {
            encoder: self.encoder.clone(),
            decoder: self.decoder.clone(),
            special_tokens: self.special_tokens.clone(),
            special_tokens_decoder: self.special_tokens_decoder.clone(),
            special_token_strings: self.special_token_strings.clone(),
            regex,
            pattern: self.pattern.clone(),
            special_matcher,
            chunk_cache,
            use_byte_level: self.use_byte_level,
            use_sentencepiece: self.use_sentencepiece,
            cache_size: self.cache_size,
            use_jit: self.use_jit,
            use_pcre2: self.use_pcre2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_tokenizer() -> Tokenizer {
        let mut encoder = FxHashMap::default();
        for b in 32u8..=126 {
            encoder.insert(vec![b], b as u32);
        }
        encoder.insert(b"Hello".to_vec(), 200);
        encoder.insert(b"World".to_vec(), 201);
        encoder.insert(b" World".to_vec(), 202);

        let mut special_tokens = FxHashMap::default();
        special_tokens.insert("<|endoftext|>".to_string(), 50256);

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
        let text = "HelloWorld";
        let tokens1 = tokenizer.encode(text);
        let tokens2 = tokenizer.encode(text);
        assert_eq!(tokens1, tokens2);
        assert!(tokenizer.cache_len() > 0);
    }

    #[test]
    fn test_clear_cache() {
        let tokenizer = make_test_tokenizer();
        tokenizer.encode("HelloWorld");
        assert!(tokenizer.cache_len() > 0);
        tokenizer.clear_cache();
        assert_eq!(tokenizer.cache_len(), 0);
    }

    #[cfg(feature = "pcre2")]
    #[test]
    fn test_pcre2_backend() {
        let tokenizer = make_test_tokenizer().pcre2(true).unwrap();
        let text = "Hello World";
        let tokens = tokenizer.encode(text);
        let decoded = tokenizer.decode(&tokens).unwrap();
        assert_eq!(decoded, text);
    }

    #[cfg(not(feature = "pcre2"))]
    #[test]
    fn test_pcre2_not_enabled() {
        let tokenizer = make_test_tokenizer();
        let result = tokenizer.pcre2(true);
        assert!(result.is_err());
    }

    #[test]
    fn test_jit_disable() {
        let tokenizer = make_test_tokenizer().jit(false).unwrap();
        let text = "Hello World";
        let tokens = tokenizer.encode(text);
        let decoded = tokenizer.decode(&tokens).unwrap();
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_jit_enable() {
        let tokenizer = make_test_tokenizer().jit(true).unwrap();
        let text = "Hello World";
        let tokens = tokenizer.encode(text);
        let decoded = tokenizer.decode(&tokens).unwrap();
        assert_eq!(decoded, text);
    }

    #[cfg(feature = "pcre2")]
    #[test]
    fn test_pcre2_switch_back_to_regexr() {
        // Start with regexr, switch to pcre2, then back to regexr
        let tokenizer = make_test_tokenizer()
            .pcre2(true)
            .unwrap()
            .pcre2(false)
            .unwrap();
        let text = "Hello World";
        let tokens = tokenizer.encode(text);
        let decoded = tokenizer.decode(&tokens).unwrap();
        assert_eq!(decoded, text);
    }

    #[cfg(feature = "pcre2")]
    #[test]
    fn test_pcre2_with_jit_disabled() {
        let tokenizer = make_test_tokenizer()
            .jit(false)
            .unwrap()
            .pcre2(true)
            .unwrap();
        let text = "Hello World";
        let tokens = tokenizer.encode(text);
        let decoded = tokenizer.decode(&tokens).unwrap();
        assert_eq!(decoded, text);
    }

    const _: () = {
        assert!(super::cl100k_agent_tokens::SYSTEM > 100276);
        assert!(super::cl100k_agent_tokens::SUMMARY_END == 100330);
        assert!(super::o200k_agent_tokens::SYSTEM > 200018);
        assert!(super::o200k_agent_tokens::SUMMARY_END == 200072);
        assert!(super::cl100k_agent_tokens::USER == super::cl100k_agent_tokens::SYSTEM + 1);
        assert!(super::o200k_agent_tokens::USER == super::o200k_agent_tokens::SYSTEM + 1);
    };
}
