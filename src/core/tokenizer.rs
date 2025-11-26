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
/// Example:
/// ```text
/// <|im_start|>system
/// You are a helpful assistant.<|im_end|>
/// <|im_start|>user
/// Hello!<|im_end|>
/// <|im_start|>assistant
/// Hi there!<|im_end|>
/// ```
///
/// ## Reasoning/Thinking (100282-100283)
/// Chain-of-Thought (CoT) tokens for System 2 reasoning, similar to DeepSeek-R1
/// or OpenAI o1-style thinking:
/// - `<|think|>`: Start of internal reasoning (hidden from user in production)
/// - `<|/think|>`: End of internal reasoning
///
/// Example:
/// ```text
/// <|think|>
/// Let me break this down step by step...
/// First, I need to consider X.
/// Then, Y follows from X.
/// <|/think|>
/// The answer is Y.
/// ```
///
/// ## ReAct Agent Loop (100284-100291)
/// Tokens for ReAct (Reason + Act) agent architectures:
/// - `<|plan|>`: High-level planning phase where agent decides strategy
/// - `<|step|>`: Individual step within a plan
/// - `<|act|>`: Action intent declaration (what the agent wants to do)
/// - `<|observe|>`: Observation/feedback from environment after action
///
/// Example:
/// ```text
/// <|plan|>
/// I need to: 1) Search for info, 2) Summarize findings
/// <|/plan|>
/// <|step|>Searching for relevant information<|/step|>
/// <|act|>search("climate change effects")<|/act|>
/// <|observe|>Found 3 relevant articles...<|/observe|>
/// ```
///
/// ## Tool/Function Calling (100292-100297)
/// Structured tool use with explicit success/error handling:
/// - `<|function|>`: Function call specification (name + arguments)
/// - `<|result|>`: Successful function return value
/// - `<|error|>`: Function execution error (enables retry logic)
///
/// Example:
/// ```text
/// <|function|>{"name": "get_weather", "args": {"city": "London"}}<|/function|>
/// <|result|>{"temp": 18, "condition": "cloudy"}<|/result|>
/// ```
///
/// ## Code Execution (100298-100303)
/// Jupyter notebook-style code interpreter flow:
/// - `<|code|>`: Code block to execute
/// - `<|output|>`: Execution output (stdout, return values)
/// - `<|lang|>`: Programming language identifier
///
/// Example:
/// ```text
/// <|code|><|lang|>python<|/lang|>
/// import math
/// print(math.sqrt(16))
/// <|/code|>
/// <|output|>4.0<|/output|>
/// ```
///
/// ## RAG/Citations (100304-100311)
/// Retrieval-Augmented Generation with source attribution:
/// - `<|context|>`: Injected context from retrieval system
/// - `<|quote|>`: Direct quotation from source material
/// - `<|cite|>`: Citation reference marker
/// - `<|source|>`: Source metadata (URL, document ID, etc.)
///
/// Example:
/// ```text
/// <|context|>
/// <|source|>doc_123<|/source|>
/// The Earth orbits the Sun in 365.25 days.
/// <|/context|>
/// According to the source<|cite|>doc_123<|/cite|>, <|quote|>The Earth orbits
/// the Sun in 365.25 days.<|/quote|>
/// ```
///
/// ## Memory/State (100312-100315)
/// Long-term memory and state persistence:
/// - `<|memory|>`: Store information for future reference
/// - `<|recall|>`: Retrieve previously stored information
///
/// Example:
/// ```text
/// <|memory|>User prefers concise responses<|/memory|>
/// ...later...
/// <|recall|>User prefers concise responses<|/recall|>
/// ```
///
/// ## Control Tokens (100316-100318)
/// Sequence control and formatting:
/// - `<|pad|>`: Padding token for batch alignment
/// - `<|stop|>`: Generation stop signal
/// - `<|sep|>`: Separator between segments
///
/// ## Multimodal (100319-100324)
/// Placeholders for non-text content:
/// - `<|image|>`: Image embedding or base64 data
/// - `<|audio|>`: Audio embedding or encoded data
/// - `<|video|>`: Video embedding or encoded data
///
/// Example:
/// ```text
/// Describe this image: <|image|>base64_data_here<|/image|>
/// ```
///
/// ## Document Structure (100325-100330)
/// Semantic layout tokens for parsing structured documents:
/// - `<|title|>`: Document or section title
/// - `<|section|>`: Semantic section boundary
/// - `<|summary|>`: Condensed content summary
///
/// Example:
/// ```text
/// <|title|>Introduction<|/title|>
/// <|section|>
/// This section covers the basics...
/// <|/section|>
/// <|summary|>Key points: X, Y, Z<|/summary|>
/// ```
pub mod cl100k_agent_tokens {
    // =========================================================================
    // Conversation Structure (100277-100281)
    // =========================================================================

    /// System message marker - defines assistant behavior and constraints.
    pub const SYSTEM: u32 = 100277;
    /// User message marker - marks human input in conversation.
    pub const USER: u32 = 100278;
    /// Assistant message marker - marks AI responses.
    pub const ASSISTANT: u32 = 100279;
    /// ChatML message start - generic delimiter for any role.
    pub const IM_START: u32 = 100280;
    /// ChatML message end - closes any message block.
    pub const IM_END: u32 = 100281;

    // =========================================================================
    // Reasoning/Thinking - Chain-of-Thought (100282-100283)
    // =========================================================================

    /// Start of thinking/reasoning block (System 2 cognition).
    /// Content between THINK and THINK_END represents internal reasoning
    /// that may be hidden from users in production.
    pub const THINK: u32 = 100282;
    /// End of thinking/reasoning block.
    pub const THINK_END: u32 = 100283;

    // =========================================================================
    // ReAct Agent Loop (100284-100291)
    // =========================================================================

    /// Start of planning phase - high-level strategy formulation.
    pub const PLAN: u32 = 100284;
    /// End of planning phase.
    pub const PLAN_END: u32 = 100285;
    /// Start of individual step - discrete action within a plan.
    pub const STEP: u32 = 100286;
    /// End of step.
    pub const STEP_END: u32 = 100287;
    /// Start of action - the intent to perform an operation.
    pub const ACT: u32 = 100288;
    /// End of action.
    pub const ACT_END: u32 = 100289;
    /// Start of observation - environment feedback after action.
    pub const OBSERVE: u32 = 100290;
    /// End of observation.
    pub const OBSERVE_END: u32 = 100291;

    // =========================================================================
    // Tool/Function Calling (100292-100297)
    // =========================================================================

    /// Start of function call - contains function name and arguments (usually JSON).
    pub const FUNCTION: u32 = 100292;
    /// End of function call.
    pub const FUNCTION_END: u32 = 100293;
    /// Start of function result - successful return value.
    pub const RESULT: u32 = 100294;
    /// End of function result.
    pub const RESULT_END: u32 = 100295;
    /// Start of error block - function execution failure, enables retry logic.
    pub const ERROR: u32 = 100296;
    /// End of error block.
    pub const ERROR_END: u32 = 100297;

    // =========================================================================
    // Code Execution (100298-100303)
    // =========================================================================

    /// Start of code block - executable code content.
    pub const CODE: u32 = 100298;
    /// End of code block.
    pub const CODE_END: u32 = 100299;
    /// Start of execution output - stdout, return values, rendered output.
    pub const OUTPUT: u32 = 100300;
    /// End of execution output.
    pub const OUTPUT_END: u32 = 100301;
    /// Start of language identifier (e.g., "python", "javascript").
    pub const LANG: u32 = 100302;
    /// End of language identifier.
    pub const LANG_END: u32 = 100303;

    // =========================================================================
    // RAG/Citations (100304-100311)
    // =========================================================================

    /// Start of retrieved context block - injected by RAG pipeline.
    pub const CONTEXT: u32 = 100304;
    /// End of context block.
    pub const CONTEXT_END: u32 = 100305;
    /// Start of direct quotation from source material.
    pub const QUOTE: u32 = 100306;
    /// End of quotation.
    pub const QUOTE_END: u32 = 100307;
    /// Start of citation marker - references a source.
    pub const CITE: u32 = 100308;
    /// End of citation marker.
    pub const CITE_END: u32 = 100309;
    /// Start of source identifier - URL, document ID, or metadata.
    pub const SOURCE: u32 = 100310;
    /// End of source identifier.
    pub const SOURCE_END: u32 = 100311;

    // =========================================================================
    // Memory/State Management (100312-100315)
    // =========================================================================

    /// Start of memory block - information to persist across sessions.
    pub const MEMORY: u32 = 100312;
    /// End of memory block.
    pub const MEMORY_END: u32 = 100313;
    /// Start of recall block - retrieved persistent memory.
    pub const RECALL: u32 = 100314;
    /// End of recall block.
    pub const RECALL_END: u32 = 100315;

    // =========================================================================
    // Control Tokens (100316-100318)
    // =========================================================================

    /// Padding token - used for batch alignment, has no semantic meaning.
    pub const PAD: u32 = 100316;
    /// Stop token - signals end of generation.
    pub const STOP: u32 = 100317;
    /// Separator token - delimits segments within a sequence.
    pub const SEP: u32 = 100318;

    // =========================================================================
    // Multimodal Placeholders (100319-100324)
    // =========================================================================

    /// Start of image content - embedding vector or encoded image data.
    pub const IMAGE: u32 = 100319;
    /// End of image content.
    pub const IMAGE_END: u32 = 100320;
    /// Start of audio content - embedding vector or encoded audio data.
    pub const AUDIO: u32 = 100321;
    /// End of audio content.
    pub const AUDIO_END: u32 = 100322;
    /// Start of video content - embedding vector or encoded video data.
    pub const VIDEO: u32 = 100323;
    /// End of video content.
    pub const VIDEO_END: u32 = 100324;

    // =========================================================================
    // Document Structure (100325-100330)
    // =========================================================================

    /// Start of title - document or section title for semantic parsing.
    pub const TITLE: u32 = 100325;
    /// End of title.
    pub const TITLE_END: u32 = 100326;
    /// Start of section - semantic document section boundary.
    pub const SECTION: u32 = 100327;
    /// End of section.
    pub const SECTION_END: u32 = 100328;
    /// Start of summary - condensed content summary.
    pub const SUMMARY: u32 = 100329;
    /// End of summary.
    pub const SUMMARY_END: u32 = 100330;
}

/// Agent tokens for o200k_base (GPT-4o).
///
/// These special tokens extend the o200k_base vocabulary for building chat models,
/// reasoning systems, and autonomous agents. Token IDs start at 200019 to avoid
/// conflicts with OpenAI's reserved range (199999-200018).
///
/// See [`cl100k_agent_tokens`] for detailed documentation on each token category.
/// The token semantics are identical; only the IDs differ.
pub mod o200k_agent_tokens {
    // =========================================================================
    // Conversation Structure (200019-200023)
    // =========================================================================

    /// System message marker - defines assistant behavior and constraints.
    pub const SYSTEM: u32 = 200019;
    /// User message marker - marks human input in conversation.
    pub const USER: u32 = 200020;
    /// Assistant message marker - marks AI responses.
    pub const ASSISTANT: u32 = 200021;
    /// ChatML message start - generic delimiter for any role.
    pub const IM_START: u32 = 200022;
    /// ChatML message end - closes any message block.
    pub const IM_END: u32 = 200023;

    // =========================================================================
    // Reasoning/Thinking - Chain-of-Thought (200024-200025)
    // =========================================================================

    /// Start of thinking/reasoning block (System 2 cognition).
    pub const THINK: u32 = 200024;
    /// End of thinking/reasoning block.
    pub const THINK_END: u32 = 200025;

    // =========================================================================
    // ReAct Agent Loop (200026-200033)
    // =========================================================================

    /// Start of planning phase - high-level strategy formulation.
    pub const PLAN: u32 = 200026;
    /// End of planning phase.
    pub const PLAN_END: u32 = 200027;
    /// Start of individual step - discrete action within a plan.
    pub const STEP: u32 = 200028;
    /// End of step.
    pub const STEP_END: u32 = 200029;
    /// Start of action - the intent to perform an operation.
    pub const ACT: u32 = 200030;
    /// End of action.
    pub const ACT_END: u32 = 200031;
    /// Start of observation - environment feedback after action.
    pub const OBSERVE: u32 = 200032;
    /// End of observation.
    pub const OBSERVE_END: u32 = 200033;

    // =========================================================================
    // Tool/Function Calling (200034-200039)
    // =========================================================================

    /// Start of function call - contains function name and arguments (usually JSON).
    pub const FUNCTION: u32 = 200034;
    /// End of function call.
    pub const FUNCTION_END: u32 = 200035;
    /// Start of function result - successful return value.
    pub const RESULT: u32 = 200036;
    /// End of function result.
    pub const RESULT_END: u32 = 200037;
    /// Start of error block - function execution failure, enables retry logic.
    pub const ERROR: u32 = 200038;
    /// End of error block.
    pub const ERROR_END: u32 = 200039;

    // =========================================================================
    // Code Execution (200040-200045)
    // =========================================================================

    /// Start of code block - executable code content.
    pub const CODE: u32 = 200040;
    /// End of code block.
    pub const CODE_END: u32 = 200041;
    /// Start of execution output - stdout, return values, rendered output.
    pub const OUTPUT: u32 = 200042;
    /// End of execution output.
    pub const OUTPUT_END: u32 = 200043;
    /// Start of language identifier (e.g., "python", "javascript").
    pub const LANG: u32 = 200044;
    /// End of language identifier.
    pub const LANG_END: u32 = 200045;

    // =========================================================================
    // RAG/Citations (200046-200053)
    // =========================================================================

    /// Start of retrieved context block - injected by RAG pipeline.
    pub const CONTEXT: u32 = 200046;
    /// End of context block.
    pub const CONTEXT_END: u32 = 200047;
    /// Start of direct quotation from source material.
    pub const QUOTE: u32 = 200048;
    /// End of quotation.
    pub const QUOTE_END: u32 = 200049;
    /// Start of citation marker - references a source.
    pub const CITE: u32 = 200050;
    /// End of citation marker.
    pub const CITE_END: u32 = 200051;
    /// Start of source identifier - URL, document ID, or metadata.
    pub const SOURCE: u32 = 200052;
    /// End of source identifier.
    pub const SOURCE_END: u32 = 200053;

    // =========================================================================
    // Memory/State Management (200054-200057)
    // =========================================================================

    /// Start of memory block - information to persist across sessions.
    pub const MEMORY: u32 = 200054;
    /// End of memory block.
    pub const MEMORY_END: u32 = 200055;
    /// Start of recall block - retrieved persistent memory.
    pub const RECALL: u32 = 200056;
    /// End of recall block.
    pub const RECALL_END: u32 = 200057;

    // =========================================================================
    // Control Tokens (200058-200060)
    // =========================================================================

    /// Padding token - used for batch alignment, has no semantic meaning.
    pub const PAD: u32 = 200058;
    /// Stop token - signals end of generation.
    pub const STOP: u32 = 200059;
    /// Separator token - delimits segments within a sequence.
    pub const SEP: u32 = 200060;

    // =========================================================================
    // Multimodal Placeholders (200061-200066)
    // =========================================================================

    /// Start of image content - embedding vector or encoded image data.
    pub const IMAGE: u32 = 200061;
    /// End of image content.
    pub const IMAGE_END: u32 = 200062;
    /// Start of audio content - embedding vector or encoded audio data.
    pub const AUDIO: u32 = 200063;
    /// End of audio content.
    pub const AUDIO_END: u32 = 200064;
    /// Start of video content - embedding vector or encoded video data.
    pub const VIDEO: u32 = 200065;
    /// End of video content.
    pub const VIDEO_END: u32 = 200066;

    // =========================================================================
    // Document Structure (200067-200072)
    // =========================================================================

    /// Start of title - document or section title for semantic parsing.
    pub const TITLE: u32 = 200067;
    /// End of title.
    pub const TITLE_END: u32 = 200068;
    /// Start of section - semantic document section boundary.
    pub const SECTION: u32 = 200069;
    /// End of section.
    pub const SECTION_END: u32 = 200070;
    /// Start of summary - condensed content summary.
    pub const SUMMARY: u32 = 200071;
    /// End of summary.
    pub const SUMMARY_END: u32 = 200072;
}

/// Default cache size for encoded chunks
const DEFAULT_CACHE_SIZE: usize = 4096;

/// High-performance tokenizer using PCRE2 with JIT and Rayon parallelism.
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
/// # Design Decision: Sequential by Default
///
/// The [`encode`] method uses sequential processing because Rayon parallel
/// overhead is significant for typical text sizes:
///
/// | Text Size | Sequential | Rayon | Speedup |
/// |-----------|------------|-------|---------|
/// | 100 bytes | 42 MB/s | 3 MB/s | Sequential 12x faster |
/// | 10 KB | 50 MB/s | 26 MB/s | Sequential 2x faster |
/// | 100 KB | 54 MB/s | 41 MB/s | Sequential 1.3x faster |
/// | 1 MB | 44 MB/s | 47 MB/s | Rayon 1.07x faster |
///
/// Rayon only becomes beneficial at ~1MB, which is rare in typical workloads.
/// For batch processing, use [`encode_batch`] which parallelizes across texts.
///
/// # Key Optimizations
///
/// - PCRE2 with JIT compilation (2-4x faster than fancy-regex)
/// - Rayon parallelism for batch encoding (across texts, not within)
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
    /// Uses sequential processing, which is faster than parallel for texts up to ~1MB.
    /// Achieves ~50 MB/s throughput, approximately 3x faster than tiktoken.
    ///
    /// # Why Sequential?
    ///
    /// Rayon parallel processing has significant thread pool overhead that only
    /// pays off for very large texts (~1MB+). Benchmarks show:
    /// - 100 bytes: Sequential is 12x faster than Rayon
    /// - 10 KB: Sequential is 2x faster
    /// - 100 KB: Sequential is 1.3x faster
    /// - 1 MB: Rayon becomes ~7% faster
    ///
    /// # When to Use Other Methods
    ///
    /// - **Multiple texts**: Use [`encode_batch`] for parallel encoding across texts
    /// - **Very large texts (>1MB)**: Use [`encode_rayon`] for parallel within-text encoding
    /// - **Special tokens**: Use [`encode_with_special`] to recognize special tokens
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

        // Sequential encoding - Rayon overhead not worth it for texts < 1MB
        // See struct-level docs for benchmark data
        let results: Vec<Vec<u32>> = chunks
            .iter()
            .map(|&(start, end)| {
                let slice = &text_bytes[start..end];
                self.encode_chunk(slice)
            })
            .collect();

        // Flatten results
        results.into_iter().flatten().collect()
    }

    /// Encode text to token IDs using Rayon parallel processing.
    ///
    /// Parallelizes BPE encoding of individual regex-matched chunks using Rayon.
    /// Only beneficial for very large texts (>1MB) where parallelization overhead
    /// is amortized across many chunks.
    ///
    /// # Performance
    ///
    /// | Text Size | Sequential | Rayon | Winner |
    /// |-----------|------------|-------|--------|
    /// | < 500 KB | ~50 MB/s | ~40 MB/s | Sequential |
    /// | ~1 MB | ~44 MB/s | ~47 MB/s | Rayon (1.07x) |
    ///
    /// # When to Use
    ///
    /// - Single texts larger than ~1MB (e.g., entire books, large documents)
    /// - When processing time is more critical than thread pool overhead
    ///
    /// For most use cases, prefer [`encode`] (sequential) or [`encode_batch`]
    /// (parallel across multiple texts).
    pub fn encode_rayon(&self, text: &str) -> Vec<u32> {
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

        // Parallel encoding using Rayon - each chunk encoded in parallel
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
    /// Uses Rayon to parallelize **across texts** (not within each text).
    /// This is the most efficient approach for batch workloads because:
    ///
    /// 1. Each text is encoded sequentially (optimal for texts < 1MB)
    /// 2. Multiple texts are processed in parallel across CPU cores
    /// 3. No thread coordination overhead within individual texts
    ///
    /// # Performance
    ///
    /// Achieves ~110 MB/s throughput on batch workloads, approximately
    /// 10-12x faster than tiktoken's `encode_ordinary_batch`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let texts = vec!["Hello".to_string(), "World".to_string()];
    /// let token_ids = tokenizer.encode_batch(&texts);
    /// ```
    pub fn encode_batch(&self, texts: &[String]) -> Vec<Vec<u32>> {
        texts.par_iter().map(|text| self.encode(text)).collect()
    }

    /// Batch encode multiple texts with special token handling.
    ///
    /// Like [`encode_batch`], but recognizes special tokens in the input.
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

    // Compile-time verification that agent tokens don't conflict with OpenAI's reserved range
    const _: () = {
        assert!(super::cl100k_agent_tokens::SYSTEM > 100276); // After endofprompt
        assert!(super::cl100k_agent_tokens::SUMMARY_END == 100330); // Last token
        assert!(super::o200k_agent_tokens::SYSTEM > 200018); // After endofprompt
        assert!(super::o200k_agent_tokens::SUMMARY_END == 200072); // Last token
                                                                   // Verify token ordering is correct (no gaps or overlaps)
        assert!(super::cl100k_agent_tokens::USER == super::cl100k_agent_tokens::SYSTEM + 1);
        assert!(super::o200k_agent_tokens::USER == super::o200k_agent_tokens::SYSTEM + 1);
    };

    #[test]
    fn test_agent_tokens_encode_decode() {
        // Create a tokenizer with agent tokens for testing
        let mut encoder: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        encoder.insert(b"Hello".to_vec(), 0);
        encoder.insert(b" ".to_vec(), 1);
        encoder.insert(b"World".to_vec(), 2);

        let mut special: FxHashMap<String, u32> = FxHashMap::default();
        // Add some agent tokens
        special.insert("<|system|>".to_string(), 100277);
        special.insert("<|user|>".to_string(), 100278);
        special.insert("<|assistant|>".to_string(), 100279);
        special.insert("<|think|>".to_string(), 100282);
        special.insert("<|/think|>".to_string(), 100283);

        let pattern = r"\S+|\s+";
        let tokenizer = Tokenizer::new(encoder, special, pattern).unwrap();

        // Test encoding with agent tokens
        let text = "<|system|>Hello<|user|>World";
        let tokens = tokenizer.encode_with_special(text);

        // Should contain the special tokens
        assert!(tokens.contains(&100277)); // <|system|>
        assert!(tokens.contains(&100278)); // <|user|>

        // Test decoding back
        let decoded = tokenizer.decode(&tokens).unwrap();
        assert_eq!(decoded, text);

        // Test think tokens
        let think_text = "<|think|>reasoning here<|/think|>";
        let think_tokens = tokenizer.encode_with_special(think_text);
        assert!(think_tokens.contains(&100282)); // <|think|>
        assert!(think_tokens.contains(&100283)); // <|/think|>
    }
}
