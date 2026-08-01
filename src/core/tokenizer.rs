use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use lru::LruCache;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use regexr::{Regex as RegexrRegex, RegexBuilder};
use rustc_hash::FxHashMap;
use rustc_hash::FxHasher;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};
use thiserror::Error;

#[cfg(feature = "pcre2")]
use pcre2::bytes::Regex as Pcre2Regex;

use super::bpe::{byte_pair_encode, byte_pair_encode_with_ranks};
use super::byte_level::{byte_level_decode_bytes, byte_level_encode};
use super::vocab::{build_decoder, load_tiktoken_bpe, load_tiktoken_bpe_file, VocabError};

/// Build the special/added-token automaton with leftmost-longest semantics, so a
/// longer added token (e.g. a 24-space run) wins over a shorter one (a 2-space
/// run) that starts at the same position — matching HuggingFace's added-token
/// matching. Default `AhoCorasick` (Standard) would instead report the
/// earliest-ending match, splitting the run into several short tokens.
fn build_special_matcher<S: AsRef<[u8]>>(
    patterns: &[S],
) -> Result<AhoCorasick, aho_corasick::BuildError> {
    AhoCorasickBuilder::new()
        .match_kind(MatchKind::LeftmostLongest)
        .build(patterns)
}

#[derive(Error, Debug)]
pub enum TokenizerError {
    #[error("Regex compilation error (regexr): {0}")]
    RegexrError(#[from] regexr::Error),
    #[cfg(feature = "pcre2")]
    #[error("Regex compilation error (PCRE2): {0}")]
    Pcre2Error(#[from] pcre2::Error),
    #[error("Vocabulary error: {0}")]
    VocabError(#[from] VocabError),
    #[error("SentencePiece BPE error: {0}")]
    SpmError(#[from] crate::core::spm::SpmError),
    #[error("Decoding error: invalid UTF-8")]
    Utf8Error,
    #[error("Aho-Corasick build error: {0}")]
    AhoCorasickError(#[from] aho_corasick::BuildError),
    #[error("PCRE2 feature not enabled. Compile with --features pcre2")]
    Pcre2NotEnabled,
    #[error("Unknown pretrained model: {0}")]
    UnknownPretrained(String),
    #[error("Pre-tokenizer pattern list is empty")]
    EmptyPatternList,
}

/// Default regex pattern for cl100k_base (GPT-4, GPT-3.5-turbo).
///
/// Transcribed verbatim from tiktoken's cl100k pattern, with possessive
/// quantifiers (`?+`/`++`/`*+`) lowered to greedy — proven split-equivalent over
/// 40k+ random strings, and greedy compiles on the regexr backend (which has no
/// possessive support). Note the `\s+$` end-anchored branch and trailing bare
/// `\s`, which an earlier hand-simplified approximation omitted.
pub const CL100K_BASE_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s+$|\s*[\r\n]|\s+(?!\S)|\s";

/// Default regex pattern for o200k_base (GPT-4o). Transcribed from tiktoken's
/// o200k pattern (already greedy upstream).
pub const O200K_BASE_PATTERN: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Pre-tokenizer pattern for Llama 3/3.1/3.2/3.3.
///
/// Transcribed verbatim from the `Split` pre-tokenizer's `Regex` in Meta's
/// `llama-3.2-1b/tokenizer.json` — the pattern the model was actually trained
/// with. llama.cpp records the same string byte-for-byte as the
/// "original regex from tokenizer.json" for `LLAMA_VOCAB_PRE_TYPE_LLAMA3`
/// (`llama-vocab.cpp:286`); the expression it feeds its own engine
/// (`llama-vocab.cpp:289`) differs only by expanding `(?i:'s|'t|…)` into
/// `(?:'[sS]|'[tT]|…)`, which is the same language.
///
/// This is NOT [`O200K_BASE_PATTERN`] and must never be re-aliased to it.
/// o200k's two leading `[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}…]+` branches
/// split letter runs on upper/lower case boundaries — an OpenAI convention
/// Llama 3 does not share, since its pre-tokenizer takes whole letter runs with
/// a plain `\p{L}+`. Aliasing the two breaks every camelCase merge: with the
/// o200k split `XMLHttpRequest` encodes as `[10833, 2977, 1939]` instead of the
/// correct `[10833, 27459]`.
///
/// Identical to [`QWEN2_PATTERN`] apart from `\p{N}{1,3}` (digit runs of up to
/// three) versus Qwen's single-digit `\p{N}`, so the two are not
/// interchangeable either.
pub const LLAMA3_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

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

/// Regex pattern for Mistral V3/Tekken tokenizer.
///
/// This pattern is specifically from the Mistral NeMo tokenizer and differs from O200K:
/// - No English contraction handling (`'s`, `'t`, `'re`, etc.)
/// - Single digit numbers `\p{N}` instead of `\p{N}{1,3}`
/// - Otherwise similar Unicode category handling
pub const MISTRAL_V3_PATTERN: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// GPT-2 style pre-tokenizer pattern used by Whisper.
///
/// Whisper's `tokenizer.json` declares a `ByteLevel` pre-tokenizer which applies
/// this regex to split text before BPE merging.
pub const GPT2_PATTERN: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

/// Pre-tokenizer pattern for Qwen2 / Qwen3 (llama.cpp's `qwen2` pre-tokenizer).
///
/// Identical to [`LLAMA3_PATTERN`] except that digits split one at a time
/// (`\p{N}`) rather than in runs of up to three (`\p{N}{1,3}`). That single
/// difference changes the resulting tokens, so the two are not interchangeable
/// and must stay separate constants.
pub const QWEN2_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

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

/// Compile one pre-tokenizer expression on the selected backend.
///
/// The single place that knows how each backend is configured (PCRE2 needs
/// `utf`+`ucp` to give `\p{…}` and `\s` their Unicode meanings), so a chained
/// pre-tokenizer's later passes are compiled exactly like its first one.
fn compile_pattern(
    pattern: &str,
    use_pcre2: bool,
    use_jit: bool,
) -> Result<RegexBackend, TokenizerError> {
    #[cfg(feature = "pcre2")]
    if use_pcre2 {
        let mut regex_builder = pcre2::bytes::RegexBuilder::new();
        if use_jit {
            regex_builder.jit_if_available(true);
        }
        regex_builder.utf(true);
        regex_builder.ucp(true);
        return Ok(RegexBackend::Pcre2(regex_builder.build(pattern)?));
    }
    #[cfg(not(feature = "pcre2"))]
    let _ = use_pcre2;

    let regex = RegexBuilder::new(pattern).jit(use_jit).build()?;
    Ok(RegexBackend::Regexr(Box::new(regex)))
}

/// One pass of llama.cpp's `unicode_regex_split` (`unicode.cpp:990-1088`).
///
/// Every span produced by the previous pass is re-matched **independently** —
/// the expression sees only that span's text, so `^`, `$` and lookaround treat
/// the span's edges as the edges of the world — and each span is replaced by the
/// ordered sequence of its matches AND the gaps between them
/// (`unicode_regex_split_stl`, `unicode.cpp:486-505`: an unmatched prefix is
/// emitted before every match, and any unmatched tail after the last one).
///
/// So a later expression NEVER re-examines the whole text and never merges
/// anything: it can only cut existing pieces finer. Nothing is exempted from a
/// later pass either — a gap left by pass 1 is an ordinary span that pass 2
/// subdivides like any other. That is why a list of N expressions is not the
/// alternation of those N expressions.
fn subdivide(re: &RegexBackend, text: &str, spans: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(spans.len());
    for &(span_start, span_end) in spans {
        let Some(piece) = text.get(span_start..span_end) else {
            continue;
        };
        let mut last = 0;
        for (start, end) in re.find_iter(piece) {
            if start > last {
                out.push((span_start + last, span_start + start));
            }
            // A zero-width match would emit an empty piece upstream too; it
            // carries no bytes, so it is simply not recorded.
            if end > start {
                out.push((span_start + start, span_start + end));
            }
            last = end;
        }
        if last < piece.len() {
            out.push((span_start + last, span_end));
        }
    }
    out
}

/// High-performance BPE tokenizer with regexr backend (default) or PCRE2 (optional).
///
/// # Performance Characteristics
///
/// This tokenizer is optimized for high throughput across different workloads:
///
/// - **Single text encoding**: Uses sequential processing via [`Tokenizer::encode`].
///   Benchmarks show sequential is faster for texts up to ~1MB due to Rayon
///   thread pool overhead. Sequential achieves ~50 MB/s consistently.
///
/// - **Batch encoding**: Uses Rayon parallelism via [`Tokenizer::encode_batch`].
///   Parallelizes across texts (not within a single text), achieving ~110 MB/s
///   on batch workloads - approximately 10-12x faster than tiktoken.
///
/// - **Very large single texts (>1MB)**: Use [`Tokenizer::encode_rayon`] for texts larger
///   than ~1MB where Rayon parallelization within the text becomes beneficial.
///
/// # Regex Backend
///
/// By default, uses the `regexr` backend (pure Rust with JIT and SIMD support).
/// To use PCRE2 instead, enable the `pcre2` feature and call `.pcre2(true)`:
///
/// ```rust
/// use splintr::{from_pretrained, Backend};
///
/// // Default (regexr) — `from_pretrained` returns an `AnyTokenizer`; reach the
/// // concrete `Tokenizer` (needed for `.pcre2()`) through `into_backend()`.
/// let any = from_pretrained("cl100k_base")?;
/// let Backend::Bpe(tokenizer) = any.into_backend() else {
///     unreachable!("cl100k_base loads as a BPE backend");
/// };
///
/// // With PCRE2 (requires --features pcre2; without it this returns an error
/// // rather than switching backends).
/// match tokenizer.pcre2(true) {
///     Ok(_pcre2_tokenizer) => { /* now using the PCRE2 backend */ }
///     Err(_) => { /* `pcre2` feature not enabled at compile time */ }
/// }
/// # Ok::<(), splintr::TokenizerError>(())
/// ```
///
/// # Key Optimizations
///
/// - Regexr with JIT compilation and SIMD acceleration (default)
/// - Optional PCRE2 with JIT (industry-standard backend, via the `pcre2` feature)
/// - Rayon parallelism for batch encoding (across texts, not within)
/// - Linked-list BPE algorithm (avoids O(N²) on pathological inputs)
/// - FxHashMap for fast lookups
/// - Aho-Corasick for fast multi-pattern special token matching
/// - LRU cache for frequently encoded chunks
/// - Optional ByteLevel encoding for GPT-2/Llama/DeepSeek style tokenizers
/// - Optional SentencePiece mode for Mistral/Gemma style tokenizers (▁ → space)
pub struct Tokenizer {
    encoder: FxHashMap<Vec<u8>, u32>,
    /// Optional separate merge-priority map (bytes → merge rank). When present,
    /// BPE merges by this rank instead of by token id — required for HuggingFace
    /// BPE models whose ids don't follow merge order (e.g. RoBERTa). `None`
    /// means tiktoken-style (id doubles as merge rank).
    merge_ranks: Option<FxHashMap<Vec<u8>, u32>>,
    decoder: FxHashMap<u32, Vec<u8>>,
    special_tokens: FxHashMap<String, u32>,
    special_tokens_decoder: FxHashMap<u32, String>,
    special_token_strings: Vec<String>,
    regex: RegexBackend,
    pattern: String,
    /// Pre-tokenizer expressions applied AFTER [`Tokenizer::regex`], in order —
    /// llama.cpp's `regex_exprs` list beyond its first entry (see [`subdivide`]).
    ///
    /// Empty for a single-expression pre-tokenizer, which is every tiktoken-style
    /// vocabulary and the throughput-critical case; [`Tokenizer::split_chunks`]
    /// then runs the untouched single-regex path. Behind an `Arc` so cloning a
    /// tokenizer shares the compiled passes instead of recompiling them.
    chain: Arc<[RegexBackend]>,
    /// Source expressions for [`Tokenizer::chain`], kept so switching backend or
    /// JIT recompiles the later passes the same way it recompiles the first.
    chain_patterns: Arc<[String]>,
    special_matcher: Option<AhoCorasick>,
    chunk_cache: Mutex<LruCache<u64, Vec<u32>>>,
    use_byte_level: bool,
    use_sentencepiece: bool,
    /// Prepend a space to input before tokenizing (HF ByteLevel `add_prefix_space`).
    add_prefix_space: bool,
    /// Optional multi-stage pre-tokenizer pipeline (HF `pre_tokenizer` graphs
    /// beyond a single regex, e.g. Digits/Punctuation/Sequence). When set, it
    /// produces the (already byte-level-encoded) chunks instead of the regex.
    pre_tokenizer: Option<std::sync::Arc<super::pretokenizer::PreTokenizer>>,
    /// When true, `encode` first matches `special_tokens` (HF always recognizes
    /// added tokens in the input).
    match_added_tokens: bool,
    /// Ids of `special=true` added tokens to drop on decode (HF default
    /// skip_special_tokens=true). Non-special added tokens are still rendered.
    special_decode_ids: rustc_hash::FxHashSet<u32>,
    /// Optional text normalizer (HF `normalizer`, e.g. NFC) applied to content
    /// before splitting. Applied per content gap, never to special-token matches.
    normalizer: Option<std::sync::Arc<super::normalizer::Normalizer>>,
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

    /// Create a ByteLevel tokenizer whose pre-tokenizer is a SEQUENCE of
    /// expressions applied in order, llama.cpp's `regex_exprs` list.
    ///
    /// Each expression subdivides the pieces the previous one produced rather
    /// than re-reading the whole text, and the gaps a pass leaves unmatched stay
    /// as pieces of their own — see [`subdivide`] for the exact semantics and
    /// their source. A one-element list is exactly [`Tokenizer::new_byte_level`]
    /// and keeps the single-regex fast path, so callers can pass a list
    /// unconditionally without paying for the general machinery.
    ///
    /// Vocabularies that need this cannot be expressed as one alternation:
    /// `falcon` splits punctuation runs, then applies the GPT-2 split to the
    /// pieces, then cuts digit runs into groups of three.
    pub fn new_byte_level_chain(
        encoder: FxHashMap<Vec<u8>, u32>,
        special_tokens: FxHashMap<String, u32>,
        patterns: &[&str],
    ) -> Result<Self, TokenizerError> {
        let (first, rest) = patterns
            .split_first()
            .ok_or(TokenizerError::EmptyPatternList)?;
        let mut tokenizer = Self::new_byte_level(encoder, special_tokens, first)?;
        tokenizer.set_chain(rest)?;
        Ok(tokenizer)
    }

    /// Compile and install the later pre-tokenizer passes on the current backend.
    fn set_chain(&mut self, patterns: &[&str]) -> Result<(), TokenizerError> {
        let compiled = patterns
            .iter()
            .map(|p| compile_pattern(p, self.use_pcre2, self.use_jit))
            .collect::<Result<Vec<_>, _>>()?;
        self.chain = Arc::from(compiled);
        self.chain_patterns = patterns.iter().map(|p| (*p).to_owned()).collect();
        Ok(())
    }

    /// Recompile the later pre-tokenizer passes after a backend or JIT change.
    fn rebuild_chain(&mut self) -> Result<(), TokenizerError> {
        if self.chain_patterns.is_empty() {
            return Ok(());
        }
        let compiled = self
            .chain_patterns
            .iter()
            .map(|p| compile_pattern(p, self.use_pcre2, self.use_jit))
            .collect::<Result<Vec<_>, _>>()?;
        self.chain = Arc::from(compiled);
        Ok(())
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
        let regex = compile_pattern(pattern, false, true)?;

        // Build Aho-Corasick automaton for special tokens
        let special_token_strings: Vec<String> = special_tokens.keys().cloned().collect();
        let special_matcher = if special_token_strings.is_empty() {
            None
        } else {
            Some(build_special_matcher(&special_token_strings)?)
        };

        // Initialize LRU cache
        let cache_size_nz = NonZeroUsize::new(cache_size.max(1)).unwrap();
        let chunk_cache = Mutex::new(LruCache::new(cache_size_nz));

        Ok(Self {
            encoder,
            merge_ranks: None,
            decoder,
            special_tokens,
            special_tokens_decoder,
            special_token_strings,
            regex,
            pattern: pattern.to_string(),
            chain: Arc::from(Vec::new()),
            chain_patterns: Arc::from(Vec::new()),
            special_matcher,
            chunk_cache,
            use_byte_level,
            use_sentencepiece,
            add_prefix_space: false,
            pre_tokenizer: None,
            match_added_tokens: false,
            special_decode_ids: rustc_hash::FxHashSet::default(),
            normalizer: None,
            cache_size,
            use_jit: true,
            use_pcre2: false,
        })
    }

    /// Attach a separate merge-priority map (bytes → merge rank) so BPE merges
    /// by this order rather than by token id. Use for HuggingFace BPE models
    /// whose ids don't follow merge order (e.g. RoBERTa).
    pub fn with_merge_ranks(mut self, merge_ranks: FxHashMap<Vec<u8>, u32>) -> Self {
        self.merge_ranks = Some(merge_ranks);
        self
    }

    /// Enable HF ByteLevel `add_prefix_space`: a leading space is prepended to
    /// the input before tokenizing (unless it already starts with whitespace).
    pub fn with_prefix_space(mut self, add_prefix_space: bool) -> Self {
        self.add_prefix_space = add_prefix_space;
        self
    }

    /// Attach a multi-stage pre-tokenizer pipeline. Its pieces are already
    /// byte-level-encoded, so this tokenizer must have `use_byte_level=false`.
    pub fn with_pre_tokenizer(mut self, pt: super::pretokenizer::PreTokenizer) -> Self {
        self.pre_tokenizer = Some(std::sync::Arc::new(pt));
        self
    }

    /// Make `encode` recognize `special_tokens` (added tokens) in the input,
    /// matching HuggingFace, which always recognizes added tokens.
    pub fn with_added_token_matching(mut self, enabled: bool) -> Self {
        self.match_added_tokens = enabled;
        self
    }

    /// Set the ids of `special=true` added tokens to drop on decode (HF default
    /// `skip_special_tokens=true`). Non-special added tokens stay rendered.
    pub fn with_special_decode_ids(mut self, ids: rustc_hash::FxHashSet<u32>) -> Self {
        self.special_decode_ids = ids;
        self
    }

    /// Attach a text normalizer (HF `normalizer`, e.g. NFC) applied to content
    /// before splitting. An empty normalizer is treated as absent.
    pub fn with_normalizer(mut self, normalizer: super::normalizer::Normalizer) -> Self {
        self.normalizer = (!normalizer.is_empty()).then(|| std::sync::Arc::new(normalizer));
        self
    }

    /// The raw surface string of a token id (the vocab key, byte-level-encoded for
    /// byte-level vocabs), or the special-token text. Used to drive a
    /// configuration-declared decoder pipeline.
    pub fn token_surface(&self, id: u32) -> Option<String> {
        if let Some(bytes) = self.decoder.get(&id) {
            Some(String::from_utf8_lossy(bytes).into_owned())
        } else {
            self.special_tokens_decoder.get(&id).cloned()
        }
    }

    /// Apply `add_prefix_space` to an input, borrowing when no change is needed.
    #[inline]
    fn prefixed<'a>(&self, text: &'a str) -> std::borrow::Cow<'a, str> {
        if self.add_prefix_space && !text.starts_with(|c: char| c.is_whitespace()) {
            std::borrow::Cow::Owned(format!(" {text}"))
        } else {
            std::borrow::Cow::Borrowed(text)
        }
    }

    /// Switch to PCRE2 regex backend.
    ///
    /// PCRE2 is an alternative regex backend. Requires the `pcre2` feature
    /// to be enabled at compile time.
    ///
    /// # Example
    /// ```rust
    /// use splintr::{from_pretrained, Backend};
    ///
    /// let any = from_pretrained("cl100k_base")?;
    /// let Backend::Bpe(tokenizer) = any.into_backend() else {
    ///     unreachable!("cl100k_base loads as a BPE backend");
    /// };
    /// let tokenizer = tokenizer.pcre2(true)?;
    /// # Ok::<(), splintr::TokenizerError>(())
    /// ```
    ///
    /// # Errors
    /// Returns an error if `pcre2` feature is not enabled or regex compilation fails.
    #[cfg(feature = "pcre2")]
    pub fn pcre2(mut self, use_pcre2: bool) -> Result<Self, TokenizerError> {
        self.use_pcre2 = use_pcre2;
        self.regex = compile_pattern(&self.pattern, use_pcre2, self.use_jit)?;
        self.rebuild_chain()?;
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
    /// ```rust
    /// use splintr::{from_pretrained, Backend};
    ///
    /// let any = from_pretrained("cl100k_base")?;
    /// let Backend::Bpe(tokenizer) = any.into_backend() else {
    ///     unreachable!("cl100k_base loads as a BPE backend");
    /// };
    /// let tokenizer = tokenizer.jit(false)?;
    /// # Ok::<(), splintr::TokenizerError>(())
    /// ```
    #[cfg(feature = "pcre2")]
    pub fn jit(mut self, use_jit: bool) -> Result<Self, TokenizerError> {
        self.use_jit = use_jit;
        self.regex = compile_pattern(&self.pattern, self.use_pcre2, use_jit)?;
        self.rebuild_chain()?;
        Ok(self)
    }

    /// Enable or disable JIT compilation (non-pcre2 version).
    #[cfg(not(feature = "pcre2"))]
    pub fn jit(mut self, use_jit: bool) -> Result<Self, TokenizerError> {
        self.use_jit = use_jit;
        self.regex = compile_pattern(&self.pattern, self.use_pcre2, use_jit)?;
        self.rebuild_chain()?;
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
            Some(build_special_matcher(&special_token_strings)?)
        };

        // Initialize LRU cache
        let cache_size_nz = NonZeroUsize::new(DEFAULT_CACHE_SIZE.max(1)).unwrap();
        let chunk_cache = Mutex::new(LruCache::new(cache_size_nz));

        Ok(Self {
            encoder,
            merge_ranks: None,
            decoder,
            special_tokens,
            special_tokens_decoder,
            special_token_strings,
            regex: RegexBackend::Regexr(Box::new(regex)),
            pattern: pattern.to_string(),
            chain: Arc::from(Vec::new()),
            chain_patterns: Arc::from(Vec::new()),
            special_matcher,
            chunk_cache,
            use_byte_level: false,
            use_sentencepiece: true,
            add_prefix_space: false,
            pre_tokenizer: None,
            match_added_tokens: false,
            special_decode_ids: rustc_hash::FxHashSet::default(),
            normalizer: None,
            cache_size: DEFAULT_CACHE_SIZE,
            use_jit: true,
            use_pcre2: false,
        })
    }

    /// Split `text` into pre-token spans.
    ///
    /// The overwhelmingly common case is a single pre-tokenizer expression, and
    /// that case is the original code verbatim: one `find_iter` over the whole
    /// text, matches only. The multi-pass machinery costs exactly one
    /// `is_empty()` test per `encode` call — not per chunk, not per byte — and
    /// the chained branch is never entered by a single-expression tokenizer, so
    /// its spans are byte-identical to before.
    #[inline]
    fn split_chunks(&self, text: &str) -> Vec<(usize, usize)> {
        if self.chain.is_empty() {
            return self.regex.find_iter(text);
        }
        self.split_chunks_chained(text)
    }

    /// [`Tokenizer::split_chunks`] for a multi-expression pre-tokenizer.
    ///
    /// Kept out of line so the single-expression path stays a straight call to
    /// `find_iter`. Unlike that path this keeps unmatched gaps as spans, because
    /// llama.cpp's `unicode_regex_split_stl` does — including on the FIRST pass,
    /// whose leftovers a later pass still gets to cut.
    fn split_chunks_chained(&self, text: &str) -> Vec<(usize, usize)> {
        let mut spans = subdivide(&self.regex, text, &[(0, text.len())]);
        for pass in self.chain.iter() {
            spans = subdivide(pass, text, &spans);
        }
        spans
    }

    /// Run BPE on a piece, honoring a separate merge-rank map when present.
    #[inline]
    fn bpe(&self, bytes: &[u8]) -> Vec<u32> {
        match &self.merge_ranks {
            Some(ranks) => byte_pair_encode_with_ranks(bytes, ranks, &self.encoder),
            None => byte_pair_encode(bytes, &self.encoder),
        }
    }

    /// Compute a fast hash for a byte slice to use as an LRU cache key.
    #[inline]
    fn hash_slice(slice: &[u8]) -> u64 {
        let mut hasher = FxHasher::default();
        slice.hash(&mut hasher);
        hasher.finish()
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
        let result = self.bpe(bytes);

        // Store in cache
        if let Ok(mut cache) = self.chunk_cache.lock() {
            cache.put(hash, result.clone());
        }

        result
    }

    /// Encode a single text chunk with LRU caching.
    fn encode_chunk(&self, slice: &[u8]) -> Vec<u32> {
        // Apply ByteLevel preprocessing if enabled. When a pre-tokenizer engine
        // is attached it has already byte-level-encoded the pieces, so we must
        // NOT re-encode here (but `use_byte_level` stays true so `decode` still
        // reverses the byte-level mapping).
        let bytes_to_encode: std::borrow::Cow<[u8]> =
            if self.use_byte_level && self.pre_tokenizer.is_none() {
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
        let result = self.bpe(bytes_to_encode.as_ref());

        // Store in cache
        if let Ok(mut cache) = self.chunk_cache.lock() {
            cache.put(hash, result.clone());
        }

        result
    }

    /// Encode text to token IDs.
    ///
    /// By default special tokens in the input are treated as ordinary text. When
    /// the tokenizer was built with added-token matching (HF `tokenizer.json`
    /// loaders), `added_tokens` are recognized first.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if self.match_added_tokens {
            self.encode_with_special(text)
        } else {
            self.encode_ordinary(text)
        }
    }

    /// Encode text to token IDs, always treating special tokens as ordinary text.
    ///
    /// Uses sequential processing, which is faster than parallel for texts up to ~1MB.
    pub fn encode_ordinary(&self, text: &str) -> Vec<u32> {
        // Apply the HF `normalizer` (e.g. NFC) to content before splitting. This
        // runs on content gaps (special tokens are extracted upstream), matching
        // HuggingFace's extract-then-normalize order.
        let normalized;
        let text = if let Some(norm) = &self.normalizer {
            normalized = norm.normalize(text);
            normalized.as_str()
        } else {
            text
        };

        // Multi-stage pre-tokenizer path (Digits/Punctuation/Sequence/…): the
        // engine produces already byte-level-encoded pieces; BPE each directly.
        if let Some(pt) = &self.pre_tokenizer {
            let mut out = Vec::new();
            for piece in pt.split(text) {
                out.extend(self.encode_chunk(piece.as_bytes()));
            }
            return out;
        }

        let text = self.prefixed(text);
        let text = text.as_ref();
        let text_bytes = text.as_bytes();
        let chunks = self.split_chunks(text);

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
                    self.encode_chunk(slice)
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
        if self.use_sentencepiece || self.pre_tokenizer.is_some() {
            // SentencePiece and the multi-stage pre-tokenizer need sequential logic.
            return self.encode(text);
        }

        let text = self.prefixed(text);
        let text = text.as_ref();
        let text_bytes = text.as_bytes();
        let chunks = self.split_chunks(text);

        if chunks.is_empty() {
            return vec![];
        }

        #[cfg(feature = "rayon")]
        let results: Vec<Vec<u32>> = chunks
            .par_iter()
            .map(|&(start, end)| {
                let slice = &text_bytes[start..end];
                self.encode_chunk(slice)
            })
            .collect();

        #[cfg(not(feature = "rayon"))]
        let results: Vec<Vec<u32>> = chunks
            .iter()
            .map(|&(start, end)| {
                let slice = &text_bytes[start..end];
                self.encode_chunk(slice)
            })
            .collect();

        results.into_iter().flatten().collect()
    }

    /// Encode text with special token handling.
    ///
    /// Special tokens in the input are encoded directly without BPE.
    pub fn encode_with_special(&self, text: &str) -> Vec<u32> {
        let Some(ref special_matcher) = self.special_matcher else {
            return self.encode_ordinary(text);
        };

        let text_bytes = text.as_bytes();
        let mut result = Vec::new();
        let mut last_end = 0;

        for m in special_matcher.find_iter(text_bytes) {
            let start = m.start();
            let end = m.end();

            if start > last_end {
                let slice = &text[last_end..start];
                result.extend(self.encode_ordinary(slice));
            }

            let pattern_idx = m.pattern().as_usize();
            let token_str = &self.special_token_strings[pattern_idx];
            if let Some(&rank) = self.special_tokens.get(token_str) {
                result.push(rank);
            }

            last_end = end;
        }

        if last_end < text.len() {
            result.extend(self.encode_ordinary(&text[last_end..]));
        }

        result
    }

    /// Decode token IDs back to bytes.
    pub fn decode_bytes(&self, tokens: &[u32]) -> Vec<u8> {
        let mut result = Vec::with_capacity(tokens.len() * 4);

        for &token in tokens {
            // Drop `special=true` added tokens (HF default skip_special_tokens).
            if self.special_decode_ids.contains(&token) {
                continue;
            }
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

    /// Batch encode multiple texts (parallel when rayon is enabled).
    pub fn encode_batch(&self, texts: &[String]) -> Vec<Vec<u32>> {
        #[cfg(feature = "rayon")]
        {
            texts.par_iter().map(|text| self.encode(text)).collect()
        }
        #[cfg(not(feature = "rayon"))]
        {
            texts.iter().map(|text| self.encode(text)).collect()
        }
    }

    /// Batch encode multiple texts with special token handling.
    pub fn encode_batch_with_special(&self, texts: &[String]) -> Vec<Vec<u32>> {
        #[cfg(feature = "rayon")]
        {
            texts
                .par_iter()
                .map(|text| self.encode_with_special(text))
                .collect()
        }
        #[cfg(not(feature = "rayon"))]
        {
            texts
                .iter()
                .map(|text| self.encode_with_special(text))
                .collect()
        }
    }

    /// Batch decode multiple token lists.
    pub fn decode_batch(&self, token_lists: &[Vec<u32>]) -> Result<Vec<String>, TokenizerError> {
        #[cfg(feature = "rayon")]
        {
            token_lists
                .par_iter()
                .map(|tokens| self.decode(tokens))
                .collect()
        }
        #[cfg(not(feature = "rayon"))]
        {
            token_lists
                .iter()
                .map(|tokens| self.decode(tokens))
                .collect()
        }
    }

    /// Batch decode multiple token lists, replacing invalid UTF-8.
    pub fn decode_batch_lossy(&self, token_lists: &[Vec<u32>]) -> Vec<String> {
        #[cfg(feature = "rayon")]
        {
            token_lists
                .par_iter()
                .map(|tokens| self.decode_lossy(tokens))
                .collect()
        }
        #[cfg(not(feature = "rayon"))]
        {
            token_lists
                .iter()
                .map(|tokens| self.decode_lossy(tokens))
                .collect()
        }
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
            Some(build_special_matcher(&self.special_token_strings).unwrap())
        };

        Self {
            encoder: self.encoder.clone(),
            merge_ranks: self.merge_ranks.clone(),
            decoder: self.decoder.clone(),
            special_tokens: self.special_tokens.clone(),
            special_tokens_decoder: self.special_tokens_decoder.clone(),
            special_token_strings: self.special_token_strings.clone(),
            regex,
            pattern: self.pattern.clone(),
            // The later passes are immutable once compiled, so a clone shares
            // them rather than repeating the (fallible) compilation.
            chain: Arc::clone(&self.chain),
            chain_patterns: Arc::clone(&self.chain_patterns),
            special_matcher,
            chunk_cache,
            use_byte_level: self.use_byte_level,
            use_sentencepiece: self.use_sentencepiece,
            add_prefix_space: self.add_prefix_space,
            pre_tokenizer: self.pre_tokenizer.clone(),
            match_added_tokens: self.match_added_tokens,
            special_decode_ids: self.special_decode_ids.clone(),
            normalizer: self.normalizer.clone(),
            cache_size: self.cache_size,
            use_jit: self.use_jit,
            use_pcre2: self.use_pcre2,
        }
    }
}

impl super::tokenize::Tokenize for Tokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.encode(text)
    }

    fn decode(&self, ids: &[u32]) -> Result<String, super::tokenize::TokenizeError> {
        self.decode(ids).map_err(|e| match e {
            TokenizerError::Utf8Error => super::tokenize::TokenizeError::Utf8Error,
            other => super::tokenize::TokenizeError::Other(other.to_string()),
        })
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size()
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

    // ── Multi-pass pre-tokenizer (llama.cpp `unicode_regex_split`) ───────────

    /// Build a tokenizer over `patterns` and report the pieces it splits `text`
    /// into, so a pass composition can be asserted as text rather than ids.
    fn pieces(patterns: &[&str], text: &str) -> Vec<String> {
        let tokenizer =
            Tokenizer::new_byte_level_chain(FxHashMap::default(), FxHashMap::default(), patterns)
                .expect("patterns compile");
        tokenizer
            .split_chunks(text)
            .into_iter()
            .filter_map(|(s, e)| text.get(s..e).map(str::to_owned))
            .collect()
    }

    /// A one-expression list must take the single-regex path and behave exactly
    /// like the plain constructor — matches only, unmatched text dropped.
    #[test]
    fn single_expression_list_keeps_the_original_split() {
        let one = Tokenizer::new_byte_level_chain(
            FxHashMap::default(),
            FxHashMap::default(),
            &[GPT2_PATTERN],
        )
        .expect("compiles");
        assert!(
            one.chain.is_empty(),
            "a one-expression list must not engage the chained path"
        );

        let plain =
            Tokenizer::new_byte_level(FxHashMap::default(), FxHashMap::default(), GPT2_PATTERN)
                .expect("compiles");
        let text = "Hello, world! 1234\n\n  trailing";
        assert_eq!(one.split_chunks(text), plain.split_chunks(text));
    }

    /// The defining property: a later pass only subdivides what an earlier pass
    /// produced. `\p{N}` first cuts every digit apart, so the GPT-2 split's
    /// ` ?\p{N}+` can no longer take `123` as one piece — which is precisely why
    /// `starcoder` is not the GPT-2 pre-tokenizer.
    #[test]
    fn later_pass_subdivides_earlier_pieces_and_cannot_re_merge() {
        // One expression: ` ?\p{N}+` takes the whole digit run with its space.
        assert_eq!(pieces(&[GPT2_PATTERN], "abc 123"), vec!["abc", " 123"]);
        // Two: `\p{N}` has already cut the digits apart AND left `"abc "` as a
        // gap, so pass 2 can only split that gap — it can never reunite the
        // space with a digit.
        assert_eq!(
            pieces(&[r"\p{N}", GPT2_PATTERN], "abc 123"),
            vec!["abc", " ", "1", "2", "3"],
        );
    }

    /// Text a pass leaves unmatched is kept as a piece of its own rather than
    /// dropped, and stays eligible for the passes that follow.
    #[test]
    fn unmatched_gaps_are_kept_and_still_subdivided() {
        // Pass 1 matches only the digits; the letters survive as gaps. Pass 2
        // then cuts those gaps on the letter/space boundary.
        assert_eq!(
            pieces(&[r"\p{N}+", r"\p{L}+"], "ab12cd"),
            vec!["ab", "12", "cd"],
        );
        // With no second pass the same gaps are still pieces, not losses.
        assert_eq!(
            pieces(&[r"\p{N}+", r"\p{N}+"], "ab12cd"),
            vec!["ab", "12", "cd"]
        );
    }

    /// Each pass sees one span in isolation, so an anchor or lookahead resolves
    /// against the span's edges — llama.cpp matches over `[start, start+offset)`
    /// only (unicode.cpp:487). Here pass 1 isolates the digits, and `^.` in
    /// pass 2 therefore fires inside EVERY resulting span, not once per text.
    #[test]
    fn each_pass_matches_within_a_span_not_across_the_text() {
        assert_eq!(
            pieces(&[r"\p{N}+", r"^."], "ab12cd"),
            vec!["a", "b", "1", "2", "c", "d"],
        );
    }

    /// Falcon's three passes compose: punctuation runs first, then the GPT-2
    /// split inside the remaining pieces, then digit runs chopped into threes
    /// from the left of each piece pass 2 produced.
    #[test]
    fn falcon_three_pass_composition() {
        let falcon = [r"[\p{P}\$\+<=>\^~\|`]+", GPT2_PATTERN, r"[0-9][0-9][0-9]"];
        assert_eq!(pieces(&falcon, "a=1234"), vec!["a", "=", "123", "4"]);
        // The alternation of the same three expressions cannot do this: it takes
        // `1234` whole via ` ?\p{N}+` and never revisits it.
        assert_eq!(
            pieces(&[r"[\p{P}\$\+<=>\^~\|`]+|'s| ?\p{L}+| ?\p{N}+"], "a=1234"),
            vec!["a", "=", "1234"],
        );
    }

    /// An empty list has no first expression to compile and is refused rather
    /// than silently becoming a no-op split.
    #[test]
    fn empty_pattern_list_is_refused() {
        assert!(matches!(
            Tokenizer::new_byte_level_chain(FxHashMap::default(), FxHashMap::default(), &[]),
            Err(TokenizerError::EmptyPatternList)
        ));
    }

    /// Switching JIT recompiles the later passes too, so the split is unchanged.
    #[test]
    fn toggling_jit_preserves_a_chained_split() {
        let patterns = [r"\p{N}", GPT2_PATTERN];
        let tokenizer =
            Tokenizer::new_byte_level_chain(FxHashMap::default(), FxHashMap::default(), &patterns)
                .expect("compiles");
        let text = "abc 123";
        let before = tokenizer.split_chunks(text);
        let tokenizer = tokenizer.jit(false).expect("recompiles");
        assert_eq!(tokenizer.chain.len(), 1);
        assert_eq!(tokenizer.split_chunks(text), before);
    }

    /// Cloning shares the compiled passes and keeps the split identical.
    #[test]
    fn cloning_preserves_a_chained_split() {
        let patterns = [r"\p{N}", GPT2_PATTERN];
        let tokenizer =
            Tokenizer::new_byte_level_chain(FxHashMap::default(), FxHashMap::default(), &patterns)
                .expect("compiles");
        let text = "abc 123";
        assert_eq!(
            tokenizer.clone().split_chunks(text),
            tokenizer.split_chunks(text)
        );
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
