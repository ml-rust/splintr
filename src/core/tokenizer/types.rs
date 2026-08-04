use crate::core::added::AddedTokens;
use crate::core::policy::{PolicyError, SpecialMode};
use lru::LruCache;
use rustc_hash::{FxHashMap, FxHasher};
use std::hash::BuildHasherDefault;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

use super::backend::RegexBackend;
use super::error::TokenizerError;

/// How a character the BPE merge vocabulary cannot represent is rendered.
///
/// HuggingFace resolves this **per character**, not through an all-or-nothing
/// 256-entry table: a character whose every byte has a `<0xNN>` entry is
/// emitted as those byte tokens, and otherwise the whole character collapses to
/// a single `model.unk_token` id. Either half may be missing — a vocabulary
/// with only some `<0xNN>` entries is a valid file, not a malformed one — so
/// both are optional here and a character neither half can render is dropped,
/// which is `byte_pair_encode_pieces`' behavior without any fallback at all.
///
/// An entirely empty `byte_ids` with a `Some(unk_id)` is the ordinary shape for
/// a `tokenizer.json` that declares `model.unk_token` without
/// `model.byte_fallback`: HuggingFace gates the `<0xNN>` branch on that flag but
/// never gates the unk branch on it.
#[derive(Clone, Debug)]
pub struct ByteFallback {
    /// Token id per byte value, `None` where the vocabulary declares no
    /// `<0xNN>` entry for that byte.
    pub(super) byte_ids: Box<[Option<u32>; 256]>,
    /// `model.unk_token`'s id, for a character the byte entries cannot cover.
    pub(super) unk_id: Option<u32>,
}

impl ByteFallback {
    /// Build a fallback from a per-byte `<0xNN>` id table and an unk id.
    pub fn new(byte_ids: [Option<u32>; 256], unk_id: Option<u32>) -> Self {
        Self {
            byte_ids: Box::new(byte_ids),
            unk_id,
        }
    }
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
/// - Linked-list BPE with a binary heap for merge selection (O(N log N))
/// - FxHashMap for fast lookups
/// - Aho-Corasick for fast multi-pattern special token matching
/// - LRU cache for frequently encoded chunks
/// - Optional ByteLevel encoding for GPT-2/Llama/DeepSeek style tokenizers
/// - Optional metaspace decoder for Mistral/Gemma style tokenizers (▁ → space)
pub struct Tokenizer {
    pub(super) encoder: FxHashMap<Vec<u8>, u32>,
    /// Optional separate merge-priority map (bytes → merge rank). When present,
    /// BPE merges by this rank instead of by token id — required for HuggingFace
    /// BPE models whose ids don't follow merge order (e.g. RoBERTa). `None`
    /// means tiktoken-style (id doubles as merge rank).
    pub(super) merge_ranks: Option<FxHashMap<Vec<u8>, u32>>,
    pub(super) decoder: FxHashMap<u32, Vec<u8>>,
    pub(super) special_tokens: FxHashMap<String, u32>,
    pub(super) special_tokens_decoder: FxHashMap<u32, String>,
    /// Behind an `Arc` so cloning a tokenizer shares the compiled regex
    /// instead of recompiling it (and re-running JIT on the pcre2 backend).
    pub(super) regex: Arc<RegexBackend>,
    pub(super) pattern: String,
    /// Pre-tokenizer expressions applied AFTER [`Tokenizer::regex`], in order —
    /// llama.cpp's `regex_exprs` list beyond its first entry (see [`subdivide`]).
    ///
    /// Empty for a single-expression pre-tokenizer, which is every tiktoken-style
    /// vocabulary and the throughput-critical case; [`Tokenizer::split_chunks`]
    /// then runs the untouched single-regex path. Behind an `Arc` so cloning a
    /// tokenizer shares the compiled passes instead of recompiling them.
    pub(super) chain: Arc<[RegexBackend]>,
    /// Source expressions for [`Tokenizer::chain`], kept so switching backend or
    /// JIT recompiles the later passes the same way it recompiles the first.
    pub(super) chain_patterns: Arc<[String]>,
    /// Special-token matcher shared with the SentencePiece/SPM/WordPiece
    /// backends (see [`crate::core::added::AddedTokens`]) — leftmost-longest
    /// Aho-Corasick over `special_tokens`, `None` when it's empty.
    pub(super) special_matcher: Option<AddedTokens>,
    /// Keyed by the chunk bytes themselves (not a bare hash) so a hash
    /// collision cannot return another chunk's token ids — the `lru` crate
    /// hashes AND compares the key, making a wrong-chunk hit structurally
    /// impossible. FxHash stays as the hasher for throughput on this hot
    /// path; only the key type changed.
    pub(super) chunk_cache: Mutex<LruCache<Vec<u8>, Vec<u32>, BuildHasherDefault<FxHasher>>>,
    pub(super) use_byte_level: bool,
    /// BPE with a metaspace (▁) decoder — see [`Tokenizer::new_with_metaspace_decoder`].
    /// This is NOT SentencePiece (Unigram/Viterbi); for that use
    /// [`crate::core::sentencepiece::SentencePieceTokenizer`] or
    /// [`crate::core::spm::SpmTokenizer`].
    pub(super) use_metaspace_decoder: bool,
    /// Prepend a space to input before tokenizing (HF ByteLevel `add_prefix_space`).
    pub(super) add_prefix_space: bool,
    /// Optional multi-stage pre-tokenizer pipeline (HF `pre_tokenizer` graphs
    /// beyond a single regex, e.g. Digits/Punctuation/Sequence). When set, it
    /// produces the (already byte-level-encoded) chunks instead of the regex.
    pub(super) pre_tokenizer: Option<std::sync::Arc<crate::core::pretokenizer::PreTokenizer>>,
    /// When true, `encode` first matches `special_tokens` (HF always recognizes
    /// added tokens in the input).
    pub(super) match_added_tokens: bool,
    /// Ids of `special=true` added tokens to drop on decode (HF default
    /// skip_special_tokens=true). Non-special added tokens are still rendered.
    pub(super) special_decode_ids: rustc_hash::FxHashSet<u32>,
    /// Optional text normalizer (HF `normalizer`, e.g. NFC) applied to content
    /// before splitting. Applied per content gap, never to special-token matches.
    pub(super) normalizer: Option<std::sync::Arc<crate::core::normalizer::Normalizer>>,
    pub(super) cache_size: usize,
    pub(super) use_jit: bool,
    pub(super) use_pcre2: bool,
    /// `<0xNN>`/`<unk>` resolution for a piece the merge cannot represent, so
    /// it is emitted through those ids instead of being dropped. `None` when
    /// the vocabulary declares no byte fallback — every ByteLevel BPE model,
    /// which has full alphabet coverage and needs none.
    pub(super) byte_fallback: Option<ByteFallback>,
}

impl Clone for Tokenizer {
    fn clone(&self) -> Self {
        // The compiled regex is immutable once built (every backend/JIT toggle
        // replaces the `Arc` rather than mutating through it), so a clone
        // shares it instead of recompiling — and, on the pcre2 path, re-JITing.
        let regex = Arc::clone(&self.regex);

        // Create a new empty cache (caches are not shared).
        // `.max(1)` guarantees the value is already >= 1, so `new` cannot
        // fail; `unwrap_or` avoids an unwrap on the (unreachable) None arm.
        let cache_size_nz = NonZeroUsize::new(self.cache_size.max(1)).unwrap_or(NonZeroUsize::MIN);
        let chunk_cache = Mutex::new(LruCache::with_hasher(
            cache_size_nz,
            BuildHasherDefault::<FxHasher>::default(),
        ));

        // Clone the already-built matcher directly: `AddedTokens` is `Clone`,
        // rebuilding here would mean this infallible `Clone` impl would need to
        // swallow a hypothetical build failure (or panic on it), and it's
        // strictly cheaper to clone the automaton than to rebuild it.
        let special_matcher = self.special_matcher.clone();

        Self {
            encoder: self.encoder.clone(),
            merge_ranks: self.merge_ranks.clone(),
            decoder: self.decoder.clone(),
            special_tokens: self.special_tokens.clone(),
            special_tokens_decoder: self.special_tokens_decoder.clone(),
            regex,
            pattern: self.pattern.clone(),
            // The later passes are immutable once compiled, so a clone shares
            // them rather than repeating the (fallible) compilation.
            chain: Arc::clone(&self.chain),
            chain_patterns: Arc::clone(&self.chain_patterns),
            special_matcher,
            chunk_cache,
            use_byte_level: self.use_byte_level,
            use_metaspace_decoder: self.use_metaspace_decoder,
            add_prefix_space: self.add_prefix_space,
            pre_tokenizer: self.pre_tokenizer.clone(),
            match_added_tokens: self.match_added_tokens,
            special_decode_ids: self.special_decode_ids.clone(),
            normalizer: self.normalizer.clone(),
            cache_size: self.cache_size,
            use_jit: self.use_jit,
            use_pcre2: self.use_pcre2,
            byte_fallback: self.byte_fallback.clone(),
        }
    }
}

impl crate::core::tokenize::Tokenize for Tokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.encode(text)
    }

    fn encode_with(&self, text: &str, mode: &SpecialMode<'_>) -> Result<Vec<u32>, PolicyError> {
        self.encode_with(text, mode)
    }

    fn decode(&self, ids: &[u32]) -> Result<String, crate::core::tokenize::TokenizeError> {
        self.decode(ids).map_err(|e| match e {
            TokenizerError::Utf8Error => crate::core::tokenize::TokenizeError::Utf8Error,
            TokenizerError::InvalidTokenId(id) => {
                crate::core::tokenize::TokenizeError::InvalidTokenId(id)
            }
            other => crate::core::tokenize::TokenizeError::Other(other.to_string()),
        })
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size()
    }
}

impl Tokenizer {
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

    /// Whether this tokenizer has a [`ByteFallback`] configured, so a piece the
    /// BPE merge cannot represent is emitted through its `<0xNN>`/`<unk>` ids
    /// instead of being dropped.
    pub fn has_byte_fallback(&self) -> bool {
        self.byte_fallback.is_some()
    }
}
