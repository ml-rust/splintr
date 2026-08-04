use super::backend::{compile_pattern, RegexBackend};
use super::error::TokenizerError;
use super::types::{ByteFallback, Tokenizer};
use crate::core::added::{AddedTokenSet, AddedTokens};
use crate::core::vocab::{build_decoder, load_tiktoken_bpe, load_tiktoken_bpe_file};
use lru::LruCache;
use regexr::RegexBuilder;
use rustc_hash::{FxHashMap, FxHasher};
use std::hash::BuildHasherDefault;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};

/// Default cache size for encoded chunks
const DEFAULT_CACHE_SIZE: usize = 4096;

impl Tokenizer {
    /// Create a new tokenizer from encoder map, special tokens, and regex pattern.
    ///
    /// Uses regexr as the default regex backend.
    ///
    /// # Arguments
    /// * `encoder` - Map of byte sequences to token IDs
    /// * `special_tokens` - The added tokens: an [`AddedTokenSet`] when the
    ///   `lstrip`/`rstrip` flags matter (a `tokenizer.json`), or a plain name→id
    ///   map when they cannot be declared at all (tiktoken vocabularies, GGUF)
    /// * `pattern` - Regex pattern for tokenization
    pub fn new(
        encoder: FxHashMap<Vec<u8>, u32>,
        special_tokens: impl Into<AddedTokenSet>,
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
        special_tokens: impl Into<AddedTokenSet>,
        pattern: &str,
    ) -> Result<Self, TokenizerError> {
        Self::with_options(encoder, special_tokens, pattern, DEFAULT_CACHE_SIZE, true)
    }

    /// Create a ByteLevel tokenizer whose pre-tokenizer is a SEQUENCE of
    /// expressions applied in order, llama.cpp's `regex_exprs` list.
    ///
    /// Each expression subdivides the pieces the previous one produced rather
    /// than re-reading the whole text, and the gaps a pass leaves unmatched stay
    /// as pieces of their own — see `subdivide` for the exact semantics and
    /// their source. A one-element list is exactly [`Tokenizer::new_byte_level`]
    /// and keeps the single-regex fast path, so callers can pass a list
    /// unconditionally without paying for the general machinery.
    ///
    /// Vocabularies that need this cannot be expressed as one alternation:
    /// `falcon` splits punctuation runs, then applies the GPT-2 split to the
    /// pieces, then cuts digit runs into groups of three.
    pub fn new_byte_level_chain(
        encoder: FxHashMap<Vec<u8>, u32>,
        special_tokens: impl Into<AddedTokenSet>,
        patterns: &[&str],
    ) -> Result<Self, TokenizerError> {
        let (first, rest) = Self::split_chain_patterns(patterns)?;
        let mut tokenizer = Self::new_byte_level(encoder, special_tokens, first)?;
        tokenizer.set_chain(rest)?;
        Ok(tokenizer)
    }

    /// Create a tokenizer whose pre-tokenizer is a SEQUENCE of expressions
    /// applied in order, llama.cpp's `regex_exprs` list.
    ///
    /// Identical to [`Tokenizer::new_byte_level_chain`] except the head
    /// tokenizer is built with [`Tokenizer::new`] rather than
    /// [`Tokenizer::new_byte_level`], for vocabularies that do not use
    /// ByteLevel encoding.
    pub fn new_chain(
        encoder: FxHashMap<Vec<u8>, u32>,
        special_tokens: impl Into<AddedTokenSet>,
        patterns: &[&str],
    ) -> Result<Self, TokenizerError> {
        let (first, rest) = Self::split_chain_patterns(patterns)?;
        let mut tokenizer = Self::new(encoder, special_tokens, first)?;
        tokenizer.set_chain(rest)?;
        Ok(tokenizer)
    }

    /// Split a pre-tokenizer pattern list into its head (compiled as the
    /// primary regex) and the remaining passes (installed via [`Self::set_chain`]).
    fn split_chain_patterns<'a>(
        patterns: &'a [&'a str],
    ) -> Result<(&'a str, &'a [&'a str]), TokenizerError> {
        patterns
            .split_first()
            .map(|(first, rest)| (*first, rest))
            .ok_or(TokenizerError::EmptyPatternList)
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

    /// Create a new tokenizer with a metaspace (▁) decoder enabled.
    ///
    /// This is byte-level BPE, not SentencePiece — despite the historical name,
    /// it is the linked-list BPE algorithm in this file with a decode-time
    /// ▁ (U+2581) → space substitution bolted on. It is required for Mistral,
    /// Gemma, and similar tokenizers that use ▁ as a word-boundary marker in
    /// their vocab. For real SentencePiece (Unigram, Viterbi decoding) use
    /// [`crate::core::sentencepiece::SentencePieceTokenizer`]; for SPM-BPE
    /// (merge-by-rank) vocabularies use [`crate::core::spm::SpmTokenizer`].
    pub fn new_with_metaspace_decoder(
        encoder: FxHashMap<Vec<u8>, u32>,
        special_tokens: impl Into<AddedTokenSet>,
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
        special_tokens: impl Into<AddedTokenSet>,
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
        special_tokens: impl Into<AddedTokenSet>,
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

    /// Create a new tokenizer with all configuration options including the
    /// metaspace decoder.
    ///
    /// # Arguments
    /// * `encoder` - Map of byte sequences to token IDs
    /// * `special_tokens` - Map of special token strings to token IDs
    /// * `pattern` - Regex pattern for tokenization
    /// * `cache_size` - Size of the LRU cache for encoded chunks
    /// * `use_byte_level` - Enable ByteLevel encoding for GPT-2/Llama/DeepSeek style tokenizers
    /// * `use_metaspace_decoder` - Enable the metaspace decoder (▁ → space during decode);
    ///   NOT SentencePiece, see [`Tokenizer::new_with_metaspace_decoder`]
    pub fn with_full_options(
        encoder: FxHashMap<Vec<u8>, u32>,
        special_tokens: impl Into<AddedTokenSet>,
        pattern: &str,
        cache_size: usize,
        use_byte_level: bool,
        use_metaspace_decoder: bool,
    ) -> Result<Self, TokenizerError> {
        // Build decoder maps
        let decoder = build_decoder(&encoder);

        // Compile regex with regexr (default backend)
        let regex = Arc::new(compile_pattern(pattern, false, true)?);

        // Build the special-token matcher (shared with the other backends) from
        // the declared set — the only place the `lstrip`/`rstrip` flags are
        // consulted — then reduce the set to the plain name→id map the decode
        // tables speak. Reducing *after* building means the flags never have to
        // be carried in a second field that could drift out of step with it.
        let added: AddedTokenSet = special_tokens.into();
        let special_matcher = AddedTokens::new(&added)?;
        let special_tokens = added.into_id_map();
        let special_tokens_decoder: FxHashMap<u32, String> = special_tokens
            .iter()
            .map(|(k, v)| (*v, k.clone()))
            .collect();

        // Initialize LRU cache
        // `.max(1)` already guarantees a nonzero value; the fallback is unreachable.
        let cache_size_nz = NonZeroUsize::new(cache_size.max(1)).unwrap_or(NonZeroUsize::MIN);
        let chunk_cache = Mutex::new(LruCache::with_hasher(
            cache_size_nz,
            BuildHasherDefault::<FxHasher>::default(),
        ));

        Ok(Self {
            encoder,
            merge_ranks: None,
            decoder,
            special_tokens,
            special_tokens_decoder,
            regex,
            pattern: pattern.to_string(),
            chain: Arc::from(Vec::new()),
            chain_patterns: Arc::from(Vec::new()),
            special_matcher,
            chunk_cache,
            use_byte_level,
            use_metaspace_decoder,
            add_prefix_space: false,
            pre_tokenizer: None,
            match_added_tokens: false,
            special_decode_ids: rustc_hash::FxHashSet::default(),
            normalizer: None,
            cache_size,
            use_jit: true,
            use_pcre2: false,
            byte_fallback: None,
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

    /// Attach a multi-stage pre-tokenizer pipeline. When the pipeline contains a
    /// `ByteLevel` stage the engine byte-encodes the pieces itself and `encode`
    /// skips re-encoding, so this tokenizer's `use_byte_level` governs only
    /// `decode` — set it to match the engine. An empty pipeline is treated as
    /// absent.
    pub fn with_pre_tokenizer(mut self, pt: crate::core::pretokenizer::PreTokenizer) -> Self {
        self.pre_tokenizer = (!pt.is_empty()).then(|| std::sync::Arc::new(pt));
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
    pub fn with_normalizer(mut self, normalizer: crate::core::normalizer::Normalizer) -> Self {
        self.normalizer = (!normalizer.is_empty()).then(|| std::sync::Arc::new(normalizer));
        self
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
        self.regex = Arc::new(compile_pattern(&self.pattern, use_pcre2, self.use_jit)?);
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
        self.regex = Arc::new(compile_pattern(&self.pattern, self.use_pcre2, use_jit)?);
        self.rebuild_chain()?;
        Ok(self)
    }

    /// Enable or disable JIT compilation (non-pcre2 version).
    #[cfg(not(feature = "pcre2"))]
    pub fn jit(mut self, use_jit: bool) -> Result<Self, TokenizerError> {
        self.use_jit = use_jit;
        self.regex = Arc::new(compile_pattern(&self.pattern, self.use_pcre2, use_jit)?);
        self.rebuild_chain()?;
        Ok(self)
    }

    /// Create a tokenizer from a tiktoken vocabulary file.
    pub fn from_file(
        vocab_path: &str,
        pattern: &str,
        special_tokens: impl Into<AddedTokenSet>,
    ) -> Result<Self, TokenizerError> {
        let encoder = load_tiktoken_bpe_file(vocab_path)?;
        Self::new(encoder, special_tokens, pattern)
    }

    /// Create a tokenizer from raw vocabulary bytes.
    pub fn from_bytes(
        vocab_data: &[u8],
        pattern: &str,
        special_tokens: impl Into<AddedTokenSet>,
    ) -> Result<Self, TokenizerError> {
        let encoder = load_tiktoken_bpe(vocab_data)?;
        Self::new(encoder, special_tokens, pattern)
    }

    /// Create a tokenizer from raw vocabulary bytes with ByteLevel encoding.
    pub fn from_bytes_byte_level(
        vocab_data: &[u8],
        pattern: &str,
        special_tokens: impl Into<AddedTokenSet>,
    ) -> Result<Self, TokenizerError> {
        let encoder = load_tiktoken_bpe(vocab_data)?;
        Self::new_byte_level(encoder, special_tokens, pattern)
    }

    /// Create a tokenizer from raw vocabulary bytes with a chained pre-tokenizer
    /// pattern sequence. See [`Tokenizer::new_chain`].
    pub fn from_bytes_chain(
        vocab_data: &[u8],
        patterns: &[&str],
        special_tokens: impl Into<AddedTokenSet>,
    ) -> Result<Self, TokenizerError> {
        let encoder = load_tiktoken_bpe(vocab_data)?;
        Self::new_chain(encoder, special_tokens, patterns)
    }

    /// Create a tokenizer from raw vocabulary bytes with ByteLevel encoding and
    /// a chained pre-tokenizer pattern sequence. See [`Tokenizer::new_byte_level_chain`].
    pub fn from_bytes_byte_level_chain(
        vocab_data: &[u8],
        patterns: &[&str],
        special_tokens: impl Into<AddedTokenSet>,
    ) -> Result<Self, TokenizerError> {
        let encoder = load_tiktoken_bpe(vocab_data)?;
        Self::new_byte_level_chain(encoder, special_tokens, patterns)
    }

    /// Create a tokenizer from raw vocabulary bytes with the metaspace decoder.
    ///
    /// This is byte-level BPE, not SentencePiece — see
    /// [`Tokenizer::new_with_metaspace_decoder`]. It converts ▁ (U+2581) to
    /// space during decoding. Used for Mistral, Gemma, and similar tokenizers.
    pub fn from_bytes_with_metaspace_decoder(
        vocab_data: &[u8],
        pattern: &str,
        special_tokens: impl Into<AddedTokenSet>,
    ) -> Result<Self, TokenizerError> {
        let encoder = load_tiktoken_bpe(vocab_data)?;
        Self::new_with_metaspace_decoder(encoder, special_tokens, pattern)
    }

    /// Create a metaspace-decoder BPE tokenizer (see
    /// [`Tokenizer::new_with_metaspace_decoder`] — NOT SentencePiece) with an
    /// explicit decoder to preserve all token IDs.
    ///
    /// This is used for vocabs with duplicate byte sequences (like Mistral V2 where byte fallback
    /// tokens may duplicate BPE merges). The decoder preserves ALL token IDs, while the encoder
    /// only keeps the lowest ID for each byte sequence.
    pub fn from_bytes_with_metaspace_decoder_preserving_ids(
        vocab_data: &[u8],
        pattern: &str,
        special_tokens: impl Into<AddedTokenSet>,
    ) -> Result<Self, TokenizerError> {
        use crate::core::vocab::load_tiktoken_bpe_with_decoder;
        let (encoder, mut decoder) = load_tiktoken_bpe_with_decoder(vocab_data)?;

        // Compile regex
        let regex = RegexBuilder::new(pattern).jit(true).build()?;

        // Build the special-token matcher (shared with the other backends) from
        // the declared set, then reduce it to the plain name→id map the decode
        // tables speak — see `with_full_options`.
        let added: AddedTokenSet = special_tokens.into();
        let special_matcher = AddedTokens::new(&added)?;
        let special_tokens = added.into_id_map();

        // Add special tokens to decoder
        for (token_str, id) in &special_tokens {
            decoder.insert(*id, token_str.as_bytes().to_vec());
        }

        // Build the tokenizer manually with explicit decoder
        let special_tokens_decoder: FxHashMap<u32, String> = special_tokens
            .iter()
            .map(|(k, v)| (*v, k.clone()))
            .collect();

        // Initialize LRU cache
        // `.max(1)` already guarantees a nonzero value; the fallback is unreachable.
        let cache_size_nz =
            NonZeroUsize::new(DEFAULT_CACHE_SIZE.max(1)).unwrap_or(NonZeroUsize::MIN);
        let chunk_cache = Mutex::new(LruCache::with_hasher(
            cache_size_nz,
            BuildHasherDefault::<FxHasher>::default(),
        ));

        Ok(Self {
            encoder,
            merge_ranks: None,
            decoder,
            special_tokens,
            special_tokens_decoder,
            regex: Arc::new(RegexBackend::Regexr(Box::new(regex))),
            pattern: pattern.to_string(),
            chain: Arc::from(Vec::new()),
            chain_patterns: Arc::from(Vec::new()),
            special_matcher,
            chunk_cache,
            use_byte_level: false,
            use_metaspace_decoder: true,
            add_prefix_space: false,
            pre_tokenizer: None,
            match_added_tokens: false,
            special_decode_ids: rustc_hash::FxHashSet::default(),
            normalizer: None,
            cache_size: DEFAULT_CACHE_SIZE,
            use_jit: true,
            use_pcre2: false,
            byte_fallback: None,
        })
    }

    /// Attach the [`ByteFallback`] resolution, so a BPE piece the merge cannot
    /// represent is emitted through its `<0xNN>`/`<unk>` ids instead of being
    /// dropped. `None` disables it — the correct choice for any vocabulary that
    /// declares no byte fallback, and *required* for every ByteLevel BPE model:
    /// `Tokenizer::bpe` discards `byte_fallback` outright whenever
    /// `use_byte_level` is true (the `<0xNN>` table is keyed by RAW byte value,
    /// the wrong space once input has already been byte-level-encoded), so a
    /// `Some` here would never fire and callers should not build one (see
    /// `build_bpe` in `src/core/hf_json/loader.rs`, which skips the call).
    pub fn with_byte_fallback(mut self, byte_fallback: Option<ByteFallback>) -> Self {
        self.byte_fallback = byte_fallback;
        self
    }

    /// Derive a [`ByteFallback`] from an encoder and a resolved `unk` id, by
    /// looking up the 256 `<0xNN>` token spellings HuggingFace byte-fallback
    /// vocabularies declare. Mirrors `SpmTokenizer::new`'s identical lookup
    /// over its own vocab (see `src/core/spm.rs`) so the two backends agree on
    /// the table's construction.
    ///
    /// A partial set is kept as a partial set: HuggingFace resolves fallback
    /// per character, emitting `<0xNN>` where the entry exists and the unk id
    /// where it does not, so a vocabulary declaring only some `<0xNN>` entries
    /// is a valid file. Returns `None` only when neither half exists (no byte
    /// entries and no unk id), where there is nothing to fall back *to* and
    /// dropping — the no-fallback behavior — is already the answer.
    ///
    /// `declares_byte_fallback` is the model's own `byte_fallback` flag and
    /// gates the `<0xNN>` table ONLY. When it is false the table is left empty
    /// even if the vocabulary spells `<0xNN>` tokens — HuggingFace's BPE model
    /// consults them only under the flag — while `unk_id` still applies, since
    /// the unk branch is not gated on the flag at all (measured against
    /// `tokenizers` 0.22.1; see `build_bpe` in `src/core/hf_json/loader.rs`).
    pub(crate) fn byte_fallback_from_encoder(
        encoder: &FxHashMap<Vec<u8>, u32>,
        unk_id: Option<u32>,
        declares_byte_fallback: bool,
    ) -> Option<ByteFallback> {
        let mut byte_ids = [None; 256];
        let mut any = false;
        if declares_byte_fallback {
            for (b, slot) in byte_ids.iter_mut().enumerate() {
                *slot = encoder.get(format!("<0x{b:02X}>").as_bytes()).copied();
                any |= slot.is_some();
            }
        }
        (any || unk_id.is_some()).then(|| ByteFallback::new(byte_ids, unk_id))
    }
}
