use super::backend::subdivide;
use super::types::Tokenizer;
use crate::core::added::AddedTokens;
use crate::core::bpe::{byte_pair_encode_pieces, byte_pair_encode_pieces_seeded, Piece};
use crate::core::byte_level::byte_level_encode;
use crate::core::policy::{PolicyError, SpecialMode};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

impl Tokenizer {
    /// Apply `add_prefix_space` to an input, borrowing when no change is needed.
    #[inline]
    fn prefixed<'a>(&self, text: &'a str) -> std::borrow::Cow<'a, str> {
        if self.add_prefix_space && !text.starts_with(|c: char| c.is_whitespace()) {
            std::borrow::Cow::Owned(format!(" {text}"))
        } else {
            std::borrow::Cow::Borrowed(text)
        }
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
    pub(super) fn split_chunks(&self, text: &str) -> Vec<(usize, usize)> {
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

    /// Run BPE on a piece, honoring a separate merge-rank map when present,
    /// and mapping any span the vocabulary could not represent through the
    /// `<0xNN>` byte-fallback table when one is configured (instead of
    /// silently dropping it, `byte_pair_encode_with_ranks`'s contract).
    ///
    /// **Byte-space note:** `bytes` is in RAW input-byte space when
    /// `use_byte_level` is false, but is the UTF-8 of the *ByteLevel-encoded*
    /// text when `use_byte_level` is true (see `encode_chunk`). The
    /// `<0xNN>` table maps a raw byte value to its fallback token id, so
    /// mapping a `Piece::Unresolved` span through it in ByteLevel space would
    /// emit the wrong id. This is guarded explicitly below (`!self.use_byte_level`)
    /// rather than relied on by construction: in practice every ByteLevel
    /// vocabulary has full 256-char alphabet coverage, so `Unresolved` never
    /// actually occurs for them today, but a future gap in one must still
    /// drop the byte (matching prior behavior) rather than silently emit a
    /// fallback id computed in the wrong byte space.
    #[inline]
    fn bpe(&self, bytes: &[u8]) -> Vec<u32> {
        let table = (!self.use_byte_level)
            .then_some(self.byte_fallback_ids.as_deref())
            .flatten();

        // Seed BPE by character when and only when this vocabulary has its own
        // merge list AND is not ByteLevel. Both conjuncts are load-bearing:
        //
        // - `merge_ranks.is_some()` alone would not do: `!use_byte_level` is
        //   also true of the bundled tiktoken vocabularies (cl100k_base,
        //   o200k_base, llama3, deepseek_v3), whose merges genuinely operate on
        //   bytes and whose vocabularies contain tokens that are not valid
        //   UTF-8 at all — character seeding could never produce those. They
        //   all have `merge_ranks == None`, so the first conjunct excludes them.
        // - `!use_byte_level` is required because the ByteLevel HF-json and
        //   GGUF-gpt2 routes DO carry merge ranks, but `bytes` is then in
        //   ByteLevel space (see the byte-space note above), whose alphabet is
        //   entirely ≤2 UTF-8 bytes and therefore unaffected by the ≥3-byte
        //   stranding character seeding exists to fix.
        let pieces = match &self.merge_ranks {
            Some(ranks) => {
                byte_pair_encode_pieces_seeded(bytes, ranks, &self.encoder, !self.use_byte_level)
            }
            None => byte_pair_encode_pieces(bytes, &self.encoder, &self.encoder),
        };

        let mut out = Vec::with_capacity(pieces.len());
        for piece in pieces {
            match piece {
                Piece::Token(id) => out.push(id),
                Piece::Unresolved { start, len } => {
                    if let Some(table) = table {
                        if let Some(span) = bytes.get(start..start + len) {
                            out.extend(span.iter().map(|&b| table[b as usize]));
                        }
                    }
                    // else: dropped, matching the prior (and still preserved)
                    // byte_pair_encode_with_ranks drop contract.
                }
            }
        }
        out
    }

    /// Encode bytes with BPE and caching.
    fn encode_bytes_with_cache(&self, bytes: &[u8]) -> Vec<u32> {
        // Fast path: check if entire chunk is a known token
        if let Some(&rank) = self.encoder.get(bytes) {
            return vec![rank];
        }

        // Check cache. Keyed by the chunk bytes themselves (via the `Vec<u8>:
        // Borrow<[u8]>` impl, so this lookup allocates nothing on a hit).
        if let Ok(mut cache) = self.chunk_cache.lock() {
            if let Some(cached) = cache.get(bytes) {
                return cached.clone();
            }
        }

        // Perform BPE encoding
        let result = self.bpe(bytes);

        // Store in cache
        if let Ok(mut cache) = self.chunk_cache.lock() {
            cache.put(bytes.to_vec(), result.clone());
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

        // Check cache. Keyed by the chunk bytes themselves (via the `Vec<u8>:
        // Borrow<[u8]>` impl, so this lookup allocates nothing on a hit).
        if let Ok(mut cache) = self.chunk_cache.lock() {
            if let Some(cached) = cache.get(bytes_to_encode.as_ref()) {
                return cached.clone();
            }
        }

        // Perform BPE encoding
        let result = self.bpe(bytes_to_encode.as_ref());

        // Store in cache
        if let Ok(mut cache) = self.chunk_cache.lock() {
            cache.put(bytes_to_encode.as_ref().to_vec(), result.clone());
        }

        result
    }

    /// Map each `(start, end)` chunk span over `text_bytes` through
    /// [`Tokenizer::encode_chunk`] and flatten the results, in parallel via
    /// rayon when `parallel` is true and the `rayon` feature is enabled.
    ///
    /// When the `rayon` feature is disabled, `parallel` is ignored and the
    /// map always runs sequentially — there is no rayon thread pool to use.
    #[inline]
    fn map_chunks(&self, text_bytes: &[u8], chunks: &[(usize, usize)], parallel: bool) -> Vec<u32> {
        #[cfg(feature = "rayon")]
        {
            if parallel {
                return chunks
                    .par_iter()
                    .flat_map(|&(start, end)| {
                        let slice = &text_bytes[start..end];
                        self.encode_chunk(slice)
                    })
                    .collect();
            }
        }
        #[cfg(not(feature = "rayon"))]
        let _ = parallel;

        chunks
            .iter()
            .flat_map(|&(start, end)| {
                let slice = &text_bytes[start..end];
                self.encode_chunk(slice)
            })
            .collect()
    }

    /// Canonical content-encoding pipeline: normalizer, then the pre-tokenizer
    /// / metaspace / plain-chunk fork.
    ///
    /// `parallel` only ever affects the plain-chunk fork (via
    /// [`Tokenizer::map_chunks`]). The pre-tokenizer and metaspace-decoder
    /// forks are deliberately sequential-only regardless of `parallel`: this
    /// is a strategy choice, not an oversight. The pre-tokenizer engine owns
    /// its own iteration over `pt.split(text)`, and the metaspace decoder
    /// accumulates `pending_underscores` as a strictly left-to-right fold
    /// over chunks — both are stateful across chunks and cannot be
    /// parallelized without changing output.
    fn encode_content(&self, text: &str, parallel: bool) -> Vec<u32> {
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

        if self.use_metaspace_decoder {
            self.encode_metaspace_chunks(text_bytes, &chunks)
        } else {
            // No metaspace decoder: use original logic
            self.map_chunks(text_bytes, &chunks, parallel)
        }
    }

    /// Metaspace-decoder chunk fold: spaces accumulate into `▁` prefixes for
    /// the next word (may merge into `▁▁▁`-style runs), non-space whitespace
    /// is encoded as its own byte token, and words are encoded together with
    /// any accumulated `▁` prefix. Always sequential — see
    /// [`Tokenizer::encode_content`] for why.
    fn encode_metaspace_chunks(&self, text_bytes: &[u8], chunks: &[(usize, usize)]) -> Vec<u32> {
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
                            results.extend(self.encode_bytes_with_cache(underscores.as_bytes()));
                            pending_underscores = 0;
                        }
                        // Encode the non-space whitespace as a byte
                        results.extend(self.encode_bytes_with_cache(&[b]));
                    }
                }
            } else {
                // Word chunk - prepend accumulated ▁ characters and encode together
                if pending_underscores > 0 {
                    let mut with_prefix = Vec::with_capacity(pending_underscores * 3 + slice.len());
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
        self.encode_content(text, false)
    }

    /// Encode text to token IDs using Rayon parallel processing.
    ///
    /// Produces exactly the same ids as [`Tokenizer::encode`] — same
    /// normalizer, same added-token dispatch, same pre-tokenizer/metaspace/
    /// plain-chunk fork — and differs only in execution strategy: the
    /// plain-chunk fork's BPE calls run in parallel via rayon rather than
    /// sequentially. Only beneficial for very large texts (>1MB).
    ///
    /// Metaspace-decoder tokenizers and tokenizers with a multi-stage
    /// pre-tokenizer still run sequentially regardless of this method,
    /// because their per-chunk state (`pending_underscores`, the
    /// pre-tokenizer engine's own iteration) is a left-to-right fold that
    /// cannot be parallelized without changing output — see
    /// [`Tokenizer::encode_content`].
    pub fn encode_rayon(&self, text: &str) -> Vec<u32> {
        if self.match_added_tokens {
            AddedTokens::dispatch(&self.special_matcher, text, |gap| {
                self.encode_content(gap, true)
            })
        } else {
            self.encode_content(text, true)
        }
    }

    /// Encode text with special token handling.
    ///
    /// Special tokens in the input are encoded directly without BPE, via the
    /// same `AddedTokens` matcher the SentencePiece/SPM/WordPiece backends
    /// use.
    pub fn encode_with_special(&self, text: &str) -> Vec<u32> {
        AddedTokens::dispatch(&self.special_matcher, text, |gap| self.encode_ordinary(gap))
    }

    /// Encode text to token IDs under an explicit [`SpecialMode`], governing
    /// whether `special_tokens` found in the input text are matched.
    ///
    /// This only concerns added-token matching in the content — it says
    /// nothing about boundary tokens (BOS/EOS/CLS/SEP), which this backend
    /// has no notion of; those come from [`SpecialPolicy`](crate::core::SpecialPolicy)
    /// via [`AnyTokenizer::encode_with`](crate::core::AnyTokenizer::encode_with).
    ///
    /// If this tokenizer was never configured for added-token matching
    /// ([`with_added_token_matching`](Self::with_added_token_matching) is
    /// `false`, [`Tokenizer::encode`]'s default), [`SpecialMode::All`] is read
    /// as "there is no matching to turn on" and falls back to the ordinary
    /// encoding — the same behavior [`Tokenizer::encode`] already gives in
    /// that configuration. [`SpecialMode::Ordinary`] and
    /// [`SpecialMode::Allow`] are the caller stating an explicit choice rather
    /// than asking for this tokenizer's default, so they always take effect
    /// regardless of that flag.
    pub fn encode_with(&self, text: &str, mode: &SpecialMode<'_>) -> Result<Vec<u32>, PolicyError> {
        if matches!(mode, SpecialMode::All) && !self.match_added_tokens {
            return Ok(self.encode_ordinary(text));
        }
        AddedTokens::dispatch_with_mode(&self.special_matcher, text, mode, |gap| {
            self.encode_ordinary(gap)
        })
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
}
