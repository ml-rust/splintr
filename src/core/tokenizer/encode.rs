use super::backend::subdivide;
use super::types::{ByteFallback, Tokenizer};
use crate::core::added::AddedTokens;
use crate::core::bpe::{
    byte_pair_encode_pieces, byte_pair_encode_pieces_presegmented, byte_pair_encode_pieces_seeded,
    Piece, Seed,
};
use crate::core::byte_level::{byte_level_decode, byte_level_encode};
use crate::core::policy::{PolicyError, SpecialMode};
use crate::core::precompiled::utf8_len;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

impl Tokenizer {
    /// Apply `add_prefix_space` to an input, borrowing when no change is needed.
    ///
    /// The guard is a literal **space**, not whitespace in general — both HF
    /// nodes that set this flag suppress the prefix only on an existing leading
    /// space, and prepend across every other whitespace character:
    ///
    /// - `ByteLevel::pre_tokenize` tests `!normalized.get().starts_with(' ')`.
    /// - `Metaspace::pre_tokenize` replaces spaces with the replacement first
    ///   and then prepends unless the result already starts with it, which is
    ///   the same test one step later.
    ///
    /// Measured against `tokenizers` 0.22.1 — on mistral-7b-v0.3 (`Metaspace`,
    /// `prepend_scheme: "first"`, `split: false`), `"\n\n\n"` is
    /// `[29473, 781, 781, 781]` (`▁`, `<0x0A>`×3) and `"\ta"` is
    /// `[29473, 780, 29476]`, both keeping the `▁` a whitespace-wide guard
    /// dropped, while `" a"` stays `[1032]` (`▁a`) and `"  a"` stays
    /// `[29473, 1032]`. A ByteLevel fixture with `add_prefix_space: true`
    /// behaves identically (`"\ta"` → `Ġ ĉ a`, `" a"` → `Ġa`).
    #[inline]
    fn prefixed<'a>(&self, text: &'a str) -> std::borrow::Cow<'a, str> {
        if self.add_prefix_space && !text.starts_with(' ') {
            std::borrow::Cow::Owned(format!(" {text}"))
        } else {
            std::borrow::Cow::Borrowed(text)
        }
    }

    /// Apply the declared HF `normalizer` to an input, borrowing when none is.
    ///
    /// The one place the pipeline normalizes, so that
    /// [`normalize`](Self::normalize) reports the same string
    /// [`encode_content`](Self::encode_content) goes on to split rather than a
    /// second rendering of it — HF's `Prepend` and `Replace` normalizers are not
    /// idempotent, so a drifted copy would not merely be stale, it would be
    /// unrecoverable from the outside.
    #[inline]
    fn normalized<'a>(&self, text: &'a str) -> std::borrow::Cow<'a, str> {
        match &self.normalizer {
            Some(norm) => std::borrow::Cow::Owned(norm.normalize(text)),
            None => std::borrow::Cow::Borrowed(text),
        }
    }

    /// The input as this tokenizer's `normalizer` leaves it — the text the rest
    /// of the encode path, [`pre_tokenize`](Self::pre_tokenize) included, is
    /// driven with.
    ///
    /// Exists for the same reason [`pre_tokenize`](Self::pre_tokenize) does: the
    /// stage is otherwise unobservable from outside the crate, so a normalizer
    /// pipeline that drifts from the `tokenizer.json` it was parsed out of stays
    /// invisible until it happens to move a token id.
    /// `tests/reference_parity.rs` pins this against the reference tokenizers'
    /// own `normalizer.normalize_str`.
    ///
    /// # What it does and does not include
    ///
    /// The declared `normalizer` and nothing else. `add_prefix_space` is *not*
    /// applied here — HuggingFace hangs that flag off its `ByteLevel` /
    /// `Metaspace` pre-tokenizer nodes, and so does this crate, which is why
    /// [`pre_tokenize`](Self::pre_tokenize) applies it instead. Added-token
    /// extraction is not included either: it runs upstream on the raw input, and
    /// this is what one content gap becomes. A tokenizer that declares no
    /// normalizer (every vocabulary in [`crate::pretrained`]) returns `text`
    /// unchanged.
    pub fn normalize(&self, text: &str) -> String {
        self.normalized(text).into_owned()
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

    /// The pre-token pieces this tokenizer's encode path splits `text` into,
    /// before any BPE merge runs.
    ///
    /// Exists because the split is otherwise unobservable from outside the
    /// crate — only the ids that come out the far end of BPE are — so a
    /// pre-tokenizer pattern that drifts from the reference it was transcribed
    /// from stays invisible until it happens to move a token id.
    /// `tests/reference_parity.rs` pins these pieces against the reference
    /// tokenizers' own split for exactly that reason.
    ///
    /// This calls the same two steps `encode` calls — the `add_prefix_space`
    /// guard and then the pre-tokenizer — rather than reconstructing them, so
    /// the answer here and the split BPE is actually fed cannot drift apart.
    ///
    /// # What `text` must already be
    ///
    /// The text as the *pre-tokenizer* sees it. `encode` runs the configured HF
    /// `normalizer` first and this deliberately does not, so a caller holding a
    /// reference implementation's normalizer output can pass it straight in
    /// without normalizing twice — HF's `Prepend` and `Replace` normalizers are
    /// not idempotent, so doing it twice is not harmless. A tokenizer that
    /// declares no normalizer (every vocabulary in [`crate::pretrained`]) makes
    /// the distinction moot.
    ///
    /// `add_prefix_space` sits on the other side of that line: it is applied
    /// *here*, which is where HuggingFace puts it too (its `ByteLevel` and
    /// `Metaspace` pre-tokenizer nodes own the flag), so a prepended space
    /// shows up in the first piece exactly as it does in the reference's.
    ///
    /// # Piece space
    ///
    /// Raw input text, never the ByteLevel alphabet — the two forks of the
    /// encode path disagree on this internally and are reconciled here. The
    /// regex fork slices the text and maps each chunk through
    /// [`byte_level_encode`](crate::core::byte_level::byte_level_encode) only
    /// later; the multi-stage pre-tokenizer engine maps up front, and that
    /// mapping (a bijection over the same bytes) is undone below.
    pub fn pre_tokenize(&self, text: &str) -> Vec<String> {
        if let Some(pt) = &self.pre_tokenizer {
            return pt
                .split(text)
                .into_iter()
                .map(|piece| unmap_byte_level(piece, self.use_byte_level))
                .collect();
        }

        let text = self.prefixed(text);
        self.split_chunks(&text)
            .into_iter()
            // `get` rather than indexing: the spans are on char boundaries by
            // construction, and `subdivide` already skips a span it cannot
            // resolve rather than panicking.
            .filter_map(|(start, end)| text.get(start..end).map(str::to_owned))
            .collect()
    }

    /// Run BPE on a piece, honoring a separate merge-rank map when present,
    /// and rendering any span the vocabulary could not represent through the
    /// [`ByteFallback`](super::types::ByteFallback) resolution when one is
    /// configured (instead of silently dropping it, `byte_pair_encode_pieces`'
    /// contract).
    ///
    /// **Ordering note:** the resolution here runs after the merge, which is
    /// the cheap order and the right answer whenever no merge takes a
    /// `<0xNN>`/`<unk>` token as an operand. When a vocabulary's merge list
    /// does, HuggingFace's order (resolve first, then merge) is observably
    /// different, and the chunk is redone through
    /// [`bpe_fallback_first`](Self::bpe_fallback_first) — see its doc for the
    /// measurement and for why no shelf vocabulary reaches it.
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
        let fallback = (!self.use_byte_level)
            .then_some(self.byte_fallback.as_ref())
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
        let Some(fallback) = fallback else {
            // No fallback configured: an unrepresentable span is dropped,
            // matching the prior (and still preserved) drop contract.
            for piece in pieces {
                if let Piece::Token(id) = piece {
                    out.push(id);
                }
            }
            return out;
        };

        // HuggingFace resolves the fallback BEFORE merging, so its `<0xNN>`/
        // `<unk>` tokens are ordinary word symbols the merge list may combine
        // with their neighbours; the resolution below runs AFTER, so it cannot.
        // The two agree unless some merge takes one of those tokens as an
        // operand, which is why the redo is gated on there being something
        // unresolved at all: with nothing to substitute, the orders coincide by
        // construction and the merge already ran over exactly HuggingFace's
        // symbols. (`merge_ranks.is_some()` is the whole of the "is this a
        // HuggingFace-shaped vocabulary" test here — `fallback` being `Some`
        // already implies `!use_byte_level`, so the two conjuncts together are
        // the character-seeded path.)
        if self.merge_ranks.is_some()
            && pieces
                .iter()
                .any(|piece| matches!(piece, Piece::Unresolved { .. }))
        {
            if let Some(ids) = self.bpe_fallback_first(bytes, fallback) {
                return ids;
            }
        }

        // HuggingFace resolves fallback per CHARACTER over the whole word, with
        // one wrinkle reproduced deliberately here: a pending unk is flushed by
        // a *vocabulary* hit, never by a `<0xNN>` hit — `merge_word` in
        // `tokenizers`' BPE model adds the byte tokens directly while the unk
        // stays pending until the next vocabulary hit or the end of the word.
        // So `▁hello\n` over a vocab whose only byte token is `<0x0A>` gives
        // `▁ <unk> <unk> <unk> <unk> <0x0A> <unk>`, with the newline ahead of
        // the final unk. Measured against `tokenizers` 0.22.1; a "tidier"
        // strictly positional order would disagree with HuggingFace on every
        // partial-fallback vocabulary. `pending_unk` holds the id to emit, so a
        // vocabulary with no `unk_token` at all pends `None` and the character
        // is dropped — the no-fallback behavior, which is also what HF does
        // there (measured).
        //
        // `model.fuse_unk` collapses a *run* of unk-resolved characters into one
        // unk, and it is exactly this flush that it suppresses: the run is
        // delimited by a vocabulary hit (which still flushes) and spans any
        // `<0xNN>` hit in between, since those never flush either way. Measured
        // against `tokenizers` 0.22.1 — see [`ByteFallback::fuse_unk`].
        let mut pending_unk: Option<u32> = None;
        for piece in pieces {
            match piece {
                Piece::Token(id) => {
                    out.extend(pending_unk.take());
                    out.push(id);
                }
                Piece::Unresolved { start, len } => {
                    let Some(span) = bytes.get(start..start + len) else {
                        continue;
                    };
                    let mut i = 0;
                    while i < span.len() {
                        // `min` guards a span that is not whole characters: the
                        // tiktoken path (`merge_ranks == None`) splits by byte,
                        // so a span can start or end mid-character. A truncated
                        // or continuation byte is then handled on its own, which
                        // is exactly the granularity BPE reported it at.
                        let n = utf8_len(span[i]).min(span.len() - i);
                        let ch = &span[i..i + n];
                        // A character is emitted as its bytes only when EVERY
                        // one of them has a `<0xNN>` entry, and otherwise
                        // collapses to a single unk: `é` over a vocab declaring
                        // only `<0xC3>` is one `<unk>`, not `<0xC3> <unk>`.
                        if ch.iter().all(|&b| fallback.byte_ids[b as usize].is_some()) {
                            out.extend(ch.iter().filter_map(|&b| fallback.byte_ids[b as usize]));
                        } else {
                            if !fallback.fuse_unk {
                                out.extend(pending_unk.take());
                            }
                            pending_unk = fallback.unk_id;
                        }
                        i += n;
                    }
                }
            }
        }
        out.extend(pending_unk.take());
        out
    }

    /// [`Tokenizer::bpe`] in HuggingFace's order: resolve the byte fallback
    /// FIRST, then merge over the result.
    ///
    /// `tokenizers`' `BPE::merge_word` builds the word one character at a time,
    /// and a character the vocabulary cannot represent is added to that word as
    /// its `<0xNN>` byte tokens (or as the unk) right there — *before*
    /// `merge_all` runs. Those tokens are therefore ordinary symbols the merge
    /// list may combine with their neighbours. `Tokenizer::bpe` merges the raw
    /// characters first and maps the leftovers through the table afterwards, so
    /// they never can.
    ///
    /// Measured against `tokenizers` 0.22.1 on a `{"<unk>": 0, "a": 1, "b": 2,
    /// "<0x7A>": 3, "<0x7A>b": 4, "a<0x7A>": 5}` vocab with
    /// `byte_fallback: true` and `merges` `[["<0x7A>","b"], ["a","<0x7A>"]]`
    /// (`z` is 0x7A and is absent from the vocabulary): `encode("zb")` is
    /// `['<0x7A>b']` and `encode("az")` is `['a<0x7A>']` — one token each, from
    /// a merge whose operand is a byte-fallback token. Resolving after the
    /// merge gives `['<0x7A>', 'b']` and `['a', '<0x7A>']` instead.
    ///
    /// No published vocabulary on the shelf distinguishes the two: neither
    /// `mistral-7b-v0.3` nor `embeddinggemma-300m` — the byte-fallback models
    /// this project verifies against — has a single merge whose concatenated
    /// key so much as *contains* `<0x` or `<unk>`, and a merge can only take a
    /// substituted symbol as an operand if its key contains that symbol's whole
    /// spelling. So this path returns byte-identical ids for them; it is the
    /// partial or unusual vocabulary it exists for.
    ///
    /// Returns `None` when the substitution cannot be carried out faithfully —
    /// non-UTF-8 input (there are no characters to walk), a fallback id with no
    /// spelling in the vocabulary (it cannot be a merge operand, so there is
    /// nothing for this ordering to change), or a merged surface that does not
    /// resolve. The caller then keeps the resolve-after-merge path, which is
    /// this method's answer too in all of those cases.
    fn bpe_fallback_first(&self, bytes: &[u8], fallback: &ByteFallback) -> Option<Vec<u32>> {
        let ranks = self.merge_ranks.as_ref()?;
        let text = std::str::from_utf8(bytes).ok()?;

        // The rewritten buffer: input characters the vocabulary has, and the
        // vocabulary SPELLINGS (`<0x7A>`, `<unk>`) of the tokens the ones it
        // lacks resolve to — because the merge list refers to those tokens by
        // exactly those spellings. `seeds` cuts it back at the symbol
        // boundaries, so a 6-byte `<0x7A>` is one symbol rather than six.
        let mut buf: Vec<u8> = Vec::with_capacity(bytes.len());
        let mut seeds: Vec<Seed> = Vec::new();

        // Append a resolved token by its vocabulary spelling. `None` when the
        // id has no spelling — see this method's own `None` contract.
        let push_token = |buf: &mut Vec<u8>, seeds: &mut Vec<Seed>, id: u32| -> Option<()> {
            let spelling = self.decoder.get(&id)?;
            seeds.push(Seed {
                start: buf.len(),
                len: spelling.len(),
                id: Some(id),
            });
            buf.extend_from_slice(spelling);
            Some(())
        };

        // The same per-character walk `Tokenizer::bpe`'s resolution does, with
        // the same deliberate wrinkle: a pending unk is flushed by a
        // *vocabulary* hit, never by a `<0xNN>` hit, and `fuse_unk` collapses a
        // run of unk-resolved characters into one. See `Tokenizer::bpe` and
        // `ByteFallback::fuse_unk` for the measurements behind both.
        let mut pending_unk: Option<u32> = None;
        for ch in text.chars() {
            let mut encoded = [0u8; 4];
            let ch_bytes = ch.encode_utf8(&mut encoded).as_bytes();

            if let Some(&id) = self.encoder.get(ch_bytes) {
                if let Some(unk) = pending_unk.take() {
                    push_token(&mut buf, &mut seeds, unk)?;
                }
                // Seeded by its own bytes rather than through `push_token`: a
                // vocabulary hit's spelling IS the character.
                seeds.push(Seed {
                    start: buf.len(),
                    len: ch_bytes.len(),
                    id: Some(id),
                });
                buf.extend_from_slice(ch_bytes);
                continue;
            }

            // A character is emitted as its bytes only when EVERY one of them
            // has a `<0xNN>` entry, and otherwise collapses to a single unk.
            if ch_bytes
                .iter()
                .all(|&b| fallback.byte_ids[b as usize].is_some())
            {
                for &b in ch_bytes {
                    if let Some(id) = fallback.byte_ids[b as usize] {
                        push_token(&mut buf, &mut seeds, id)?;
                    }
                }
                continue;
            }

            if !fallback.fuse_unk {
                if let Some(unk) = pending_unk.take() {
                    push_token(&mut buf, &mut seeds, unk)?;
                }
            }
            pending_unk = fallback.unk_id;
        }
        if let Some(unk) = pending_unk.take() {
            push_token(&mut buf, &mut seeds, unk)?;
        }

        let pieces = byte_pair_encode_pieces_presegmented(&buf, &seeds, ranks, &self.encoder);
        let mut out = Vec::with_capacity(pieces.len());
        for piece in pieces {
            match piece {
                Piece::Token(id) => out.push(id),
                // Every seed carries its own id, so only a merged surface the
                // vocabulary does not contain can land here — which a merge
                // list built from that same vocabulary cannot produce. Bail
                // rather than map bytes of a rewritten buffer through a table
                // keyed by RAW input bytes.
                Piece::Unresolved { .. } => return None,
            }
        }
        Some(out)
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
        let normalized = self.normalized(text);
        let text = normalized.as_ref();

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
    /// cannot be parallelized without changing output — see the internal
    /// `encode_content`.
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

/// Undo the ByteLevel mapping a pre-tokenizer engine applied to a piece.
///
/// `byte_level` is the tokenizer's own flag, which the loader keeps equal to
/// the engine's (`with_pre_tokenizer`), so `false` means the piece is already
/// raw text. When it is `true` the piece is one span of
/// `byte_level_encode(text)` and the inverse is total; the piece is returned
/// unchanged rather than panicking if a future engine ever breaks that
/// invariant.
fn unmap_byte_level(piece: String, byte_level: bool) -> String {
    if !byte_level {
        return piece;
    }
    byte_level_decode(&piece)
        .and_then(|bytes| String::from_utf8(bytes).ok())
        .unwrap_or(piece)
}
