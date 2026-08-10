use super::super::types::{ByteFallback, Tokenizer};
use crate::core::bpe::{
    byte_pair_encode_ids_seeded_into, byte_pair_encode_pieces_presegmented,
    byte_pair_encode_pieces_seeded, PairRanks, Piece, RankLookup, Seed, Seeding,
};
use crate::core::precompiled::utf8_len;

impl Tokenizer {
    /// Where the merge loop reads ranks from.
    ///
    /// A model's own `merges` list when it has one, the encoder otherwise
    /// (tiktoken-style vocabularies, where a token's id *is* its rank), fronted
    /// by the two-byte index built from whichever of the two that was.
    #[inline]
    fn rank_lookup(&self) -> RankLookup<'_> {
        RankLookup::with_pairs(
            self.merge_ranks.as_ref().unwrap_or(&self.encoder),
            &self.byte_pair_ranks,
        )
        .with_ids(
            self.pair_ranks
                .get_or_init(|| {
                    PairRanks::build(
                        self.merge_ranks.as_ref().unwrap_or(&self.encoder),
                        &self.encoder,
                        self.raw_encoder.as_ref(),
                    )
                })
                .as_ref(),
        )
    }
    /// Whether chunks reach the cache and the merge as **raw** input bytes
    /// rather than mapped into ByteLevel space.
    ///
    /// One decision for the whole tokenizer, deliberately: the chunk cache is
    /// keyed by whatever bytes it is handed, and ASCII maps to itself, so a
    /// tokenizer that fed it both spaces would mostly agree and occasionally
    /// hand back another chunk's ids — a raw piece can spell some other piece's
    /// ByteLevel form. Every entry point therefore asks this, and none decides
    /// for itself.
    ///
    /// Requires the id-keyed table with a complete raw alphabet: the merge needs
    /// its symbols' ids, and only that table can supply them without the
    /// mapping. A pipeline that emits already-mapped pieces has no raw form to
    /// offer and keeps the mapped space.
    ///
    /// Answered once and remembered: every chunk asks, and the question reaches
    /// through the lazily-built id table to do it.
    #[inline]
    pub(super) fn merges_raw(&self) -> bool {
        *self.raw_space.get_or_init(|| {
            self.use_byte_level
                && self.raw_encoder.is_some()
                && self.pre_tokenizer.as_ref().is_none_or(|pt| pt.emits_raw())
                && self.rank_lookup().by_id().is_some_and(|t| t.seeds_raw())
        })
    }

    /// The vocabulary keyed in the space chunks arrive in.
    #[inline]
    pub(super) fn chunk_encoder(&self) -> &crate::core::encoder::Encoder {
        match self.merges_raw() {
            true => self.raw_encoder.as_ref().unwrap_or(&self.encoder),
            false => &self.encoder,
        }
    }

    /// Run BPE on a piece, honoring a separate merge-rank map when present,
    /// and rendering any span the vocabulary could not represent through the
    /// [`ByteFallback`](super::types::ByteFallback) resolution when one is
    /// configured (instead of silently dropping it, `byte_pair_encode_pieces_seeded`'
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
    pub(super) fn bpe_into(&self, bytes: &[u8], out: &mut Vec<u32>) {
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
        let char_granular = self.merge_ranks.is_some() && !self.use_byte_level;
        let ranks = self.rank_lookup();

        let Some(fallback) = fallback else {
            let seeding = match (self.merges_raw(), char_granular) {
                (true, _) => match ranks.by_id().is_some_and(|t| t.seeds_chars()) {
                    true => Seeding::RawChars,
                    false => Seeding::RawBytes,
                },
                (false, true) => Seeding::Chars,
                (false, false) => Seeding::Bytes,
            };
            // No fallback configured: an unrepresentable span is dropped,
            // matching the prior (and still preserved) drop contract. Nothing
            // downstream needs to know *which* spans those were, so the merge
            // reports ids directly instead of building a `Vec<Piece>` to filter.
            // This is the path every ByteLevel and every tiktoken-style
            // vocabulary takes — i.e. almost all traffic.
            byte_pair_encode_ids_seeded_into(bytes, ranks, self.chunk_encoder(), seeding, out);
            return;
        };

        let pieces = byte_pair_encode_pieces_seeded(bytes, ranks, &self.encoder, char_granular);
        out.reserve(pieces.len());

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
                out.extend_from_slice(&ids);
                return;
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
    }

    /// [`Tokenizer::bpe_into`] in HuggingFace's order: resolve the byte fallback
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
        // Presence of a merge list is what makes this path meaningful at all;
        // the lookup itself then reads through the same source `bpe_into` uses.
        self.merge_ranks.as_ref()?;
        let ranks = self.rank_lookup();
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
            let spelling = self.decoder.get(id)?;
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

            if let Some(id) = self.encoder.get(ch_bytes) {
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
}
