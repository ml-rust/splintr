use super::super::types::Tokenizer;
use crate::core::byte_level::{byte_level_encode, byte_level_encode_into};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

impl Tokenizer {
    /// Encode already-prepared chunk bytes through the chunk cache, appending
    /// the ids to `out`.
    ///
    /// The single place the cache protocol lives: whole-chunk vocabulary hit,
    /// then cache, then merge, then record. `bytes` must already be in the
    /// space the vocabulary is keyed in — see [`Tokenizer::encode_chunk_into`]
    /// for the ByteLevel half of that.
    ///
    /// Appending rather than returning is the point. A text is many chunks and
    /// their ids are concatenated, so a per-chunk `Vec` is an allocation and a
    /// copy per chunk for a result that is immediately spliced into its
    /// neighbours. Writing through to one buffer removes both, on every branch:
    /// the whole-chunk hit pushes one id, a cache hit copies from under the
    /// shard lock, and the merge writes its output in place.
    pub(super) fn encode_bytes_into(&self, bytes: &[u8], out: &mut Vec<u32>) {
        // One hash of the chunk serves every question asked about it: the
        // vocabulary and the cache are keyed on the same one.
        let hash = crate::core::encoder::Encoder::hash_of(bytes);
        // Fast path: the entire chunk is one known token. Ahead of the cache
        // deliberately — it is a single hash lookup against a map that is
        // already hot, so caching its answer would cost more than recomputing.
        if let Some(rank) = self.chunk_encoder().get_with_hash(bytes, hash) {
            out.push(rank);
            return;
        }
        // A metaspace pipeline can hand this a chunk spanning many words, and
        // merging that whole is the expensive way to the same ids — see
        // [`RunSplit`].
        //
        // Only the pre-tokenizer fork asks. A `Metaspace` node cuts its own text
        // at the same boundaries before any chunk reaches here, so asking again
        // would rescan every word to find the cut already taken; and a
        // ByteLevel vocabulary maps every byte into an alphabet `▁` is not in,
        // so a marker in a chunk there is not the marker at all. What is left is
        // the shape Gemma ships: a `Split` pre-tokenizer looking for the space
        // its own normalizer has already replaced, which therefore splits
        // nothing.
        if self.pre_tokenizer.is_some()
            && !self.use_byte_level
            && self.split_marker_runs_into(bytes, out)
        {
            return;
        }
        self.encode_unresolved_bytes_into(bytes, hash, out);
    }

    /// Cut `bytes` at every marker-run start the vocabulary proves free, encode
    /// the pieces, and report whether any cut was made.
    ///
    /// `false` leaves `out` untouched and the chunk for the caller to merge
    /// whole — either because this vocabulary licenses no cut, or because this
    /// chunk holds none. The latter is the ordinary case once the cutting has
    /// happened: a piece produced here opens with a marker run and carries no
    /// further free run start.
    fn split_marker_runs_into(&self, bytes: &[u8], out: &mut Vec<u32>) -> bool {
        let split = self.marker_run_split();
        if matches!(split, crate::core::tokenizer::types::RunSplit::Never) {
            return false;
        }

        let marker = super::content::MARKER.as_bytes();
        let mut start = 0;
        let mut prev_end = usize::MAX;
        for at in memchr::memmem::find_iter(bytes, marker) {
            let mid_run = at == prev_end;
            prev_end = at + marker.len();
            // `at == 0` is the piece's own opening run, not a cut inside it.
            if mid_run || at == 0 || !split.cuts_at(bytes, at) {
                continue;
            }
            self.encode_piece_into(&bytes[start..at], out);
            start = at;
        }
        if start == 0 {
            return false;
        }
        self.encode_piece_into(&bytes[start..], out);
        true
    }

    /// [`Tokenizer::encode_bytes_into`] for bytes already known to hold no cut,
    /// which every piece [`Tokenizer::split_marker_runs_into`] produces is.
    fn encode_piece_into(&self, bytes: &[u8], out: &mut Vec<u32>) {
        let hash = crate::core::encoder::Encoder::hash_of(bytes);
        match self.chunk_encoder().get_with_hash(bytes, hash) {
            Some(rank) => out.push(rank),
            None => self.encode_unresolved_bytes_into(bytes, hash, out),
        }
    }

    /// [`Tokenizer::encode_bytes_into`] for a caller that has already asked the
    /// vocabulary about the whole chunk and been told no.
    ///
    /// Split out because in raw space that question is asked against the very
    /// map this would ask again — the two used to be different maps in different
    /// spaces, and are now one. `hash` is that question's hash, reused here.
    fn encode_unresolved_bytes_into(&self, bytes: &[u8], hash: u64, out: &mut Vec<u32>) {
        if self.chunk_cache.extend_into(hash, bytes, out) {
            return;
        }

        // The merge appends in place, so what it produced for this chunk is the
        // tail of `out` — which is exactly what the cache needs to record, with
        // no intermediate vector to hold it.
        let start = out.len();
        self.bpe_into(bytes, out);
        self.chunk_cache.put(hash, bytes, &out[start..]);
    }

    /// Encode one pre-token chunk into `out`, applying ByteLevel encoding first
    /// when this tokenizer owns that step.
    ///
    /// When a pre-tokenizer engine is attached it has already
    /// byte-level-encoded the pieces, so we must NOT re-encode here (but
    /// `use_byte_level` stays true so `decode` still reverses the mapping).
    pub(super) fn encode_chunk_into(&self, slice: &[u8], out: &mut Vec<u32>) {
        if self.use_byte_level && self.pre_tokenizer.is_none() && !self.merges_raw() {
            let encoded = byte_level_encode(slice);
            self.encode_bytes_into(encoded.as_bytes(), out);
            return;
        }
        self.encode_bytes_into(slice, out);
    }

    /// Encode one **raw** (unmapped) pre-token chunk from a ByteLevel
    /// pipeline, mapping it into ByteLevel space only if it has to.
    ///
    /// The whole-piece vocabulary hit resolves 92.5% of pre-tokens on ordinary
    /// prose, and `raw_encoder` answers it without any mapping at all. Only the
    /// remaining 7.5% — the ones headed for the chunk cache or the merge loop,
    /// both of which are keyed in ByteLevel space — pay for `scratch`.
    ///
    /// Falls back to mapping everything when `raw_encoder` is absent, which is
    /// exactly the old behavior.
    pub(super) fn encode_raw_chunk_into(
        &self,
        raw: &[u8],
        out: &mut Vec<u32>,
        scratch: &mut String,
    ) {
        // Short chunks are answered by an index rather than a hash and a probe.
        if let Some(id) = self.short_chunk_id(raw) {
            out.push(id);
            return;
        }
        let hash = crate::core::encoder::Encoder::hash_of(raw);
        if let Some(raw_encoder) = &self.raw_encoder {
            if let Some(rank) = raw_encoder.get_with_hash(raw, hash) {
                out.push(rank);
                return;
            }
        }
        // The merge works from ids, so a vocabulary that can supply them for raw
        // bytes never needs the mapping at all — not for the cache key either,
        // which is then the input's own bytes rather than the one-to-two-byte
        // expansion of them.
        if self.merges_raw() {
            // The whole-chunk question was just asked, against the same map
            // `encode_bytes_into` would ask — so go straight past it, hash and
            // all.
            self.encode_unresolved_bytes_into(raw, hash, out);
            return;
        }
        scratch.clear();
        byte_level_encode_into(scratch, raw);
        self.encode_bytes_into(scratch.as_bytes(), out);
    }

    /// Map each `(start, end)` chunk span over `text_bytes` through
    /// [`Tokenizer::encode_chunk_into`] and concatenate the results, in
    /// parallel via rayon when `parallel` is true and the `rayon` feature is
    /// enabled.
    ///
    /// The sequential path fills a single buffer, so the whole text costs one
    /// growing allocation rather than one per chunk. The parallel path cannot
    /// share a buffer, so it gives each rayon task its own and lets rayon
    /// concatenate them — one per task, not one per chunk.
    ///
    /// When the `rayon` feature is disabled, `parallel` is ignored and the
    /// map always runs sequentially — there is no rayon thread pool to use.
    #[inline]
    pub(super) fn map_chunks(
        &self,
        text_bytes: &[u8],
        chunks: &[(usize, usize)],
        parallel: bool,
    ) -> Vec<u32> {
        #[cfg(feature = "rayon")]
        {
            if parallel {
                return chunks
                    .par_iter()
                    .fold(Vec::new, |mut acc, &(start, end)| {
                        self.encode_chunk_into(&text_bytes[start..end], &mut acc);
                        acc
                    })
                    .reduce(Vec::new, |mut a, b| {
                        a.extend_from_slice(&b);
                        a
                    });
            }
        }
        #[cfg(not(feature = "rayon"))]
        let _ = parallel;

        // One id per chunk is the floor, not the estimate, and the two scripts
        // miss it in opposite directions — hence two terms and a `max`.
        //
        // Latin text lands just above one id per chunk: English prose runs
        // 1.076 and JSON 1.015-1.031, so an exact `chunks.len()` held only 33%
        // and 50% of texts, and the rest doubled and copied the whole id
        // buffer. An eighth of headroom holds 100% of both.
        //
        // CJK misses by multiples instead — 3.3-4.9 ids per chunk, since a run
        // of Han is one chunk and many tokens — and no headroom expressed in
        // chunks can follow that. Bytes can: dense scripts spend ~3-4 bytes per
        // token, so `len / 4` tracks them while staying under the chunk term
        // for the Latin text it would otherwise inflate. Measured per corpus,
        // it is worth 9.4%/13.2% on Chinese and 6.3%/7.8% on mixed-script text
        // (cl100k/o200k), and costs code and JSON nothing.
        //
        // `len / 3` was tried, to fit cl100k Chinese exactly rather than merely
        // closely. It won another 10% there and lost roughly 1-2% on every
        // o200k corpus including Chinese, so the tighter divisor is not carried.
        let mut out =
            Vec::with_capacity((chunks.len() + chunks.len() / 8).max(text_bytes.len() / 4 + 8));
        for &(start, end) in chunks {
            self.encode_chunk_into(&text_bytes[start..end], &mut out);
        }
        out
    }
}
