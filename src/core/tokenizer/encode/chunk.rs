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
        // Fast path: the entire chunk is one known token. Ahead of the cache
        // deliberately — it is a single hash lookup against a map that is
        // already hot, so caching its answer would cost more than recomputing.
        if let Some(&rank) = self.encoder.get(bytes) {
            out.push(rank);
            return;
        }

        if self.chunk_cache.extend_into(bytes, out) {
            return;
        }

        // The merge appends in place, so what it produced for this chunk is the
        // tail of `out` — which is exactly what the cache needs to record, with
        // no intermediate vector to hold it.
        let start = out.len();
        self.bpe_into(bytes, out);
        self.chunk_cache.put(bytes, &out[start..]);
    }

    /// Encode one pre-token chunk into `out`, applying ByteLevel encoding first
    /// when this tokenizer owns that step.
    ///
    /// When a pre-tokenizer engine is attached it has already
    /// byte-level-encoded the pieces, so we must NOT re-encode here (but
    /// `use_byte_level` stays true so `decode` still reverses the mapping).
    pub(super) fn encode_chunk_into(&self, slice: &[u8], out: &mut Vec<u32>) {
        if self.use_byte_level && self.pre_tokenizer.is_none() {
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
        if let Some(raw_encoder) = &self.raw_encoder {
            if let Some(&rank) = raw_encoder.get(raw) {
                out.push(rank);
                return;
            }
        }
        scratch.clear();
        byte_level_encode_into(scratch, raw);
        self.encode_bytes_into(scratch.as_bytes(), out);
    }

    /// [`Tokenizer::encode_bytes_into`] as a standalone call, for the few
    /// callers that genuinely need an owned vector of one chunk's ids.
    pub(super) fn encode_bytes_with_cache(&self, bytes: &[u8]) -> Vec<u32> {
        let mut out = Vec::new();
        self.encode_bytes_into(bytes, &mut out);
        out
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

        // One id per chunk is the floor; most chunks produce one or two, so this
        // usually holds the whole text without regrowing.
        let mut out = Vec::with_capacity(chunks.len());
        for &(start, end) in chunks {
            self.encode_chunk_into(&text_bytes[start..end], &mut out);
        }
        out
    }
}
