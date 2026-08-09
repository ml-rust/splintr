use super::super::types::Tokenizer;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

impl Tokenizer {
    /// Canonical content-encoding pipeline: normalizer, then the pre-tokenizer
    /// / metaspace / plain-chunk fork.
    ///
    /// `parallel` affects the plain-chunk fork (via [`Tokenizer::map_chunks`])
    /// and the pre-tokenizer fork, whose pieces are independent of each other
    /// once the engine has produced them.
    ///
    /// The metaspace-decoder fork stays sequential regardless: it accumulates
    /// `pending_underscores` as a strictly left-to-right fold over chunks, so it
    /// is genuinely stateful and cannot be parallelized without changing output.
    pub(super) fn encode_content(&self, text: &str, parallel: bool) -> Vec<u32> {
        // Apply the HF `normalizer` (e.g. NFC) to content before splitting. This
        // runs on content gaps (special tokens are extracted upstream), matching
        // HuggingFace's extract-then-normalize order.
        let normalized = self.normalized(text);
        let text = normalized.as_ref();

        // Multi-stage pre-tokenizer path (Digits/Punctuation/Sequence/…): the
        // engine produces already byte-level-encoded pieces; BPE each directly.
        //
        // `split_pieces` rather than `split`: the pieces are consumed
        // immediately and never stored, so materializing a `String` for each
        // one is pure waste on the path every `tokenizer.json` model takes.
        if let Some(pt) = &self.pre_tokenizer {
            #[cfg(feature = "rayon")]
            if parallel {
                // Splitting into a `Vec` first is what lets the pieces be shared
                // out across threads; the streaming path below cannot, since it
                // hands back one reused buffer.
                let pieces = pt.split_pieces(text);
                return pieces
                    .par_iter()
                    .fold(Vec::new, |mut acc, piece| {
                        self.encode_chunk_into(piece.as_bytes(), &mut acc);
                        acc
                    })
                    .reduce(Vec::new, |mut a, b| {
                        a.extend_from_slice(&b);
                        a
                    });
            }

            // One id per pre-token is the floor, so sizing from the text holds
            // the whole result without regrowing. The streaming path has no
            // piece count to size from — that was the point of not building one
            // — so it estimates from the same rule the pipeline uses.
            let mut out = Vec::with_capacity(crate::core::pretokenizer::estimated_pieces(text));
            // When the pipeline ends in ByteLevel, take the pieces unmapped and
            // let `encode_raw_chunk_into` map only the ones that need it.
            if self.use_byte_level && self.raw_encoder.is_some() && pt.emits_raw() {
                let mut scratch = String::new();
                pt.for_each_raw_piece(text, |piece| {
                    self.encode_raw_chunk_into(piece.as_bytes(), &mut out, &mut scratch)
                });
                return out;
            }
            pt.for_each_piece(text, |piece| {
                self.encode_chunk_into(piece.as_bytes(), &mut out)
            });
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
}
