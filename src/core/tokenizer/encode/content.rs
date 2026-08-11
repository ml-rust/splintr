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
        let mut out = Vec::new();
        self.encode_content_into(text, parallel, true, &mut out);
        out
    }

    /// [`Tokenizer::encode_content`] appending to a buffer the caller owns.
    ///
    /// This is what added-token dispatch calls, once per gap between special
    /// tokens — so on text those are dense in, a returned `Vec` per gap is an
    /// allocation, a copy and a free for a handful of ids. The streaming
    /// pre-tokenizer paths write straight through; the rest keep their own
    /// buffer and are copied over, which is what they did before.
    ///
    /// `is_first` says whether this gap opens the input, which `Metaspace`'s
    /// `prepend_scheme: "first"` needs and nothing else reads: HuggingFace
    /// prepends the replacement to the sequence's first split only, so a gap
    /// that follows an added token does not get one. A gap that follows an added
    /// token is never the first split, and neither is anything after it — which
    /// is exactly what the pointer test at the call sites reports.
    pub(super) fn encode_content_into(
        &self,
        text: &str,
        parallel: bool,
        is_first: bool,
        out: &mut Vec<u32>,
    ) {
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
                //
                // Unmapped when the merge works from raw bytes — the same
                // choice the sequential path makes below, and it has to be the
                // same one: both feed the chunk cache, which is keyed by the
                // bytes it is handed and cannot hold two spaces at once.
                let pieces = match self.merges_raw() {
                    true => pt.split_raw_pieces(text),
                    false => pt.split_pieces(text),
                };
                out.extend(
                    pieces
                        .par_iter()
                        .fold(Vec::new, |mut acc, piece| {
                            self.encode_chunk_into(piece.as_bytes(), &mut acc);
                            acc
                        })
                        .reduce(Vec::new, |mut a, b| {
                            a.extend_from_slice(&b);
                            a
                        }),
                );
                return;
            }

            // One id per pre-token is the floor, so sizing from the text holds
            // the whole result without regrowing. The streaming path has no
            // piece count to size from — that was the point of not building one
            // — so it estimates from the same rule the pipeline uses.
            out.reserve(crate::core::pretokenizer::estimated_pieces(text));
            // When the pipeline ends in ByteLevel, take the pieces unmapped and
            // let `encode_raw_chunk_into` map only the ones that need it.
            if self.use_byte_level && self.raw_encoder.is_some() && pt.emits_raw() {
                // `scratch` is only touched when the merge still needs the
                // ByteLevel form; a tokenizer that merges raw never fills it.
                // Sized rather than grown from empty. It is cleared and refilled
                // per piece, so it settles at the longest pre-token in the text
                // — but it reaches that by doubling from zero on the first few
                // pieces, and on macOS each of those regrows is a fresh
                // allocation plus a copy. 64 bytes covers a pre-token in every
                // script the bundled vocabularies cover; a longer one still
                // grows, exactly as before.
                // Empty when the merge works from raw bytes, which never asks
                // for the mapping: `String::new` does not allocate, and this
                // runs once per gap.
                let mut scratch = match self.merges_raw() {
                    true => String::new(),
                    false => String::with_capacity(64),
                };
                pt.for_each_raw_piece(text, |piece| {
                    self.encode_raw_chunk_into(piece.as_bytes(), out, &mut scratch)
                });
                return;
            }
            pt.for_each_piece(text, |piece| self.encode_chunk_into(piece.as_bytes(), out));
            return;
        }

        if self.use_metaspace_decoder {
            return crate::core::scratch::with_text(|buf| {
                self.metaspace_transform_at(text, is_first, buf);
                self.for_each_metaspace_piece(buf, |piece| {
                    self.encode_chunk_into(piece.as_bytes(), out)
                });
            });
        }

        let text = self.prefixed(text);
        let text = text.as_ref();
        let text_bytes = text.as_bytes();

        // The spans are built, walked and dropped inside this call, which is
        // exactly the shape a per-thread buffer serves: the vector outlives the
        // encode and is reused by the next one on this thread. Nothing escapes
        // — both arms return ids, never a borrow of the spans.
        crate::core::scratch::with_spans(|chunks| {
            self.split_chunks_into(text, chunks);

            if chunks.is_empty() {
                return;
            }

            out.extend(self.map_chunks(text_bytes, chunks, parallel));
        })
    }

    /// Write `text` as HuggingFace's `Metaspace` node leaves it: every **space**
    /// (U+0020, and nothing else that is whitespace) becomes `▁`, then a leading
    /// `▁` is prepended unless the result already opens with one.
    ///
    /// Both halves are literal readings of `Metaspace::pre_tokenize`, and both
    /// matter. Replacing anything wider than a space would eat a tab or a
    /// newline the vocabulary spells with its own byte token; guarding the
    /// prepend on the *replaced* text rather than the input is what makes
    /// `" a"` and `"▁a"` agree, while `"\ta"` still takes the prefix.
    pub(in crate::core::tokenizer) fn metaspace_transform(&self, text: &str, out: &mut String) {
        self.metaspace_transform_at(text, true, out)
    }

    /// [`Tokenizer::metaspace_transform`] told whether this text opens the
    /// sequence.
    ///
    /// `prepend_scheme: "first"` — every metaspace vocabulary here — prepends to
    /// the first split and no other, so a content gap that follows an added
    /// token is transformed without a prefix. Measured on mistral-7b-v0.3:
    /// `"([0-5]"` is `['▁(', '[', …]` while `"<s>([0-5]"` is `['<s>', '([', …]`,
    /// which is a different first token, not merely a missing one.
    pub(in crate::core::tokenizer) fn metaspace_transform_at(
        &self,
        text: &str,
        is_first: bool,
        out: &mut String,
    ) {
        out.reserve(text.len() + 3);
        for ch in text.chars() {
            match ch {
                ' ' => out.push('▁'),
                _ => out.push(ch),
            }
        }
        if self.add_prefix_space && is_first && !out.starts_with('▁') {
            out.insert(0, '▁');
        }
    }

    /// Run `f` on every piece a `Metaspace` node splits its (already
    /// transformed) text into.
    ///
    /// With `split: false` that is the whole text, once — which is the shape
    /// Mistral ships, and the reason this cannot be a whitespace regex: a merge
    /// there may legitimately span what looks like a word boundary.
    ///
    /// With `split: true` it is HuggingFace's `MergedWithNext` split on the
    /// replacement, so each `▁` opens a piece and carries into the word behind
    /// it. Text before the first `▁` is a piece of its own.
    pub(in crate::core::tokenizer) fn for_each_metaspace_piece(
        &self,
        text: &str,
        mut f: impl FnMut(&str),
    ) {
        if !self.metaspace_split {
            if !text.is_empty() {
                f(text);
            }
            return;
        }
        let mut start = 0;
        // `match_indices` rather than `split`: the delimiter stays with the
        // piece that follows it, so the boundaries are what is wanted, not the
        // fragments between them.
        for (at, _) in text.match_indices('▁') {
            if at > start {
                f(&text[start..at]);
            }
            start = at;
        }
        if start < text.len() {
            f(&text[start..]);
        }
    }
}
