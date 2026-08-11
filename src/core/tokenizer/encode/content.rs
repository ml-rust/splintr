use super::super::types::Tokenizer;

/// SentencePiece's metaspace marker, `U+2581`. Spelled once: it is three bytes
/// wide, and the split below steps by exactly that.
pub(in crate::core::tokenizer) const MARKER: &str = "▁";
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
        // Every space grows by two bytes, so a text that is all spaces triples.
        // Sizing for the text plus a marker covers ordinary prose in one
        // allocation and leaves the pathological case to grow, which is what a
        // per-thread buffer settles into anyway.
        out.reserve(text.len() + MARKER.len());

        // The prefix is decided from the INPUT rather than inserted into the
        // finished buffer: `String::insert(0, ..)` shifts every byte already
        // written, which on a whole document is a memmove the size of the
        // document. `out` starts with the marker exactly when the text starts
        // with a space (which becomes one) or with a marker already.
        if self.add_prefix_space && is_first && !text.starts_with(' ') && !text.starts_with(MARKER)
        {
            out.push_str(MARKER);
        }

        // Copy the runs between spaces wholesale. Pushing one `char` at a time
        // re-encodes every character of the document through `char::encode_utf8`
        // for the one byte value that changes.
        let bytes = text.as_bytes();
        let mut start = 0;
        for at in memchr::memchr_iter(b' ', bytes) {
            out.push_str(&text[start..at]);
            out.push_str(MARKER);
            start = at + 1;
        }
        out.push_str(&text[start..]);
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
    ///
    /// The `split: false` case still cuts when
    /// [`metaspace_runs_are_boundaries`](Self::metaspace_runs_are_boundaries)
    /// proves the cuts change nothing — at the start of each marker RUN, never
    /// inside one, which is the weaker split that proof licenses.
    pub(in crate::core::tokenizer) fn for_each_metaspace_piece(
        &self,
        text: &str,
        mut f: impl FnMut(&str),
    ) {
        let run_starts_only = if self.metaspace_split {
            false
        } else if self.metaspace_runs_are_boundaries() {
            true
        } else {
            if !text.is_empty() {
                f(text);
            }
            return;
        };

        let mut start = 0;
        let mut prev_end = usize::MAX;
        // `match_indices` rather than `split`: the delimiter stays with the
        // piece that follows it, so the boundaries are what is wanted, not the
        // fragments between them.
        for (at, _) in text.match_indices(MARKER) {
            // A marker directly behind this one means we are mid-run, and a run
            // may merge into a `▁▁`-style token — so only its first marker is a
            // boundary when the split was proven rather than declared.
            let mid_run = run_starts_only && at == prev_end;
            prev_end = at + MARKER.len();
            if mid_run {
                continue;
            }
            if at > start {
                f(&text[start..at]);
            }
            start = at;
        }
        if start < text.len() {
            f(&text[start..]);
        }
    }

    /// Whether cutting a `split: false` metaspace text at the start of every
    /// marker run yields the same ids as never cutting it at all.
    ///
    /// It does when no vocabulary token carries a marker **after** a non-marker
    /// character. A merge that spanned a run's start would have to produce a
    /// symbol shaped `…x▁…`; if the vocabulary holds no such token, that merge
    /// can never form, so the cut removes nothing that could have happened. Runs
    /// themselves are left whole because `▁▁`, `▁▁▁`, … *are* tokens.
    ///
    /// mistral-7b-v0.3 passes: of 32,768 tokens exactly 14 hold an interior
    /// marker and every one is a pure run. A vocabulary that fails keeps the
    /// literal whole-text merge.
    ///
    /// Proven once per tokenizer, on first encode.
    fn metaspace_runs_are_boundaries(&self) -> bool {
        *self.metaspace_run_split.get_or_init(|| {
            self.encoder.keys().all(|token| {
                // Past the leading run, no marker may appear.
                let rest = token
                    .chunks_exact(MARKER.len())
                    .position(|c| c != MARKER.as_bytes())
                    .map_or(token.len(), |runs| runs * MARKER.len());
                memchr::memmem::find(&token[rest..], MARKER.as_bytes()).is_none()
            })
        })
    }
}
