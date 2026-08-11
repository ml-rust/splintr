use super::super::types::{Guard, RunSplit, Tokenizer};

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
        // Declared `split: true` cuts at every marker; a proven split cuts only
        // at run starts, and only where no token spans the cut — which for most
        // vocabularies is everywhere, so the guard check is skipped entirely
        // rather than asked per marker.
        let (run_starts_only, guarded) = match self.metaspace_split {
            true => (false, None),
            false => match self.marker_run_split() {
                RunSplit::Never => {
                    if !text.is_empty() {
                        f(text);
                    }
                    return;
                }
                split @ RunSplit::Allowed { guards, .. } => {
                    (true, (!guards.is_empty()).then_some(split))
                }
            },
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
            if let Some(split) = guarded {
                if at > 0 && !split.cuts_at(text.as_bytes(), at) {
                    continue;
                }
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

    /// Where this vocabulary's own tokens allow a chunk to be cut at the start
    /// of a marker run. See [`RunSplit`] for why such a cut can be free.
    ///
    /// A token can only span a run start by carrying a marker that is neither
    /// its first character nor part of its leading run — `…x▁…`. Every such
    /// token becomes a [`Guard`]; a cut is taken unless a guard sits across it.
    /// mistral-7b-v0.3 yields none at all (of 32,768 tokens exactly 14 hold an
    /// interior marker and every one is a pure run), Gemma yields the single
    /// `>▁</`.
    ///
    /// Beyond [`MAX_GUARDS`] the vocabulary is not really metaspace-shaped and
    /// the per-cut check would cost more than the merge it saves, so nothing is
    /// cut. A vocabulary with no marker token at all is not metaspace at all.
    ///
    /// Proven once per tokenizer, on first encode.
    pub(in crate::core::tokenizer) fn marker_run_split(&self) -> &RunSplit {
        self.metaspace_run_split.get_or_init(|| {
            let mut guards: Vec<Guard> = Vec::new();
            let mut marker_tokens = false;

            for token in self.encoder.keys() {
                // Past the leading run: a marker there is the token's own
                // first character and spans nothing.
                let mut pos = leading_run_len(token);
                marker_tokens |= pos > 0;
                while let Some(off) = memchr::memmem::find(&token[pos..], MARKER.as_bytes()) {
                    if guards.len() == MAX_GUARDS {
                        return RunSplit::Never;
                    }
                    let at = pos + off;
                    guards.push(Guard {
                        token: token.into(),
                        at,
                    });
                    // Only a run's *start* is ever a cut, so step over the rest
                    // of this one before looking for the next.
                    pos = at + leading_run_len(&token[at..]);
                }
            }

            if !marker_tokens {
                return RunSplit::Never;
            }
            let mut preceded_by = Box::new([false; 256]);
            for guard in &guards {
                preceded_by[guard.token[guard.at - 1] as usize] = true;
            }
            RunSplit::Allowed {
                guards: guards.into(),
                preceded_by,
            }
        })
    }
}

/// How many tokens may span a marker-run start before the cut is abandoned.
const MAX_GUARDS: usize = 8;

/// Length of the run of markers `token` opens with, in bytes.
fn leading_run_len(token: &[u8]) -> usize {
    let mut len = 0;
    while token[len..].starts_with(MARKER.as_bytes()) {
        len += MARKER.len();
    }
    len
}

impl RunSplit {
    /// Whether cutting `chunk` at `pos` — the start of a marker run — leaves
    /// every id unchanged.
    ///
    /// Free unless a guard token straddles the cut. The byte before the cut
    /// answers that for almost every position without touching the list, and
    /// for the vocabularies with no guard at all it answers it for every one.
    #[inline]
    pub(in crate::core::tokenizer) fn cuts_at(&self, chunk: &[u8], pos: usize) -> bool {
        let RunSplit::Allowed {
            guards,
            preceded_by,
        } = self
        else {
            return false;
        };
        // Most metaspace vocabularies hold no token that can span a cut at all,
        // and then there is nothing to look up.
        if guards.is_empty() {
            return true;
        }
        if !preceded_by[chunk[pos - 1] as usize] {
            return true;
        }
        !guards.iter().any(|guard| {
            let Some(from) = pos.checked_sub(guard.at) else {
                return false;
            };
            chunk[from..].starts_with(&guard.token)
        })
    }
}
