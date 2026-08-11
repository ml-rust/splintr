use super::super::backend::subdivide;
use super::super::types::Tokenizer;
use crate::core::byte_level::byte_level_decode;

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
    pub(super) fn prefixed<'a>(&self, text: &'a str) -> std::borrow::Cow<'a, str> {
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
    pub(super) fn normalized<'a>(&self, text: &'a str) -> std::borrow::Cow<'a, str> {
        match &self.normalizer {
            Some(norm) => norm.normalize(text),
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
    pub(in crate::core::tokenizer) fn split_chunks(&self, text: &str) -> Vec<(usize, usize)> {
        if self.chain.is_empty() {
            return self.regex.find_iter(text);
        }
        self.split_chunks_chained(text)
    }

    /// [`Tokenizer::split_chunks`] appending into a caller-owned buffer, so the
    /// encode path can hand it one that outlives the call instead of paying a
    /// malloc and a free per text.
    ///
    /// The chained branch still builds its own vector: it re-runs the matcher
    /// over every span of the previous pass, so it needs somewhere to keep that
    /// pass while producing the next, and the buffer it was handed is where the
    /// answer has to end up.
    #[inline]
    pub(in crate::core::tokenizer) fn split_chunks_into(
        &self,
        text: &str,
        out: &mut Vec<(usize, usize)>,
    ) {
        if self.chain.is_empty() {
            self.regex.find_into(text, out);
            return;
        }
        out.extend(self.split_chunks_chained(text));
    }

    /// [`Tokenizer::split_chunks`] for a multi-expression pre-tokenizer.
    ///
    /// Kept out of line so the single-expression path stays a straight call to
    /// `find_iter`. Unlike that path this keeps unmatched gaps as spans, because
    /// llama.cpp's `unicode_regex_split_stl` does — including on the FIRST pass,
    /// whose leftovers a later pass still gets to cut.
    pub(super) fn split_chunks_chained(&self, text: &str) -> Vec<(usize, usize)> {
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

        // The metaspace fork rewrites before it splits, so its pieces are spans
        // of the transformed text and not of `text` — reported here in that
        // form, which is the one the merge loop is handed and the one the
        // reference's own `Metaspace` node reports.
        if self.use_metaspace_decoder {
            let mut pieces = Vec::new();
            self.for_each_pre_token(text, |piece| pieces.push(piece.to_owned()));
            return pieces;
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

    /// Run `f` on every pre-token of `text`, exactly as `encode` produces them.
    ///
    /// The same split as [`pre_tokenize`](Self::pre_tokenize) without either of
    /// the conveniences that method adds for its caller: no `String` per piece,
    /// and no undoing of the ByteLevel mapping. Pieces arrive in the space the
    /// merge loop receives them, which is the ByteLevel alphabet wherever the
    /// encode path uses it.
    ///
    /// That makes this the form worth *timing*. `pre_tokenize` measures the
    /// split plus an allocation and a reverse mapping per piece — real costs,
    /// but ones `encode` never pays — so timing it and subtracting would charge
    /// pre-tokenization for work that belongs to nothing at all. Everything
    /// else about the two is identical, including `add_prefix_space` and the
    /// normalizer being the caller's business rather than this method's.
    pub fn for_each_pre_token(&self, text: &str, mut f: impl FnMut(&str)) {
        if let Some(pt) = &self.pre_tokenizer {
            if self.use_byte_level && self.raw_encoder.is_some() && pt.emits_raw() {
                pt.for_each_raw_piece(text, |piece| f(piece));
            } else {
                pt.for_each_piece(text, |piece| f(piece));
            }
            return;
        }

        if self.use_metaspace_decoder {
            return crate::core::scratch::with_text(|buf| {
                self.metaspace_transform(text, buf);
                self.for_each_metaspace_piece(buf, f);
            });
        }

        let text = self.prefixed(text);
        for (start, end) in self.split_chunks(&text) {
            if let Some(piece) = text.get(start..end) {
                f(piece);
            }
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
