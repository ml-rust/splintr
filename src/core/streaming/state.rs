//! The single description of "ids → text", shared by [`Tokenizer::decode`] and
//! the [`StreamingDecoder`](super::StreamingDecoder).
//!
//! Whole-sequence decoding and streaming decoding must never disagree about
//! what an id renders to, so neither of them owns that decision: a
//! [`DecodeState`] holds the positionless [`RenderRules`] plus the ordered list
//! of text post-ops, and *both* callers drive it through the same
//! [`DecodeCursor`]. Whole-sequence decode is the degenerate drive — one
//! [`feed`](DecodeCursor::feed) of every id, then a
//! [`flush`](DecodeCursor::flush) (or their strict twins, which differ only in
//! reporting invalid UTF-8 rather than substituting U+FFFD) — so drift between
//! the two is not merely tested against, it is unrepresentable.
//!
//! [`Tokenizer::decode`]: crate::Tokenizer::decode

use super::render::{ByteFallbackRule, Lead, RenderRules, Rendered, Surfaces};
use super::utf8::{InvalidUtf8, Utf8Buffer};
use rustc_hash::{FxHashMap, FxHashSet};
use std::borrow::Borrow;
use std::sync::Arc;

/// A text post-op, applied to decoded text in list order.
///
/// Every op here must distribute over concatenation (or else be given the
/// position state it needs by the cursor), because the streaming decoder
/// applies them to each emitted chunk while whole-sequence decoding applies
/// them to one big chunk, and the two must agree.
pub(crate) enum DecodePost {
    /// Convert ▁ (U+2581, lower one eighth block) to a space.
    ///
    /// Note: Unlike some tokenizer implementations, we do NOT strip leading
    /// spaces. The ▁ character represents a word boundary and should become a
    /// space. If you need to strip leading space from the very first token in a
    /// sequence, handle that at a higher level (e.g., in your generation loop).
    ///
    /// This is a per-character substitution, so it distributes over
    /// concatenation: applying it to each streamed chunk and joining gives
    /// exactly what applying it to the joined text gives. That is what lets the
    /// streaming decoder post-process incrementally and still agree with
    /// whole-sequence decoding.
    ///
    /// Only the BPE backend lists this. Its surfaces are byte strings, so a ▁
    /// can be split across two pieces and only reassembled text can see it. The
    /// SentencePiece-shaped backends substitute per surface while rendering
    /// instead — see `RenderRules`' `use_metaspace` — because for them a ▁
    /// spelled out through `<0xNN>` byte-fallback ids must keep its own
    /// character, which text this op runs over can no longer distinguish.
    MetaspaceToSpace,
    /// Remove **one** leading space from the decoded sequence, undoing
    /// SentencePiece's `add_dummy_prefix` — see
    /// [`SpmTokenizer::decode`](crate::SpmTokenizer::decode), which documents
    /// why exactly one comes off and only when one was added.
    ///
    /// The one post-op that does *not* distribute over concatenation: "leading"
    /// is a position, so it is the cursor that decides — through its `at_start`
    /// flag — which chunk this may touch. The space it looks for is the one the
    /// metaspace substitution produces from the dummy prefix's ▁, so it must run
    /// after that substitution — which, on the backend that lists this op, has
    /// already happened while the surface was rendered.
    StripLeadingSpace,
}

/// Everything decoding consults: the per-id rules, plus what to do to the text
/// they produce.
///
/// Held behind an `Arc` by the streaming decoder, which is why that decoder
/// carries no lifetime and can be owned, moved, and stored freely. Every table
/// inside [`RenderRules`] is shared rather than copied, so building one never
/// duplicates a vocabulary-sized map.
pub(crate) struct DecodeState {
    render: RenderRules,
    post: Vec<DecodePost>,
}

impl DecodeState {
    /// Capture a tokenizer's decode configuration.
    pub(crate) fn new(render: RenderRules, post: Vec<DecodePost>) -> Self {
        Self { render, post }
    }

    /// Decode configuration shared by the SentencePiece-shaped backends
    /// (SPM-BPE and Unigram): pieces indexed by id, no separate special-token
    /// table (every id has a slot in the piece vector), `<0xNN>` byte-fallback
    /// parsed off the surface, the ▁→space substitution applied per surface
    /// while rendering (not as a post-op, so byte-fallback bytes keep the
    /// literal ▁ — see `RenderRules::new`'s `use_metaspace`), and the
    /// dummy/metaspace-prefix space stripped exactly when one was added.
    ///
    /// The two backends differ in how `skipped` is composed (their
    /// `bos_token_id`/`eos_token_id` shapes are not identical) and in what they
    /// call the leading marker, but the `DecodeState` those decisions produce
    /// is the same shape, so only this construction is shared.
    pub(crate) fn for_piece_vocab(
        id_to_token: &Arc<Vec<String>>,
        skipped: FxHashSet<u32>,
        add_prefix_space: bool,
    ) -> Self {
        let post = if add_prefix_space {
            vec![DecodePost::StripLeadingSpace]
        } else {
            Vec::new()
        };

        Self::new(
            RenderRules::new(
                Surfaces::ByIndex(Arc::clone(id_to_token)),
                Arc::new(FxHashMap::default()),
                Arc::new(skipped),
                ByteFallbackRule::ParseSurface,
                false,
                true,
            ),
            post,
        )
    }

    /// The per-id rendering rules, for a caller that wants bytes rather than
    /// text (`Tokenizer::decode_bytes`) and therefore no cursor at all.
    pub(crate) fn render(&self) -> &RenderRules {
        &self.render
    }

    /// A cursor over these rules, borrowing them, with the byte buffer
    /// pre-sized for a drive whose length is known up front — which is exactly
    /// the whole-sequence case. (A stream cannot know its length, so
    /// [`StreamingDecoder`](super::StreamingDecoder) builds its cursor from an
    /// `Arc` through [`DecodeCursor::new`] instead.)
    pub(crate) fn cursor_with_capacity(&self, capacity: usize) -> DecodeCursor<&Self> {
        DecodeCursor::with_capacity(self, capacity)
    }

    /// Apply the post-ops, in order.
    ///
    /// `at_start` is the cursor's "nothing has been emitted yet" flag, threaded
    /// in because the position-dependent ops need it — and *consumed here*, at
    /// the first chunk that carries any character at all. A chunk can be empty
    /// (a flush with nothing buffered), and a push can render nothing (every id
    /// skipped, or bytes that are still an incomplete UTF-8 sequence); neither
    /// spends the flag, because neither emitted a character for the strip to
    /// look at.
    fn postprocess(&self, text: String, at_start: &mut bool) -> String {
        if text.is_empty() {
            return text;
        }
        let text = self.post.iter().fold(text, |text, op| match op {
            // Replace ▁ with space - this preserves word boundaries
            DecodePost::MetaspaceToSpace => text.replace('\u{2581}', " "),
            // Only the very first emitted character can be the dummy prefix.
            DecodePost::StripLeadingSpace if *at_start => match text.strip_prefix(' ') {
                Some(rest) => rest.to_string(),
                None => text,
            },
            DecodePost::StripLeadingSpace => text,
        });
        *at_start = false;
        text
    }
}

/// The one place position-dependent decode state lives — the UTF-8 reassembly
/// buffer, and the "what has been emitted so far" flags the position-dependent
/// [`Lead`]/[`DecodePost`] variants need. Nothing else in decoding is allowed
/// to remember where it is in the sequence.
///
/// Generic over how the state is held so the *same* code drives both callers:
/// whole-sequence decoding borrows a `&DecodeState` it built on the spot, while
/// the streaming decoder holds an `Arc<DecodeState>` and keeps its cursor alive
/// across calls. Neither can reach the rendering rules except through here.
pub(crate) struct DecodeCursor<S> {
    state: S,
    bytes: Utf8Buffer,
    /// Whether this cursor has emitted a character yet, for the post-ops that
    /// are about position rather than about content
    /// ([`DecodePost::StripLeadingSpace`]).
    ///
    /// Spent at the first emitted *character*, not the first
    /// [`feed`](Self::feed): a push whose ids are all skipped, or whose bytes
    /// are still an incomplete UTF-8 sequence, renders nothing and leaves the
    /// flag armed — so a leading BOS cannot eat the dummy-prefix strip.
    at_start: bool,
}

impl<S: Borrow<DecodeState>> DecodeCursor<S> {
    /// A fresh cursor over `state`.
    pub(crate) fn new(state: S) -> Self {
        Self {
            state,
            bytes: Utf8Buffer::new(),
            at_start: true,
        }
    }

    /// The same, with the byte buffer pre-sized.
    pub(crate) fn with_capacity(state: S, capacity: usize) -> Self {
        Self {
            state,
            bytes: Utf8Buffer::with_capacity(capacity),
            at_start: true,
        }
    }

    /// Render `ids` into the UTF-8 buffer, deferring the "id not in any table"
    /// decision to `on_unknown` so the strict and lossy entry points cannot
    /// drift into two different notions of "unknown".
    ///
    /// Split out from [`feed`](Self::feed) so the strict drive runs the exact
    /// same rendering loop — and, crucially, in the same order: every id is
    /// rendered (and every unknown-id error raised) before the buffer is asked
    /// what it can decode, which is the order whole-sequence decoding has
    /// always reported errors in.
    fn render_into<E>(
        &mut self,
        ids: &[u32],
        on_unknown: impl Fn(u32) -> Result<(), E>,
    ) -> Result<(), E> {
        let rules = self.state.borrow().render();

        for &id in ids {
            match rules.render(id) {
                Rendered::Skipped => {}
                Rendered::Bytes { lead, bytes } => {
                    match lead {
                        Lead::None => {}
                    }
                    self.bytes.push(&bytes);
                }
                Rendered::Unknown => on_unknown(id)?,
            }
        }

        Ok(())
    }

    /// Feed ids and return whatever text became decidable, substituting U+FFFD
    /// for bytes that can never be valid UTF-8.
    ///
    /// The lossy drive: what the streaming decoder always does (a stream cannot
    /// see the future) and what `Tokenizer::decode_lossy` does in one shot.
    pub(crate) fn feed<E>(
        &mut self,
        ids: &[u32],
        on_unknown: impl Fn(u32) -> Result<(), E>,
    ) -> Result<Option<String>, E> {
        self.render_into(ids, on_unknown)?;
        match self.bytes.take_complete() {
            Some(text) => Ok(Some(self.postprocess(text))),
            None => Ok(None),
        }
    }

    /// Run the state's post-ops over one emitted chunk, handing them this
    /// cursor's position flag — the only state they are allowed to consult.
    fn postprocess(&mut self, text: String) -> String {
        // Disjoint fields: the rules are borrowed out of `state` while
        // `at_start` is borrowed mutably, which is exactly why the flag lives
        // on the cursor and not inside the shared `DecodeState`.
        let state = self.state.borrow();
        state.postprocess(text, &mut self.at_start)
    }

    /// The strict twin of [`feed`](Self::feed): a byte that is definitively not
    /// valid UTF-8 is reported through `on_invalid_utf8` instead of being
    /// substituted, so `Tokenizer::decode` stays strict when driven here.
    ///
    /// Bytes that are merely *incomplete* stay buffered, exactly as in the
    /// lossy drive; they become an error only at
    /// [`finish_strict`](Self::finish_strict), where no further bytes can
    /// arrive to complete them.
    pub(crate) fn feed_strict<E>(
        &mut self,
        ids: &[u32],
        on_unknown: impl Fn(u32) -> Result<(), E>,
        on_invalid_utf8: impl Fn() -> E,
    ) -> Result<Option<String>, E> {
        self.render_into(ids, on_unknown)?;
        match self.bytes.take_complete_strict() {
            Ok(Some(text)) => Ok(Some(self.postprocess(text))),
            Ok(None) => Ok(None),
            Err(InvalidUtf8) => Err(on_invalid_utf8()),
        }
    }

    /// Flush any remaining buffered bytes.
    ///
    /// If there are incomplete UTF-8 sequences in the buffer, they will be
    /// replaced with the Unicode replacement character (U+FFFD).
    pub(crate) fn flush(&mut self) -> String {
        let text = self.bytes.flush();
        self.postprocess(text)
    }

    /// The strict twin of [`flush`](Self::flush): a trailing sequence that is
    /// still incomplete has run out of bytes that could complete it, so it is
    /// reported through `on_invalid_utf8` rather than becoming U+FFFD.
    pub(crate) fn finish_strict<E>(
        &mut self,
        on_invalid_utf8: impl Fn() -> E,
    ) -> Result<String, E> {
        match self.bytes.flush_strict() {
            Ok(text) => Ok(self.postprocess(text)),
            Err(InvalidUtf8) => Err(on_invalid_utf8()),
        }
    }

    /// Reset the cursor, discarding any buffered bytes.
    ///
    /// The cursor is then indistinguishable from a freshly built one.
    pub(crate) fn reset(&mut self) {
        self.bytes.clear();
        self.at_start = true;
    }

    /// Check if there are buffered bytes waiting for completion.
    pub(crate) fn has_pending(&self) -> bool {
        self.bytes.has_pending()
    }

    /// The number of pending bytes in the buffer.
    pub(crate) fn pending_bytes(&self) -> usize {
        self.bytes.pending_len()
    }
}

#[cfg(test)]
mod tests {
    use super::DecodeCursor;
    use crate::core::tokenizer::Tokenizer;
    use rustc_hash::FxHashMap;
    use std::convert::Infallible;

    /// A raw BPE vocabulary whose single-byte tokens let a test split a
    /// multi-byte character across as many ids as it has bytes.
    fn make_test_tokenizer() -> Tokenizer {
        let mut encoder = FxHashMap::default();
        for b in 0u8..=255 {
            encoder.insert(vec![b], b as u32);
        }
        encoder.insert("Hello".as_bytes().to_vec(), 256);
        encoder.insert("世界".as_bytes().to_vec(), 257);

        Tokenizer::new(encoder, FxHashMap::default(), r".").expect("the test pattern compiles")
    }

    /// Drive a cursor over `ids` in chunks of `chunk`, concatenating every
    /// emission plus the final flush.
    fn drive(tokenizer: &Tokenizer, ids: &[u32], chunk: usize) -> String {
        let state = tokenizer.decode_state();
        let mut cursor = DecodeCursor::new(&state);
        let mut out = String::new();

        for group in ids.chunks(chunk.max(1)) {
            let emitted = match cursor.feed(group, |_| Ok::<(), Infallible>(())) {
                Ok(text) => text,
                // `Infallible` has no values, so this match has no arms to write.
                Err(never) => match never {},
            };
            out.push_str(&emitted.unwrap_or_default());
        }
        out.push_str(&cursor.flush());
        out
    }

    /// The invariant the whole design rests on: the cursor is the only thing
    /// that knows about position, so *how* ids reach it cannot change the text
    /// that comes out — only when.
    #[test]
    fn one_shot_drive_equals_token_by_token_drive() {
        let tokenizer = make_test_tokenizer();

        for text in ["", "Hello", "Hello 世界!", "héllo — ünïcode 🎉"] {
            let ids = tokenizer.encode(text);
            let one_shot = drive(&tokenizer, &ids, ids.len().max(1));

            assert_eq!(one_shot, drive(&tokenizer, &ids, 1), "text: {text:?}");
            // ...and every grouping in between.
            for chunk in 1..=ids.len() {
                assert_eq!(one_shot, drive(&tokenizer, &ids, chunk), "text: {text:?}");
            }
            // ...and it is what whole-sequence decoding produces.
            assert_eq!(one_shot, tokenizer.decode_lossy(&ids), "text: {text:?}");
        }
    }
}
