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
use crate::core::decoder::wordpiece_cleanup;
use crate::core::policy::SpecialDecode;
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
    /// Drop the space before `. ? ! ,` — HuggingFace `tokenizers`'
    /// WordPiece-decoder cleanup (`cleanup = true`, its default). Unlike
    /// `transformers`' `clean_up_tokenization_spaces`, it does NOT touch
    /// apostrophe contractions.
    ///
    /// The other op that does not distribute over concatenation, and for a
    /// different reason than [`StripLeadingSpace`](Self::StripLeadingSpace): the
    /// space it removes is emitted with the token *before* the punctuation,
    /// which on a stream can be a whole chunk earlier. So the cursor holds back
    /// the trailing run of spaces instead of emitting it, and the next chunk is
    /// cleaned as held+next; [`flush`](DecodeCursor::flush) emits whatever is
    /// still held.
    ///
    /// It is the whole *run*, never one space: `str::replace` is single-pass and
    /// does not rescan what it produced, so `"a  ."` is `"a ."` — holding one
    /// space would offer the replacement a different string to scan than
    /// whole-sequence decoding gives it.
    CleanupTokenization,
}

/// The [`DecodePost::CleanupTokenization`] replacement itself, over one string.
///
/// Single-pass per pattern, deliberately: this is `str::replace`, which is what
/// the reference cleanup is, and re-running it until it converges would eat the
/// second space of `"a  ."`.
fn cleanup_tokenization(s: &str) -> String {
    s.replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
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

    /// This configuration under an explicit [`SpecialDecode`].
    ///
    /// [`Skip`](SpecialDecode::Skip) is the identity — the state a backend
    /// builds already drops what it declares special — so the mode is applied
    /// here rather than threaded through every constructor above: one place
    /// decides what "render the specials" means, and whole-sequence decoding and
    /// streaming reach it through the same call, which is what keeps a stream
    /// agreeing with `decode` in *both* modes.
    pub(crate) fn with_special_decode(mut self, specials: SpecialDecode) -> Self {
        self.render = match specials {
            SpecialDecode::Skip => self.render,
            SpecialDecode::Render => self.render.rendering_specials(),
        };
        self
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
    ///
    /// `held_spaces` is the trailing run of spaces
    /// [`DecodePost::CleanupTokenization`] kept back from an earlier chunk, so
    /// that a punctuation mark arriving in this one can still consume it. An
    /// empty chunk leaves it alone: there is nothing to clean and nothing new to
    /// hold, and the run must survive until either a chunk or the flush claims
    /// it.
    fn postprocess(&self, text: String, at_start: &mut bool, held_spaces: &mut String) -> String {
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
            DecodePost::CleanupTokenization => {
                // Re-run the replacement over held+next, which is the same
                // string whole-sequence decoding hands it, and hold the new
                // trailing run in turn.
                let mut combined = std::mem::take(held_spaces);
                combined.push_str(&text);
                let mut cleaned = cleanup_tokenization(&combined);
                let kept = cleaned.trim_end_matches(' ').len();
                held_spaces.push_str(&cleaned[kept..]);
                cleaned.truncate(kept);
                cleaned
            }
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
    /// Whether a token has rendered yet, for [`Lead::SpaceUnlessFirst`] — a
    /// *different* question from [`at_start`](Self::at_start), and deliberately
    /// not the same flag: this one is about tokens, that one about characters.
    ///
    /// Spent by the first id that renders bytes at all, even zero of them: a
    /// WordPiece `##`-only piece renders the empty string and still means the
    /// next word start needs a separator, which is exactly what the
    /// whole-sequence decode's `pieces.is_empty()` predicate said. A skipped or
    /// unknown id renders nothing and leaves it armed, so a leading `[CLS]`
    /// cannot put a space in front of the first word.
    rendered_a_token: bool,
    /// The trailing run of spaces [`DecodePost::CleanupTokenization`] is holding
    /// back for a punctuation mark that may arrive in a later chunk. Emitted by
    /// [`flush`](Self::flush) when none does.
    ///
    /// Always empty on a backend that does not list that op, which is why the
    /// flush-time append costs those backends nothing.
    held_spaces: String,
    /// The run of [`Rendered::RunByte`] bytes being accumulated for
    /// [`ByteFallbackRule::DeclaredRun`], which is decoded as a whole the moment
    /// the run ends — at the first token that is not one of its bytes, or at the
    /// flush. Held here rather than pushed into the buffer because the declared
    /// step's invalid-run rule (one U+FFFD per byte) is not the buffer's, and
    /// only a whole run can be judged.
    ///
    /// The same shape as `held_spaces` above: a run's end is
    /// simply not knowable until something else arrives, so a stream carries it
    /// forward instead of guessing. Always empty on every backend that does not
    /// declare that rule, which is the whole cost they pay for it existing.
    byte_run: Vec<u8>,
}

/// End a cursor's byte-fallback run, pushing what it decodes to into the
/// buffer: valid UTF-8 is the text it spells, and an invalid run is **one U+FFFD
/// per byte** — HuggingFace's declared `ByteFallback` rule, which is
/// deliberately not the buffer's maximal-subpart rule. This is the per-run
/// decision `byte_fallback` makes in the `decoder` module, reproduced over a run
/// that is held rather than known up front.
///
/// A free function over the two fields rather than a method, so it can be called
/// while the rendering rules are borrowed out of the cursor's state.
#[inline]
fn end_byte_run(run: &mut Vec<u8>, buffer: &mut Utf8Buffer) {
    if run.is_empty() {
        return;
    }
    match std::str::from_utf8(run) {
        Ok(text) => buffer.push(text.as_bytes()),
        Err(_) => {
            for _ in 0..run.len() {
                buffer.push("\u{fffd}".as_bytes());
            }
        }
    }
    run.clear();
}

impl<S: Borrow<DecodeState>> DecodeCursor<S> {
    /// A fresh cursor over `state`.
    pub(crate) fn new(state: S) -> Self {
        Self {
            state,
            bytes: Utf8Buffer::new(),
            at_start: true,
            rendered_a_token: false,
            held_spaces: String::new(),
            byte_run: Vec::new(),
        }
    }

    /// The same, with the byte buffer pre-sized.
    pub(crate) fn with_capacity(state: S, capacity: usize) -> Self {
        Self {
            state,
            bytes: Utf8Buffer::with_capacity(capacity),
            at_start: true,
            rendered_a_token: false,
            held_spaces: String::new(),
            byte_run: Vec::new(),
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
        // Under `plain_by_id`'s shape — every tiktoken-style BPE vocabulary —
        // everything the general loop re-decides per id is a constant, so the
        // shape is established once here and the per-id work is the skip check,
        // one map lookup and the special-token fallback that `render` would
        // itself reduce to. Semantics are `render`'s, not a second set: this is
        // the same rules object, inspected rather than reimplemented.
        {
            let rules = self.state.borrow().render();
            if let Some(map) = rules.plain_by_id() {
                for &id in ids {
                    if rules.skips(id) {
                        continue;
                    }
                    // No `end_byte_run` here, and deliberately: the shape
                    // excludes every byte-fallback rule, so `Rendered::RunByte`
                    // is unreachable and `byte_run` is provably still empty.
                    // The `Lead` is likewise constantly `Lead::None` — the
                    // `ById` arm of `render` hardcodes it — so no separator is
                    // ever emitted and `rendered_a_token` is only written, never
                    // read.
                    match map.get(&id) {
                        Some(bytes) => {
                            self.bytes.push(bytes);
                            // Even when `bytes` was empty: a token rendered.
                            self.rendered_a_token = true;
                        }
                        None => match rules.special_surface(id) {
                            Some(text) => {
                                self.bytes.push(text.as_bytes());
                                self.rendered_a_token = true;
                            }
                            None => on_unknown(id)?,
                        },
                    }
                }
                return Ok(());
            }
        }

        self.render_into_general(ids, on_unknown)
    }

    /// The general rendering loop: one [`RenderRules::render`] per id, with
    /// every position-dependent decision the cursor owns applied around it.
    ///
    /// Split out from [`render_into`](Self::render_into) only so the specialized
    /// loop above has something to be checked against — it is the definition of
    /// what that loop must reproduce, and both are driven over the same ids by
    /// this module's tests.
    fn render_into_general<E>(
        &mut self,
        ids: &[u32],
        on_unknown: impl Fn(u32) -> Result<(), E>,
    ) -> Result<(), E> {
        let rules = self.state.borrow().render();

        for &id in ids {
            match rules.render(id) {
                // Neither a skipped nor an unknown id ends a byte-fallback run:
                // the declared chain never sees them at all (its token list is
                // already special-skipped), so a run must survive one.
                Rendered::Skipped => {}
                Rendered::Bytes { lead, bytes } => {
                    end_byte_run(&mut self.byte_run, &mut self.bytes);
                    // The separator goes through the same buffer the token's own
                    // bytes do, so it is part of the text the post-ops then see —
                    // which is what lets the cleanup op eat it.
                    let separated = match lead {
                        Lead::None => false,
                        Lead::SpaceUnlessFirst => self.rendered_a_token,
                    };
                    if rules.unit_cleanup() {
                        // The declared WordPiece cleanup, over the token *plus*
                        // the separator it carries — the `" {t}"` the declared
                        // decoder cleans, which is why the separator cannot
                        // simply be pushed ahead of it.
                        let text = String::from_utf8_lossy(&bytes);
                        let mut unit = String::with_capacity(text.len() + 1);
                        if separated {
                            unit.push(' ');
                        }
                        unit.push_str(&text);
                        self.bytes.push(wordpiece_cleanup(&unit).as_bytes());
                    } else {
                        if separated {
                            self.bytes.push(b" ");
                        }
                        self.bytes.push(&bytes);
                    }
                    // Even when `bytes` was empty: a token rendered.
                    self.rendered_a_token = true;
                }
                Rendered::RunByte(byte) => {
                    self.byte_run.push(byte);
                    self.rendered_a_token = true;
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
        // `at_start` and the held space run are borrowed mutably, which is
        // exactly why those live on the cursor and not inside the shared
        // `DecodeState`.
        let state = self.state.borrow();
        state.postprocess(text, &mut self.at_start, &mut self.held_spaces)
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
        // No further token can extend the byte-fallback run, so this is where it
        // ends — before the buffer is drained, since it feeds that buffer.
        end_byte_run(&mut self.byte_run, &mut self.bytes);
        let text = self.bytes.flush();
        let mut text = self.postprocess(text);
        text.push_str(&self.take_held_spaces());
        text
    }

    /// The space run [`DecodePost::CleanupTokenization`] was holding for a
    /// punctuation mark that never came, disarmed as it is taken.
    ///
    /// Always empty on every other backend, so this is the whole cost they pay
    /// for the op existing.
    fn take_held_spaces(&mut self) -> String {
        std::mem::take(&mut self.held_spaces)
    }

    /// The strict twin of [`flush`](Self::flush): a trailing sequence that is
    /// still incomplete has run out of bytes that could complete it, so it is
    /// reported through `on_invalid_utf8` rather than becoming U+FFFD.
    ///
    /// A byte-fallback run still ends the way the declared step says it does,
    /// U+FFFD and all: that substitution is the declared decoder's own
    /// output, not a UTF-8 recovery, so strictness has nothing to report about
    /// it. No backend that declares the strict drive declares that rule.
    pub(crate) fn finish_strict<E>(
        &mut self,
        on_invalid_utf8: impl Fn() -> E,
    ) -> Result<String, E> {
        end_byte_run(&mut self.byte_run, &mut self.bytes);
        match self.bytes.flush_strict() {
            Ok(text) => {
                let mut text = self.postprocess(text);
                text.push_str(&self.take_held_spaces());
                Ok(text)
            }
            Err(InvalidUtf8) => Err(on_invalid_utf8()),
        }
    }

    /// Reset the cursor, discarding any buffered bytes.
    ///
    /// The cursor is then indistinguishable from a freshly built one.
    pub(crate) fn reset(&mut self) {
        self.bytes.clear();
        self.at_start = true;
        self.rendered_a_token = false;
        self.held_spaces.clear();
        self.byte_run.clear();
    }

    /// Check if there is content that has arrived but not yet been emitted.
    ///
    /// That is all three holds, not just the UTF-8 buffer: an unfinished
    /// byte-fallback run, and a trailing run of spaces held back for a
    /// punctuation mark that may still arrive, are both text the caller has
    /// fed and cannot yet see.
    pub(crate) fn has_pending(&self) -> bool {
        self.bytes.has_pending() || !self.byte_run.is_empty() || !self.held_spaces.is_empty()
    }

    /// The number of pending bytes, across the UTF-8 buffer, an unfinished
    /// byte-fallback run and a held trailing space run.
    pub(crate) fn pending_bytes(&self) -> usize {
        self.bytes.pending_len() + self.byte_run.len() + self.held_spaces.len()
    }
}

#[cfg(test)]
mod tests {
    use super::super::render::{ByteFallbackRule, RenderRules, Surfaces};
    use super::{DecodeCursor, DecodeState};
    use crate::core::tokenizer::Tokenizer;
    use rustc_hash::{FxHashMap, FxHashSet};
    use std::cell::RefCell;
    use std::convert::Infallible;
    use std::sync::Arc;

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

    /// [`ByteFallbackRule::ParseSurface`] — the rule the SentencePiece-shaped
    /// backends run — reads a surface with the *same* strict parser the declared
    /// `ByteFallback` step uses: exactly two hex digits, either case. So `<0x1>`
    /// is text on this path too, where a second, lenient parser once made it
    /// byte `0x01` here and text under the declared step.
    ///
    /// Strictness measured against `tokenizers` 0.22.1: its
    /// `decoders.ByteFallback` decodes `<0x4a>`/`<0x4A>` to `"J"` and passes
    /// `<0x1>` and `<0x041>` through as their spelling. SentencePiece's own
    /// vocabularies never disagree — `mistral-7b-v0.3/tokenizer.model` spells
    /// all 256 byte pieces with two upper-case hex digits.
    #[test]
    fn parse_surface_byte_fallback_is_strict_two_hex_digits() {
        let pieces = Arc::new(vec![
            "<0x41>".to_string(),
            "<0x4a>".to_string(),
            "<0x1>".to_string(),
            "<0x041>".to_string(),
            "<0xG1>".to_string(),
        ]);
        let state = DecodeState::for_piece_vocab(&pieces, FxHashSet::default(), false);

        let render = state.render();
        // Two hex digits resolve to the byte, in either case.
        assert_eq!(render.token_bytes(0), Some(vec![0x41]));
        assert_eq!(render.token_bytes(1), Some(vec![0x4a]));
        // Everything else is the surface itself.
        assert_eq!(render.token_bytes(2), Some(b"<0x1>".to_vec()));
        assert_eq!(render.token_bytes(3), Some(b"<0x041>".to_vec()));
        assert_eq!(render.token_bytes(4), Some(b"<0xG1>".to_vec()));

        // ...and the same through a drive, at every chunking.
        let ids: Vec<u32> = (0..pieces.len() as u32).collect();
        for chunk in 1..=ids.len() {
            let mut cursor = state.cursor_with_capacity(ids.len() * 4);
            let mut out = String::new();
            for group in ids.chunks(chunk) {
                let emitted = match cursor.feed(group, |_| Ok::<(), Infallible>(())) {
                    Ok(text) => text,
                    Err(never) => match never {},
                };
                out.push_str(&emitted.unwrap_or_default());
            }
            out.push_str(&cursor.flush());
            assert_eq!(out, "AJ<0x1><0x041><0xG1>", "in chunks of {chunk}");
        }
    }

    /// Which of the two rendering loops a drive uses — the only thing the
    /// equivalence test below varies.
    #[derive(Clone, Copy)]
    enum Loop {
        Specialized,
        General,
    }

    /// A decode state in the shape [`RenderRules::plain_by_id`] names, holding
    /// every case the specialized loop distinguishes: ordinary surfaces, a
    /// surface that is *empty* (id 3), a skipped id (4), an id that only the
    /// special-token table knows (5), and — by omission — ids in no table at all.
    fn plain_state() -> DecodeState {
        let mut surfaces = FxHashMap::default();
        surfaces.insert(
            1u32,
            crate::core::token_bytes::TokenBytes::from(b"He".to_vec()),
        );
        surfaces.insert(
            2u32,
            crate::core::token_bytes::TokenBytes::from(b"llo".to_vec()),
        );
        surfaces.insert(3u32, crate::core::token_bytes::TokenBytes::from(Vec::new()));

        let mut specials = FxHashMap::default();
        specials.insert(5u32, "<eos>".to_string());

        let skip: FxHashSet<u32> = [4u32].into_iter().collect();

        DecodeState::new(
            RenderRules::new(
                Surfaces::ById(Arc::new(surfaces)),
                Arc::new(specials),
                Arc::new(skip),
                ByteFallbackRule::None,
                false,
                false,
            ),
            Vec::new(),
        )
    }

    /// Drive one loop over `ids`, reporting everything the two must agree on:
    /// the text rendered, the token flag, and which ids reached `on_unknown`.
    ///
    /// Unknown ids are recorded rather than raised so a single sequence can
    /// exercise the unknown arm *and* everything after it.
    fn render_with(state: &DecodeState, ids: &[u32], which: Loop) -> (String, bool, Vec<u32>) {
        let mut cursor = DecodeCursor::new(state);
        let unknown = RefCell::new(Vec::new());
        let record = |id: u32| {
            unknown.borrow_mut().push(id);
            Ok::<(), Infallible>(())
        };
        let outcome = match which {
            Loop::Specialized => cursor.render_into(ids, record),
            Loop::General => cursor.render_into_general(ids, record),
        };
        match outcome {
            Ok(()) => {}
            // `Infallible` has no values, so this match has no arms to write.
            Err(never) => match never {},
        }
        let rendered_a_token = cursor.rendered_a_token;
        (cursor.flush(), rendered_a_token, unknown.into_inner())
    }

    /// The specialization in [`DecodeCursor::render_into`] is a loop shape, not
    /// a second set of rules — so it is checked against the general loop rather
    /// than trusted to match it. Any future condition dropped from
    /// [`RenderRules::plain_by_id`] shows up here as a divergence.
    #[test]
    fn specialized_loop_agrees_with_general_loop() {
        let state = plain_state();
        assert!(
            state.render().plain_by_id().is_some(),
            "the fixture must take the specialized path, or this proves nothing"
        );

        for ids in [
            // Every arm on its own, so a divergence names its own case.
            vec![1],
            vec![3],
            vec![4],
            vec![5],
            vec![9],
            vec![],
            // ...and mixed, which is where the ordering between the surface
            // lookup, the skip set and the special table can drift.
            vec![3, 1, 4, 2, 5, 9, 4, 3, 1],
            vec![4, 9, 5, 1],
        ] {
            let specialized = render_with(&state, &ids, Loop::Specialized);
            let general = render_with(&state, &ids, Loop::General);
            assert_eq!(specialized, general, "ids: {ids:?}");
        }
    }

    /// The two cases the equivalence check above would also pass on if *both*
    /// loops were wrong, pinned to their absolute answers.
    #[test]
    fn specialized_loop_renders_the_expected_text() {
        let state = plain_state();
        let ids = [3, 1, 4, 2, 5, 9, 4, 3, 1];

        let (text, rendered_a_token, unknown) = render_with(&state, &ids, Loop::Specialized);
        assert_eq!(text, "Hello<eos>He");
        assert!(rendered_a_token);
        assert_eq!(unknown, vec![9]);

        // An empty surface still *rendered a token* — the distinction
        // `rendered_a_token` exists to make, and the one a `map.get` hit
        // returning zero bytes could silently lose.
        let (text, rendered_a_token, unknown) = render_with(&state, &[3], Loop::Specialized);
        assert_eq!(text, "");
        assert!(rendered_a_token);
        assert!(unknown.is_empty());

        // A skipped id renders nothing and leaves the flag armed.
        let (_, rendered_a_token, _) = render_with(&state, &[4], Loop::Specialized);
        assert!(!rendered_a_token);
    }
}
