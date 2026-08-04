//! The single description of "ids → text", shared by [`Tokenizer::decode`] and
//! the [`StreamingDecoder`](super::StreamingDecoder).
//!
//! Whole-sequence decoding and streaming decoding must never disagree about
//! what an id renders to, so neither of them owns that decision: both consult a
//! [`DecodeView`]. `decode` borrows the tokenizer's own tables directly; the
//! streaming decoder borrows the owned [`DecodeState`] it was handed by
//! [`Tokenizer::streaming_decoder`], which is why the stream cannot be built
//! for the wrong kind of vocabulary and cannot drift from `decode`.
//!
//! [`Tokenizer::decode`]: crate::Tokenizer::decode
//! [`Tokenizer::streaming_decoder`]: crate::Tokenizer::streaming_decoder

use crate::core::byte_level::byte_level_decode_bytes;
use rustc_hash::{FxHashMap, FxHashSet};
use std::borrow::Cow;
use std::sync::Arc;

/// What a single token id renders to.
pub(crate) enum Rendered<'a> {
    /// A `special=true` added token: dropped (HF default `skip_special_tokens`).
    /// A deliberate skip, never an unknown id.
    Skipped,
    /// The token's bytes, with any ByteLevel alphabet already unmapped — these
    /// are real bytes, ready for the UTF-8 buffer.
    Bytes(Cow<'a, [u8]>),
    /// In no table at all: neither the vocabulary nor the special tokens.
    Unknown,
}

/// `[0, 1, …, 255]`, so a resolved byte-fallback byte can be handed out as a
/// borrowed one-byte slice instead of allocating a `Vec` per token.
static BYTE_VALUES: [u8; 256] = {
    let mut values = [0u8; 256];
    let mut b = 0usize;
    while b < 256 {
        values[b] = b as u8;
        b += 1;
    }
    values
};

/// A borrowed view of everything decoding consults.
///
/// `Copy` and pointer-sized, so taking one per decode call (or per streaming
/// push) costs nothing and no table is ever cloned to obtain it.
#[derive(Clone, Copy)]
pub(crate) struct DecodeView<'a> {
    pub(crate) decoder: &'a FxHashMap<u32, Vec<u8>>,
    pub(crate) special_tokens_decoder: &'a FxHashMap<u32, String>,
    pub(crate) special_decode_ids: &'a FxHashSet<u32>,
    /// The tokenizer's own `<0xNN>` table, inverted: fallback token id → the
    /// byte value it denotes. `None` for a vocabulary that declares no byte
    /// fallback, which is then unaffected.
    pub(crate) byte_fallback: Option<&'a FxHashMap<u32, u8>>,
    pub(crate) use_byte_level: bool,
    pub(crate) use_metaspace_decoder: bool,
}

impl<'a> DecodeView<'a> {
    /// Render one id, deferring the "not in any table" decision to the caller
    /// so strict and lossy decoding cannot drift into two different notions of
    /// "unknown".
    pub(crate) fn render(&self, id: u32) -> Rendered<'a> {
        // Drop `special=true` added tokens (HF default skip_special_tokens).
        if self.special_decode_ids.contains(&id) {
            return Rendered::Skipped;
        }

        // A `<0xNN>` byte-fallback token denotes a byte, not its literal
        // spelling, so it resolves to that byte — before the vocabulary lookup,
        // which would otherwise render the spelling. The id is matched against
        // the tokenizer's own encode-side table, never against surfaces that
        // merely look like `<0x..>`, so decode is the exact inverse of what
        // encode can emit and a vocabulary holding a literal `<0x41>` token is
        // untouched. The byte goes into the same buffer every other byte does,
        // which is what lets a character split across several `<0xNN>` tokens
        // reassemble when streaming.
        if let Some(&byte) = self.byte_fallback.and_then(|table| table.get(&id)) {
            let b = byte as usize;
            return Rendered::Bytes(Cow::Borrowed(&BYTE_VALUES[b..b + 1]));
        }

        if let Some(bytes) = self.decoder.get(&id) {
            return if self.use_byte_level {
                match byte_level_decode_bytes(bytes) {
                    Some(decoded) => Rendered::Bytes(Cow::Owned(decoded)),
                    // Fallback: a surface the ByteLevel alphabet cannot explain
                    // is passed through as raw bytes.
                    None => Rendered::Bytes(Cow::Borrowed(bytes.as_slice())),
                }
            } else {
                Rendered::Bytes(Cow::Borrowed(bytes.as_slice()))
            };
        }

        // Special tokens are never ByteLevel-encoded: their text is their text.
        match self.special_tokens_decoder.get(&id) {
            Some(special) => Rendered::Bytes(Cow::Borrowed(special.as_bytes())),
            None => Rendered::Unknown,
        }
    }

    /// Post-process decoded text for metaspace-decoder tokenizers.
    ///
    /// Converts ▁ (U+2581, lower one eighth block) to space.
    ///
    /// Note: Unlike some tokenizer implementations, we do NOT strip leading spaces.
    /// The ▁ character represents a word boundary and should become a space.
    /// If you need to strip leading space from the very first token in a sequence,
    /// handle that at a higher level (e.g., in your generation loop).
    ///
    /// This is a per-character substitution, so it distributes over
    /// concatenation: applying it to each streamed chunk and joining gives
    /// exactly what applying it to the joined text gives. That is what lets the
    /// streaming decoder post-process incrementally and still agree with
    /// whole-sequence decoding.
    #[inline]
    pub(crate) fn postprocess(&self, text: String) -> String {
        if self.use_metaspace_decoder {
            // Replace ▁ with space - this preserves word boundaries
            text.replace('\u{2581}', " ")
        } else {
            text
        }
    }
}

/// The owned twin of [`DecodeView`], carrying exactly the tokenizer
/// configuration decoding consults and nothing else.
///
/// Held behind an `Arc` by the streaming decoder, which is why that decoder
/// carries no lifetime and can be owned, moved, and stored freely. The
/// vocabulary map — the one large table — is itself shared rather than copied,
/// so building a decoder never duplicates a 100k-entry map.
pub(crate) struct DecodeState {
    decoder: Arc<FxHashMap<u32, Vec<u8>>>,
    special_tokens_decoder: FxHashMap<u32, String>,
    special_decode_ids: FxHashSet<u32>,
    /// The inverted `<0xNN>` table, shared with the tokenizer rather than
    /// copied — the same reasoning as the vocabulary map above.
    byte_fallback: Option<Arc<FxHashMap<u32, u8>>>,
    use_byte_level: bool,
    use_metaspace_decoder: bool,
}

impl DecodeState {
    /// Capture a tokenizer's decode configuration.
    pub(crate) fn new(
        decoder: Arc<FxHashMap<u32, Vec<u8>>>,
        special_tokens_decoder: FxHashMap<u32, String>,
        special_decode_ids: FxHashSet<u32>,
        byte_fallback: Option<Arc<FxHashMap<u32, u8>>>,
        use_byte_level: bool,
        use_metaspace_decoder: bool,
    ) -> Self {
        Self {
            decoder,
            special_tokens_decoder,
            special_decode_ids,
            byte_fallback,
            use_byte_level,
            use_metaspace_decoder,
        }
    }

    /// Borrow this state as the same view whole-sequence decoding uses.
    pub(crate) fn view(&self) -> DecodeView<'_> {
        DecodeView {
            decoder: &self.decoder,
            special_tokens_decoder: &self.special_tokens_decoder,
            special_decode_ids: &self.special_decode_ids,
            byte_fallback: self.byte_fallback.as_deref(),
            use_byte_level: self.use_byte_level,
            use_metaspace_decoder: self.use_metaspace_decoder,
        }
    }
}
