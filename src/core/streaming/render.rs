//! The positionless half of decoding: what a single token id renders to.
//!
//! [`RenderRules`] answers "what bytes does this id stand for?" and nothing
//! else. It knows no position, holds no buffer and never sees the ids around
//! the one it is asked about, which is what lets whole-sequence decoding and
//! streaming decoding share it verbatim: an id cannot render differently just
//! because it arrived in a different chunk. Everything position-dependent —
//! UTF-8 reassembly, the ordered text post-ops, "is this the first character?"
//! — lives in [`DecodeCursor`](super::state::DecodeCursor) instead.
//!
//! The rules are expressed as backend-agnostic knobs ([`Surfaces`],
//! [`ByteFallbackRule`]) rather than as BPE fields, so the SentencePiece, SPM
//! and WordPiece backends can adopt the same shape by adding a variant rather
//! than by growing a second, drifting description of "ids → text".

use crate::core::byte_level::byte_level_decode_bytes;
use rustc_hash::{FxHashMap, FxHashSet};
use std::borrow::Cow;
use std::sync::Arc;

/// Where a token id's surface bytes come from.
///
/// Each backend keys its surfaces the way its vocabulary is keyed; the variant
/// records which, so decoding never has to guess.
pub(crate) enum Surfaces {
    /// A sparse id → bytes table: the BPE vocabulary, shared with the
    /// tokenizer rather than copied.
    ById(Arc<FxHashMap<u32, Vec<u8>>>),
}

/// How an id that denotes a raw byte rather than its own spelling is resolved.
pub(crate) enum ByteFallbackRule {
    /// The vocabulary declares no byte fallback, and is unaffected.
    None,
    /// The tokenizer's own `<0xNN>` table, inverted: fallback token id → the
    /// byte value it denotes. Shared with the tokenizer rather than copied.
    Table(Arc<FxHashMap<u32, u8>>),
}

/// A separator a rendered token carries *before* its own bytes.
///
/// Positionless on its own — whether the separator is actually emitted is the
/// cursor's decision, since only the cursor knows whether anything has been
/// rendered yet.
pub(crate) enum Lead {
    /// No separator: the token's bytes are the whole of its rendering. Every
    /// current backend, BPE included.
    None,
}

/// What a single token id renders to.
pub(crate) enum Rendered<'a> {
    /// A `special=true` added token: dropped (HF default `skip_special_tokens`).
    /// A deliberate skip, never an unknown id.
    Skipped,
    /// The token's bytes, with any ByteLevel alphabet already unmapped — these
    /// are real bytes, ready for the UTF-8 buffer.
    Bytes { lead: Lead, bytes: Cow<'a, [u8]> },
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

/// The complete per-id rendering rule set: everything decoding consults that
/// does not depend on where in the sequence an id appears.
///
/// Every table is shared (`Arc`) rather than copied, so capturing a
/// tokenizer's rules — which happens once per whole-sequence decode and once
/// per streaming decoder — never duplicates a 100k-entry map.
pub(crate) struct RenderRules {
    surfaces: Surfaces,
    special_tokens_decoder: Arc<FxHashMap<u32, String>>,
    /// Ids to drop outright (HF `skip_special_tokens`).
    skip: Arc<FxHashSet<u32>>,
    byte_fallback: ByteFallbackRule,
    use_byte_level: bool,
}

impl RenderRules {
    /// Capture a backend's per-id rendering rules.
    pub(crate) fn new(
        surfaces: Surfaces,
        special_tokens_decoder: Arc<FxHashMap<u32, String>>,
        skip: Arc<FxHashSet<u32>>,
        byte_fallback: ByteFallbackRule,
        use_byte_level: bool,
    ) -> Self {
        Self {
            surfaces,
            special_tokens_decoder,
            skip,
            byte_fallback,
            use_byte_level,
        }
    }

    /// Render one id, deferring the "not in any table" decision to the caller
    /// so strict and lossy decoding cannot drift into two different notions of
    /// "unknown".
    pub(crate) fn render(&self, id: u32) -> Rendered<'_> {
        // Drop `special=true` added tokens (HF default skip_special_tokens).
        if self.skip.contains(&id) {
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
        if let ByteFallbackRule::Table(table) = &self.byte_fallback {
            if let Some(&byte) = table.get(&id) {
                let b = byte as usize;
                return Rendered::Bytes {
                    lead: Lead::None,
                    bytes: Cow::Borrowed(&BYTE_VALUES[b..b + 1]),
                };
            }
        }

        match &self.surfaces {
            Surfaces::ById(map) => {
                if let Some(bytes) = map.get(&id) {
                    let bytes = if self.use_byte_level {
                        match byte_level_decode_bytes(bytes) {
                            Some(decoded) => Cow::Owned(decoded),
                            // Fallback: a surface the ByteLevel alphabet cannot
                            // explain is passed through as raw bytes.
                            None => Cow::Borrowed(bytes.as_slice()),
                        }
                    } else {
                        Cow::Borrowed(bytes.as_slice())
                    };
                    return Rendered::Bytes {
                        lead: Lead::None,
                        bytes,
                    };
                }
            }
        }

        // Special tokens are never ByteLevel-encoded: their text is their text.
        match self.special_tokens_decoder.get(&id) {
            Some(special) => Rendered::Bytes {
                lead: Lead::None,
                bytes: Cow::Borrowed(special.as_bytes()),
            },
            None => Rendered::Unknown,
        }
    }
}
