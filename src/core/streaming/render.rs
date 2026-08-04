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
use crate::core::metaspace::WORD_BOUNDARY;
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
    /// A dense id-indexed piece list: the SentencePiece-shaped vocabularies,
    /// whose ids *are* positions in the piece vector. Shared with the tokenizer
    /// rather than copied.
    ByIndex(Arc<Vec<String>>),
}

/// How an id that denotes a raw byte rather than its own spelling is resolved.
pub(crate) enum ByteFallbackRule {
    /// The vocabulary declares no byte fallback, and is unaffected.
    None,
    /// The tokenizer's own `<0xNN>` table, inverted: fallback token id → the
    /// byte value it denotes. Shared with the tokenizer rather than copied.
    Table(Arc<FxHashMap<u32, u8>>),
    /// The byte value is read off the *surface* — any piece spelled `<0xNN>`
    /// denotes that byte — rather than from an encode-side table.
    ///
    /// Ungated on purpose, and deliberately unlike [`Table`](Self::Table): the
    /// SentencePiece-shaped vocabularies have no separate inverse table, and
    /// their reference detokenizers (`sp.decode`, llama.cpp, and HuggingFace's
    /// declared `ByteFallback` decoder step) all parse the spelling. A
    /// vocabulary of this shape therefore cannot hold a literal `<0x41>` piece
    /// that means the text `<0x41>` — the references would not decode it that
    /// way either.
    ParseSurface,
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

/// The byte a `<0xNN>` piece denotes, or `None` for any other spelling.
///
/// The parse [`ByteFallbackRule::ParseSurface`] runs, kept beside
/// [`BYTE_VALUES`] because the two are only ever used together.
fn byte_fallback_surface(piece: &str) -> Option<u8> {
    piece
        .strip_prefix("<0x")
        .and_then(|rest| rest.strip_suffix('>'))
        .and_then(|hex| u8::from_str_radix(hex, 16).ok())
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
    /// Whether a [`Surfaces::ByIndex`] piece spells word boundaries with ▁
    /// (U+2581), so that the marker becomes a space as the surface is rendered.
    ///
    /// Deliberately a *rendering* rule rather than a
    /// [`DecodePost`](super::state::DecodePost), and deliberately consulted only
    /// where a surface is rendered: a byte produced by a `<0xNN>` token is not
    /// surface text and must survive untouched. Measured with the
    /// `sentencepiece` package 0.2.0 on Mistral's own `tokenizer.model`,
    /// `decode` of the ids for `<0xE2>`, `<0x96>`, `<0x81>` is `'▁'` — the
    /// literal character — while `decode` of the `▁` piece is `''`. A post-op
    /// over reassembled text cannot tell those two apart; this can.
    ///
    /// Exactly parallel to `use_byte_level`, which is likewise a spelling rule
    /// only one of the surface arms consults. The BPE backend keeps its own
    /// metaspace substitution as a post-op: its vocabulary is byte-keyed, so a ▁
    /// can be split across two of its pieces and only reassembled text sees it.
    use_metaspace: bool,
}

impl RenderRules {
    /// Capture a backend's per-id rendering rules.
    pub(crate) fn new(
        surfaces: Surfaces,
        special_tokens_decoder: Arc<FxHashMap<u32, String>>,
        skip: Arc<FxHashSet<u32>>,
        byte_fallback: ByteFallbackRule,
        use_byte_level: bool,
        use_metaspace: bool,
    ) -> Self {
        Self {
            surfaces,
            special_tokens_decoder,
            skip,
            byte_fallback,
            use_byte_level,
            use_metaspace,
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
        // which would otherwise render the spelling. Under `Table` the id is
        // matched against the tokenizer's own encode-side table, never against
        // surfaces that merely look like `<0x..>`, so decode is the exact
        // inverse of what encode can emit and a vocabulary holding a literal
        // `<0x41>` token is untouched. (`ParseSurface` has no table to consult
        // and resolves in the `ByIndex` arm below, where the surface it parses
        // is in hand — still ahead of that surface being rendered.) The byte
        // goes into the same buffer every other byte does, which is what lets a
        // character split across several `<0xNN>` tokens reassemble when
        // streaming.
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
            Surfaces::ByIndex(pieces) => {
                if let Some(piece) = pieces.get(id as usize) {
                    // The other half of byte-fallback resolution, and still
                    // ahead of the surface being rendered: under `ParseSurface`
                    // the byte is spelled by the piece itself, so the surface
                    // has to be in hand first — and its literal spelling is
                    // then never emitted.
                    let parsed = match self.byte_fallback {
                        ByteFallbackRule::ParseSurface => byte_fallback_surface(piece),
                        ByteFallbackRule::Table(_) | ByteFallbackRule::None => None,
                    };
                    if let Some(byte) = parsed {
                        let b = byte as usize;
                        return Rendered::Bytes {
                            lead: Lead::None,
                            bytes: Cow::Borrowed(&BYTE_VALUES[b..b + 1]),
                        };
                    }
                    // No ByteLevel arm: these vocabularies spell their pieces as
                    // text. The marker they do use (▁) is substituted here,
                    // where the text being rendered is known to be a *surface* —
                    // the byte-fallback arm above has already returned, so a ▁
                    // spelled out as `<0xE2><0x96><0x81>` keeps its own
                    // character, exactly as `sp.decode` renders it. Allocates
                    // only for the pieces that actually carry the marker.
                    let bytes = if self.use_metaspace && piece.contains(WORD_BOUNDARY) {
                        Cow::Owned(piece.replace(WORD_BOUNDARY, " ").into_bytes())
                    } else {
                        Cow::Borrowed(piece.as_bytes())
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
