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

use crate::core::byte_level::{byte_level_decode, byte_level_decode_bytes};
use crate::core::decoder::parse_byte_token;
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
    /// denotes that byte — rather than from an encode-side table. Parsed with
    /// [`parse_byte_token`], the same two-hex-digit reading every other rule
    /// uses, so `<0x5>` is text here too.
    ///
    /// Ungated on purpose, and deliberately unlike [`Table`](Self::Table): the
    /// SentencePiece-shaped vocabularies have no separate inverse table, and
    /// their reference detokenizers (`sp.decode`, llama.cpp, and HuggingFace's
    /// declared `ByteFallback` decoder step) all parse the spelling. A
    /// vocabulary of this shape therefore cannot hold a literal `<0x41>` piece
    /// that means the text `<0x41>` — the references would not decode it that
    /// way either.
    ParseSurface,
    /// HuggingFace's *declared* `ByteFallback` decoder step, as lowered by
    /// [`Decoder::lower`](crate::core::decoder::Decoder::lower).
    ///
    /// Reads the byte off the surface exactly as
    /// [`ParseSurface`](Self::ParseSurface) does — and renders
    /// it as [`Rendered::RunByte`] rather than as bytes, because the declared
    /// step does *not* decode its bytes the way the UTF-8 buffer does: a run
    /// that is not valid UTF-8 becomes one U+FFFD **per byte**, not one per
    /// maximal subpart. Measured against `tokenizers` 0.22.1 on Mistral's
    /// `tokenizer.json`: `<0xE2> <0x41>` is `"\u{FFFD}\u{FFFD}"` (std's lossy
    /// rule would give `"\u{FFFD}A"`) and `<0xF0> <0x9F>` is
    /// `"\u{FFFD}\u{FFFD}"` (std's would give a single `"\u{FFFD}"`).
    ///
    /// The run itself is held by [`DecodeCursor`](super::state::DecodeCursor),
    /// which is the only thing allowed to know where in the sequence it is — a
    /// run ends at the first non-byte token or at the flush, and neither is
    /// knowable from one id.
    DeclaredRun,
}

/// A separator a rendered token carries *before* its own bytes.
///
/// Positionless on its own — whether the separator is actually emitted is the
/// cursor's decision, since only the cursor knows whether anything has been
/// rendered yet.
pub(crate) enum Lead {
    /// No separator: the token's bytes are the whole of its rendering. The BPE
    /// and SentencePiece-shaped backends, whose surfaces spell their own
    /// spacing.
    None,
    /// A single space, unless this is the first token to render — WordPiece,
    /// whose surfaces carry no spacing of their own.
    ///
    /// "First" means the first *token*, not the first character: a `##`-only
    /// piece renders the empty string and still counts, which is what the
    /// whole-sequence decode's `pieces.is_empty()` predicate meant.
    SpaceUnlessFirst,
}

/// Whether a surface begins a word, and how that is read off its spelling.
///
/// WordPiece's alone: its vocabulary stores bare word fragments and marks
/// *continuations* rather than word starts, so the space between words exists
/// nowhere in the surfaces and has to be put back while rendering. Every other
/// backend's surfaces already spell their own spacing, which is
/// [`None`](Self::None).
pub(crate) enum WordSeparator {
    /// Surfaces spell their own spacing; no token ever carries a separator.
    None,
    /// Every surface begins a word, so every token after the first carries a
    /// separator. The GGUF-stripped WordPiece vocabularies, whose `##` markers
    /// were removed and whose continuations are therefore indistinguishable
    /// from word starts.
    EveryToken,
    /// A surface beginning with this marker (`##`) continues the previous word:
    /// the marker comes off and no separator is carried. Any other surface
    /// begins a word.
    ///
    /// Never an empty marker — every surface begins with that, so nothing would
    /// ever start a word. A vocabulary with no marker is
    /// [`EveryToken`](Self::EveryToken), which is the opposite reading.
    Continuation(String),
}

impl WordSeparator {
    /// Whether [`split`](Self::split) can only ever answer [`Lead::None`], so a
    /// caller that has no separator to emit need not consult it per token.
    #[inline]
    fn is_trivial(&self) -> bool {
        match self {
            Self::None => true,
            Self::EveryToken | Self::Continuation(_) => false,
        }
    }

    /// Split a surface into the separator it carries and the text that is left
    /// once any continuation marker is removed.
    fn split<'a>(&self, piece: &'a str) -> (Lead, &'a str) {
        match self {
            Self::None => (Lead::None, piece),
            Self::EveryToken => (Lead::SpaceUnlessFirst, piece),
            Self::Continuation(marker) => match piece.strip_prefix(marker.as_str()) {
                Some(rest) => (Lead::None, rest),
                None => (Lead::SpaceUnlessFirst, piece),
            },
        }
    }
}

/// What a single token id renders to.
pub(crate) enum Rendered<'a> {
    /// A `special=true` added token: dropped (HF default `skip_special_tokens`).
    /// A deliberate skip, never an unknown id.
    Skipped,
    /// The token's bytes, with any ByteLevel alphabet already unmapped — these
    /// are real bytes, ready for the UTF-8 buffer.
    Bytes { lead: Lead, bytes: Cow<'a, [u8]> },
    /// One byte of a [`ByteFallbackRule::DeclaredRun`] run: it joins the run the
    /// cursor is accumulating instead of going into the UTF-8 buffer, because
    /// the declared step decodes a whole run at once. Never produced by any
    /// other rule.
    RunByte(u8),
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
    /// The inclusive `(min, max)` id span `skip` covers, or `None` when `skip`
    /// is empty — recomputed by every site that assigns `skip` (see
    /// [`skip_span`]).
    ///
    /// Every vocabulary here puts its special ids in a high, narrow band well
    /// above the ordinary token ids (llama3 specials start at 128000 with
    /// ordinary ids below; cl100k at 100257), so an ordinary id almost never
    /// falls inside the span. [`skips`](Self::skips) checks the span first —
    /// two integer comparisons — before touching the hash table, which the
    /// hot decode path calls once per token id. The span only *rejects* ids
    /// the hash table would also have rejected; membership is still decided
    /// by `skip` alone, so this cannot change the answer, only skip the
    /// lookup.
    skip_span: Option<(u32, u32)>,
    byte_fallback: ByteFallbackRule,
    use_byte_level: bool,
    /// Literal substitutions applied, in order, to a [`Surfaces::ByIndex`]
    /// surface as it is rendered — the ▁ (U+2581) → space substitution every
    /// SentencePiece-shaped backend declares through
    /// [`new`](Self::new)'s `use_metaspace`, and any further per-token `Replace`
    /// a declared `tokenizer.json` decoder pipeline lowers onto here.
    ///
    /// Deliberately a *rendering* rule rather than a
    /// [`DecodePost`](super::state::DecodePost), and deliberately consulted only
    /// where a surface is rendered: a byte produced by a `<0xNN>` token is not
    /// surface text and must survive untouched. Measured with the
    /// `sentencepiece` package 0.2.0 on Mistral's own `tokenizer.model`,
    /// `decode` of the ids for `<0xE2>`, `<0x96>`, `<0x81>` is `'▁'` — the
    /// literal character — while `decode` of the `▁` piece is `''`. A post-op
    /// over reassembled text cannot tell those two apart; this can. It is also
    /// the order HuggingFace's declared chains use, where `Replace` precedes
    /// `ByteFallback`.
    ///
    /// Exactly parallel to `use_byte_level`, which is likewise a spelling rule.
    /// The BPE backend keeps its own metaspace substitution as a post-op: its
    /// vocabulary is byte-keyed, so a ▁ can be split across two of its pieces
    /// and only reassembled text sees it — which is also why this list is not
    /// consulted in the [`Surfaces::ById`] arm.
    surface_replace: Vec<(String, String)>,
    /// Whether a rendered token is cleaned *as a token* with
    /// [`wordpiece_cleanup`](crate::core::decoder::wordpiece_cleanup) — the
    /// declared `WordPiece` decoder's `cleanup`, which HuggingFace applies per
    /// token and not to the joined text.
    ///
    /// Per token is not a detail: the joined text contains `" ' "` wherever a
    /// bare apostrophe token sits between two others, and cleaning the join
    /// would collapse it, while HuggingFace — which never sees the following
    /// token's leading space — leaves `"don ' t"` alone. So the unit cleaned is
    /// the token's own text *plus* the separator it carries, which is exactly
    /// what the declared decoder's `" {t}"` is; the cursor applies it, because
    /// only the cursor knows whether the separator is emitted.
    ///
    /// The GGUF WordPiece backend does not set this: it declares
    /// [`DecodePost::CleanupTokenization`](super::state::DecodePost::CleanupTokenization)
    /// instead, which is the punctuation half over joined text.
    unit_cleanup: bool,
    /// Whether a [`Surfaces::ByIndex`] surface has to be given back the word
    /// spacing its vocabulary does not spell — see [`WordSeparator`], which only
    /// the WordPiece backend sets to anything but [`None`](WordSeparator::None).
    ///
    /// Set through a builder rather than through a [`new`](Self::new) parameter,
    /// so the backends with no word separator to declare — every other one — say
    /// nothing at all.
    word_separator: WordSeparator,
}

/// The inclusive `(min, max)` id span a skip set covers, or `None` when it is
/// empty — see `RenderRules::skip_span` for why this is worth precomputing.
#[inline]
fn skip_span(skip: &FxHashSet<u32>) -> Option<(u32, u32)> {
    skip.iter().fold(None, |span, &id| match span {
        Some((lo, hi)) => Some((lo.min(id), hi.max(id))),
        None => Some((id, id)),
    })
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
        let skip_span = skip_span(&skip);
        let mut rules = Self {
            surfaces,
            special_tokens_decoder,
            skip,
            skip_span,
            byte_fallback,
            use_byte_level,
            surface_replace: Vec::new(),
            unit_cleanup: false,
            word_separator: WordSeparator::None,
        };
        if use_metaspace {
            rules = rules.with_surface_replace(WORD_BOUNDARY.to_string(), " ".to_string());
        }
        rules
    }

    /// The rendering knobs a declared `tokenizer.json` decoder pipeline states,
    /// over an *empty* vocabulary.
    ///
    /// [`Decoder::lower`](crate::core::decoder::Decoder::lower) knows the
    /// pipeline but not the tokenizer it will run against, so the tables are
    /// supplied afterwards by [`with_vocabulary`](Self::with_vocabulary).
    /// Until they are, every id renders [`Rendered::Unknown`].
    pub(crate) fn declared(byte_fallback: ByteFallbackRule, use_byte_level: bool) -> Self {
        Self {
            surfaces: Surfaces::ByIndex(Arc::new(Vec::new())),
            special_tokens_decoder: Arc::new(FxHashMap::default()),
            skip: Arc::new(FxHashSet::default()),
            skip_span: None,
            byte_fallback,
            use_byte_level,
            surface_replace: Vec::new(),
            unit_cleanup: false,
            word_separator: WordSeparator::None,
        }
    }

    /// Give [`declared`](Self::declared) rules the tokenizer tables they render
    /// against. Every table is shared rather than copied, as in [`new`](Self::new).
    pub(crate) fn with_vocabulary(
        mut self,
        surfaces: Surfaces,
        special_tokens_decoder: Arc<FxHashMap<u32, String>>,
        skip: Arc<FxHashSet<u32>>,
    ) -> Self {
        self.surfaces = surfaces;
        self.special_tokens_decoder = special_tokens_decoder;
        self.skip_span = skip_span(&skip);
        self.skip = skip;
        self
    }

    /// Empty the skip set, so every id the vocabulary declares special renders
    /// its own spelling instead of nothing —
    /// [`SpecialDecode::Render`](crate::SpecialDecode::Render).
    ///
    /// Sound *because* the four concrete backends put nothing else in that set:
    /// each builds it from its declared `special=true` ids (plus, for the
    /// SentencePiece-shaped ones, the vocabulary's own BOS/EOS/`<unk>`), all of
    /// which have a surface to render. The one skip set that also holds ids with
    /// *no* surface is [`AnyTokenizer`](crate::AnyTokenizer)'s declared-pipeline
    /// one, which is why that path composes its set itself rather than calling
    /// this — clearing it wholesale there would render an empty slot as an empty
    /// surface, carrying a word separator with it.
    pub(crate) fn rendering_specials(mut self) -> Self {
        self.skip = Arc::new(FxHashSet::default());
        self.skip_span = None;
        self
    }

    /// Append a literal substitution applied to a surface as it is rendered —
    /// see `surface_replace`. Order is declaration order, which is the order the
    /// declared chain applies its `Replace` steps in.
    pub(crate) fn with_surface_replace(mut self, from: String, to: String) -> Self {
        self.surface_replace.push((from, to));
        self
    }

    /// Declare the per-token WordPiece cleanup — see `unit_cleanup`.
    pub(crate) fn with_unit_cleanup(mut self) -> Self {
        self.unit_cleanup = true;
        self
    }

    /// Whether a rendered token is cleaned as a token, for the cursor — which
    /// applies it, because the separator that is part of the cleaned unit is the
    /// cursor's decision.
    #[inline]
    pub(crate) fn unit_cleanup(&self) -> bool {
        self.unit_cleanup
    }

    /// Declare that this vocabulary's surfaces carry no word spacing of their
    /// own, so rendering has to put it back — see [`WordSeparator`].
    ///
    /// WordPiece's alone. Whether a declared separator is actually emitted stays
    /// the cursor's decision: [`Lead::SpaceUnlessFirst`] is a position-dependent
    /// rule, and rendering knows no position.
    pub(crate) fn with_word_separator(mut self, word_separator: WordSeparator) -> Self {
        self.word_separator = word_separator;
        self
    }

    /// The byte a surface denotes under [`ByteFallbackRule::DeclaredRun`], or
    /// `None` under every other rule and for every other spelling.
    ///
    /// Parses with [`parse_byte_token`], the one byte-token parser every rule
    /// shares: exactly two hex digits, so `<0x5>` is text, and reproducing
    /// `Decoder::decode` means reproducing that.
    #[inline]
    fn declared_run_byte(&self, surface: &[u8]) -> Option<u8> {
        match self.byte_fallback {
            ByteFallbackRule::DeclaredRun => {
                std::str::from_utf8(surface).ok().and_then(parse_byte_token)
            }
            ByteFallbackRule::ParseSurface
            | ByteFallbackRule::Table(_)
            | ByteFallbackRule::None => None,
        }
    }

    /// Whether any declared substitution has anything to do to this surface —
    /// checked first so that the pieces that carry no pattern (nearly all of
    /// them) are rendered without allocating.
    #[inline]
    fn surface_replace_applies(&self, piece: &str) -> bool {
        self.surface_replace
            .iter()
            .any(|(from, _)| piece.contains(from.as_str()))
    }

    /// The bytes one id contributes to decoded output, for the callers that
    /// want a *token* rather than a sequence — `Tokenize::decode_token_bytes`
    /// and its text sibling.
    ///
    /// Exactly [`render`](Self::render)'s answer, flattened: a skipped id
    /// contributes nothing (`Some(empty)`, a deliberate no-op and never an
    /// error), a byte-fallback run byte contributes that byte — as
    /// `Tokenizer::decode_bytes` also reads it, since the declared step's
    /// U+FFFD-per-byte rule is about text and there is no text here — and an id
    /// in no table at all is `None`, left for the caller to name.
    ///
    /// The [`Lead`] a surface carries is deliberately dropped: it is a separator
    /// *between* tokens, which is a fact about the sequence and not about this
    /// id, and only the cursor knows whether it is emitted.
    pub(crate) fn token_bytes(&self, id: u32) -> Option<Vec<u8>> {
        match self.render(id) {
            Rendered::Skipped => Some(Vec::new()),
            Rendered::Bytes { lead: _, bytes } => Some(bytes.into_owned()),
            Rendered::RunByte(byte) => Some(vec![byte]),
            Rendered::Unknown => None,
        }
    }

    /// Whether this id is dropped outright — the skip check
    /// [`render`](Self::render) makes first, exposed so a caller that has
    /// already established the rendering shape can make it without going
    /// through the full match.
    ///
    /// `skip_span` is checked first: every vocabulary here clusters its
    /// special ids into a high, narrow band, so an ordinary id — the common
    /// case on the hot decode path — is rejected by two integer comparisons
    /// and never touches the hash table. The hash lookup remains the sole
    /// authority on membership; the span can only skip a lookup that would
    /// have missed, never change the answer.
    #[inline]
    pub(crate) fn skips(&self, id: u32) -> bool {
        match self.skip_span {
            Some((lo, hi)) => (lo..=hi).contains(&id) && self.skip.contains(&id),
            None => false,
        }
    }

    /// The special token's own spelling, or `None` if the id names none — the
    /// last lookup [`render`](Self::render) makes, exposed for the same reason
    /// as [`skips`](Self::skips).
    ///
    /// Never ByteLevel-encoded, as in [`render`](Self::render): a special
    /// token's text is its text.
    #[inline]
    pub(crate) fn special_surface(&self, id: u32) -> Option<&str> {
        self.special_tokens_decoder.get(&id).map(String::as_str)
    }

    /// The id → bytes table, but only under the shape in which
    /// [`render`](Self::render) reduces to a skip-check, one map lookup and the
    /// special-token fallback — the shape every tiktoken-style BPE vocabulary
    /// has, and the one the per-id loop in
    /// [`DecodeCursor`](super::state::DecodeCursor) is specialized on.
    ///
    /// That shape is: surfaces keyed [`ById`](Surfaces::ById), no byte fallback
    /// at all (so [`Rendered::RunByte`] is unreachable and `declared_run_byte`
    /// is constantly `None`), no ByteLevel unmapping, no per-token cleanup, no
    /// surface substitutions, and a [`WordSeparator`] that can only answer
    /// [`Lead::None`]. Everything the general path re-decides per id is then a
    /// constant, which is what makes the specialized loop a loop-shape
    /// optimization rather than a second set of rules.
    pub(crate) fn plain_by_id(&self) -> Option<&FxHashMap<u32, Vec<u8>>> {
        let map = match &self.surfaces {
            Surfaces::ById(map) => map,
            Surfaces::ByIndex(_) => return None,
        };
        let plain = matches!(self.byte_fallback, ByteFallbackRule::None)
            && !self.use_byte_level
            && !self.unit_cleanup
            && self.surface_replace.is_empty()
            && self.word_separator.is_trivial();
        plain.then_some(map.as_ref())
    }

    /// Render one id, deferring the "not in any table" decision to the caller
    /// so strict and lossy decoding cannot drift into two different notions of
    /// "unknown".
    #[inline]
    pub(crate) fn render(&self, id: u32) -> Rendered<'_> {
        // Drop `special=true` added tokens (HF default skip_special_tokens).
        if self.skips(id) {
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
                    // The declared `ByteFallback` step parses the surface, so —
                    // exactly as in the `ByIndex` arm — it resolves here, with
                    // the surface in hand and ahead of it being rendered.
                    if let Some(byte) = self.declared_run_byte(bytes) {
                        return Rendered::RunByte(byte);
                    }
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
                        ByteFallbackRule::ParseSurface => parse_byte_token(piece),
                        ByteFallbackRule::DeclaredRun
                        | ByteFallbackRule::Table(_)
                        | ByteFallbackRule::None => None,
                    };
                    if let Some(byte) = parsed {
                        let b = byte as usize;
                        return Rendered::Bytes {
                            lead: Lead::None,
                            bytes: Cow::Borrowed(&BYTE_VALUES[b..b + 1]),
                        };
                    }
                    // The declared step's own byte fallback, which is a *run*
                    // rather than a byte in the buffer — see
                    // [`ByteFallbackRule::DeclaredRun`].
                    if let Some(byte) = self.declared_run_byte(piece.as_bytes()) {
                        return Rendered::RunByte(byte);
                    }
                    // The word separator is read off the surface, and the marker
                    // that decides it comes off with it — after the
                    // byte-fallback arm above, so a `<0xNN>` token is a byte and
                    // never a word start. Positionless: whether the separator is
                    // emitted is the cursor's call.
                    let (lead, piece) = self.word_separator.split(piece);
                    // These vocabularies spell their pieces as text, so the
                    // substitutions run here, where the text being rendered is
                    // known to be a *surface* — the byte-fallback arms above have
                    // already returned, so a ▁ spelled out as
                    // `<0xE2><0x96><0x81>` keeps its own character, exactly as
                    // `sp.decode` renders it. Allocates only for the pieces that
                    // actually carry one of the patterns.
                    //
                    // The ByteLevel unmapping is the alternative spelling rule,
                    // never a companion to the substitutions: a declared chain
                    // that mixes the two is not lowered at all (see
                    // `Decoder::lower`), so their order never has to be decided.
                    let bytes = if self.use_byte_level {
                        match byte_level_decode(piece) {
                            Some(decoded) => Cow::Owned(decoded),
                            // Fallback: a surface the ByteLevel alphabet cannot
                            // explain is passed through as raw bytes.
                            None => Cow::Borrowed(piece.as_bytes()),
                        }
                    } else if self.surface_replace_applies(piece) {
                        let replaced = self
                            .surface_replace
                            .iter()
                            .fold(piece.to_string(), |text, (from, to)| {
                                text.replace(from.as_str(), to.as_str())
                            });
                        Cow::Owned(replaced.into_bytes())
                    } else {
                        Cow::Borrowed(piece.as_bytes())
                    };
                    return Rendered::Bytes { lead, bytes };
                }
            }
        }

        // Special tokens are never ByteLevel-encoded: their text is their text.
        match self.special_surface(id) {
            Some(special) => Rendered::Bytes {
                lead: Lead::None,
                bytes: Cow::Borrowed(special.as_bytes()),
            },
            None => Rendered::Unknown,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{ByteFallbackRule, RenderRules, Surfaces, WordSeparator};
    use rustc_hash::{FxHashMap, FxHashSet};
    use std::sync::Arc;

    /// The BPE shape [`RenderRules::plain_by_id`] names: id-keyed surfaces and
    /// nothing else declared.
    fn plain(byte_fallback: ByteFallbackRule, use_byte_level: bool) -> RenderRules {
        let mut surfaces = FxHashMap::default();
        surfaces.insert(1u32, b"Hi".to_vec());
        RenderRules::new(
            Surfaces::ById(Arc::new(surfaces)),
            Arc::new(FxHashMap::default()),
            Arc::new(FxHashSet::default()),
            byte_fallback,
            use_byte_level,
            false,
        )
    }

    /// Each of the conditions below has its own test, so an edit that drops one
    /// from the predicate fails a named test rather than silently widening the
    /// set of vocabularies the specialized loop claims to cover.
    #[test]
    fn plain_by_id_accepts_the_plain_bpe_shape() {
        let rules = plain(ByteFallbackRule::None, false);
        let map = rules.plain_by_id().expect("the plain BPE shape qualifies");
        assert_eq!(map.get(&1).map(Vec::as_slice), Some(b"Hi".as_slice()));
    }

    #[test]
    fn plain_by_id_rejects_by_index_surfaces() {
        let rules = RenderRules::new(
            Surfaces::ByIndex(Arc::new(vec!["Hi".to_string()])),
            Arc::new(FxHashMap::default()),
            Arc::new(FxHashSet::default()),
            ByteFallbackRule::None,
            false,
            false,
        );
        assert!(rules.plain_by_id().is_none());
    }

    #[test]
    fn plain_by_id_rejects_byte_fallback() {
        // Every rule but `None` can make `render` answer something the
        // specialized loop has no arm for.
        let table = Arc::new(FxHashMap::default());
        assert!(plain(ByteFallbackRule::Table(table), false)
            .plain_by_id()
            .is_none());
        assert!(plain(ByteFallbackRule::ParseSurface, false)
            .plain_by_id()
            .is_none());
        assert!(plain(ByteFallbackRule::DeclaredRun, false)
            .plain_by_id()
            .is_none());
    }

    #[test]
    fn plain_by_id_rejects_byte_level() {
        assert!(plain(ByteFallbackRule::None, true).plain_by_id().is_none());
    }

    #[test]
    fn plain_by_id_rejects_surface_replace() {
        let rules = plain(ByteFallbackRule::None, false)
            .with_surface_replace("a".to_string(), "b".to_string());
        assert!(rules.plain_by_id().is_none());
    }

    #[test]
    fn plain_by_id_rejects_unit_cleanup() {
        let rules = plain(ByteFallbackRule::None, false).with_unit_cleanup();
        assert!(rules.plain_by_id().is_none());
    }

    #[test]
    fn plain_by_id_rejects_word_separator() {
        let every =
            plain(ByteFallbackRule::None, false).with_word_separator(WordSeparator::EveryToken);
        assert!(every.plain_by_id().is_none());
        let marked = plain(ByteFallbackRule::None, false)
            .with_word_separator(WordSeparator::Continuation("##".to_string()));
        assert!(marked.plain_by_id().is_none());
    }

    /// The `skip_span` prefilter must never change `skips`'s answer: a sparse
    /// skip set spanning a wide range should still answer true for exactly its
    /// members and false everywhere else, including ids inside the span that
    /// are not members.
    #[test]
    fn skips_is_answer_preserving_across_a_sparse_span() {
        let mut skip = FxHashSet::default();
        skip.insert(5u32);
        skip.insert(9000u32);
        let rules = RenderRules::new(
            Surfaces::ById(Arc::new(FxHashMap::default())),
            Arc::new(FxHashMap::default()),
            Arc::new(skip),
            ByteFallbackRule::None,
            false,
            false,
        );

        // Members.
        assert!(rules.skips(5));
        assert!(rules.skips(9000));

        // Non-members inside the span.
        assert!(!rules.skips(6));
        assert!(!rules.skips(4999));

        // Below the span.
        assert!(!rules.skips(0));
        assert!(!rules.skips(4));

        // Above the span.
        assert!(!rules.skips(9001));
        assert!(!rules.skips(u32::MAX));
    }

    /// An empty skip set answers false everywhere, with no span to consult.
    #[test]
    fn skips_is_false_for_an_empty_skip_set() {
        let rules = plain(ByteFallbackRule::None, false);
        assert!(!rules.skips(0));
        assert!(!rules.skips(1));
        assert!(!rules.skips(9000));
        assert!(!rules.skips(u32::MAX));
    }

    /// [`with_vocabulary`](RenderRules::with_vocabulary) replaces `skip` after
    /// construction — the one site the span could drift out of sync with the
    /// set it describes if it were not recomputed there too.
    #[test]
    fn skips_is_correct_after_with_vocabulary_replaces_skip() {
        let mut first_skip = FxHashSet::default();
        first_skip.insert(5u32);
        let rules = RenderRules::declared(ByteFallbackRule::None, false).with_vocabulary(
            Surfaces::ById(Arc::new(FxHashMap::default())),
            Arc::new(FxHashMap::default()),
            Arc::new(first_skip),
        );
        assert!(rules.skips(5));
        assert!(!rules.skips(9000));

        let mut second_skip = FxHashSet::default();
        second_skip.insert(9000u32);
        let rules = rules.with_vocabulary(
            Surfaces::ById(Arc::new(FxHashMap::default())),
            Arc::new(FxHashMap::default()),
            Arc::new(second_skip),
        );
        // The old member must no longer answer true, and the new member must.
        assert!(!rules.skips(5));
        assert!(rules.skips(9000));
    }

    /// [`rendering_specials`](RenderRules::rendering_specials) empties `skip`
    /// after construction — the other site the span must be cleared at.
    #[test]
    fn skips_is_false_after_rendering_specials_clears_skip() {
        let mut skip = FxHashSet::default();
        skip.insert(5u32);
        let rules = RenderRules::new(
            Surfaces::ById(Arc::new(FxHashMap::default())),
            Arc::new(FxHashMap::default()),
            Arc::new(skip),
            ByteFallbackRule::None,
            false,
            false,
        )
        .rendering_specials();
        assert!(!rules.skips(5));
        assert!(!rules.skips(0));
        assert!(!rules.skips(u32::MAX));
    }
}
