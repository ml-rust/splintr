//! WordPiece tokenizer for BERT-family models.
//!
//! Implements the standard BERT tokenization pipeline:
//! 1. **BasicTokenizer**: strip accents and lowercase (two independent settings,
//!    as in HuggingFace's `BertNormalizer`), split on whitespace and punctuation
//! 2. **WordPiece**: greedy longest-match subword tokenization with `##` continuation prefix
//!
//! Handles `[CLS]`, `[SEP]`, `[PAD]`, `[UNK]` special tokens.

use super::policy::{PolicyError, SpecialMode};
use super::streaming::{
    ByteFallbackRule, DecodePost, DecodeState, RenderRules, StreamingDecoder, Surfaces,
    WordSeparator,
};
use super::tokenize::{Tokenize, TokenizeError};
use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;
use thiserror::Error;

/// Errors from building a [`WordPieceTokenizer`].
#[derive(Error, Debug)]
pub enum WordPieceError {
    #[error("Failed to build added-token matcher: {0}")]
    AddedTokensError(#[from] aho_corasick::BuildError),
}

/// WordPiece tokenizer compatible with BERT-family models.
///
/// Constructed from a flat vocabulary list where index = token ID
/// (same format as GGUF `tokenizer.ggml.tokens`).
///
/// # Example
///
/// ```
/// use splintr::{WordPieceTokenizer, Tokenize};
///
/// let vocab = vec![
///     "[PAD]", "[UNK]", "[CLS]", "[SEP]",
///     "hello", "world", "##ing", "##s",
/// ].into_iter().map(String::from).collect();
/// let tok = WordPieceTokenizer::new(vocab, 1, 200, true);
/// let ids = tok.encode("hello world");
/// ```
pub struct WordPieceTokenizer {
    /// Token string → ID
    token_to_id: HashMap<String, u32>,
    /// ID → token string. Behind an `Arc` so decoding — whole-sequence and
    /// streaming alike — can share the surface table rather than copy a
    /// 30k-entry vector per decoder.
    id_to_token: Arc<Vec<String>>,
    /// Token ID for unknown tokens
    unk_token_id: u32,
    /// Maximum characters in a single word before it's treated as [UNK]
    max_word_len: usize,
    /// Whether to lowercase the input (BERT's `lowercase`). Casing only — it does
    /// NOT imply accent stripping; see [`WordPieceTokenizer::with_strip_accents`].
    do_lower_case: bool,
    /// Whether to strip accents (BERT's `strip_accents`), independent of casing.
    /// Seeded from `do_lower_case` at construction, which is HuggingFace's
    /// default for the absent/`null` form, and overridable on its own.
    strip_accents: bool,
    /// Continuation-subword prefix (e.g. `##`). Empty string means continuations
    /// are matched without a prefix (GGUF-stripped vocabs).
    continuation_prefix: String,
    /// Whether to isolate CJK ideographs as individual tokens (BERT's
    /// `handle_chinese_chars`). True for all standard BERT-family models.
    handle_chinese_chars: bool,
    /// Whether to strip control/format characters and `\0`/`�` and normalize
    /// whitespace before tokenizing (BERT's `clean_text`). Default true.
    clean_text: bool,
    /// Special token IDs for [CLS], [SEP], [PAD]
    cls_token_id: Option<u32>,
    sep_token_id: Option<u32>,
    pad_token_id: Option<u32>,
    /// Added tokens recognized in the input (HF matches these during encoding).
    added: Option<super::added::AddedTokens>,
    /// Ids dropped on decode (HF's `skip_special_tokens=True` default): the
    /// vocabulary's own bracket-named specials, resolved at construction, plus
    /// whatever the source file declares via `with_special_decode_ids`.
    special_decode: rustc_hash::FxHashSet<u32>,
}

impl WordPieceTokenizer {
    /// Create a WordPiece tokenizer from a flat vocabulary.
    ///
    /// # Arguments
    /// * `vocab` - Token strings indexed by token ID
    /// * `unk_token_id` - ID to use for unknown tokens
    /// * `max_word_len` - Words longer than this are mapped to `[UNK]`
    /// * `do_lower_case` - Whether to lowercase the input (uncased models). Accent
    ///   stripping is seeded from this flag — HuggingFace's rule for a
    ///   `BertNormalizer` whose `strip_accents` is absent/`null` — and can then be
    ///   set independently with [`with_strip_accents`](Self::with_strip_accents).
    pub fn new(
        vocab: Vec<String>,
        unk_token_id: u32,
        max_word_len: usize,
        do_lower_case: bool,
    ) -> Self {
        // Auto-detect the continuation prefix ("##" if present, else none) and
        // default `handle_chinese_chars` to true (the BERT default).
        let prefix = if vocab.iter().any(|k| k.starts_with("##")) {
            "##".to_string()
        } else {
            String::new()
        };
        Self::with_options(
            vocab,
            unk_token_id,
            max_word_len,
            do_lower_case,
            true,
            true,
            prefix,
        )
    }

    /// Like [`new`](Self::new) with explicit `handle_chinese_chars`, `clean_text`,
    /// and the continuation-subword prefix (empty string = continuations matched
    /// bare).
    #[allow(clippy::too_many_arguments)]
    pub fn with_options(
        vocab: Vec<String>,
        unk_token_id: u32,
        max_word_len: usize,
        do_lower_case: bool,
        handle_chinese_chars: bool,
        clean_text: bool,
        continuation_prefix: String,
    ) -> Self {
        let mut token_to_id = HashMap::with_capacity(vocab.len());
        for (id, token) in vocab.iter().enumerate() {
            token_to_id.insert(token.clone(), id as u32);
        }

        let cls_token_id = token_to_id.get("[CLS]").copied();
        let sep_token_id = token_to_id.get("[SEP]").copied();
        let pad_token_id = token_to_id.get("[PAD]").copied();

        // Decode-skipping is by id on every backend, so the names a BERT-family
        // vocabulary spells its specials with are resolved to ids once, here,
        // instead of re-matched against each token's surface string at decode
        // time. A vocabulary that names its specials otherwise — a GGUF file
        // declaring `<s>`/`</s>`/`<unk>` — contributes its own ids through
        // [`with_special_decode_ids`](Self::with_special_decode_ids), which
        // unions into this set rather than replacing it. Resolution walks the
        // vocab rather than `token_to_id` so a name occupying several ids marks
        // every one of them, as the surface test used to.
        let mut special_decode = rustc_hash::FxHashSet::default();
        for (id, token) in vocab.iter().enumerate() {
            if is_special_token(token) {
                special_decode.insert(id as u32);
            }
        }

        Self {
            token_to_id,
            id_to_token: Arc::new(vocab),
            unk_token_id,
            max_word_len,
            do_lower_case,
            // HuggingFace's default when `strip_accents` is absent/`null`; an
            // explicit setting arrives via `with_strip_accents`.
            strip_accents: do_lower_case,
            continuation_prefix,
            handle_chinese_chars,
            clean_text,
            cls_token_id,
            sep_token_id,
            pad_token_id,
            added: None,
            special_decode,
        }
    }

    /// Set accent stripping independently of lowercasing.
    ///
    /// Accent stripping is a setting of its own in HuggingFace's
    /// `BertNormalizer`, which computes `strip_accents.unwrap_or(lowercase)`:
    /// the absent/`null` form follows `lowercase` (what the constructors seed),
    /// but an explicit value wins on its own. Cased multilingual BERT ships
    /// `strip_accents: false`, and a vocabulary that distinguishes `café` from
    /// `cafe` resolves to different ids depending on this flag alone, so it
    /// cannot be inferred from casing.
    ///
    /// It is a builder method rather than another constructor parameter because
    /// [`with_options`](Self::with_options) already carries a
    /// `too_many_arguments` allowance, and because only callers that read an
    /// explicit value out of a config need to say anything at all.
    pub fn with_strip_accents(mut self, strip_accents: bool) -> Self {
        self.strip_accents = strip_accents;
        self
    }

    /// Attach added tokens to recognize in the input during encoding.
    ///
    /// Takes anything convertible into an [`AddedTokenSet`](super::added::AddedTokenSet),
    /// so a caller with no `lstrip`/`rstrip` flags to declare (GGUF, a bundled
    /// vocabulary, a test) can still pass a plain name→id map.
    pub fn with_added_tokens(
        mut self,
        tokens: impl Into<super::added::AddedTokenSet>,
    ) -> Result<Self, WordPieceError> {
        self.added = super::added::AddedTokens::new(&tokens.into())?;
        Ok(self)
    }

    /// Add ids of `special=true` added tokens to drop on decode (HF default).
    ///
    /// Unions rather than replaces: the constructor has already resolved the
    /// `[CLS]`/`[SEP]`/`[PAD]`/`[UNK]`/`[MASK]` names the vocabulary itself
    /// carries, and a caller stating what its *file* declares is adding to that
    /// knowledge, not correcting it. It is also the only way an `[unusedN]` id
    /// gets dropped — its spelling never earns it that.
    pub fn with_special_decode_ids(mut self, ids: rustc_hash::FxHashSet<u32>) -> Self {
        self.special_decode.extend(ids);
        self
    }

    /// Get the `[CLS]` token ID, if present in the vocabulary.
    pub fn cls_token_id(&self) -> Option<u32> {
        self.cls_token_id
    }

    /// Get the `[SEP]` token ID, if present in the vocabulary.
    pub fn sep_token_id(&self) -> Option<u32> {
        self.sep_token_id
    }

    /// Get the `[PAD]` token ID, if present in the vocabulary.
    pub fn pad_token_id(&self) -> Option<u32> {
        self.pad_token_id
    }

    /// Get the `[UNK]` token ID.
    pub fn unk_token_id(&self) -> u32 {
        self.unk_token_id
    }

    /// Pre-tokenize: clean, isolate CJK, strip accents and lowercase (each only
    /// if its own flag says so), then split on whitespace and punctuation.
    fn basic_tokenize(&self, text: &str) -> Vec<String> {
        // clean_text: drop NUL/replacement/control/format chars and turn every
        // whitespace char into a plain space, matching BERT's `_clean_text`.
        let cleaned;
        let text = if self.clean_text {
            cleaned = clean_text(text);
            cleaned.as_str()
        } else {
            text
        };

        // handle_chinese_chars: surround each CJK ideograph with spaces so it
        // becomes its own word (matching BERT's BasicTokenizer).
        let text = if self.handle_chinese_chars && text.chars().any(is_chinese_char) {
            let mut s = String::with_capacity(text.len() + 8);
            for c in text.chars() {
                if is_chinese_char(c) {
                    s.push(' ');
                    s.push(c);
                    s.push(' ');
                } else {
                    s.push(c);
                }
            }
            s
        } else {
            text.to_string()
        };

        // Accents and casing are independent settings, applied in HuggingFace's
        // own order (`BertNormalizer::normalize` strips first, then lowercases).
        let text = if self.strip_accents {
            strip_accents(&text)
        } else {
            text
        };
        let text = if self.do_lower_case {
            text.to_lowercase()
        } else {
            text
        };

        // Split on whitespace, then split each token on punctuation boundaries
        let mut tokens = Vec::new();
        for word in text.split_whitespace() {
            split_on_punctuation(word, &mut tokens);
        }
        tokens
    }

    /// WordPiece: greedily match longest subword.
    ///
    /// If the vocabulary uses `##` prefix (standard HuggingFace format),
    /// continuations are looked up with `##` prefix. Otherwise (GGUF-stripped
    /// vocabs), continuations are looked up directly.
    fn wordpiece_tokenize(&self, word: &str) -> Vec<u32> {
        let chars: Vec<char> = word.chars().collect();
        if chars.len() > self.max_word_len {
            return vec![self.unk_token_id];
        }

        let mut ids = Vec::new();
        let mut start = 0;

        while start < chars.len() {
            let mut end = chars.len();
            let mut matched = None;

            while start < end {
                let raw: String = chars[start..end].iter().collect();
                let lookup = if start == 0 || self.continuation_prefix.is_empty() {
                    raw
                } else {
                    format!("{}{}", self.continuation_prefix, raw)
                };

                if let Some(&id) = self.token_to_id.get(&lookup) {
                    matched = Some(id);
                    break;
                }

                end -= 1;
            }

            match matched {
                Some(id) => {
                    ids.push(id);
                    start = end;
                }
                // HuggingFace WordPiece maps an un-segmentable word to a single
                // `[UNK]` for the whole word — not one `[UNK]` per character.
                None => return vec![self.unk_token_id],
            }
        }

        ids
    }
}

impl WordPieceTokenizer {
    /// Encode without added-token matching (BasicTokenizer + WordPiece).
    ///
    /// Public on every backend, so a caller holding a concrete tokenizer has the
    /// same escape hatch regardless of which one it is.
    pub fn encode_ordinary(&self, text: &str) -> Vec<u32> {
        let words = self.basic_tokenize(text);
        let mut ids = Vec::new();
        for word in &words {
            ids.extend(self.wordpiece_tokenize(word));
        }
        ids
    }

    /// Encode text to token IDs under an explicit [`SpecialMode`], governing
    /// whether the added tokens attached during construction are matched in
    /// the input text. Boundary tokens (`[CLS]`/`[SEP]`) are
    /// [`SpecialPolicy`](crate::core::SpecialPolicy)'s to add via
    /// `AnyTokenizer::encode_with`, not this method's concern.
    pub fn encode_with(&self, text: &str, mode: &SpecialMode<'_>) -> Result<Vec<u32>, PolicyError> {
        super::added::AddedTokens::dispatch_with_mode(&self.added, text, mode, |gap| {
            self.encode_ordinary(gap)
        })
    }
}

impl Tokenize for WordPieceTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        // Recognize added tokens in the input first (HF behavior), then WordPiece.
        super::added::AddedTokens::dispatch(&self.added, text, |gap| self.encode_ordinary(gap))
    }

    fn encode_with(&self, text: &str, mode: &SpecialMode<'_>) -> Result<Vec<u32>, PolicyError> {
        self.encode_with(text, mode)
    }

    /// Render the pieces, joining words with spaces and dropping the space
    /// before `. ? ! ,` — the inherent [`decode`](WordPieceTokenizer::decode),
    /// which documents both, so the trait and the type can never disagree about
    /// what an id decodes to.
    fn decode(&self, ids: &[u32]) -> Result<String, TokenizeError> {
        self.decode(ids)
    }

    /// Skips ids the vocabulary does not contain — the inherent
    /// [`decode_lossy`](WordPieceTokenizer::decode_lossy), so the trait and the
    /// type can never disagree about what a sequence decodes to.
    fn decode_lossy(&self, ids: &[u32]) -> String {
        WordPieceTokenizer::decode_lossy(self, ids)
    }

    /// This backend never refuses to stream — the inherent
    /// [`streaming_decoder`](WordPieceTokenizer::streaming_decoder), wrapped in
    /// the `Ok` the trait's shape needs for
    /// [`AnyTokenizer`](crate::AnyTokenizer)'s sake.
    fn streaming_decoder(&self) -> Result<StreamingDecoder, TokenizeError> {
        Ok(WordPieceTokenizer::streaming_decoder(self))
    }

    /// The token's own text, with any `##` continuation marker removed and
    /// **without** the word separator a word-starting surface carries: that
    /// separator sits between two tokens, so it belongs to the sequence and not
    /// to this id — see [`Tokenize::decode_token_bytes`].
    fn decode_token_bytes(&self, id: u32) -> Result<Vec<u8>, TokenizeError> {
        // Rendered through the very rules `decode` drives, so a per-id answer
        // cannot drift from the sequence it emits.
        let state = self.decode_state();
        super::tokenize::token_bytes_of(state.render(), id)
    }

    fn decode_token(&self, id: u32) -> Result<String, TokenizeError> {
        super::tokenize::token_text_of(Tokenize::decode_token_bytes(self, id)?)
    }

    fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }
}

impl WordPieceTokenizer {
    /// The raw surface string of a token id (continuation tokens keep their `##`
    /// prefix). Used to drive a configuration-declared decoder pipeline.
    pub fn token_surface(&self, id: u32) -> Option<String> {
        self.id_to_token.get(id as usize).cloned()
    }

    /// This tokenizer's decode configuration, as the streaming decoder sees it.
    ///
    /// Whole-sequence decoding and streaming decoding drive the same
    /// [`DecodeState`] through the same cursor, so the two cannot disagree about
    /// what an id means or about what happens to the text it produces. The steps
    /// the two prefixed/prefix-less decode bodies used to spell out inline are
    /// exactly the knobs here: the skip set, the id-indexed surfaces, the word
    /// separator, and the cleanup pass over the text they produce.
    ///
    /// This backend has no byte layer at all — its surfaces are `String`s, so no
    /// character can be split across two of them — but the bytes still go
    /// through the cursor's UTF-8 buffer, which is what lets the streaming
    /// decoder be the same code.
    ///
    /// Cheap to build — the surface vector is shared with this tokenizer rather
    /// than copied — which is what lets `decode` capture one per call instead of
    /// the tokenizer having to cache one that could go stale.
    fn decode_state(&self) -> DecodeState {
        // An empty continuation prefix is *not* a marker that matches every
        // surface: a GGUF-stripped vocabulary has had its `##`s removed, so
        // continuations are indistinguishable from word starts and every token
        // gets a separator, which is what joining every surface with a space
        // means.
        let separator = if self.continuation_prefix.is_empty() {
            WordSeparator::EveryToken
        } else {
            WordSeparator::Continuation(self.continuation_prefix.clone())
        };

        DecodeState::new(
            RenderRules::new(
                Surfaces::ByIndex(Arc::clone(&self.id_to_token)),
                // No separate special-token table: every id this vocabulary has
                // is a slot in the surface vector, so an id outside it is
                // unknown rather than special.
                Arc::new(rustc_hash::FxHashMap::default()),
                Arc::new(self.special_decode.clone()),
                // No `<0xNN>` byte fallback and no ByteLevel or metaspace
                // spelling: a BERT-family surface is the text it stands for.
                ByteFallbackRule::None,
                false,
                false,
            )
            .with_word_separator(separator),
            vec![DecodePost::CleanupTokenization],
        )
    }

    /// A [`StreamingDecoder`] configured from this tokenizer.
    ///
    /// The only way to build one for this backend: the skipped ids, the
    /// continuation prefix and the punctuation cleanup all come from this
    /// tokenizer's configuration, so the stream cannot be pointed at the wrong
    /// kind of vocabulary and always reproduces [`decode`](Self::decode) —
    /// including across a chunk boundary that falls between a word and the
    /// punctuation mark whose space the cleanup removes.
    ///
    /// Cheap to call — the surface vector is shared, not copied — and the result
    /// borrows nothing, so it can be moved into a generation task.
    pub fn streaming_decoder(&self) -> StreamingDecoder {
        StreamingDecoder::new(Arc::new(self.decode_state()))
    }

    /// Decode token ids to text.
    ///
    /// Words are rejoined with a single space and continuations are glued back
    /// on: a surface carrying the `##` prefix loses it and follows the previous
    /// token directly, while any other surface starts a new word and is preceded
    /// by a space — unless nothing has been rendered yet. A vocabulary with no
    /// `##` at all (GGUF-stripped) cannot tell the two apart, so every token
    /// starts a word there.
    ///
    /// The declared specials produce nothing (HF's `skip_special_tokens=True`
    /// default) — see [`with_special_decode_ids`](Self::with_special_decode_ids)
    /// — and a skipped id emits no separator either, so a leading `[CLS]` does
    /// not put a space in front of the first word.
    ///
    /// Finally the `tokenizers` WordPiece cleanup drops the space before
    /// `. ? ! ,`. That step is position-dependent — the space it removes belongs
    /// to the token *before* the punctuation — so it is the cursor that holds a
    /// trailing space run back until the next chunk can claim it, and
    /// [`streaming_decoder`](Self::streaming_decoder) therefore reproduces it
    /// across chunk boundaries.
    ///
    /// Errors with [`TokenizeError::InvalidTokenId`] on an id the vocabulary
    /// does not contain — a distinct thing from the skips above, which are
    /// deliberate.
    ///
    /// The degenerate drive of the streaming cursor: one feed of every id, then
    /// a flush. Strict about ids and — like the Unigram sibling, and unlike
    /// SPM-BPE — silent about UTF-8, which costs nothing here because a surface
    /// is a `String` and can never hand the buffer a byte that is not valid. What
    /// an id renders to and what happens to the resulting text is decided by
    /// exactly the code [`streaming_decoder`](Self::streaming_decoder) uses.
    pub fn decode(&self, ids: &[u32]) -> Result<String, TokenizeError> {
        self.drive(ids, |id| Err(TokenizeError::InvalidTokenId(id)))
    }

    /// Decode token ids to text, skipping ids the vocabulary does not contain.
    ///
    /// The lenient half of the pair, over exactly the loop
    /// [`decode`](Self::decode) drives — same surfaces, same skips, same
    /// separator, same cleanup — with only an unknown id treated as something to
    /// survive rather than to report. This method never fails, so `on_unknown`
    /// is instantiated with [`Infallible`], letting the compiler prove the `Err`
    /// arm away rather than a runtime assertion claiming it.
    pub fn decode_lossy(&self, ids: &[u32]) -> String {
        match self.drive(ids, |_| Ok::<(), Infallible>(())) {
            Ok(text) => text,
            // `Infallible` has no values, so this match has no arms to write.
            Err(never) => match never {},
        }
    }

    /// The one decode loop, driven by both halves of the pair so that neither
    /// can drift: they differ only in what an id in no table means.
    ///
    /// Lossy on the UTF-8 question deliberately, and harmlessly: every surface
    /// is a `String`, so the buffer is never handed a byte that could be invalid
    /// and the substitution has nothing to substitute.
    fn drive<E>(
        &self,
        ids: &[u32],
        on_unknown: impl Fn(u32) -> Result<(), E>,
    ) -> Result<String, E> {
        let state = self.decode_state();
        let mut cursor = state.cursor_with_capacity(ids.len() * 4);

        let mut text = cursor.feed(ids, on_unknown)?.unwrap_or_default();
        text.push_str(&cursor.flush());

        Ok(text)
    }
}

/// The names BERT-family vocabularies spell their special tokens with. Read
/// once at construction to resolve them to ids — decode itself tests ids only,
/// so a vocabulary that names its specials differently is not judged by its
/// spelling.
///
/// `[unusedN]` is deliberately absent: HuggingFace does not declare those
/// special (`all-MiniLM-L6-v2`'s `tokenizer.json` lists exactly `[CLS]`,
/// `[MASK]`, `[PAD]`, `[SEP]`, `[UNK]`) and emits them from `decode` even under
/// `skip_special_tokens=True`. A file that *does* declare one special states it
/// by id, through [`WordPieceTokenizer::with_special_decode_ids`].
fn is_special_token(token: &str) -> bool {
    matches!(token, "[CLS]" | "[SEP]" | "[PAD]" | "[UNK]" | "[MASK]")
}

/// Strip accents from text, matching BERT's `BasicTokenizer._run_strip_accents`:
/// decompose (NFD) and drop only **Nonspacing_Mark (Mn)** characters. Spacing
/// combining marks (Mc) — e.g. Devanagari/Thai vowel signs — are kept, unlike a
/// blanket "all combining marks" filter which would corrupt those scripts.
fn strip_accents(text: &str) -> String {
    use unicode_general_category::{get_general_category, GeneralCategory};
    use unicode_normalization::UnicodeNormalization;
    text.nfd()
        .filter(|c| get_general_category(*c) != GeneralCategory::NonspacingMark)
        .collect()
}

/// Split a word on punctuation boundaries, pushing results into `out`.
fn split_on_punctuation(word: &str, out: &mut Vec<String>) {
    let mut current = String::new();
    for c in word.chars() {
        if is_punctuation(c) {
            if !current.is_empty() {
                out.push(std::mem::take(&mut current));
            }
            out.push(c.to_string());
        } else {
            current.push(c);
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
}

/// Check if a character is a CJK ideograph, matching BERT's `_is_chinese_char`
/// (the CJK Unified Ideographs blocks and their extensions/compatibility forms).
/// BERT `_clean_text`: drop `\0`, the replacement char, and control/format
/// characters (Unicode categories `C*`, except `\t`/`\n`/`\r`); map every
/// whitespace character (including `Zs`) to a plain space.
fn clean_text(text: &str) -> String {
    use unicode_general_category::{get_general_category, GeneralCategory};
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        if c == '\0' || c == '\u{fffd}' {
            continue;
        }
        let is_keepable_ws = matches!(c, '\t' | '\n' | '\r');
        if !is_keepable_ws {
            match get_general_category(c) {
                GeneralCategory::Control
                | GeneralCategory::Format
                | GeneralCategory::Surrogate
                | GeneralCategory::PrivateUse
                | GeneralCategory::Unassigned => continue,
                _ => {}
            }
        }
        if c == ' ' || is_keepable_ws || get_general_category(c) == GeneralCategory::SpaceSeparator
        {
            out.push(' ');
        } else {
            out.push(c);
        }
    }
    out
}

fn is_chinese_char(c: char) -> bool {
    let cp = c as u32;
    (0x4E00..=0x9FFF).contains(&cp)
        || (0x3400..=0x4DBF).contains(&cp)
        || (0x20000..=0x2A6DF).contains(&cp)
        || (0x2A700..=0x2B73F).contains(&cp)
        || (0x2B740..=0x2B81F).contains(&cp)
        || (0x2B820..=0x2CEAF).contains(&cp)
        || (0xF900..=0xFAFF).contains(&cp)
        || (0x2F800..=0x2FA1F).contains(&cp)
}

/// Check if a character is punctuation (matching BERT's definition).
fn is_punctuation(c: char) -> bool {
    // ASCII punctuation ranges
    matches!(c, '\x21'..='\x2F' | '\x3A'..='\x40' | '\x5B'..='\x60' | '\x7B'..='\x7E')
        || c.is_ascii_punctuation()
        || {
            // Unicode punctuation categories
            let cat = unicode_general_category::get_general_category(c);
            matches!(
                cat,
                unicode_general_category::GeneralCategory::ConnectorPunctuation
                    | unicode_general_category::GeneralCategory::DashPunctuation
                    | unicode_general_category::GeneralCategory::ClosePunctuation
                    | unicode_general_category::GeneralCategory::FinalPunctuation
                    | unicode_general_category::GeneralCategory::InitialPunctuation
                    | unicode_general_category::GeneralCategory::OtherPunctuation
                    | unicode_general_category::GeneralCategory::OpenPunctuation
            )
        }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn make_tokenizer() -> WordPieceTokenizer {
        let vocab = vec![
            "[PAD]".to_string(),  // 0
            "[UNK]".to_string(),  // 1
            "[CLS]".to_string(),  // 2
            "[SEP]".to_string(),  // 3
            "hello".to_string(),  // 4
            "world".to_string(),  // 5
            "##ing".to_string(),  // 6
            "##s".to_string(),    // 7
            "un".to_string(),     // 8
            "##know".to_string(), // 9
            "##n".to_string(),    // 10
            ",".to_string(),      // 11
            "the".to_string(),    // 12
            "a".to_string(),      // 13
        ];
        WordPieceTokenizer::new(vocab, 1, 200, true)
    }

    #[test]
    fn test_encode_basic() {
        let tok = make_tokenizer();
        let ids = tok.encode("hello world");
        assert_eq!(ids, vec![4, 5]);
    }

    #[test]
    fn test_encode_subwords() {
        let tok = make_tokenizer();
        let ids = tok.encode("unknown");
        // "unknown" → "un" + "##know" + "##n"
        assert_eq!(ids, vec![8, 9, 10]);
    }

    #[test]
    fn test_encode_punctuation() {
        let tok = make_tokenizer();
        let ids = tok.encode("hello, world");
        // "hello" "," "world"
        assert_eq!(ids, vec![4, 11, 5]);
    }

    #[test]
    fn test_decode_basic() {
        let tok = make_tokenizer();
        let text = tok.decode(&[4, 5]).unwrap();
        assert_eq!(text, "hello world");
    }

    #[test]
    fn test_decode_subwords() {
        let tok = make_tokenizer();
        let text = tok.decode(&[8, 9, 10]).unwrap();
        assert_eq!(text, "unknown");
    }

    #[test]
    fn test_decode_skips_special() {
        let tok = make_tokenizer();
        let text = tok.decode(&[2, 4, 5, 3]).unwrap();
        assert_eq!(text, "hello world");
    }

    /// The byte-identical guarantee: a standard bracket-named vocabulary drops
    /// exactly the ids the old surface-string rule dropped — `[CLS]`, `[SEP]`,
    /// `[PAD]`, `[UNK]` and `[MASK]` — now resolved to ids at construction.
    #[test]
    fn standard_bracket_named_vocab_decodes_unchanged() {
        let vocab = vec![
            "[PAD]".to_string(),  // 0
            "[UNK]".to_string(),  // 1
            "[CLS]".to_string(),  // 2
            "[SEP]".to_string(),  // 3
            "[MASK]".to_string(), // 4
            "hello".to_string(),  // 5
            "world".to_string(),  // 6
            "##ing".to_string(),  // 7
        ];
        let tok = WordPieceTokenizer::new(vocab, 1, 200, true);
        assert_eq!(tok.decode(&[2, 5, 6, 3]).unwrap(), "hello world");
        assert_eq!(tok.decode(&[0, 1, 4]).unwrap(), "");
        assert_eq!(tok.decode(&[5, 6, 7]).unwrap(), "hello worlding");
    }

    /// Ids handed in by a loader join the names resolved at construction rather
    /// than replacing them — a `tokenizer.json` stating its `special = true` ids
    /// must not silently un-skip `[CLS]`.
    #[test]
    fn declared_special_ids_join_the_resolved_names() {
        let tok = make_tokenizer().with_special_decode_ids([13u32].into_iter().collect());
        // 13 is the content token "a", declared special by the caller; 2/3 are
        // [CLS]/[SEP], resolved from the vocabulary itself.
        assert_eq!(tok.decode(&[2, 4, 13, 5, 3]).unwrap(), "hello world");
    }

    /// A token spelled `[unusedN]` that no file declares special is ordinary
    /// content and survives decode. Reference (`tokenizers`, on
    /// `all-MiniLM-L6-v2/tokenizer.json`, whose declared specials are exactly
    /// `['[CLS]', '[MASK]', '[PAD]', '[SEP]', '[UNK]']` — `[unused*]` is not
    /// among its 994 such tokens):
    ///
    /// ```text
    /// ids = [vocab["hello"], vocab["[unused0]"], vocab["world"]]  # [7592, 1, 2088]
    /// decode(ids, skip_special_tokens=False) -> 'hello [unused0] world'
    /// decode(ids, skip_special_tokens=True)  -> 'hello [unused0] world'
    /// ```
    #[test]
    fn unused_spelled_content_token_survives_decode() {
        let vocab = vec![
            "[UNK]".to_string(),     // 0
            "[unused7]".to_string(), // 1
            "hello".to_string(),     // 2
            "world".to_string(),     // 3
        ];
        let tok = WordPieceTokenizer::new(vocab, 0, 200, true);
        assert_eq!(tok.decode(&[2, 1, 3]).unwrap(), "hello [unused7] world");
    }

    /// …and an `[unusedN]` its *file* declares special (HF-json `added_tokens`
    /// with `"special": true`, or a GGUF declared special id) is still dropped.
    /// The declaration decides, not the spelling — the same id-based path that
    /// drops any other declared special.
    #[test]
    fn declared_special_unused_token_is_dropped() {
        let vocab = vec![
            "[UNK]".to_string(),     // 0
            "[unused7]".to_string(), // 1
            "hello".to_string(),     // 2
            "world".to_string(),     // 3
        ];
        let tok = WordPieceTokenizer::new(vocab, 0, 200, true)
            .with_special_decode_ids([1u32].into_iter().collect());
        assert_eq!(tok.decode(&[2, 1, 3]).unwrap(), "hello world");
    }

    #[test]
    fn test_vocab_size() {
        let tok = make_tokenizer();
        assert_eq!(tok.vocab_size(), 14);
    }

    #[test]
    fn test_special_token_ids() {
        let tok = make_tokenizer();
        assert_eq!(tok.cls_token_id(), Some(2));
        assert_eq!(tok.sep_token_id(), Some(3));
        assert_eq!(tok.pad_token_id(), Some(0));
        assert_eq!(tok.unk_token_id(), 1);
    }

    #[test]
    fn clean_text_strips_control_and_format_chars() {
        // Zero-width space (Cf), ZWNJ (Cf), BOM (Cf), NUL and replacement char
        // are removed; \t/\n become spaces; ordinary text is untouched.
        assert_eq!(
            clean_text("a\u{200b}b\u{200c}\u{feff}c\0\u{fffd}d\te"),
            "abcd e"
        );
        assert_eq!(clean_text("plain text"), "plain text");
    }

    #[test]
    fn test_unknown_word() {
        let tok = make_tokenizer();
        // An un-segmentable word maps to a single [UNK] (HuggingFace behavior),
        // not one [UNK] per character.
        assert_eq!(tok.encode("xyz"), vec![1]);
    }

    #[test]
    fn test_handle_chinese_chars() {
        // Each CJK ideograph is isolated into its own word; with none in the
        // vocab here, each becomes its own [UNK] (one per char, since they are
        // separate words — distinct from the whole-word rule above).
        let tok = make_tokenizer();
        assert_eq!(tok.encode("hello世界world"), vec![4, 1, 1, 5]);
    }

    #[test]
    fn test_lowercase() {
        let tok = make_tokenizer();
        let ids = tok.encode("Hello WORLD");
        assert_eq!(ids, vec![4, 5]);
    }

    #[test]
    fn test_case_sensitive() {
        let vocab = vec![
            "[UNK]".to_string(), // 0
            "Hello".to_string(), // 1
            "hello".to_string(), // 2
        ];
        let tok = WordPieceTokenizer::new(vocab, 0, 200, false);
        let ids = tok.encode("Hello");
        assert_eq!(ids, vec![1]);
        let ids = tok.encode("hello");
        assert_eq!(ids, vec![2]);
    }

    /// Vocabulary that keeps every casing/accent variant apart, so which of the
    /// two normalization flags ran is readable straight off the id.
    fn accent_vocab() -> Vec<String> {
        vec![
            "[UNK]".to_string(), // 0
            "cafe".to_string(),  // 1
            "café".to_string(),  // 2
            "Cafe".to_string(),  // 3
            "Café".to_string(),  // 4
            "naive".to_string(), // 5
            "naïve".to_string(), // 6
        ]
    }

    /// The `null`/absent shape: accent stripping is *seeded* from `do_lower_case`,
    /// which is HuggingFace's `strip_accents.unwrap_or(lowercase)` default.
    /// Reference (`tokenizers` 0.22.1, `lowercase: true, strip_accents: null`):
    /// `"Café"` and `"café"` both reach the unaccented `cafe` entry.
    #[test]
    fn strip_accents_defaults_to_lowercasing() {
        let tok = WordPieceTokenizer::new(accent_vocab(), 0, 200, true);
        assert_eq!(tok.encode("Café"), vec![1]);
        assert_eq!(tok.encode("café"), vec![1]);
        assert_eq!(tok.encode("naïve"), vec![5]);

        let cased = WordPieceTokenizer::new(accent_vocab(), 0, 200, false);
        assert_eq!(cased.encode("Café"), vec![4]);
        assert_eq!(cased.encode("naïve"), vec![6]);
    }

    /// Lowercasing with accent stripping explicitly OFF — the cased-multilingual
    /// shape (`strip_accents: false`). Reference (`tokenizers` 0.22.1,
    /// `lowercase: true, strip_accents: false`): `"Café"` -> `café`, not `cafe`.
    #[test]
    fn lowercasing_does_not_force_accent_stripping() {
        let tok = WordPieceTokenizer::new(accent_vocab(), 0, 200, true).with_strip_accents(false);
        assert_eq!(tok.encode("Café"), vec![2]);
        assert_eq!(tok.encode("café"), vec![2]);
        assert_eq!(tok.encode("naïve"), vec![6]);
        // Lowercasing still runs — it is only accents that were turned off.
        assert_eq!(tok.encode("Cafe"), vec![1]);
    }

    /// Accent stripping with lowercasing OFF: casing survives, accents do not.
    /// Reference (`tokenizers` 0.22.1, `lowercase: false, strip_accents: true`):
    /// `"Café"` -> `Cafe` and `"café"` -> `cafe`.
    #[test]
    fn accent_stripping_does_not_force_lowercasing() {
        let tok = WordPieceTokenizer::new(accent_vocab(), 0, 200, false).with_strip_accents(true);
        assert_eq!(tok.encode("Café"), vec![3]);
        assert_eq!(tok.encode("café"), vec![1]);
        assert_eq!(tok.encode("naïve"), vec![5]);
    }

    #[test]
    fn test_decode_invalid_id() {
        let tok = make_tokenizer();
        let result = tok.decode(&[999]);
        assert!(result.is_err());
    }

    /// An id the vocabulary does not contain is skipped rather than reported,
    /// which is the only thing `decode_lossy` does differently from `decode`.
    #[test]
    fn decode_lossy_skips_an_unknown_id_that_decode_reports() {
        let tok = make_tokenizer();
        assert!(tok.decode(&[4, 999, 5]).is_err());
        assert_eq!(tok.decode_lossy(&[4, 999, 5]), "hello world");
        // ...and agrees with `decode` everywhere `decode` succeeds.
        assert_eq!(tok.decode_lossy(&[2, 4, 5, 3]), "hello world");
    }

    // =========================================================================
    // The streaming decoder: concat(stream) == decode, at every chunk size
    // =========================================================================

    /// A `##` vocabulary carrying everything the stream has to get right: a
    /// continuation, specials dropped on decode, the punctuation the cleanup
    /// rewrites — as a word of its own (`,`, `.`) and as a *continuation*
    /// (`##.`), which is the shape that leaves the space it eats in an earlier
    /// chunk. The empty surface (id 10) renders a bare separator, so a chunk can
    /// consist of nothing but spaces.
    fn stream_vocab() -> Vec<String> {
        [
            "[PAD]", // 0
            "[UNK]", // 1
            "[CLS]", // 2
            "[SEP]", // 3
            "hello", // 4
            "world", // 5
            "##ing", // 6
            ",",     // 7
            ".",     // 8
            "a",     // 9
            "",      // 10
            "##.",   // 11
            "?",     // 12
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect()
    }

    /// The `##` path, whose continuations are marked.
    fn stream_tokenizer() -> WordPieceTokenizer {
        WordPieceTokenizer::new(stream_vocab(), 1, 200, true)
    }

    /// The GGUF-stripped path: the same vocabulary with no continuation prefix
    /// declared, so every token starts a word.
    fn bare_stream_tokenizer() -> WordPieceTokenizer {
        WordPieceTokenizer::with_options(stream_vocab(), 1, 200, true, true, true, String::new())
    }

    /// Feed `ids` through a streaming decoder in the given chunk sizes and
    /// concatenate every emission plus the final flush.
    fn drive_strict(tokenizer: &WordPieceTokenizer, ids: &[u32], chunk: usize) -> String {
        let mut decoder = tokenizer.streaming_decoder();
        let mut out = String::new();
        for group in ids.chunks(chunk.max(1)) {
            if let Some(text) = decoder.add_tokens(group).expect("ids are all known") {
                out.push_str(&text);
            }
        }
        out.push_str(&decoder.flush());
        out
    }

    /// Same, one id at a time through the lossy entry point.
    fn drive_lossy(tokenizer: &WordPieceTokenizer, ids: &[u32]) -> String {
        let mut decoder = tokenizer.streaming_decoder();
        let mut out = String::new();
        for &id in ids {
            if let Some(text) = decoder.add_token_lossy(id) {
                out.push_str(&text);
            }
        }
        out.push_str(&decoder.flush());
        out
    }

    /// Id sequences covering a continuation, a dropped special at position 0
    /// (which must not emit a leading separator), punctuation the cleanup
    /// rewrites, and runs of empty surfaces.
    const STREAM_IDS: &[&[u32]] = &[
        &[],
        &[4, 5],
        &[4, 6],
        &[2, 4, 6, 7, 5, 3],
        &[2, 4, 5, 8],
        &[4, 7, 5, 12],
        &[9, 10, 8],
        &[9, 10, 10, 8],
        &[9, 10, 11],
        &[10, 4],
        &[2, 3, 0],
    ];

    /// The point of the factory: streaming reproduces `decode` exactly, under
    /// every grouping and through both entry points — on the `##` path and on
    /// the prefix-less one.
    #[test]
    fn stream_matches_decode_at_every_chunk_size() {
        for tokenizer in [stream_tokenizer(), bare_stream_tokenizer()] {
            for ids in STREAM_IDS {
                let expected = tokenizer.decode(ids).expect("ids are all known");

                for chunk in 1..=ids.len().max(1) {
                    assert_eq!(
                        drive_strict(&tokenizer, ids, chunk),
                        expected,
                        "ids: {ids:?}, chunk: {chunk}"
                    );
                }
                assert_eq!(
                    drive_lossy(&tokenizer, ids),
                    tokenizer.decode_lossy(ids),
                    "ids: {ids:?}"
                );
                assert_eq!(tokenizer.decode_lossy(ids), expected, "ids: {ids:?}");
            }
        }
    }

    /// A skipped special at position 0 must not emit a separator: the flag the
    /// separator consults is "a token has rendered", and a skip renders nothing.
    /// The failure mode is a leading space on every BERT sequence.
    #[test]
    fn a_leading_special_does_not_emit_a_separator() {
        let tok = stream_tokenizer();
        let mut decoder = tok.streaming_decoder();

        assert_eq!(decoder.add_token(2).expect("[CLS] is known"), None);
        assert_eq!(
            decoder.add_token(4).expect("known id"),
            Some("hello".to_string())
        );
        assert_eq!(drive_strict(&tok, &[2, 4, 5, 3], 1), "hello world");
        assert_eq!(tok.decode(&[2, 4, 5, 3]).unwrap(), "hello world");
    }

    /// The case the held space run exists for: the space the cleanup removes is
    /// emitted with the token *before* the punctuation, so when the two arrive
    /// in separate `add_token` calls the stream can only agree with `decode` by
    /// holding that space back.
    ///
    /// `##.` is the sharp form — a *continuation* punctuation carries no
    /// separator of its own, so the only space available is the one already
    /// emitted. Without the hold this streams as `"a ."` where `decode` is
    /// `"a."`.
    #[test]
    fn punctuation_straddling_a_chunk_boundary_matches_decode() {
        let tok = stream_tokenizer();

        let ids = [9u32, 10, 11]; // "a", "", "##."
        assert_eq!(tok.decode(&ids).unwrap(), "a.");

        let mut decoder = tok.streaming_decoder();
        let mut streamed = String::new();
        for id in ids {
            streamed.push_str(&decoder.add_token(id).expect("known id").unwrap_or_default());
        }
        streamed.push_str(&decoder.flush());
        assert_eq!(streamed, "a.");

        // ...and the ordinary shape, where the punctuation is a word of its own
        // and brings its own separator: `"hello"` then `","`.
        let ids = [4u32, 7];
        assert_eq!(tok.decode(&ids).unwrap(), "hello,");
        let mut decoder = tok.streaming_decoder();
        let mut streamed = String::new();
        for id in ids {
            streamed.push_str(&decoder.add_token(id).expect("known id").unwrap_or_default());
        }
        streamed.push_str(&decoder.flush());
        assert_eq!(streamed, "hello,");
    }

    /// The cleanup is `str::replace`, which is single-pass and does not rescan
    /// what it produced: `"a  ."` keeps one of its two spaces. The stream has to
    /// hold the whole trailing *run* back to be handed the same string — holding
    /// one space would offer the replacement a different one.
    #[test]
    fn the_whole_trailing_space_run_is_held() {
        let tok = stream_tokenizer();

        // "a" + sep + "" + sep + "." renders `"a  ."`, and exactly one space
        // comes off.
        let ids = [9u32, 10, 8];
        assert_eq!(tok.decode(&ids).unwrap(), "a .");
        for chunk in 1..=ids.len() {
            assert_eq!(drive_strict(&tok, &ids, chunk), "a .", "chunk: {chunk}");
        }

        // A second empty surface adds a second space-only chunk, so the run
        // spans two emissions and still yields one deletion.
        let ids = [9u32, 10, 10, 8];
        assert_eq!(tok.decode(&ids).unwrap(), "a  .");
        for chunk in 1..=ids.len() {
            assert_eq!(drive_strict(&tok, &ids, chunk), "a  .", "chunk: {chunk}");
        }
    }

    /// A held run that no punctuation ever claims is emitted by `flush` rather
    /// than swallowed — the trailing space `decode` also keeps.
    #[test]
    fn a_held_space_run_survives_the_flush() {
        let tok = stream_tokenizer();
        let ids = [9u32, 10];

        assert_eq!(tok.decode(&ids).unwrap(), "a ");

        let mut decoder = tok.streaming_decoder();
        let emitted = decoder
            .add_tokens(&ids)
            .expect("known ids")
            .unwrap_or_default();
        assert_eq!(emitted, "a", "the trailing space is still held");
        assert_eq!(decoder.flush(), " ");
    }

    /// The prefix-less (GGUF-stripped) path joins every token with a space, so
    /// its punctuation always arrives with a separator of its own — and the
    /// space-run hold still has to reproduce `decode` across chunks.
    #[test]
    fn the_prefixless_path_streams_like_decode() {
        let tok = bare_stream_tokenizer();

        // `##.` is not a continuation here: it is a word spelled `##.`.
        assert_eq!(tok.decode(&[9, 11]).unwrap(), "a ##.");
        assert_eq!(tok.decode(&[4, 6]).unwrap(), "hello ##ing");
        assert_eq!(tok.decode(&[9, 10, 10, 8]).unwrap(), "a  .");

        for ids in [vec![9u32, 11], vec![4, 6], vec![9, 10, 10, 8]] {
            let expected = tok.decode(&ids).expect("known ids");
            for chunk in 1..=ids.len() {
                assert_eq!(drive_strict(&tok, &ids, chunk), expected, "chunk: {chunk}");
            }
        }
    }

    proptest! {
        /// Chunk-partition invariance on the `##` path: arbitrary grouping
        /// through `add_tokens` gives what one-at-a-time gives, and both give
        /// `decode`.
        #[test]
        fn prop_chunking_matches_decode(
            ids in prop::collection::vec(0u32..13, 0..32),
            chunk in 1usize..8,
        ) {
            let tokenizer = stream_tokenizer();
            let expected = tokenizer.decode(&ids).expect("every id is in range");

            prop_assert_eq!(drive_strict(&tokenizer, &ids, 1), expected.clone());
            prop_assert_eq!(drive_strict(&tokenizer, &ids, chunk), expected);
        }

        /// The same on the prefix-less path, whose separator rule differs.
        #[test]
        fn prop_chunking_matches_decode_without_prefix(
            ids in prop::collection::vec(0u32..13, 0..32),
            chunk in 1usize..8,
        ) {
            let tokenizer = bare_stream_tokenizer();
            let expected = tokenizer.decode(&ids).expect("every id is in range");

            prop_assert_eq!(drive_strict(&tokenizer, &ids, 1), expected.clone());
            prop_assert_eq!(drive_strict(&tokenizer, &ids, chunk), expected);
        }

        /// Arbitrary ids — unknown ones included — stream lossily to exactly
        /// what `decode_lossy` produces.
        #[test]
        fn prop_arbitrary_ids_match_decode_lossy(
            ids in prop::collection::vec(0u32..40, 0..48),
        ) {
            let tokenizer = stream_tokenizer();
            prop_assert_eq!(drive_lossy(&tokenizer, &ids), tokenizer.decode_lossy(&ids));
        }

        /// `reset()` purity: a used-then-reset decoder behaves byte-identically
        /// to a freshly built one — the "a token has rendered" flag and the held
        /// space run included, which is why the dirty prefix is fed before the
        /// reset rather than after.
        #[test]
        fn prop_reset_matches_a_fresh_decoder(
            dirty in prop::collection::vec(0u32..40, 0..16),
            ids in prop::collection::vec(0u32..40, 0..32),
        ) {
            let tokenizer = stream_tokenizer();

            let mut reused = tokenizer.streaming_decoder();
            reused.add_tokens_lossy(&dirty);
            reused.reset();
            prop_assert!(!reused.has_pending());
            prop_assert_eq!(reused.pending_bytes(), 0);

            let mut fresh = tokenizer.streaming_decoder();

            let mut from_reused = String::new();
            let mut from_fresh = String::new();
            for &id in &ids {
                let a = reused.add_token_lossy(id);
                let b = fresh.add_token_lossy(id);
                prop_assert_eq!(&a, &b);
                prop_assert_eq!(reused.pending_bytes(), fresh.pending_bytes());
                from_reused.push_str(&a.unwrap_or_default());
                from_fresh.push_str(&b.unwrap_or_default());
            }
            from_reused.push_str(&reused.flush());
            from_fresh.push_str(&fresh.flush());

            prop_assert_eq!(from_reused, from_fresh);
        }
    }

    // =========================================================================
    // Per-id decoding: `Tokenize::decode_token_bytes` / `decode_token`
    // =========================================================================

    /// The three answers the method distinguishes: an ordinary surface renders
    /// its text (a `##` continuation without its marker, which is a fact about
    /// that surface's spelling and not about the sequence), a skipped special
    /// contributes an empty `Vec` rather than an error — it really does
    /// contribute nothing — and an id the vocabulary has no slot for is reported.
    #[test]
    fn decode_token_bytes_separates_content_skip_and_unknown() {
        let tok = make_tokenizer();

        assert_eq!(tok.decode_token_bytes(4).unwrap(), b"hello".to_vec());
        assert_eq!(tok.decode_token(4).unwrap(), "hello");
        // The `##` comes off: `##ing` contributes `ing`.
        assert_eq!(tok.decode_token(6).unwrap(), "ing");

        // `[CLS]` and `[SEP]` are dropped on decode.
        for skipped in [2, 3] {
            assert_eq!(tok.decode_token_bytes(skipped).unwrap(), Vec::<u8>::new());
            assert_eq!(tok.decode_token(skipped).unwrap(), "");
        }

        assert!(matches!(
            tok.decode_token_bytes(999),
            Err(TokenizeError::InvalidTokenId(999))
        ));
        assert!(matches!(
            tok.decode_token(999),
            Err(TokenizeError::InvalidTokenId(999))
        ));
    }

    /// Agreement, and the one place it is not literal equality.
    ///
    /// This backend's surfaces carry no spacing of their own, so decoding puts a
    /// separator *between* word-starting tokens. That separator belongs to the
    /// sequence, not to any one id, and `decode_token_bytes` therefore leaves it
    /// out by contract. So the invariant is stated over a sequence whose tokens
    /// carry no separator — a leading skipped special plus continuations, where
    /// exact equality holds — and the word-start case is pinned separately as
    /// "the decoded text differs by exactly the separators".
    #[test]
    fn concatenated_token_bytes_equal_the_decoded_sequence_without_separators() {
        let tok = make_tokenizer();

        // `[CLS] un ##know ##n [SEP]`: nothing after the first word start, so no
        // separator is ever emitted and the concatenation is the decoded text.
        let glued = [2, 8, 9, 10, 3];
        let joined: Vec<u8> = glued
            .iter()
            .flat_map(|&id| tok.decode_token_bytes(id).expect("every id is known"))
            .collect();
        assert_eq!(joined, tok.decode_lossy(&glued).into_bytes());
        assert_eq!(String::from_utf8(joined).unwrap(), "unknown");

        // `hello world`: two word starts, so decoding inserts the one separator
        // the per-id bytes do not carry.
        let separated = [4, 5];
        let joined: String = separated
            .iter()
            .map(|&id| tok.decode_token(id).expect("every id is known"))
            .collect();
        assert_eq!(joined, "helloworld");
        assert_eq!(tok.decode_lossy(&separated), "hello world");
    }

    /// The trait's `decode_lossy` and `streaming_decoder` are the inherent ones.
    /// Before this, `decode_lossy` was inherent-only and therefore unreachable
    /// for any caller holding this backend through the trait.
    #[test]
    fn trait_decode_lossy_and_streaming_decoder_match_the_inherent_pair() {
        let tok = make_tokenizer();
        let ids = [2, 4, 999, 5, 3];

        assert_eq!(Tokenize::decode_lossy(&tok, &ids), "hello world");
        assert_eq!(
            Tokenize::decode_lossy(&tok, &ids),
            WordPieceTokenizer::decode_lossy(&tok, &ids)
        );

        let mut streamed = Tokenize::streaming_decoder(&tok).expect("WordPiece always streams");
        let mut out = streamed.add_tokens_lossy(&ids).unwrap_or_default();
        out.push_str(&streamed.flush());
        assert_eq!(out, "hello world");
    }
}
