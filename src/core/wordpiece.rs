//! WordPiece tokenizer for BERT-family models.
//!
//! Implements the standard BERT tokenization pipeline:
//! 1. **BasicTokenizer**: lowercase, strip accents, split on whitespace and punctuation
//! 2. **WordPiece**: greedy longest-match subword tokenization with `##` continuation prefix
//!
//! Handles `[CLS]`, `[SEP]`, `[PAD]`, `[UNK]` special tokens.

use super::tokenize::{Tokenize, TokenizeError};
use std::collections::HashMap;
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
    /// ID → token string
    id_to_token: Vec<String>,
    /// Token ID for unknown tokens
    unk_token_id: u32,
    /// Maximum characters in a single word before it's treated as [UNK]
    max_word_len: usize,
    /// Whether to lowercase and strip accents (for uncased models)
    do_lower_case: bool,
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
    /// Ids of `special=true` added tokens dropped on decode (HF default).
    special_decode: rustc_hash::FxHashSet<u32>,
}

impl WordPieceTokenizer {
    /// Create a WordPiece tokenizer from a flat vocabulary.
    ///
    /// # Arguments
    /// * `vocab` - Token strings indexed by token ID
    /// * `unk_token_id` - ID to use for unknown tokens
    /// * `max_word_len` - Words longer than this are mapped to `[UNK]`
    /// * `do_lower_case` - Whether to lowercase and strip accents (for uncased models)
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

        Self {
            token_to_id,
            id_to_token: vocab,
            unk_token_id,
            max_word_len,
            do_lower_case,
            continuation_prefix,
            handle_chinese_chars,
            clean_text,
            cls_token_id,
            sep_token_id,
            pad_token_id,
            added: None,
            special_decode: rustc_hash::FxHashSet::default(),
        }
    }

    /// Attach added tokens to recognize in the input during encoding.
    pub fn with_added_tokens(
        mut self,
        map: &rustc_hash::FxHashMap<String, u32>,
    ) -> Result<Self, WordPieceError> {
        self.added = super::added::AddedTokens::new(map)?;
        Ok(self)
    }

    /// Set ids of `special=true` added tokens to drop on decode (HF default).
    pub fn with_special_decode_ids(mut self, ids: rustc_hash::FxHashSet<u32>) -> Self {
        self.special_decode = ids;
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

    /// Pre-tokenize: lowercase, strip accents, split on whitespace and punctuation.
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

        let text = if self.do_lower_case {
            let lowered = text.to_lowercase();
            strip_accents(&lowered)
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
    fn encode_ordinary(&self, text: &str) -> Vec<u32> {
        let words = self.basic_tokenize(text);
        let mut ids = Vec::new();
        for word in &words {
            ids.extend(self.wordpiece_tokenize(word));
        }
        ids
    }
}

impl Tokenize for WordPieceTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        // Recognize added tokens in the input first (HF behavior), then WordPiece.
        super::added::AddedTokens::dispatch(&self.added, text, |gap| self.encode_ordinary(gap))
    }

    fn decode(&self, ids: &[u32]) -> Result<String, TokenizeError> {
        if self.continuation_prefix.is_empty() {
            self.decode_without_prefix(ids)
        } else {
            self.decode_with_prefix(ids)
        }
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

    /// Decode when vocab uses `##` prefix — use prefix presence to detect continuations.
    fn decode_with_prefix(&self, ids: &[u32]) -> Result<String, TokenizeError> {
        let mut pieces = Vec::with_capacity(ids.len());

        for &id in ids {
            let token = self
                .id_to_token
                .get(id as usize)
                .ok_or(TokenizeError::InvalidTokenId(id))?;

            if is_special_token(token) || self.special_decode.contains(&id) {
                continue;
            }

            if let Some(stripped) = token.strip_prefix(self.continuation_prefix.as_str()) {
                pieces.push(stripped.to_string());
            } else {
                if !pieces.is_empty() {
                    pieces.push(" ".to_string());
                }
                pieces.push(token.to_string());
            }
        }

        Ok(cleanup_tokenization(&pieces.join("")))
    }

    /// Decode when vocab has no `##` prefix (GGUF-stripped).
    /// Without `##`, we can't distinguish continuations from word starts,
    /// so we just join with spaces between each token.
    fn decode_without_prefix(&self, ids: &[u32]) -> Result<String, TokenizeError> {
        let mut parts = Vec::with_capacity(ids.len());

        for &id in ids {
            let token = self
                .id_to_token
                .get(id as usize)
                .ok_or(TokenizeError::InvalidTokenId(id))?;

            if is_special_token(token) || self.special_decode.contains(&id) {
                continue;
            }

            parts.push(token.as_str());
        }

        Ok(cleanup_tokenization(&parts.join(" ")))
    }
}

/// HuggingFace `tokenizers` WordPiece-decoder cleanup (`cleanup=true`, the
/// default): drop the space before `. ? ! ,`. (Unlike `transformers`'
/// `clean_up_tokenization_spaces`, the `tokenizers` decoder does NOT touch
/// apostrophe contractions.)
fn cleanup_tokenization(s: &str) -> String {
    s.replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
}

fn is_special_token(token: &str) -> bool {
    matches!(token, "[CLS]" | "[SEP]" | "[PAD]" | "[UNK]" | "[MASK]")
        || (token.starts_with("[unused") && token.ends_with(']'))
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

    #[test]
    fn test_decode_invalid_id() {
        let tok = make_tokenizer();
        let result = tok.decode(&[999]);
        assert!(result.is_err());
    }
}
