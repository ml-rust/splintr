//! WordPiece tokenizer for BERT-family models.
//!
//! Implements the standard BERT tokenization pipeline:
//! 1. **BasicTokenizer**: lowercase, strip accents, split on whitespace and punctuation
//! 2. **WordPiece**: greedy longest-match subword tokenization with `##` continuation prefix
//!
//! Handles `[CLS]`, `[SEP]`, `[PAD]`, `[UNK]` special tokens.

use super::tokenize::{Tokenize, TokenizeError};
use std::collections::HashMap;

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
    /// Whether the vocabulary uses `##` prefix for continuation tokens.
    /// If false, continuations are looked up without prefix (GGUF-stripped vocabs).
    has_continuation_prefix: bool,
    /// Special token IDs for [CLS], [SEP], [PAD]
    cls_token_id: Option<u32>,
    sep_token_id: Option<u32>,
    pad_token_id: Option<u32>,
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
        let mut token_to_id = HashMap::with_capacity(vocab.len());
        for (id, token) in vocab.iter().enumerate() {
            token_to_id.insert(token.clone(), id as u32);
        }

        // Auto-detect whether vocab uses ## prefix for continuations
        let has_continuation_prefix = token_to_id.keys().any(|k| k.starts_with("##"));

        let cls_token_id = token_to_id.get("[CLS]").copied();
        let sep_token_id = token_to_id.get("[SEP]").copied();
        let pad_token_id = token_to_id.get("[PAD]").copied();

        Self {
            token_to_id,
            id_to_token: vocab,
            unk_token_id,
            max_word_len,
            do_lower_case,
            has_continuation_prefix,
            cls_token_id,
            sep_token_id,
            pad_token_id,
        }
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
        let text = if self.do_lower_case {
            let lowered = text.to_lowercase();
            strip_accents(&lowered)
        } else {
            text.to_string()
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
            let mut found = false;

            while start < end {
                let raw: String = chars[start..end].iter().collect();
                let lookup = if start == 0 || !self.has_continuation_prefix {
                    raw
                } else {
                    format!("##{}", raw)
                };

                if let Some(&id) = self.token_to_id.get(&lookup) {
                    ids.push(id);
                    found = true;
                    start = end;
                    break;
                }

                end -= 1;
            }

            if !found {
                ids.push(self.unk_token_id);
                start += 1;
            }
        }

        ids
    }
}

impl Tokenize for WordPieceTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        let words = self.basic_tokenize(text);
        let mut ids = Vec::new();

        for word in &words {
            let word_ids = self.wordpiece_tokenize(word);
            ids.extend(word_ids);
        }

        ids
    }

    fn decode(&self, ids: &[u32]) -> Result<String, TokenizeError> {
        if self.has_continuation_prefix {
            self.decode_with_prefix(ids)
        } else {
            self.decode_without_prefix(ids)
        }
    }

    fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }
}

impl WordPieceTokenizer {
    /// Decode when vocab uses `##` prefix — use prefix presence to detect continuations.
    fn decode_with_prefix(&self, ids: &[u32]) -> Result<String, TokenizeError> {
        let mut pieces = Vec::with_capacity(ids.len());

        for &id in ids {
            let token = self
                .id_to_token
                .get(id as usize)
                .ok_or(TokenizeError::InvalidTokenId(id))?;

            if is_special_token(token) {
                continue;
            }

            if let Some(stripped) = token.strip_prefix("##") {
                pieces.push(stripped.to_string());
            } else {
                if !pieces.is_empty() {
                    pieces.push(" ".to_string());
                }
                pieces.push(token.to_string());
            }
        }

        Ok(pieces.join(""))
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

            if is_special_token(token) {
                continue;
            }

            parts.push(token.as_str());
        }

        Ok(parts.join(" "))
    }
}

fn is_special_token(token: &str) -> bool {
    matches!(token, "[CLS]" | "[SEP]" | "[PAD]" | "[UNK]" | "[MASK]")
        || (token.starts_with("[unused") && token.ends_with(']'))
}

/// Strip Unicode combining marks (accents) from text.
fn strip_accents(text: &str) -> String {
    use unicode_normalization::UnicodeNormalization;
    text.nfd()
        .filter(|c| !unicode_normalization::char::is_combining_mark(*c))
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
    fn test_unknown_word() {
        let tok = make_tokenizer();
        // "xyz" has no vocab entries → each char becomes [UNK]
        let ids = tok.encode("xyz");
        assert!(ids.iter().all(|&id| id == 1));
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
