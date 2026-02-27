//! SentencePiece-compatible unigram tokenizer.
//!
//! This tokenizer implements greedy longest-match encoding with score-based
//! tie-breaking, compatible with SentencePiece unigram models used by
//! Llama, Mistral, and other models distributed in GGUF format.
//!
//! Unlike BPE (which merges byte pairs iteratively), unigram tokenization
//! greedily selects the longest matching token at each position, using
//! scores to break ties between equal-length matches.

use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SentencePieceError {
    #[error("Empty vocabulary")]
    EmptyVocab,
    #[error("Scores length ({scores}) does not match tokens length ({tokens})")]
    ScoreMismatch { scores: usize, tokens: usize },
    #[error("Decoding error: token ID {0} out of range")]
    InvalidTokenId(u32),
}

/// SentencePiece-compatible unigram tokenizer.
///
/// Accepts a raw vocabulary (token strings, scores, special token IDs) and
/// performs greedy longest-match encoding with SentencePiece word boundary
/// markers (▁ U+2581).
///
/// # Example
///
/// ```
/// use splintr::SentencePieceTokenizer;
///
/// let tokens = vec!["▁Hello".to_string(), "▁world".to_string(), "H".to_string()];
/// let scores = vec![0.0; 3];
/// let tok = SentencePieceTokenizer::new(tokens, scores, None, 2).unwrap();
/// ```
pub struct SentencePieceTokenizer {
    /// Token string -> ID mapping
    token_to_id: HashMap<String, u32>,
    /// ID -> Token string mapping
    id_to_token: Vec<String>,
    /// Scores for each token (used for tie-breaking)
    scores: Vec<f32>,
    /// BOS token ID
    bos_token_id: Option<u32>,
    /// EOS token ID
    eos_token_id: u32,
}

impl SentencePieceTokenizer {
    /// Create a tokenizer from raw vocabulary data.
    ///
    /// # Arguments
    /// * `tokens` - Token strings, indexed by token ID
    /// * `scores` - Score per token (for tie-breaking). If empty, defaults to all zeros.
    /// * `bos_token_id` - Optional beginning-of-sequence token ID
    /// * `eos_token_id` - End-of-sequence token ID
    pub fn new(
        tokens: Vec<String>,
        scores: Vec<f32>,
        bos_token_id: Option<u32>,
        eos_token_id: u32,
    ) -> Result<Self, SentencePieceError> {
        if tokens.is_empty() {
            return Err(SentencePieceError::EmptyVocab);
        }

        let scores = if scores.is_empty() {
            vec![0.0; tokens.len()]
        } else if scores.len() != tokens.len() {
            return Err(SentencePieceError::ScoreMismatch {
                scores: scores.len(),
                tokens: tokens.len(),
            });
        } else {
            scores
        };

        let mut token_to_id = HashMap::with_capacity(tokens.len());
        for (id, token) in tokens.iter().enumerate() {
            token_to_id.insert(token.clone(), id as u32);
        }

        Ok(Self {
            token_to_id,
            id_to_token: tokens,
            scores,
            bos_token_id,
            eos_token_id,
        })
    }

    /// Encode text to token IDs using greedy longest-match.
    ///
    /// Prepends BOS token if configured. Replaces spaces with ▁ (U+2581)
    /// following the SentencePiece convention.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();

        if let Some(bos_id) = self.bos_token_id {
            tokens.push(bos_id);
        }

        // SentencePiece: prepend ▁ and replace spaces with ▁
        let processed = format!("▁{}", text.replace(' ', "▁"));
        let chars: Vec<char> = processed.chars().collect();
        let mut pos = 0;
        let mut substr_buf = String::with_capacity(256 * 4);

        while pos < chars.len() {
            let mut best_len = 0;
            let mut best_id = None;
            let mut best_score = f32::NEG_INFINITY;

            substr_buf.clear();
            for end in (pos + 1)..=chars.len().min(pos + 256) {
                substr_buf.push(chars[end - 1]);
                if let Some(&id) = self.token_to_id.get(&substr_buf) {
                    let score = self.scores.get(id as usize).copied().unwrap_or(0.0);
                    let len = end - pos;
                    if len > best_len || (len == best_len && score > best_score) {
                        best_len = len;
                        best_id = Some(id);
                        best_score = score;
                    }
                }
            }

            if let Some(id) = best_id {
                tokens.push(id);
                pos += best_len;
            } else {
                let c = chars[pos];
                let byte_tokens = self.encode_char_as_bytes(c);
                if !byte_tokens.is_empty() {
                    tokens.extend(byte_tokens);
                }
                pos += 1;
            }
        }

        tokens
    }

    /// Encode a character as individual byte tokens using `<0xNN>` format.
    ///
    /// Each UTF-8 byte of the character is looked up as a token (e.g., `<0xFF>`).
    /// Bytes not present in the vocabulary are silently skipped.
    fn encode_char_as_bytes(&self, c: char) -> Vec<u32> {
        let mut result = Vec::new();
        let mut buf = [0u8; 4];
        let bytes = c.encode_utf8(&mut buf);

        for b in bytes.as_bytes() {
            let byte_token = format!("<0x{:02X}>", b);
            if let Some(&id) = self.token_to_id.get(&byte_token) {
                result.push(id);
            }
        }

        result
    }

    /// Decode token IDs to text.
    ///
    /// Skips BOS/EOS tokens and converts ▁ back to spaces.
    pub fn decode(&self, ids: &[u32]) -> Result<String, SentencePieceError> {
        let mut result = String::new();

        for &id in ids {
            let token = self
                .id_to_token
                .get(id as usize)
                .ok_or(SentencePieceError::InvalidTokenId(id))?;

            if Some(id) == self.bos_token_id || id == self.eos_token_id {
                continue;
            }
            result.push_str(&token.replace('▁', " "));
        }

        // Remove leading space (artifact of SentencePiece encoding)
        if result.starts_with(' ') {
            result.remove(0);
        }

        Ok(result)
    }

    /// Decode token IDs to text, skipping invalid IDs.
    pub fn decode_lossy(&self, ids: &[u32]) -> String {
        let mut result = String::new();

        for &id in ids {
            if let Some(token) = self.id_to_token.get(id as usize) {
                if Some(id) == self.bos_token_id || id == self.eos_token_id {
                    continue;
                }
                result.push_str(&token.replace('▁', " "));
            }
        }

        if result.starts_with(' ') {
            result.remove(0);
        }

        result
    }

    /// Check if a token is the EOS token.
    pub fn is_eos(&self, token_id: u32) -> bool {
        token_id == self.eos_token_id
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    /// Get EOS token ID.
    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    /// Get BOS token ID.
    pub fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tokenizer() -> SentencePieceTokenizer {
        // Minimal vocab: ▁Hello, ▁world, ▁, <0x48> (byte fallback for 'H')
        let tokens = vec![
            "<unk>".to_string(),  // 0
            "<s>".to_string(),    // 1 (BOS)
            "</s>".to_string(),   // 2 (EOS)
            "▁Hello".to_string(), // 3
            "▁world".to_string(), // 4
            "▁".to_string(),      // 5
            "H".to_string(),      // 6
            "e".to_string(),      // 7
            "l".to_string(),      // 8
            "o".to_string(),      // 9
        ];
        let scores = vec![0.0; tokens.len()];
        SentencePieceTokenizer::new(tokens, scores, Some(1), 2).unwrap()
    }

    #[test]
    fn test_encode_basic() {
        let tok = make_tokenizer();
        let ids = tok.encode("Hello world");
        // BOS(1), ▁Hello(3), ▁world(4)
        assert_eq!(ids, vec![1, 3, 4]);
    }

    #[test]
    fn test_decode_basic() {
        let tok = make_tokenizer();
        let text = tok.decode(&[1, 3, 4]).unwrap();
        assert_eq!(text, "Hello world");
    }

    #[test]
    fn test_decode_skips_bos_eos() {
        let tok = make_tokenizer();
        let text = tok.decode(&[1, 3, 2]).unwrap();
        assert_eq!(text, "Hello");
    }

    #[test]
    fn test_roundtrip() {
        let tok = make_tokenizer();
        let ids = tok.encode("Hello world");
        let text = tok.decode(&ids).unwrap();
        assert_eq!(text, "Hello world");
    }

    #[test]
    fn test_vocab_size() {
        let tok = make_tokenizer();
        assert_eq!(tok.vocab_size(), 10);
    }

    #[test]
    fn test_is_eos() {
        let tok = make_tokenizer();
        assert!(tok.is_eos(2));
        assert!(!tok.is_eos(1));
    }

    #[test]
    fn test_empty_scores_defaults() {
        let tokens = vec!["▁a".to_string(), "▁b".to_string()];
        let tok = SentencePieceTokenizer::new(tokens, vec![], None, 1).unwrap();
        assert_eq!(tok.vocab_size(), 2);
    }

    #[test]
    fn test_empty_vocab_errors() {
        let result = SentencePieceTokenizer::new(vec![], vec![], None, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_score_mismatch_errors() {
        let tokens = vec!["a".to_string()];
        let result = SentencePieceTokenizer::new(tokens, vec![1.0, 2.0], None, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_encode_empty_string() {
        let tok = make_tokenizer();
        let ids = tok.encode("");
        // BOS + ▁ (SentencePiece always prepends ▁, which matches token 5)
        assert_eq!(ids, vec![1, 5]);
    }

    #[test]
    fn test_encode_empty_string_no_bos() {
        let tokens = vec!["▁a".to_string(), "▁b".to_string()];
        let tok = SentencePieceTokenizer::new(tokens, vec![], None, 1).unwrap();
        let ids = tok.encode("");
        assert!(ids.is_empty());
    }

    #[test]
    fn test_decode_lossy_skips_invalid_tokens() {
        let tok = make_tokenizer();
        // 999 is out of range, should be skipped
        let text = tok.decode_lossy(&[1, 3, 999, 4]);
        assert_eq!(text, "Hello world");
    }

    #[test]
    fn test_decode_lossy_all_invalid() {
        let tok = make_tokenizer();
        let text = tok.decode_lossy(&[999, 1000, 1001]);
        assert_eq!(text, "");
    }

    #[test]
    fn test_decode_invalid_token_id_errors() {
        let tok = make_tokenizer();
        let result = tok.decode(&[1, 999]);
        assert!(result.is_err());
    }
}
