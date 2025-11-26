//! UTF-8 safe streaming decoder for token-by-token LLM output.
//!
//! This module provides a stateful decoder that buffers incomplete UTF-8 sequences
//! and only emits complete, valid UTF-8 characters. This is critical for streaming
//! LLM output where token boundaries may not align with character boundaries.
//!
//! # ByteLevel Support
//!
//! For tokenizers using ByteLevel encoding (GPT-2, Llama, DeepSeek V3), the
//! [`ByteLevelStreamingDecoder`] handles the ByteLevel-to-bytes conversion
//! before UTF-8 assembly.

use super::byte_level::byte_level_decode_bytes;
use super::tokenizer::Tokenizer;

/// A streaming decoder that handles incomplete UTF-8 sequences across token boundaries.
///
/// When decoding tokens one at a time (as in streaming LLM output), a token's bytes
/// may end in the middle of a multi-byte UTF-8 character. This decoder buffers
/// incomplete sequences and only returns complete, valid UTF-8 strings.
///
/// # Example
///
/// ```ignore
/// let tokenizer = Tokenizer::from_pretrained("cl100k_base")?;
/// let mut decoder = StreamingDecoder::new(&tokenizer);
///
/// for token_id in token_stream {
///     if let Some(text) = decoder.add_token(token_id) {
///         print!("{}", text);
///     }
/// }
/// // Flush any remaining buffered bytes
/// print!("{}", decoder.flush());
/// ```
pub struct StreamingDecoder<'a> {
    tokenizer: &'a Tokenizer,
    buffer: Vec<u8>,
}

impl<'a> StreamingDecoder<'a> {
    /// Create a new streaming decoder for the given tokenizer.
    pub fn new(tokenizer: &'a Tokenizer) -> Self {
        Self {
            tokenizer,
            buffer: Vec::with_capacity(16),
        }
    }

    /// Add a token and return any complete UTF-8 characters.
    ///
    /// Returns `Some(string)` if there are complete characters to emit,
    /// or `None` if the current bytes are still incomplete.
    pub fn add_token(&mut self, token_id: u32) -> Option<String> {
        // Get bytes for this token
        let bytes = if let Some(b) = self.tokenizer.decoder().get(&token_id) {
            b.as_slice()
        } else if let Some(s) = self.tokenizer.special_tokens_decoder().get(&token_id) {
            s.as_bytes()
        } else {
            return None;
        };

        // Add to buffer
        self.buffer.extend_from_slice(bytes);

        // Try to extract complete UTF-8 characters
        self.extract_complete_utf8()
    }

    /// Add multiple tokens at once and return complete UTF-8 characters.
    pub fn add_tokens(&mut self, token_ids: &[u32]) -> Option<String> {
        for &token_id in token_ids {
            // Get bytes for this token
            let bytes = if let Some(b) = self.tokenizer.decoder().get(&token_id) {
                b.as_slice()
            } else if let Some(s) = self.tokenizer.special_tokens_decoder().get(&token_id) {
                s.as_bytes()
            } else {
                continue;
            };

            self.buffer.extend_from_slice(bytes);
        }

        self.extract_complete_utf8()
    }

    /// Flush any remaining buffered bytes.
    ///
    /// If there are incomplete UTF-8 sequences in the buffer, they will be
    /// replaced with the Unicode replacement character (U+FFFD).
    pub fn flush(&mut self) -> String {
        if self.buffer.is_empty() {
            return String::new();
        }

        let result = String::from_utf8_lossy(&self.buffer).into_owned();
        self.buffer.clear();
        result
    }

    /// Reset the decoder state, discarding any buffered bytes.
    pub fn reset(&mut self) {
        self.buffer.clear();
    }

    /// Check if there are buffered bytes waiting for completion.
    pub fn has_pending(&self) -> bool {
        !self.buffer.is_empty()
    }

    /// Get the number of pending bytes in the buffer.
    pub fn pending_bytes(&self) -> usize {
        self.buffer.len()
    }

    /// Extract complete UTF-8 characters from the buffer.
    ///
    /// This function finds the longest valid UTF-8 prefix of the buffer,
    /// returns it as a string, and keeps any incomplete trailing bytes.
    fn extract_complete_utf8(&mut self) -> Option<String> {
        if self.buffer.is_empty() {
            return None;
        }

        // Find the longest valid UTF-8 prefix
        let valid_len = self.find_valid_utf8_len();

        if valid_len == 0 {
            return None;
        }

        // Extract the valid portion
        let valid_bytes: Vec<u8> = self.buffer.drain(..valid_len).collect();

        // SAFETY: We've verified this is valid UTF-8
        let result = unsafe { String::from_utf8_unchecked(valid_bytes) };

        Some(result)
    }

    /// Find the length of the longest valid UTF-8 prefix.
    ///
    /// This accounts for incomplete multi-byte sequences at the end.
    fn find_valid_utf8_len(&self) -> usize {
        let bytes = &self.buffer;
        let len = bytes.len();

        if len == 0 {
            return 0;
        }

        // First, try to validate the entire buffer
        if std::str::from_utf8(bytes).is_ok() {
            return len;
        }

        // Find how many bytes at the end might be an incomplete sequence
        // UTF-8 sequences can be 1-4 bytes long
        // We need to check if the last 1-3 bytes could be the start of an incomplete sequence

        for incomplete_len in 1..=3.min(len) {
            let check_len = len - incomplete_len;
            if check_len == 0 {
                continue;
            }

            // Check if prefix is valid UTF-8
            if std::str::from_utf8(&bytes[..check_len]).is_ok() {
                // Check if the trailing bytes could be an incomplete sequence
                if self.could_be_incomplete_sequence(&bytes[check_len..]) {
                    return check_len;
                }
            }
        }

        // If nothing works, find the last position that's valid
        // This handles cases with invalid bytes in the middle
        for i in (0..len).rev() {
            if std::str::from_utf8(&bytes[..=i]).is_ok() {
                return i + 1;
            }
        }

        0
    }

    /// Check if bytes could be the start of an incomplete UTF-8 sequence.
    fn could_be_incomplete_sequence(&self, bytes: &[u8]) -> bool {
        if bytes.is_empty() {
            return false;
        }

        let first = bytes[0];

        // Check if first byte indicates a multi-byte sequence
        // and we don't have all the continuation bytes
        match first {
            // 2-byte sequence: 110xxxxx
            0xC0..=0xDF => bytes.len() < 2,
            // 3-byte sequence: 1110xxxx
            0xE0..=0xEF => bytes.len() < 3,
            // 4-byte sequence: 11110xxx
            0xF0..=0xF7 => bytes.len() < 4,
            // Continuation byte or invalid - not the start of an incomplete sequence
            _ => false,
        }
    }
}

/// A streaming decoder for ByteLevel-encoded tokenizers (GPT-2, Llama, DeepSeek V3).
///
/// This decoder handles the ByteLevel encoding used by some tokenizers where raw bytes
/// are mapped to printable Unicode characters. It first decodes the ByteLevel representation
/// back to raw bytes, then assembles them into valid UTF-8 strings.
///
/// # Example
///
/// ```ignore
/// let tokenizer = Tokenizer::from_pretrained("deepseek_v3")?;
/// let mut decoder = ByteLevelStreamingDecoder::new(&tokenizer);
///
/// for token_id in token_stream {
///     if let Some(text) = decoder.add_token(token_id) {
///         print!("{}", text);
///     }
/// }
/// // Flush any remaining buffered bytes
/// print!("{}", decoder.flush());
/// ```
pub struct ByteLevelStreamingDecoder<'a> {
    tokenizer: &'a Tokenizer,
    buffer: Vec<u8>,
}

impl<'a> ByteLevelStreamingDecoder<'a> {
    /// Create a new ByteLevel streaming decoder for the given tokenizer.
    pub fn new(tokenizer: &'a Tokenizer) -> Self {
        Self {
            tokenizer,
            buffer: Vec::with_capacity(16),
        }
    }

    /// Add a token and return any complete UTF-8 characters.
    ///
    /// The token's ByteLevel-encoded bytes are first decoded to raw bytes,
    /// then assembled into valid UTF-8 strings.
    ///
    /// Returns `Some(string)` if there are complete characters to emit,
    /// or `None` if the current bytes are still incomplete.
    pub fn add_token(&mut self, token_id: u32) -> Option<String> {
        // Get bytes for this token
        if let Some(encoded_bytes) = self.tokenizer.decoder().get(&token_id) {
            // Decode ByteLevel encoding to raw bytes
            if let Some(raw_bytes) = byte_level_decode_bytes(encoded_bytes) {
                self.buffer.extend_from_slice(&raw_bytes);
            } else {
                // Fallback: treat as raw bytes if ByteLevel decode fails
                self.buffer.extend_from_slice(encoded_bytes);
            }
        } else if let Some(special) = self.tokenizer.special_tokens_decoder().get(&token_id) {
            // Special tokens are NOT ByteLevel-encoded, add directly
            self.buffer.extend_from_slice(special.as_bytes());
        } else {
            return None;
        }

        // Try to extract complete UTF-8 characters
        self.extract_complete_utf8()
    }

    /// Add multiple tokens at once and return complete UTF-8 characters.
    pub fn add_tokens(&mut self, token_ids: &[u32]) -> Option<String> {
        for &token_id in token_ids {
            if let Some(encoded_bytes) = self.tokenizer.decoder().get(&token_id) {
                // Decode ByteLevel encoding to raw bytes
                if let Some(raw_bytes) = byte_level_decode_bytes(encoded_bytes) {
                    self.buffer.extend_from_slice(&raw_bytes);
                } else {
                    self.buffer.extend_from_slice(encoded_bytes);
                }
            } else if let Some(special) = self.tokenizer.special_tokens_decoder().get(&token_id) {
                self.buffer.extend_from_slice(special.as_bytes());
            }
        }

        self.extract_complete_utf8()
    }

    /// Flush any remaining buffered bytes.
    ///
    /// If there are incomplete UTF-8 sequences in the buffer, they will be
    /// replaced with the Unicode replacement character (U+FFFD).
    pub fn flush(&mut self) -> String {
        if self.buffer.is_empty() {
            return String::new();
        }

        let result = String::from_utf8_lossy(&self.buffer).into_owned();
        self.buffer.clear();
        result
    }

    /// Reset the decoder state, discarding any buffered bytes.
    pub fn reset(&mut self) {
        self.buffer.clear();
    }

    /// Check if there are buffered bytes waiting for completion.
    pub fn has_pending(&self) -> bool {
        !self.buffer.is_empty()
    }

    /// Get the number of pending bytes in the buffer.
    pub fn pending_bytes(&self) -> usize {
        self.buffer.len()
    }

    /// Extract complete UTF-8 characters from the buffer.
    fn extract_complete_utf8(&mut self) -> Option<String> {
        if self.buffer.is_empty() {
            return None;
        }

        // Find the longest valid UTF-8 prefix
        let valid_len = self.find_valid_utf8_len();

        if valid_len == 0 {
            return None;
        }

        // Extract the valid portion
        let valid_bytes: Vec<u8> = self.buffer.drain(..valid_len).collect();

        // SAFETY: We've verified this is valid UTF-8
        let result = unsafe { String::from_utf8_unchecked(valid_bytes) };

        Some(result)
    }

    /// Find the length of the longest valid UTF-8 prefix.
    fn find_valid_utf8_len(&self) -> usize {
        let bytes = &self.buffer;
        let len = bytes.len();

        if len == 0 {
            return 0;
        }

        // First, try to validate the entire buffer
        if std::str::from_utf8(bytes).is_ok() {
            return len;
        }

        // Find how many bytes at the end might be an incomplete sequence
        for incomplete_len in 1..=3.min(len) {
            let check_len = len - incomplete_len;
            if check_len == 0 {
                continue;
            }

            if std::str::from_utf8(&bytes[..check_len]).is_ok()
                && self.could_be_incomplete_sequence(&bytes[check_len..])
            {
                return check_len;
            }
        }

        // Find the last position that's valid
        for i in (0..len).rev() {
            if std::str::from_utf8(&bytes[..=i]).is_ok() {
                return i + 1;
            }
        }

        0
    }

    /// Check if bytes could be the start of an incomplete UTF-8 sequence.
    fn could_be_incomplete_sequence(&self, bytes: &[u8]) -> bool {
        if bytes.is_empty() {
            return false;
        }

        let first = bytes[0];

        match first {
            0xC0..=0xDF => bytes.len() < 2, // 2-byte sequence
            0xE0..=0xEF => bytes.len() < 3, // 3-byte sequence
            0xF0..=0xF7 => bytes.len() < 4, // 4-byte sequence
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustc_hash::FxHashMap;

    fn make_test_tokenizer() -> Tokenizer {
        let mut encoder = FxHashMap::default();
        // Add all single bytes as tokens
        for b in 0u8..=255 {
            encoder.insert(vec![b], b as u32);
        }
        // Add some multi-byte tokens
        encoder.insert("Hello".as_bytes().to_vec(), 256);
        encoder.insert("世界".as_bytes().to_vec(), 257);

        let special_tokens = FxHashMap::default();
        let pattern = r".";

        Tokenizer::new(encoder, special_tokens, pattern).unwrap()
    }

    #[test]
    fn test_simple_ascii() {
        let tokenizer = make_test_tokenizer();
        let mut decoder = StreamingDecoder::new(&tokenizer);

        // ASCII is single-byte, should return immediately
        assert_eq!(decoder.add_token(b'H' as u32), Some("H".to_string()));
        assert_eq!(decoder.add_token(b'i' as u32), Some("i".to_string()));
        assert!(!decoder.has_pending());
    }

    #[test]
    fn test_multi_byte_complete() {
        let tokenizer = make_test_tokenizer();
        let mut decoder = StreamingDecoder::new(&tokenizer);

        // "世界" token should return the complete string
        assert_eq!(decoder.add_token(257), Some("世界".to_string()));
        assert!(!decoder.has_pending());
    }

    #[test]
    fn test_multi_byte_split() {
        let tokenizer = make_test_tokenizer();
        let mut decoder = StreamingDecoder::new(&tokenizer);

        // "世" in UTF-8 is: 0xE4 0xB8 0x96 (3 bytes)
        // Feed them one at a time
        assert_eq!(decoder.add_token(0xE4), None); // First byte of 3-byte sequence
        assert!(decoder.has_pending());
        assert_eq!(decoder.pending_bytes(), 1);

        assert_eq!(decoder.add_token(0xB8), None); // Second byte
        assert_eq!(decoder.pending_bytes(), 2);

        assert_eq!(decoder.add_token(0x96), Some("世".to_string())); // Third byte completes it
        assert!(!decoder.has_pending());
    }

    #[test]
    fn test_flush_incomplete() {
        let tokenizer = make_test_tokenizer();
        let mut decoder = StreamingDecoder::new(&tokenizer);

        // Add incomplete sequence
        decoder.add_token(0xE4); // First byte of 3-byte sequence
        decoder.add_token(0xB8); // Second byte

        // Flush should return replacement character
        let flushed = decoder.flush();
        assert!(flushed.contains('\u{FFFD}')); // Replacement character
        assert!(!decoder.has_pending());
    }

    #[test]
    fn test_reset() {
        let tokenizer = make_test_tokenizer();
        let mut decoder = StreamingDecoder::new(&tokenizer);

        decoder.add_token(0xE4);
        assert!(decoder.has_pending());

        decoder.reset();
        assert!(!decoder.has_pending());
    }

    #[test]
    fn test_mixed_complete_incomplete() {
        let tokenizer = make_test_tokenizer();
        let mut decoder = StreamingDecoder::new(&tokenizer);

        // Add complete ASCII - should return immediately
        let result1 = decoder.add_token(b'H' as u32);
        assert_eq!(result1, Some("H".to_string()));
        assert!(!decoder.has_pending());

        // Add incomplete UTF-8 byte - should buffer it
        let result2 = decoder.add_token(0xE4); // Start of 3-byte sequence
        assert_eq!(result2, None);
        assert!(decoder.has_pending());
    }

    #[test]
    fn test_add_tokens_batch() {
        let tokenizer = make_test_tokenizer();
        let mut decoder = StreamingDecoder::new(&tokenizer);

        // Add multiple tokens at once
        let result = decoder.add_tokens(&[b'H' as u32, b'i' as u32, b'!' as u32]);
        assert_eq!(result, Some("Hi!".to_string()));
    }

    // =========================================================================
    // ByteLevelStreamingDecoder tests
    // =========================================================================

    use super::super::byte_level::byte_level_encode;

    fn make_byte_level_tokenizer() -> Tokenizer {
        let mut encoder = FxHashMap::default();

        // Add ByteLevel-encoded tokens
        // "Hello" -> each byte maps to itself for ASCII
        encoder.insert(byte_level_encode(b"Hello").into_bytes(), 100);
        // " world" -> space (0x20) becomes Ġ (U+0120)
        encoder.insert(byte_level_encode(b" world").into_bytes(), 101);
        // "你好" in ByteLevel encoding
        encoder.insert(byte_level_encode("你好".as_bytes()).into_bytes(), 102);

        // Add individual ByteLevel-encoded bytes for split character tests
        // "你" in UTF-8 is [0xE4, 0xBD, 0xA0] - 3 bytes
        // Each byte gets ByteLevel encoded
        let ni_bytes = "你".as_bytes();
        for (i, &b) in ni_bytes.iter().enumerate() {
            let byte_level = byte_level_encode(&[b]);
            encoder.insert(byte_level.into_bytes(), 200 + i as u32);
        }

        let mut special_tokens = FxHashMap::default();
        special_tokens.insert("<|think|>".to_string(), 1000);

        let pattern = r".";

        Tokenizer::new_byte_level(encoder, special_tokens, pattern).unwrap()
    }

    #[test]
    fn test_byte_level_simple_ascii() {
        let tokenizer = make_byte_level_tokenizer();
        let mut decoder = ByteLevelStreamingDecoder::new(&tokenizer);

        // "Hello" token should decode to "Hello"
        let result = decoder.add_token(100);
        assert_eq!(result, Some("Hello".to_string()));
        assert!(!decoder.has_pending());
    }

    #[test]
    fn test_byte_level_with_space() {
        let tokenizer = make_byte_level_tokenizer();
        let mut decoder = ByteLevelStreamingDecoder::new(&tokenizer);

        // " world" with ByteLevel-encoded space
        let result = decoder.add_token(101);
        assert_eq!(result, Some(" world".to_string()));
    }

    #[test]
    fn test_byte_level_chinese() {
        let tokenizer = make_byte_level_tokenizer();
        let mut decoder = ByteLevelStreamingDecoder::new(&tokenizer);

        // "你好" as ByteLevel-encoded token
        let result = decoder.add_token(102);
        assert_eq!(result, Some("你好".to_string()));
    }

    #[test]
    fn test_byte_level_split_chinese() {
        let tokenizer = make_byte_level_tokenizer();
        let mut decoder = ByteLevelStreamingDecoder::new(&tokenizer);

        // "你" is 3 UTF-8 bytes, feed them one at a time as ByteLevel tokens
        // First byte
        let result1 = decoder.add_token(200);
        assert_eq!(result1, None);
        assert!(decoder.has_pending());

        // Second byte
        let result2 = decoder.add_token(201);
        assert_eq!(result2, None);
        assert!(decoder.has_pending());

        // Third byte completes the character
        let result3 = decoder.add_token(202);
        assert_eq!(result3, Some("你".to_string()));
        assert!(!decoder.has_pending());
    }

    #[test]
    fn test_byte_level_special_token() {
        let tokenizer = make_byte_level_tokenizer();
        let mut decoder = ByteLevelStreamingDecoder::new(&tokenizer);

        // Special tokens are NOT ByteLevel-encoded
        let result = decoder.add_token(1000);
        assert_eq!(result, Some("<|think|>".to_string()));
    }

    #[test]
    fn test_byte_level_mixed() {
        let tokenizer = make_byte_level_tokenizer();
        let mut decoder = ByteLevelStreamingDecoder::new(&tokenizer);

        // Mix of regular and special tokens
        let result = decoder.add_tokens(&[100, 1000, 101]);
        assert_eq!(result, Some("Hello<|think|> world".to_string()));
    }

    #[test]
    fn test_byte_level_flush() {
        let tokenizer = make_byte_level_tokenizer();
        let mut decoder = ByteLevelStreamingDecoder::new(&tokenizer);

        // Add incomplete sequence (first 2 bytes of "你")
        decoder.add_token(200);
        decoder.add_token(201);
        assert!(decoder.has_pending());

        // Flush should produce replacement character
        let flushed = decoder.flush();
        assert!(flushed.contains('\u{FFFD}'));
        assert!(!decoder.has_pending());
    }

    #[test]
    fn test_byte_level_reset() {
        let tokenizer = make_byte_level_tokenizer();
        let mut decoder = ByteLevelStreamingDecoder::new(&tokenizer);

        decoder.add_token(200);
        assert!(decoder.has_pending());

        decoder.reset();
        assert!(!decoder.has_pending());
    }
}
