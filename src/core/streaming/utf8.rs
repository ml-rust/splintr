//! Incremental UTF-8 buffering shared by the streaming decoders.
//!
//! Token bytes arrive in arbitrary chunks that may cut a multi-byte UTF-8
//! character in half. `Utf8Buffer` accumulates those bytes and hands back only
//! the complete, valid UTF-8 prefix, keeping the incomplete tail for the next
//! push.

/// A byte buffer that emits only complete, valid UTF-8.
pub(super) struct Utf8Buffer {
    buffer: Vec<u8>,
}

impl Utf8Buffer {
    /// Create an empty buffer.
    pub(super) fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(16),
        }
    }

    /// Append raw bytes to the buffer.
    pub(super) fn push(&mut self, bytes: &[u8]) {
        self.buffer.extend_from_slice(bytes);
    }

    /// Clear the buffer, discarding any buffered bytes.
    pub(super) fn clear(&mut self) {
        self.buffer.clear();
    }

    /// Check if there are buffered bytes waiting for completion.
    pub(super) fn has_pending(&self) -> bool {
        !self.buffer.is_empty()
    }

    /// Get the number of pending bytes in the buffer.
    pub(super) fn pending_len(&self) -> usize {
        self.buffer.len()
    }

    /// Flush any remaining buffered bytes.
    ///
    /// If there are incomplete UTF-8 sequences in the buffer, they will be
    /// replaced with the Unicode replacement character (U+FFFD).
    pub(super) fn flush(&mut self) -> String {
        if self.buffer.is_empty() {
            return String::new();
        }

        let result = String::from_utf8_lossy(&self.buffer).into_owned();
        self.buffer.clear();
        result
    }

    /// Extract complete UTF-8 characters from the buffer.
    ///
    /// This function finds the longest valid UTF-8 prefix of the buffer,
    /// returns it as a string, and keeps any incomplete trailing bytes.
    pub(super) fn take_complete(&mut self) -> Option<String> {
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
            if std::str::from_utf8(&bytes[..check_len]).is_ok()
                && could_be_incomplete_sequence(&bytes[check_len..])
            {
                // The trailing bytes could be an incomplete sequence
                return check_len;
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
}

/// Check if bytes could be the start of an incomplete UTF-8 sequence.
fn could_be_incomplete_sequence(bytes: &[u8]) -> bool {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Push bytes and immediately take whatever became complete.
    fn push_take(buf: &mut Utf8Buffer, bytes: &[u8]) -> Option<String> {
        buf.push(bytes);
        buf.take_complete()
    }

    #[test]
    fn test_ascii_is_emitted_immediately() {
        let mut buf = Utf8Buffer::new();

        assert_eq!(push_take(&mut buf, b"Hi!"), Some("Hi!".to_string()));
        assert!(!buf.has_pending());
    }

    #[test]
    fn test_multi_byte_split_across_pushes() {
        let mut buf = Utf8Buffer::new();

        // "世" in UTF-8 is: 0xE4 0xB8 0x96 (3 bytes)
        assert_eq!(push_take(&mut buf, &[0xE4]), None); // First byte of 3-byte sequence
        assert!(buf.has_pending());
        assert_eq!(buf.pending_len(), 1);

        assert_eq!(push_take(&mut buf, &[0xB8]), None); // Second byte
        assert_eq!(buf.pending_len(), 2);

        assert_eq!(push_take(&mut buf, &[0x96]), Some("世".to_string())); // Third byte completes it
        assert!(!buf.has_pending());
    }

    #[test]
    fn test_complete_prefix_with_incomplete_tail() {
        let mut buf = Utf8Buffer::new();

        // "H" plus the first byte of a 3-byte sequence: only "H" is complete
        assert_eq!(push_take(&mut buf, &[b'H', 0xE4]), Some("H".to_string()));
        assert!(buf.has_pending());
        assert_eq!(buf.pending_len(), 1);
    }

    #[test]
    fn test_flush_incomplete_yields_replacement_char() {
        let mut buf = Utf8Buffer::new();

        buf.push(&[0xE4, 0xB8]); // First two bytes of a 3-byte sequence

        let flushed = buf.flush();
        assert!(flushed.contains('\u{FFFD}')); // Replacement character
        assert!(!buf.has_pending());
    }

    #[test]
    fn test_flush_empty_is_empty_string() {
        let mut buf = Utf8Buffer::new();

        assert_eq!(buf.flush(), String::new());
    }

    #[test]
    fn test_clear_discards_pending() {
        let mut buf = Utf8Buffer::new();

        buf.push(&[0xE4]);
        assert!(buf.has_pending());

        buf.clear();
        assert!(!buf.has_pending());
    }
}
