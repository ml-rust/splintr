//! Incremental UTF-8 buffering shared by the streaming decoders.
//!
//! Token bytes arrive in arbitrary chunks that may cut a multi-byte UTF-8
//! character in half. `Utf8Buffer` accumulates those bytes and hands back only
//! the complete, valid UTF-8 prefix, keeping the incomplete tail for the next
//! push.
//!
//! # Contract
//!
//! For any byte sequence, feeding it to a `Utf8Buffer` in *any* chunking and
//! concatenating everything [`Utf8Buffer::take_complete`] emits plus a final
//! [`Utf8Buffer::flush`] equals `String::from_utf8_lossy` of the whole
//! sequence. Chunk boundaries affect only *when* text is emitted, never *what*
//! is emitted. `std::str::from_utf8` is the single authority on validity here:
//! there is no second, hand-rolled notion of what a valid sequence looks like.

/// A byte buffer that emits only complete, valid UTF-8.
pub(crate) struct Utf8Buffer {
    buffer: Vec<u8>,
}

impl Utf8Buffer {
    /// Create an empty buffer.
    pub(crate) fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(16),
        }
    }

    /// Append raw bytes to the buffer.
    pub(crate) fn push(&mut self, bytes: &[u8]) {
        self.buffer.extend_from_slice(bytes);
    }

    /// Clear the buffer, discarding any buffered bytes.
    pub(crate) fn clear(&mut self) {
        self.buffer.clear();
    }

    /// Check if there are buffered bytes waiting for completion.
    pub(crate) fn has_pending(&self) -> bool {
        !self.buffer.is_empty()
    }

    /// Get the number of pending bytes in the buffer.
    pub(crate) fn pending_len(&self) -> usize {
        self.buffer.len()
    }

    /// Flush any remaining buffered bytes.
    ///
    /// If there are incomplete UTF-8 sequences in the buffer, they will be
    /// replaced with the Unicode replacement character (U+FFFD).
    pub(crate) fn flush(&mut self) -> String {
        if self.buffer.is_empty() {
            return String::new();
        }

        let result = String::from_utf8_lossy(&self.buffer).into_owned();
        self.buffer.clear();
        result
    }

    /// Extract every character the buffer can already decide on.
    ///
    /// Valid text is emitted as-is. A byte that can never begin (or continue)
    /// a valid sequence is definitively invalid *now*, no matter what arrives
    /// later, so it is replaced with U+FFFD and scanning continues past it —
    /// otherwise a single bad byte at the head would stall the buffer until
    /// [`flush`](Self::flush). Only a trailing sequence that is incomplete but
    /// still *possible* stays buffered.
    ///
    /// Returns `None` when nothing could be decided yet.
    pub(crate) fn take_complete(&mut self) -> Option<String> {
        if self.buffer.is_empty() {
            return None;
        }

        let mut out = String::new();
        let mut consumed = 0;

        loop {
            let rest = &self.buffer[consumed..];
            if rest.is_empty() {
                break;
            }

            let (valid_up_to, error_len) = match std::str::from_utf8(rest) {
                // The whole remainder is valid: emit it and stop.
                Ok(valid) => {
                    out.push_str(valid);
                    consumed += valid.len();
                    break;
                }
                Err(e) => (e.valid_up_to(), e.error_len()),
            };

            // `Utf8Error` guarantees the prefix is valid, so the lossy
            // conversion borrows it unchanged rather than replacing anything.
            out.push_str(&String::from_utf8_lossy(&rest[..valid_up_to]));
            consumed += valid_up_to;

            match error_len {
                // Incomplete but still possible: keep the tail for the next push.
                None => break,
                // Definitively invalid: one U+FFFD, skip the bad bytes, keep going.
                Some(invalid_len) => {
                    out.push(char::REPLACEMENT_CHARACTER);
                    consumed += invalid_len;
                }
            }
        }

        self.buffer.drain(..consumed);

        if out.is_empty() {
            None
        } else {
            Some(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// Push bytes and immediately take whatever became complete.
    fn push_take(buf: &mut Utf8Buffer, bytes: &[u8]) -> Option<String> {
        buf.push(bytes);
        buf.take_complete()
    }

    /// Feed `input` in the given chunks and concatenate every emission plus the
    /// final flush — the left-hand side of the lossy-decoding contract.
    fn drive_chunks(input: &[u8], chunks: &[&[u8]]) -> String {
        let mut buf = Utf8Buffer::new();
        let mut out = String::new();

        debug_assert_eq!(
            chunks.concat(),
            input,
            "chunks must reassemble the input exactly"
        );

        for chunk in chunks {
            if let Some(text) = push_take(&mut buf, chunk) {
                out.push_str(&text);
            }
        }
        out.push_str(&buf.flush());
        out
    }

    /// Feed `input` one byte at a time through [`drive_chunks`].
    fn drive_byte_by_byte(input: &[u8]) -> String {
        let chunks: Vec<&[u8]> = input.chunks(1).collect();
        drive_chunks(input, &chunks)
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

    #[test]
    fn test_truncated_lead_recovers_on_next_byte() {
        let mut buf = Utf8Buffer::new();

        // A 3-byte lead that is never completed must not withhold the ASCII
        // that follows it: the lead is invalid the moment 'A' arrives.
        assert_eq!(push_take(&mut buf, &[0xE4]), None);
        assert_eq!(push_take(&mut buf, b"A"), Some("\u{FFFD}A".to_string()));
        assert_eq!(push_take(&mut buf, b"B"), Some("B".to_string()));
        assert_eq!(push_take(&mut buf, b"C"), Some("C".to_string()));
        assert!(!buf.has_pending());
    }

    #[test]
    fn test_stray_continuation_byte_recovers() {
        let mut buf = Utf8Buffer::new();

        // 0x80 can never start a sequence, so it is invalid on arrival.
        assert_eq!(push_take(&mut buf, &[0x80]), Some("\u{FFFD}".to_string()));
        assert!(!buf.has_pending());
        assert_eq!(push_take(&mut buf, b"ok"), Some("ok".to_string()));
    }

    #[test]
    fn test_never_valid_byte_recovers() {
        let mut buf = Utf8Buffer::new();

        // 0xFF appears in no valid UTF-8 sequence at all.
        assert_eq!(push_take(&mut buf, &[0xFF]), Some("\u{FFFD}".to_string()));
        assert!(!buf.has_pending());
        assert_eq!(push_take(&mut buf, b"ok"), Some("ok".to_string()));
    }

    #[test]
    fn test_invalid_lead_bytes_are_not_buffered_as_possible_leads() {
        // 0xC0/0xC1 would only ever encode overlong 2-byte forms and 0xF5 is
        // beyond U+10FFFF, so none of them can begin a sequence.
        for &lead in &[0xC0u8, 0xC1, 0xF5] {
            let mut buf = Utf8Buffer::new();

            assert_eq!(
                push_take(&mut buf, &[lead]),
                Some("\u{FFFD}".to_string()),
                "0x{lead:02X} must be rejected immediately"
            );
            assert!(!buf.has_pending(), "0x{lead:02X} must not stay buffered");
        }
    }

    #[test]
    fn test_overlong_encoding_is_rejected() {
        // 0xE0 0x80 0xAF is an overlong encoding of '/' (U+002F).
        let input = [0xE0, 0x80, 0xAF];
        let decoded = drive_byte_by_byte(&input);

        assert!(!decoded.contains('/'), "overlong form must not decode");
        assert_eq!(decoded, String::from_utf8_lossy(&input));
    }

    #[test]
    fn test_surrogate_encoding_is_rejected() {
        // 0xED 0xA0 0x80 is the CESU-8 style encoding of the surrogate U+D800.
        let input = [0xED, 0xA0, 0x80];
        let decoded = drive_byte_by_byte(&input);

        assert!(decoded.chars().all(|c| c == '\u{FFFD}'));
        assert_eq!(decoded, String::from_utf8_lossy(&input));
    }

    #[test]
    fn test_invalid_bytes_interleaved_with_split_multi_byte_char() {
        let mut buf = Utf8Buffer::new();

        // Bad byte, then "世" (0xE4 0xB8 0x96) split across two pushes, then a
        // second bad byte followed by ASCII.
        assert_eq!(push_take(&mut buf, &[0xFF]), Some("\u{FFFD}".to_string()));
        assert_eq!(push_take(&mut buf, &[0xE4, 0xB8]), None);
        assert_eq!(push_take(&mut buf, &[0x96]), Some("世".to_string()));
        assert_eq!(
            push_take(&mut buf, &[0x80, b'z']),
            Some("\u{FFFD}z".to_string())
        );
        assert!(!buf.has_pending());

        // ...and the same byte stream matches std when driven byte by byte.
        let input = [0xFF, 0xE4, 0xB8, 0x96, 0x80, b'z'];
        assert_eq!(drive_byte_by_byte(&input), String::from_utf8_lossy(&input));
    }

    proptest! {
        /// The oracle: byte-at-a-time feeding reproduces `from_utf8_lossy`.
        #[test]
        fn prop_byte_at_a_time_matches_lossy(input in prop::collection::vec(any::<u8>(), 0..64)) {
            prop_assert_eq!(
                drive_byte_by_byte(&input),
                String::from_utf8_lossy(&input).into_owned()
            );
        }

        /// Same oracle under arbitrary chunking, which is what proves chunk
        /// boundaries cannot change the decoded result.
        #[test]
        fn prop_arbitrary_chunking_matches_lossy(
            chunks in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..8), 0..16)
        ) {
            let input: Vec<u8> = chunks.concat();
            let chunk_refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();

            prop_assert_eq!(
                drive_chunks(&input, &chunk_refs),
                String::from_utf8_lossy(&input).into_owned()
            );
        }
    }
}
