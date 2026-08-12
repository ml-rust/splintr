//! The streaming decoder itself.
//!
//! Nothing about decoding is decided here: the decoder is a long-lived
//! [`DecodeCursor`](super::state::DecodeCursor) over the same
//! [`DecodeState`] whole-sequence decoding drives, so id rendering, UTF-8
//! reassembly and the text post-ops all live in exactly one place. This type
//! only chooses *when* to feed it and what an unknown id means. Byte-level
//! unmapping happens before the UTF-8 buffer — the buffer only ever sees real
//! bytes.

use super::state::{DecodeCursor, DecodeState};
use crate::core::tokenize::TokenizeError;
use std::convert::Infallible;
use std::sync::Arc;

/// A streaming decoder that handles incomplete UTF-8 sequences across token
/// boundaries.
///
/// When decoding tokens one at a time (as in streaming LLM output), a token's
/// bytes may end in the middle of a multi-byte UTF-8 character. This decoder
/// buffers incomplete sequences and only returns complete, valid UTF-8 strings.
///
/// # Obtaining one
///
/// There is no public constructor: a decoder is built by a tokenizer's own
/// `streaming_decoder` —
/// [`Tokenizer::streaming_decoder`](crate::Tokenizer::streaming_decoder),
/// [`SpmTokenizer::streaming_decoder`](crate::SpmTokenizer::streaming_decoder)
/// or
/// [`SentencePieceTokenizer::streaming_decoder`](crate::SentencePieceTokenizer::streaming_decoder)
/// or
/// [`WordPieceTokenizer::streaming_decoder`](crate::WordPieceTokenizer::streaming_decoder) —
/// which takes the surfaces, the skipped-special-token set and every spelling
/// rule (byte level, byte fallback, metaspace) from that tokenizer's own
/// configuration. A decoder therefore cannot be paired with the wrong kind of
/// vocabulary — the mistake that used to turn a byte-level stream into mojibake
/// is not expressible.
///
/// A tokenizer loaded from a `tokenizer.json` has one more factory,
/// [`AnyTokenizer::streaming_decoder`](crate::AnyTokenizer::streaming_decoder):
/// it takes those same rules from the file's *declared* `decoder` pipeline when
/// one is declared, and delegates to the backend factory above when none is —
/// the same choice `AnyTokenizer::decode` makes.
///
/// # Agreement with whole-sequence decoding
///
/// For any ids, concatenating every emission plus the final [`flush`](Self::flush)
/// equals [`Tokenizer::decode_lossy`](crate::Tokenizer::decode_lossy) of the same
/// ids, and equals [`Tokenizer::decode`](crate::Tokenizer::decode) whenever that
/// succeeds. (A stream cannot see the future, so bytes that are still invalid at
/// `flush` become U+FFFD instead of an error — that is the one and only case
/// where strict whole-sequence decoding reports something the stream renders
/// lossily.) Chunk boundaries affect only *when* text is emitted, never *what*.
///
/// The decoder owns its state and borrows nothing, so it can be moved into a
/// generation task and outlive the scope the tokenizer was created in.
///
/// # Example
///
/// ```rust
/// use splintr::{from_pretrained, Backend};
///
/// let any = from_pretrained("cl100k_base")?;
/// let Backend::Bpe(tokenizer) = any.into_backend() else {
///     unreachable!("cl100k_base loads as a BPE backend");
/// };
/// let mut decoder = tokenizer.streaming_decoder();
///
/// for token_id in tokenizer.encode("Hello, world!") {
///     if let Some(text) = decoder.add_token(token_id)? {
///         print!("{}", text);
///     }
/// }
/// // Flush any remaining buffered bytes
/// print!("{}", decoder.flush());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct StreamingDecoder {
    /// A cursor over the state shared with the tokenizer that produced this
    /// decoder, so building one never copies the vocabulary map — and so a
    /// stream renders ids through exactly the code whole-sequence decoding
    /// renders them through. The cursor is long-lived: it *is* the decoder's
    /// position in the stream.
    cursor: DecodeCursor<Arc<DecodeState>>,
}

impl StreamingDecoder {
    /// Build a decoder over an already-captured decode configuration.
    ///
    /// Crate-internal on purpose: see the type-level docs.
    pub(crate) fn new(state: Arc<DecodeState>) -> Self {
        Self {
            cursor: DecodeCursor::new(state),
        }
    }

    /// Add a token and return any complete UTF-8 characters.
    ///
    /// Returns `Some(string)` if there are complete characters to emit, or
    /// `None` if the current bytes are still incomplete.
    ///
    /// Errors with [`TokenizeError::InvalidTokenId`] if the id is in neither
    /// the vocabulary nor the special tokens, matching
    /// [`Tokenizer::decode`](crate::Tokenizer::decode). Ids already accepted
    /// stay buffered; [`reset`](Self::reset) discards them.
    pub fn add_token(&mut self, id: u32) -> Result<Option<String>, TokenizeError> {
        self.add_tokens(&[id])
    }

    /// Add multiple tokens at once and return complete UTF-8 characters.
    ///
    /// Feeding ids in groups is indistinguishable from feeding them one by one:
    /// only the emission points differ, never the concatenated text.
    pub fn add_tokens(&mut self, ids: &[u32]) -> Result<Option<String>, TokenizeError> {
        self.cursor
            .feed(ids, |id| Err(TokenizeError::InvalidTokenId(id)))
    }

    /// Add a token, skipping it if it is in no table.
    ///
    /// The permissive twin of [`add_token`](Self::add_token), matching
    /// [`Tokenizer::decode_lossy`](crate::Tokenizer::decode_lossy). This never
    /// fails, so `on_unknown` is instantiated with [`Infallible`], letting the
    /// compiler prove the `Err` arm away rather than a runtime assertion
    /// claiming it.
    pub fn add_token_lossy(&mut self, id: u32) -> Option<String> {
        self.add_tokens_lossy(&[id])
    }

    /// Add multiple tokens at once, skipping any that are in no table.
    pub fn add_tokens_lossy(&mut self, ids: &[u32]) -> Option<String> {
        match self.cursor.feed(ids, |_| Ok::<(), Infallible>(())) {
            Ok(text) => text,
            // `Infallible` has no values, so this match has no arms to write.
            Err(never) => match never {},
        }
    }

    /// Flush any remaining buffered bytes.
    ///
    /// If there are incomplete UTF-8 sequences in the buffer, they will be
    /// replaced with the Unicode replacement character (U+FFFD).
    pub fn flush(&mut self) -> String {
        self.cursor.flush()
    }

    /// Reset the decoder state, discarding any buffered bytes.
    ///
    /// The decoder is then indistinguishable from a freshly built one.
    pub fn reset(&mut self) {
        self.cursor.reset();
    }

    /// Check if there are buffered bytes waiting for completion.
    pub fn has_pending(&self) -> bool {
        self.cursor.has_pending()
    }

    /// Get the number of pending bytes in the buffer.
    pub fn pending_bytes(&self) -> usize {
        self.cursor.pending_bytes()
    }
}

#[cfg(test)]
mod tests {
    use crate::core::any_tokenizer::Backend;
    use crate::core::byte_level::byte_level_encode;
    use crate::core::pretrained::from_pretrained;
    use crate::core::tokenize::TokenizeError;
    use crate::core::tokenizer::Tokenizer;
    use proptest::prelude::*;
    use rustc_hash::FxHashMap;
    use std::sync::OnceLock;

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

    /// A metaspace-decoder BPE vocabulary: ▁ marks word boundaries and must
    /// become a space on decode, streaming included.
    fn make_metaspace_tokenizer() -> Tokenizer {
        let mut encoder = FxHashMap::default();
        encoder.insert("\u{2581}Hello".as_bytes().to_vec(), 10);
        encoder.insert("\u{2581}world".as_bytes().to_vec(), 11);
        // A ▁ split across two tokens: the ▁ is 3 UTF-8 bytes (E2 96 81), so
        // the substitution has to survive the buffer reassembling it.
        encoder.insert(vec![0xE2, 0x96], 12);
        encoder.insert(vec![0x81, b'x'], 13);

        Tokenizer::new_with_metaspace_decoder(encoder, FxHashMap::default(), r".").unwrap()
    }

    /// A byte-fallback BPE vocabulary in the shape mistral-7b's
    /// `tokenizer.json` declares: the four bytes of `𐍈` (U+10348) are
    /// reachable only through their `<0xNN>` entries, so that one character
    /// encodes to four separate tokens.
    fn make_byte_fallback_tokenizer() -> Tokenizer {
        let mut encoder = FxHashMap::default();
        encoder.insert(b"a".to_vec(), 1);
        encoder.insert(b"c".to_vec(), 2);
        for (i, b) in [0xF0u8, 0x90, 0x8D, 0x88].into_iter().enumerate() {
            encoder.insert(format!("<0x{b:02X}>").into_bytes(), 10 + i as u32);
        }

        let lookup = crate::core::encoder::encoder_from_owned(encoder.clone());
        let byte_fallback =
            Tokenizer::byte_fallback_from(|spelling| lookup.get(spelling), None, true);
        Tokenizer::new(encoder, FxHashMap::default(), r"\S+|\s+")
            .expect("the test pattern compiles")
            .with_byte_fallback(byte_fallback)
    }

    /// The concrete BPE backend behind a bundled vocabulary.
    fn pretrained_bpe(name: &str) -> Tokenizer {
        let any = from_pretrained(name).expect("bundled vocabulary loads");
        match any.into_backend() {
            Backend::Bpe(tokenizer) => tokenizer,
            _ => panic!("{name} is a BPE vocabulary"),
        }
    }

    /// The bundled vocabularies, built once: a proptest case must not pay for
    /// parsing a 100k-entry vocabulary on every iteration.
    fn cl100k_base() -> &'static Tokenizer {
        static TOKENIZER: OnceLock<Tokenizer> = OnceLock::new();
        TOKENIZER.get_or_init(|| pretrained_bpe("cl100k_base"))
    }

    /// The ByteLevel counterpart, built once for the same reason.
    fn deepseek_v3() -> &'static Tokenizer {
        static TOKENIZER: OnceLock<Tokenizer> = OnceLock::new();
        TOKENIZER.get_or_init(|| pretrained_bpe("deepseek_v3"))
    }

    /// Texts exercising ASCII, multi-byte scripts, emoji (4-byte sequences) and
    /// combining marks — every shape that can straddle a token boundary.
    const AGREEMENT_TEXTS: &[&str] = &[
        "",
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog. 1234567890",
        "こんにちは世界、これはテストです。",
        "Привет, мир! Здравствуйте.",
        "🎉🚀 emoji 👨‍👩‍👧‍👦 family, and é combining e\u{0301}.",
        "混合 mixed 텍스트 with\ttabs\nand  spaces   ",
        "def f(x):\n    return x ** 2  # code",
    ];

    /// Feed `ids` through a streaming decoder in the given chunk sizes and
    /// concatenate every emission plus the final flush.
    fn drive_strict(tokenizer: &Tokenizer, ids: &[u32], chunk: usize) -> String {
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
    fn drive_lossy(tokenizer: &Tokenizer, ids: &[u32]) -> String {
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

    // =========================================================================
    // Raw (non-ByteLevel) vocabularies
    // =========================================================================

    #[test]
    fn test_simple_ascii() {
        let tokenizer = make_test_tokenizer();
        let mut decoder = tokenizer.streaming_decoder();

        // ASCII is single-byte, should return immediately
        assert_eq!(
            decoder.add_token(b'H' as u32).unwrap(),
            Some("H".to_string())
        );
        assert_eq!(
            decoder.add_token(b'i' as u32).unwrap(),
            Some("i".to_string())
        );
        assert!(!decoder.has_pending());
    }

    #[test]
    fn test_multi_byte_complete() {
        let tokenizer = make_test_tokenizer();
        let mut decoder = tokenizer.streaming_decoder();

        // "世界" token should return the complete string
        assert_eq!(decoder.add_token(257).unwrap(), Some("世界".to_string()));
        assert!(!decoder.has_pending());
    }

    #[test]
    fn test_multi_byte_split() {
        let tokenizer = make_test_tokenizer();
        let mut decoder = tokenizer.streaming_decoder();

        // "世" in UTF-8 is: 0xE4 0xB8 0x96 (3 bytes)
        // Feed them one at a time
        assert_eq!(decoder.add_token(0xE4).unwrap(), None); // First byte of 3-byte sequence
        assert!(decoder.has_pending());
        assert_eq!(decoder.pending_bytes(), 1);

        assert_eq!(decoder.add_token(0xB8).unwrap(), None); // Second byte
        assert_eq!(decoder.pending_bytes(), 2);

        // Third byte completes it
        assert_eq!(decoder.add_token(0x96).unwrap(), Some("世".to_string()));
        assert!(!decoder.has_pending());
    }

    #[test]
    fn test_flush_incomplete() {
        let tokenizer = make_test_tokenizer();
        let mut decoder = tokenizer.streaming_decoder();

        // Add incomplete sequence
        decoder.add_token(0xE4).unwrap(); // First byte of 3-byte sequence
        decoder.add_token(0xB8).unwrap(); // Second byte

        // Flush should return replacement character
        let flushed = decoder.flush();
        assert!(flushed.contains('\u{FFFD}')); // Replacement character
        assert!(!decoder.has_pending());
    }

    #[test]
    fn test_reset() {
        let tokenizer = make_test_tokenizer();
        let mut decoder = tokenizer.streaming_decoder();

        decoder.add_token(0xE4).unwrap();
        assert!(decoder.has_pending());

        decoder.reset();
        assert!(!decoder.has_pending());
    }

    #[test]
    fn test_mixed_complete_incomplete() {
        let tokenizer = make_test_tokenizer();
        let mut decoder = tokenizer.streaming_decoder();

        // Add complete ASCII - should return immediately
        let result1 = decoder.add_token(b'H' as u32).unwrap();
        assert_eq!(result1, Some("H".to_string()));
        assert!(!decoder.has_pending());

        // Add incomplete UTF-8 byte - should buffer it
        let result2 = decoder.add_token(0xE4).unwrap(); // Start of 3-byte sequence
        assert_eq!(result2, None);
        assert!(decoder.has_pending());
    }

    #[test]
    fn test_add_tokens_batch() {
        let tokenizer = make_test_tokenizer();
        let mut decoder = tokenizer.streaming_decoder();

        // Add multiple tokens at once
        let result = decoder
            .add_tokens(&[b'H' as u32, b'i' as u32, b'!' as u32])
            .unwrap();
        assert_eq!(result, Some("Hi!".to_string()));
    }

    // =========================================================================
    // ByteLevel vocabularies — reached through the same factory, which picks
    // the ByteLevel path from the tokenizer's own configuration.
    // =========================================================================

    #[test]
    fn test_byte_level_simple_ascii() {
        let tokenizer = make_byte_level_tokenizer();
        let mut decoder = tokenizer.streaming_decoder();

        // "Hello" token should decode to "Hello"
        let result = decoder.add_token(100).unwrap();
        assert_eq!(result, Some("Hello".to_string()));
        assert!(!decoder.has_pending());
    }

    #[test]
    fn test_byte_level_with_space() {
        let tokenizer = make_byte_level_tokenizer();
        let mut decoder = tokenizer.streaming_decoder();

        // " world" with ByteLevel-encoded space
        let result = decoder.add_token(101).unwrap();
        assert_eq!(result, Some(" world".to_string()));
    }

    #[test]
    fn test_byte_level_chinese() {
        let tokenizer = make_byte_level_tokenizer();
        let mut decoder = tokenizer.streaming_decoder();

        // "你好" as ByteLevel-encoded token
        let result = decoder.add_token(102).unwrap();
        assert_eq!(result, Some("你好".to_string()));
    }

    #[test]
    fn test_byte_level_split_chinese() {
        let tokenizer = make_byte_level_tokenizer();
        let mut decoder = tokenizer.streaming_decoder();

        // "你" is 3 UTF-8 bytes, feed them one at a time as ByteLevel tokens
        // First byte
        let result1 = decoder.add_token(200).unwrap();
        assert_eq!(result1, None);
        assert!(decoder.has_pending());

        // Second byte
        let result2 = decoder.add_token(201).unwrap();
        assert_eq!(result2, None);
        assert!(decoder.has_pending());

        // Third byte completes the character
        let result3 = decoder.add_token(202).unwrap();
        assert_eq!(result3, Some("你".to_string()));
        assert!(!decoder.has_pending());
    }

    #[test]
    fn test_byte_level_special_token() {
        let tokenizer = make_byte_level_tokenizer();
        let mut decoder = tokenizer.streaming_decoder();

        // Special tokens are NOT ByteLevel-encoded
        let result = decoder.add_token(1000).unwrap();
        assert_eq!(result, Some("<|think|>".to_string()));
    }

    #[test]
    fn test_byte_level_mixed() {
        let tokenizer = make_byte_level_tokenizer();
        let mut decoder = tokenizer.streaming_decoder();

        // Mix of regular and special tokens
        let result = decoder.add_tokens(&[100, 1000, 101]).unwrap();
        assert_eq!(result, Some("Hello<|think|> world".to_string()));
    }

    #[test]
    fn test_byte_level_flush() {
        let tokenizer = make_byte_level_tokenizer();
        let mut decoder = tokenizer.streaming_decoder();

        // Add incomplete sequence (first 2 bytes of "你")
        decoder.add_token(200).unwrap();
        decoder.add_token(201).unwrap();
        assert!(decoder.has_pending());

        // Flush should produce replacement character
        let flushed = decoder.flush();
        assert!(flushed.contains('\u{FFFD}'));
        assert!(!decoder.has_pending());
    }

    #[test]
    fn test_byte_level_reset() {
        let tokenizer = make_byte_level_tokenizer();
        let mut decoder = tokenizer.streaming_decoder();

        decoder.add_token(200).unwrap();
        assert!(decoder.has_pending());

        decoder.reset();
        assert!(!decoder.has_pending());
    }

    // =========================================================================
    // The configuration the caller used to have to guess
    // =========================================================================

    /// A `special=true` added token is dropped by the stream exactly as
    /// whole-sequence decoding drops it — and it is a skip, not an unknown id,
    /// so even the strict entry point accepts it.
    #[test]
    fn test_special_decode_ids_are_skipped_like_decode() {
        let skipped: rustc_hash::FxHashSet<u32> = [1000u32].into_iter().collect();
        let tokenizer = make_byte_level_tokenizer().with_special_decode_ids(skipped);

        let ids = [100, 1000, 101];
        let mut decoder = tokenizer.streaming_decoder();
        let streamed = decoder.add_tokens(&ids).unwrap().unwrap_or_default() + &decoder.flush();

        assert_eq!(streamed, "Hello world");
        assert_eq!(streamed, tokenizer.decode(&ids).unwrap());
    }

    /// The metaspace substitution runs on the stream too, including when the ▁
    /// itself is split across two tokens.
    #[test]
    fn test_metaspace_decoder_applies_to_the_stream() {
        let tokenizer = make_metaspace_tokenizer();

        for ids in [vec![10, 11], vec![12, 13]] {
            let expected = tokenizer.decode(&ids).unwrap();
            assert_eq!(drive_strict(&tokenizer, &ids, 1), expected);
        }
        assert_eq!(drive_strict(&tokenizer, &[10, 11], 1), " Hello world");
        // ▁ reassembled across the token boundary still becomes a space.
        assert_eq!(drive_strict(&tokenizer, &[12, 13], 1), " x");
    }

    /// D23: a character split across several `<0xNN>` byte-fallback tokens
    /// reassembles across `add_token` calls — the resolved bytes go through the
    /// same UTF-8 buffer every other byte does, so nothing is emitted until the
    /// character is complete, and `concat(stream) == decode` still holds.
    #[test]
    fn test_byte_fallback_char_reassembles_across_add_token_calls() {
        let tokenizer = make_byte_fallback_tokenizer();
        let ids = tokenizer.encode("a𐍈c");
        assert_eq!(ids, vec![1, 10, 11, 12, 13, 2]);

        let mut decoder = tokenizer.streaming_decoder();
        assert_eq!(decoder.add_token(1).unwrap(), Some("a".to_string()));
        // The three leading bytes of `𐍈` stay buffered: an incomplete
        // character is never emitted.
        for id in [10, 11, 12] {
            assert_eq!(decoder.add_token(id).unwrap(), None);
            assert!(decoder.has_pending());
        }
        assert_eq!(decoder.add_token(13).unwrap(), Some("𐍈".to_string()));
        assert!(!decoder.has_pending());
        assert_eq!(decoder.add_token(2).unwrap(), Some("c".to_string()));
        assert_eq!(decoder.flush(), "");

        // And the agreement property, under every grouping.
        let expected = tokenizer.decode(&ids).expect("real ids decode");
        assert_eq!(expected, "a𐍈c");
        for chunk in 1..=ids.len() {
            assert_eq!(drive_strict(&tokenizer, &ids, chunk), expected);
        }
        assert_eq!(drive_lossy(&tokenizer, &ids), tokenizer.decode_lossy(&ids));
    }

    /// Strict streaming reports an id in no table, mirroring `decode`; the
    /// lossy twin skips it, mirroring `decode_lossy`.
    #[test]
    fn test_unknown_id_strict_errors_and_lossy_skips() {
        let tokenizer = make_byte_level_tokenizer();
        let unknown = 999_999;

        let mut strict = tokenizer.streaming_decoder();
        assert!(matches!(
            strict.add_token(unknown),
            Err(TokenizeError::InvalidTokenId(id)) if id == unknown
        ));
        assert!(tokenizer.decode(&[unknown]).is_err());

        let mut lossy = tokenizer.streaming_decoder();
        let emitted = lossy
            .add_tokens_lossy(&[100, unknown, 101])
            .unwrap_or_default();
        let text = emitted + &lossy.flush();
        assert_eq!(text, "Hello world");
        assert_eq!(text, tokenizer.decode_lossy(&[100, unknown, 101]));
    }

    /// The decoder owns everything it needs, so it outlives the scope its
    /// tokenizer was created in — this compiles only because it carries no
    /// lifetime.
    #[test]
    fn test_decoder_is_owned_and_outlives_its_tokenizer() {
        let mut decoder = {
            let tokenizer = make_test_tokenizer();
            tokenizer.streaming_decoder()
        };
        assert_eq!(
            decoder.add_token(b'H' as u32).unwrap(),
            Some("H".to_string())
        );
    }

    // =========================================================================
    // The agreement property: concat(stream) == decode
    // =========================================================================

    /// A raw vocabulary: streaming a real encoding one token at a time must
    /// reproduce `decode` exactly.
    #[test]
    fn test_stream_matches_decode_cl100k_base() {
        let tokenizer = cl100k_base();
        for text in AGREEMENT_TEXTS {
            let ids = tokenizer.encode(text);
            let expected = tokenizer.decode(&ids).expect("real ids decode");
            assert_eq!(drive_strict(tokenizer, &ids, 1), expected, "text: {text:?}");
            assert_eq!(drive_lossy(tokenizer, &ids), tokenizer.decode_lossy(&ids));
        }
    }

    /// A ByteLevel vocabulary, where picking the wrong decoder used to produce
    /// mojibake silently.
    #[test]
    fn test_stream_matches_decode_deepseek_v3() {
        let tokenizer = deepseek_v3();
        for text in AGREEMENT_TEXTS {
            let ids = tokenizer.encode(text);
            let expected = tokenizer.decode(&ids).expect("real ids decode");
            assert_eq!(drive_strict(tokenizer, &ids, 1), expected, "text: {text:?}");
            assert_eq!(drive_lossy(tokenizer, &ids), tokenizer.decode_lossy(&ids));
        }
    }

    proptest! {
        /// Chunk-partition invariance on a raw vocabulary: arbitrary grouping
        /// through `add_tokens` gives what one-at-a-time gives, and both give
        /// `decode`.
        #[test]
        fn prop_chunking_matches_decode_cl100k_base(
            text in ".{0,120}",
            chunk in 1usize..8,
        ) {
            let tokenizer = cl100k_base();
            let ids = tokenizer.encode(&text);
            let expected = tokenizer.decode(&ids).expect("real ids decode");

            prop_assert_eq!(drive_strict(tokenizer, &ids, 1), expected.clone());
            prop_assert_eq!(drive_strict(tokenizer, &ids, chunk), expected);
        }

        /// Same on a ByteLevel vocabulary: the ByteLevel unmapping happens
        /// before the UTF-8 buffer, so a character split across tokens still
        /// reassembles under any grouping.
        #[test]
        fn prop_chunking_matches_decode_deepseek_v3(
            text in ".{0,120}",
            chunk in 1usize..8,
        ) {
            let tokenizer = deepseek_v3();
            let ids = tokenizer.encode(&text);
            let expected = tokenizer.decode(&ids).expect("real ids decode");

            prop_assert_eq!(drive_strict(tokenizer, &ids, 1), expected.clone());
            prop_assert_eq!(drive_strict(tokenizer, &ids, chunk), expected);
        }

        /// Arbitrary ids — including unknown ones and mid-character splits —
        /// stream lossily to exactly what `decode_lossy` produces.
        #[test]
        fn prop_arbitrary_ids_match_decode_lossy(
            ids in prop::collection::vec(0u32..300, 0..48),
        ) {
            let tokenizer = make_test_tokenizer();
            prop_assert_eq!(drive_lossy(&tokenizer, &ids), tokenizer.decode_lossy(&ids));
        }

        /// `reset()` purity: a used-then-reset decoder behaves byte-identically
        /// to a freshly built one on the same following ids.
        #[test]
        fn prop_reset_matches_a_fresh_decoder(
            dirty in prop::collection::vec(0u32..300, 0..16),
            ids in prop::collection::vec(0u32..300, 0..32),
        ) {
            let tokenizer = make_test_tokenizer();

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
}
