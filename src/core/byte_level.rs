//! ByteLevel encoding/decoding for BPE tokenizers.
//!
//! This module implements the ByteLevel preprocessing used by GPT-2, Llama, DeepSeek,
//! and other modern tokenizers. It provides a bijective mapping between raw bytes
//! (0-255) and printable Unicode characters.
//!
//! # Why ByteLevel?
//!
//! BPE tokenizers need to handle arbitrary byte sequences, but many bytes (like
//! control characters 0x00-0x1F) are problematic:
//! - They may be invisible or cause display issues
//! - They can interfere with text processing
//! - They're hard to distinguish visually
//!
//! ByteLevel solves this by mapping each byte to a unique, printable Unicode character.
//!
//! # Mapping Strategy
//!
//! The mapping preserves printable ASCII and Latin-1 characters (they map to themselves),
//! while non-printable bytes get mapped to characters starting at U+0100:
//!
//! - Bytes 33-126 (`!` to `~`): Map to themselves
//! - Bytes 161-172 (`¡` to `¬`): Map to themselves
//! - Bytes 174-255 (`®` to `ÿ`): Map to themselves
//! - Other bytes (0-32, 127-160, 173): Map to U+0100 onwards
//!
//! This is compatible with HuggingFace tokenizers and GPT-2's approach.
//!
//! # Example
//!
//! ```ignore
//! // Space (0x20 = 32) maps to 'Ġ' (U+0120)
//! let encoded = byte_level_encode(b" ");
//! assert_eq!(encoded, "Ġ");
//!
//! // Chinese text gets converted to ByteLevel representation
//! let encoded = byte_level_encode("你好".as_bytes());
//! assert_eq!(encoded, "ä½łå¥½");
//! ```

use rustc_hash::FxHashMap;
use std::sync::LazyLock;

/// Byte to Unicode character mapping (256 entries).
/// Maps each byte value (0-255) to a unique Unicode character.
static BYTE_TO_CHAR: LazyLock<[char; 256]> = LazyLock::new(|| {
    let mut mapping = ['\0'; 256];

    // Printable ASCII and Latin-1 characters that map to themselves
    let mut direct_chars: Vec<u8> = Vec::new();

    // ASCII printable: ! (33) to ~ (126)
    direct_chars.extend(33u8..=126);
    // Latin-1 printable: ¡ (161) to ¬ (172)
    direct_chars.extend(161u8..=172);
    // Latin-1 printable: ® (174) to ÿ (255)
    direct_chars.extend(174u8..=255);

    // First, mark direct mappings
    for &b in &direct_chars {
        mapping[b as usize] = b as char;
    }

    // Map remaining bytes to U+0100 onwards
    let mut next_char = 256u32; // Start at U+0100
    for b in 0u8..=255 {
        if !direct_chars.contains(&b) {
            mapping[b as usize] = char::from_u32(next_char).unwrap();
            next_char += 1;
        }
    }

    mapping
});

/// Unicode character to byte mapping (reverse of BYTE_TO_CHAR).
static CHAR_TO_BYTE: LazyLock<FxHashMap<char, u8>> = LazyLock::new(|| {
    BYTE_TO_CHAR
        .iter()
        .enumerate()
        .map(|(byte, &ch)| (ch, byte as u8))
        .collect()
});

/// Encode a byte slice using ByteLevel encoding.
///
/// Converts each byte to its corresponding Unicode character representation.
/// The resulting string can be safely used for BPE tokenization.
///
/// # Arguments
/// * `bytes` - Raw bytes to encode
///
/// # Returns
/// A String where each byte is represented as a Unicode character
///
/// # Example
/// ```ignore
/// // Space becomes 'Ġ'
/// assert_eq!(byte_level_encode(b" hello"), "Ġhello");
///
/// // Chinese UTF-8 bytes become ByteLevel characters
/// assert_eq!(byte_level_encode("你".as_bytes()), "ä½ł");
/// ```
#[inline]
pub fn byte_level_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|&b| BYTE_TO_CHAR[b as usize]).collect()
}

/// Decode a ByteLevel-encoded string back to raw bytes.
///
/// Converts each Unicode character back to its original byte value.
///
/// # Arguments
/// * `text` - ByteLevel-encoded string
///
/// # Returns
/// The original raw bytes, or None if the string contains invalid ByteLevel characters
///
/// # Example
/// ```ignore
/// let bytes = byte_level_decode("Ġhello").unwrap();
/// assert_eq!(bytes, b" hello");
/// ```
#[inline]
pub fn byte_level_decode(text: &str) -> Option<Vec<u8>> {
    text.chars()
        .map(|ch| CHAR_TO_BYTE.get(&ch).copied())
        .collect()
}

/// Decode ByteLevel-encoded bytes back to raw bytes.
///
/// This is useful when you have the ByteLevel representation as bytes
/// (e.g., from token decoding) and need the original bytes.
///
/// # Arguments
/// * `encoded_bytes` - UTF-8 bytes of the ByteLevel-encoded string
///
/// # Returns
/// The original raw bytes, or None if decoding fails
#[inline]
pub fn byte_level_decode_bytes(encoded_bytes: &[u8]) -> Option<Vec<u8>> {
    // First convert to string (the ByteLevel encoding is valid UTF-8)
    let text = std::str::from_utf8(encoded_bytes).ok()?;
    byte_level_decode(text)
}

/// Check if a character is part of the ByteLevel alphabet.
#[inline]
pub fn is_byte_level_char(ch: char) -> bool {
    CHAR_TO_BYTE.contains_key(&ch)
}

/// Get the ByteLevel character for a specific byte value.
#[inline]
pub fn get_byte_level_char(byte: u8) -> char {
    BYTE_TO_CHAR[byte as usize]
}

/// Get the byte value for a ByteLevel character.
#[inline]
pub fn get_byte_level_byte(ch: char) -> Option<u8> {
    CHAR_TO_BYTE.get(&ch).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_level_mapping_is_bijective() {
        // Every byte should map to a unique character
        let mut seen: std::collections::HashSet<char> = std::collections::HashSet::new();
        for b in 0u8..=255 {
            let ch = BYTE_TO_CHAR[b as usize];
            assert!(!seen.contains(&ch), "Duplicate mapping for byte {}", b);
            seen.insert(ch);
        }
        assert_eq!(seen.len(), 256);
    }

    #[test]
    fn test_byte_level_roundtrip() {
        // Test roundtrip for all bytes
        for b in 0u8..=255 {
            let encoded = byte_level_encode(&[b]);
            let decoded = byte_level_decode(&encoded).unwrap();
            assert_eq!(decoded, vec![b], "Roundtrip failed for byte {}", b);
        }
    }

    #[test]
    fn test_space_mapping() {
        // Space (0x20 = 32) should map to 'Ġ' (U+0120)
        let space_char = BYTE_TO_CHAR[32];
        assert_eq!(space_char, 'Ġ');
        assert_eq!(space_char as u32, 0x0120);
    }

    #[test]
    fn test_printable_ascii_preserved() {
        // Printable ASCII (33-126) should map to themselves
        for b in 33u8..=126 {
            let ch = BYTE_TO_CHAR[b as usize];
            assert_eq!(ch as u8, b, "ASCII {} should map to itself", b);
        }
    }

    #[test]
    fn test_encode_hello() {
        let encoded = byte_level_encode(b"Hello");
        assert_eq!(encoded, "Hello"); // All printable ASCII
    }

    #[test]
    fn test_encode_with_space() {
        let encoded = byte_level_encode(b" hello");
        assert_eq!(encoded, "Ġhello"); // Space becomes Ġ
    }

    #[test]
    fn test_encode_chinese() {
        // "你好" in UTF-8 is [228, 189, 160, 229, 165, 189]
        let text = "你好";
        let encoded = byte_level_encode(text.as_bytes());
        assert_eq!(encoded, "ä½łå¥½");
    }

    #[test]
    fn test_decode_hello() {
        let decoded = byte_level_decode("Hello").unwrap();
        assert_eq!(decoded, b"Hello");
    }

    #[test]
    fn test_decode_with_space() {
        let decoded = byte_level_decode("Ġhello").unwrap();
        assert_eq!(decoded, b" hello");
    }

    #[test]
    fn test_decode_chinese() {
        let decoded = byte_level_decode("ä½łå¥½").unwrap();
        assert_eq!(String::from_utf8(decoded).unwrap(), "你好");
    }

    #[test]
    fn test_full_roundtrip_string() {
        let original = "Hello, 世界! 🌍";
        let encoded = byte_level_encode(original.as_bytes());
        let decoded_bytes = byte_level_decode(&encoded).unwrap();
        let decoded = String::from_utf8(decoded_bytes).unwrap();
        assert_eq!(decoded, original);
    }
}
