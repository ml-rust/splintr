//! Vocabulary loading utilities for tiktoken BPE format.
//!
//! This module handles loading BPE vocabularies from the tiktoken file format
//! used by OpenAI's tokenizers (GPT-3.5, GPT-4, GPT-4o, etc.).
//!
//! # Tiktoken Format
//!
//! The tiktoken format is a simple text-based format where each line contains:
//! - A base64-encoded token (the byte sequence)
//! - A space separator
//! - An integer rank (the token's priority in BPE merging)
//!
//! Lower ranks indicate higher priority - tokens with lower ranks are merged
//! first during the BPE encoding process.
//!
//! # Example Format
//!
//! ```text
//! SGVsbG8= 0
//! V29ybGQ= 1
//! IQ== 2
//! ```
//!
//! Where:
//! - `SGVsbG8=` decodes to `Hello` (rank 0, highest priority)
//! - `V29ybGQ=` decodes to `World` (rank 1)
//! - `IQ==` decodes to `!` (rank 2)
//!
//! # Vocabulary Files
//!
//! OpenAI provides vocabulary files for their models:
//! - `cl100k_base.tiktoken`: ~100k tokens for GPT-4, GPT-3.5-turbo
//! - `o200k_base.tiktoken`: ~200k tokens for GPT-4o

use base64::{engine::general_purpose::STANDARD, Engine};
use rustc_hash::FxHashMap;
use thiserror::Error;

/// Type alias for encoder/decoder pair returned by `load_tiktoken_bpe_with_decoder`.
pub type EncoderDecoderPair = (FxHashMap<Vec<u8>, u32>, FxHashMap<u32, Vec<u8>>);

/// Errors that can occur when loading vocabulary files.
#[derive(Error, Debug)]
pub enum VocabError {
    #[error("Invalid base64 encoding: {0}")]
    Base64Error(#[from] base64::DecodeError),
    #[error("Invalid line format: {0}")]
    ParseError(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Vocabulary is empty")]
    EmptyVocab,
    #[error("Vocabulary has no token for id {0}")]
    MissingId(u32),
    #[error(
        "Vocabulary has no run of 256 consecutive single-byte tokens, so it carries no \
         SentencePiece byte fallback"
    )]
    MissingByteFallback,
    #[error("Token {id} is not valid UTF-8 and lies outside the byte-fallback run")]
    NonUtf8Token { id: u32 },
    #[error("Special token {name:?} claims id {id}, which the vocabulary spells {found:?}")]
    SpecialTokenConflict {
        id: u32,
        name: String,
        found: String,
    },
}

/// Load a tiktoken BPE vocabulary from raw bytes.
///
/// Format: `base64_token rank\n` per line
/// Example: `SGVsbG8= 0` (where "SGVsbG8=" decodes to "Hello")
pub fn load_tiktoken_bpe(data: &[u8]) -> Result<FxHashMap<Vec<u8>, u32>, VocabError> {
    let mut encoder = FxHashMap::default();

    for line in data.split(|&b| b == b'\n') {
        if line.is_empty() {
            continue;
        }

        // Find the space separator
        let space_pos = line
            .iter()
            .rposition(|&b| b == b' ')
            .ok_or_else(|| VocabError::ParseError("Missing space separator".to_string()))?;

        let token_b64 = &line[..space_pos];
        let rank_str = &line[space_pos + 1..];

        // Decode base64 token
        let token = STANDARD.decode(token_b64)?;

        // Parse rank
        let rank_str = std::str::from_utf8(rank_str)
            .map_err(|_| VocabError::ParseError("Invalid UTF-8 in rank".to_string()))?;
        let rank: u32 = rank_str
            .trim()
            .parse()
            .map_err(|_| VocabError::ParseError(format!("Invalid rank: {}", rank_str)))?;

        encoder.insert(token, rank);
    }

    Ok(encoder)
}

/// Load a tiktoken BPE vocabulary from a file path.
pub fn load_tiktoken_bpe_file(path: &str) -> Result<FxHashMap<Vec<u8>, u32>, VocabError> {
    let data = std::fs::read(path)?;
    load_tiktoken_bpe(&data)
}

/// Load a tiktoken BPE vocabulary and build both encoder and decoder.
///
/// This function preserves all token IDs in the decoder, even if multiple IDs map to the same
/// byte sequence. The encoder will only keep the FIRST occurrence of each byte sequence (lowest ID).
pub fn load_tiktoken_bpe_with_decoder(data: &[u8]) -> Result<EncoderDecoderPair, VocabError> {
    let mut encoder = FxHashMap::default();
    let mut decoder = FxHashMap::default();

    for line in data.split(|&b| b == b'\n') {
        if line.is_empty() {
            continue;
        }

        // Find the space separator
        let space_pos = line
            .iter()
            .rposition(|&b| b == b' ')
            .ok_or_else(|| VocabError::ParseError("Missing space separator".to_string()))?;

        let token_b64 = &line[..space_pos];
        let rank_str = &line[space_pos + 1..];

        // Decode base64 token
        let token = STANDARD.decode(token_b64)?;

        // Parse rank
        let rank_str = std::str::from_utf8(rank_str)
            .map_err(|_| VocabError::ParseError("Invalid UTF-8 in rank".to_string()))?;
        let rank: u32 = rank_str
            .trim()
            .parse()
            .map_err(|_| VocabError::ParseError(format!("Invalid rank: {}", rank_str)))?;

        // Always add to decoder (preserves all token IDs)
        decoder.insert(rank, token.clone());

        // Only add to encoder if this byte sequence isn't already mapped
        // This keeps the FIRST (lowest ID) occurrence
        encoder.entry(token).or_insert(rank);
    }

    Ok((encoder, decoder))
}

/// Load a tiktoken file as a **SentencePiece piece list**, indexed by token id.
///
/// A `.tiktoken` file stores every token as raw bytes, which is the right shape
/// for byte-level BPE and the wrong shape for SentencePiece. SentencePiece
/// merges *pieces*, and its word-boundary marker `▁` (U+2581 = `E2 96 81`) can
/// only be produced by merging characters: `E2 96` is not a piece any
/// SentencePiece vocabulary was ever trained on, so a byte-level merger can
/// never build `▁` and every word boundary shatters into three byte-fallback
/// tokens. Converting the file to pieces up front is what lets
/// [`SpmTokenizer`](super::spm::SpmTokenizer) merge the way llama.cpp does.
///
/// The 256 raw single-byte tokens these files carry are re-spelled `<0xNN>`
/// (uppercase hex), the GGUF/SentencePiece byte-fallback spelling that
/// `SpmTokenizer` recognizes. They are located by scanning for the run of 256
/// consecutive ids holding `0x00..=0xFF` in order rather than assuming a fixed
/// offset — the bundled Mistral files disagree on where it starts (V1 at id 3,
/// V2 at id 771) — and a file without such a run is rejected rather than
/// silently loaded with no byte fallback.
///
/// Every token outside that run must be valid UTF-8; one that is not means the
/// file is not a SentencePiece vocabulary, and is reported instead of being
/// lossily converted.
pub fn load_tiktoken_spm_pieces(data: &[u8]) -> Result<Vec<String>, VocabError> {
    let (_, decoder) = load_tiktoken_bpe_with_decoder(data)?;
    let max_id = decoder
        .keys()
        .copied()
        .max()
        .ok_or(VocabError::EmptyVocab)?;

    // Dense id → bytes. A hole means the file cannot be indexed by id at all.
    let mut slots: Vec<Option<Vec<u8>>> = vec![None; max_id as usize + 1];
    for (id, bytes) in decoder {
        if let Some(slot) = slots.get_mut(id as usize) {
            *slot = Some(bytes);
        }
    }
    let mut raw: Vec<Vec<u8>> = Vec::with_capacity(slots.len());
    for (id, slot) in slots.into_iter().enumerate() {
        match slot {
            Some(bytes) => raw.push(bytes),
            None => return Err(VocabError::MissingId(id as u32)),
        }
    }

    let start = byte_fallback_start(&raw).ok_or(VocabError::MissingByteFallback)?;

    let mut pieces = Vec::with_capacity(raw.len());
    for (id, bytes) in raw.iter().enumerate() {
        match id.checked_sub(start).filter(|offset| *offset < 256) {
            Some(byte) => pieces.push(format!("<0x{byte:02X}>")),
            None => match std::str::from_utf8(bytes) {
                Ok(piece) => pieces.push(piece.to_string()),
                Err(_) => return Err(VocabError::NonUtf8Token { id: id as u32 }),
            },
        }
    }
    Ok(pieces)
}

/// The id at which the 256 single-byte tokens `0x00..=0xFF` start, in order.
fn byte_fallback_start(raw: &[Vec<u8>]) -> Option<usize> {
    (0..raw.len()).find(|&start| {
        (0..=255u8).all(|b| {
            raw.get(start + b as usize)
                .is_some_and(|token| token.len() == 1 && token.first() == Some(&b))
        })
    })
}

/// Place named special tokens into a piece list at the ids they claim.
///
/// Bundled vocabularies carry special tokens the vocabulary *file* does not:
/// splintr's agent tokens sit above the file's last id. Without a slot in the
/// piece list those ids exist only in the matcher — encodable but not
/// decodable, and absent from `vocab_size`. The list is grown to cover them
/// (holes stay empty, which no input can produce and no merge can look up).
///
/// A special that lands on an id the file already spells differently is a
/// disagreement between the two, not something to overwrite: it is reported, so
/// a vocabulary bump that shifts ids fails loudly instead of quietly mapping a
/// chat marker onto a real word.
pub fn place_special_pieces(
    pieces: &mut Vec<String>,
    special: &FxHashMap<String, u32>,
) -> Result<(), VocabError> {
    let Some(&max_id) = special.values().max() else {
        return Ok(());
    };
    if pieces.len() <= max_id as usize {
        pieces.resize(max_id as usize + 1, String::new());
    }
    for (name, &id) in special {
        let Some(slot) = pieces.get_mut(id as usize) else {
            continue;
        };
        if slot.is_empty() {
            *slot = name.clone();
        } else if slot.as_str() != name.as_str() {
            return Err(VocabError::SpecialTokenConflict {
                id,
                name: name.clone(),
                found: slot.clone(),
            });
        }
    }
    Ok(())
}

/// Build a decoder map (token ID → bytes) from an encoder map (bytes → token ID).
///
/// This creates the inverse mapping needed for decoding tokens back to text.
/// The decoder is used during the decode phase to convert token IDs back to
/// their original byte sequences.
pub fn build_decoder(encoder: &FxHashMap<Vec<u8>, u32>) -> FxHashMap<u32, Vec<u8>> {
    encoder.iter().map(|(k, v)| (*v, k.clone())).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_tiktoken_bpe() {
        // "Hello" base64 = "SGVsbG8="
        // "World" base64 = "V29ybGQ="
        let data = b"SGVsbG8= 0\nV29ybGQ= 1\n";
        let encoder = load_tiktoken_bpe(data).unwrap();

        assert_eq!(encoder.get(b"Hello".as_slice()), Some(&0));
        assert_eq!(encoder.get(b"World".as_slice()), Some(&1));
        assert_eq!(encoder.len(), 2);
    }

    /// Build a `.tiktoken` blob from tokens listed in id order.
    fn tiktoken_blob(tokens: &[&[u8]]) -> Vec<u8> {
        let mut out = Vec::new();
        for (id, token) in tokens.iter().enumerate() {
            out.extend_from_slice(STANDARD.encode(token).as_bytes());
            out.extend_from_slice(format!(" {id}\n").as_bytes());
        }
        out
    }

    /// A SentencePiece-shaped vocabulary: a leading control token, the 256 raw
    /// bytes, then real pieces carrying the `▁` word-boundary marker.
    fn spm_shaped(byte_run_start: usize) -> Vec<Vec<u8>> {
        let mut tokens: Vec<Vec<u8>> = Vec::new();
        for i in 0..byte_run_start {
            tokens.push(format!("<ctrl_{i}>").into_bytes());
        }
        for b in 0..=255u8 {
            tokens.push(vec![b]);
        }
        tokens.push("▁the".as_bytes().to_vec());
        tokens.push("▁sour".as_bytes().to_vec());
        tokens
    }

    fn pieces_of(tokens: &[Vec<u8>]) -> Result<Vec<String>, VocabError> {
        let refs: Vec<&[u8]> = tokens.iter().map(Vec::as_slice).collect();
        load_tiktoken_spm_pieces(&tiktoken_blob(&refs))
    }

    /// The whole point of the conversion: the raw byte tokens become `<0xNN>`
    /// (the spelling `SpmTokenizer` recognizes as byte fallback) while real
    /// pieces keep their `▁` marker intact, so a word boundary is one piece
    /// rather than three bytes.
    #[test]
    fn spm_pieces_respell_the_byte_run_and_keep_the_word_boundary_marker() {
        let tokens = spm_shaped(3);
        let pieces = pieces_of(&tokens).unwrap();

        assert_eq!(pieces.len(), 3 + 256 + 2);
        assert_eq!(pieces[0], "<ctrl_0>");
        assert_eq!(pieces[3], "<0x00>");
        assert_eq!(pieces[3 + 0xE2], "<0xE2>");
        assert_eq!(pieces[3 + 255], "<0xFF>");
        assert_eq!(pieces[259], "▁the");
        assert_eq!(pieces[260], "▁sour");
        // Uppercase hex, matching GGUF/SentencePiece — `<0xab>` would not be found.
        assert_eq!(pieces[3 + 0xAB], "<0xAB>");
    }

    /// The run is located by scanning, not by a fixed offset: the bundled
    /// Mistral files start it at id 3 (V1) and id 771 (V2).
    #[test]
    fn spm_pieces_find_the_byte_run_wherever_it_starts() {
        for start in [0, 3, 771] {
            let pieces = pieces_of(&spm_shaped(start)).unwrap();
            assert_eq!(pieces[start], "<0x00>", "run start {start}");
            assert_eq!(pieces[start + 255], "<0xFF>", "run start {start}");
            assert_eq!(pieces[start + 256], "▁the", "run start {start}");
        }
    }

    /// Without a complete byte run there is no byte fallback, so arbitrary input
    /// could not be encoded. That must be an error, not a silent load.
    #[test]
    fn spm_pieces_reject_a_vocabulary_without_a_byte_run() {
        let mut tokens = spm_shaped(3);
        tokens.remove(3 + 200);
        assert!(matches!(
            pieces_of(&tokens),
            Err(VocabError::MissingByteFallback)
        ));
    }

    /// A non-UTF-8 token outside the byte run means the file is not a
    /// SentencePiece vocabulary; converting it lossily would invent a piece.
    #[test]
    fn spm_pieces_reject_non_utf8_outside_the_byte_run() {
        let mut tokens = spm_shaped(3);
        // `E2 96` — the first two bytes of `▁`, which is not a piece anywhere.
        tokens.push(vec![0xE2, 0x96]);
        assert!(matches!(
            pieces_of(&tokens),
            Err(VocabError::NonUtf8Token { id: 261 })
        ));
    }

    /// Special tokens above the file's last id must gain a slot, so they decode
    /// and count towards `vocab_size` rather than existing only in the matcher.
    #[test]
    fn place_special_pieces_grows_the_list_to_cover_added_ids() {
        let mut pieces = vec!["<unk>".to_string(), "a".to_string()];
        let mut special = FxHashMap::default();
        special.insert("<unk>".to_string(), 0);
        special.insert("<|pad|>".to_string(), 4);

        place_special_pieces(&mut pieces, &special).unwrap();
        assert_eq!(pieces.len(), 5);
        assert_eq!(pieces[4], "<|pad|>");
        // The hole stays empty — no input produces it and no merge looks it up.
        assert_eq!(pieces[2], "");
        assert_eq!(pieces[3], "");
    }

    /// A special landing on an id the file already spells differently is a
    /// disagreement between the two. Overwriting would silently map a chat
    /// marker onto a real word, so it is reported.
    #[test]
    fn place_special_pieces_reports_a_claimed_id_that_holds_another_token() {
        let mut pieces = vec!["<unk>".to_string(), "▁the".to_string()];
        let mut special = FxHashMap::default();
        special.insert("<|im_start|>".to_string(), 1);

        assert!(matches!(
            place_special_pieces(&mut pieces, &special),
            Err(VocabError::SpecialTokenConflict { id: 1, .. })
        ));
    }

    #[test]
    fn test_build_decoder() {
        let mut encoder = FxHashMap::default();
        encoder.insert(b"Hello".to_vec(), 0);
        encoder.insert(b"World".to_vec(), 1);

        let decoder = build_decoder(&encoder);
        assert_eq!(decoder.get(&0), Some(&b"Hello".to_vec()));
        assert_eq!(decoder.get(&1), Some(&b"World".to_vec()));
    }
}
