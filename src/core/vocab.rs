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
