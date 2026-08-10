//! Vocabulary loading utilities for the bundled vocabulary formats.
//!
//! Two formats live here:
//!
//! - **tiktoken** (`.tiktoken`) — `base64(token_bytes) rank` per line, used by
//!   OpenAI's byte-level tokenizers (GPT-3.5, GPT-4, GPT-4o, …).
//! - **SentencePiece** (`.spm`) — `base64(piece) score` per line in id order,
//!   see [`load_spm_vocab`]. A SentencePiece vocabulary cannot be stored in the
//!   tiktoken format without losing its scores and its `<0xNN>` byte-fallback
//!   spellings, which is why it has a format of its own.
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

use super::token_bytes::{Decoder, Encoder, TokenBytes};

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
    #[error("Special token {name:?} claims id {id}, which the vocabulary spells {found:?}")]
    SpecialTokenConflict {
        id: u32,
        name: String,
        found: String,
    },
    #[error("SentencePiece vocabulary line for id {id} has no space separating piece from score")]
    SpmMissingScore { id: u32 },
    #[error("SentencePiece piece for id {id} is not valid base64: {source}")]
    SpmBase64 {
        id: u32,
        source: base64::DecodeError,
    },
    #[error("SentencePiece score for id {id} is not a number: {value:?}")]
    SpmScore { id: u32, value: String },
    #[error("SentencePiece piece for id {id} is not valid UTF-8")]
    SpmNonUtf8 { id: u32 },
}

/// Load a bundled SentencePiece vocabulary (`.spm`) as pieces and scores.
///
/// # Format
///
/// One line per token id, **in ascending id order with no gaps**:
///
/// ```text
/// <base64 of the piece, UTF-8 encoded> <score>
/// ```
///
/// The id is the line's position, so it cannot be non-monotonic or duplicated
/// by construction — there is no id field to disagree with the ordering. The
/// piece is SentencePiece's own `id_to_piece`, so byte fallback keeps its real
/// `<0x41>` spelling instead of being reconstructed from a run of raw bytes,
/// and the `▁` word-boundary runs keep theirs.
///
/// # Why not `.tiktoken`
///
/// A `.tiktoken` line is `base64(token_bytes) rank`, which throws away the
/// score. SentencePiece merges by score, not by id order, and the 15 whitespace
/// pieces (`▁`, `▁▁`, …) carry a `-1e9` "never merge" sentinel that id order
/// inverts: with id-order merge ranks, `" Hello world"` comes out as
/// `▁▁` + `Hello` + `▁world` instead of `▁` + `▁Hello` + `▁world`.
///
/// Scores are written as the shortest decimal that round-trips the value, and
/// the vocabularies bundled here hold whole numbers plus the `-1e9` sentinel —
/// all exactly representable in `f32`, so the parse is lossless.
///
/// # Errors
///
/// Returns [`VocabError`] when the data is empty, a line has no separator, a
/// piece is not valid base64 or not valid UTF-8, or a score does not parse.
pub fn load_spm_vocab(data: &[u8]) -> Result<(Vec<String>, Vec<f32>), VocabError> {
    let mut pieces = Vec::new();
    let mut scores = Vec::new();

    for line in data.split(|&b| b == b'\n') {
        // Tolerate a trailing newline and CRLF line endings; a blank line
        // carries no id, so it is skipped rather than filling a slot.
        let line = match line.strip_suffix(b"\r") {
            Some(stripped) => stripped,
            None => line,
        };
        if line.is_empty() {
            continue;
        }
        let id = pieces.len() as u32;

        let space = line
            .iter()
            .rposition(|&b| b == b' ')
            .ok_or(VocabError::SpmMissingScore { id })?;
        let (Some(piece_b64), Some(score_bytes)) = (line.get(..space), line.get(space + 1..))
        else {
            return Err(VocabError::SpmMissingScore { id });
        };

        let bytes = STANDARD
            .decode(piece_b64)
            .map_err(|source| VocabError::SpmBase64 { id, source })?;
        let piece = String::from_utf8(bytes).map_err(|_| VocabError::SpmNonUtf8 { id })?;

        let score_str = std::str::from_utf8(score_bytes)
            .map_err(|_| VocabError::SpmScore {
                id,
                value: String::from_utf8_lossy(score_bytes).into_owned(),
            })?
            .trim();
        let score: f32 = score_str.parse().map_err(|_| VocabError::SpmScore {
            id,
            value: score_str.to_string(),
        })?;

        pieces.push(piece);
        scores.push(score);
    }

    if pieces.is_empty() {
        return Err(VocabError::EmptyVocab);
    }
    Ok((pieces, scores))
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

/// Magic identifying a packed vocabulary. Bumping the trailing digit is how a
/// format change announces itself, so an old crate refuses a new file instead
/// of misreading it.
const PACKED_MAGIC: &[u8; 8] = b"SPLNTRV1";

/// Load a BPE vocabulary from the packed form `scripts/pack_vocabs.py` writes.
///
/// # Format
///
/// ```text
/// magic    8 bytes   b"SPLNTRV1"
/// count    u32 LE    number of entries
/// entries  count x   varint(rank), varint(len), len raw token bytes
/// ```
///
/// # Why this exists alongside [`load_tiktoken_bpe`]
///
/// The text form spends four base64 characters per three token bytes and writes
/// each rank in decimal, so it is ~47% larger than the ranks it carries. That
/// pushed the published crate past the 10 MiB crates.io limit. This form is what
/// the bundled vocabularies embed; `.tiktoken` remains the interchange format
/// that [`load_tiktoken_bpe`] and `Tokenizer::from_file` read, and
/// `tests/vocab_packed_parity.rs` fails if the two ever disagree.
///
/// Decoding is also strictly less work than the text path — no base64, no
/// decimal parse — which `from_pretrained` pays on every process start.
///
/// This form **copies** each token, for callers whose data is not `'static`.
/// The bundled vocabularies use [`load_packed_bpe_borrowed`] instead and copy
/// nothing.
///
/// Ranks are absolute rather than implied by position. Every bundled vocabulary
/// happens to be contiguous, but the tiktoken format guarantees no such thing,
/// and a positional format would silently renumber a vocabulary with a gap
/// instead of refusing it.
pub fn load_packed_bpe(data: &[u8]) -> Result<Encoder, VocabError> {
    let count = packed_header(data)?;
    // Sized up front: this map is 100k-200k entries and growing it by doubling
    // rehashes the whole table several times on the load path.
    let mut encoder = Encoder::with_capacity_and_hasher(count, rustc_hash::FxBuildHasher);
    walk_packed(data, count, |token, rank| {
        encoder.insert(TokenBytes::from(token.to_vec()), rank);
    })?;
    Ok(encoder)
}

/// Load a packed vocabulary **without copying any token bytes**.
///
/// The zero-copy counterpart to [`load_packed_bpe`], and the reason the packed
/// format exists in the shape it does: every token sits contiguously inside
/// `data`, so a key can point at it instead of owning a copy. `data` must be
/// `'static` — in practice the `include_bytes!` payload in `pretrained.rs`.
///
/// Measured against the copying form on `cl100k_base`, ~3.7x faster. The saving
/// is 100k-200k small allocations and their `memcpy`s, not parsing: both walk
/// the same bytes in the same order.
///
/// The text form has no equivalent, and cannot: base64 tokens do not exist as
/// contiguous bytes anywhere until something decodes them.
pub fn load_packed_bpe_borrowed(data: &'static [u8]) -> Result<Encoder, VocabError> {
    let count = packed_header(data)?;
    let mut encoder = Encoder::with_capacity_and_hasher(count, rustc_hash::FxBuildHasher);
    walk_packed(data, count, |token, rank| {
        encoder.insert(TokenBytes::Static(token), rank);
    })?;
    Ok(encoder)
}

/// Validate a packed header and return its entry count.
fn packed_header(data: &[u8]) -> Result<usize, VocabError> {
    if data.len() < 12 || &data[..8] != PACKED_MAGIC {
        return Err(VocabError::ParseError(
            "not a packed vocabulary: bad magic".to_string(),
        ));
    }
    let count = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
    if count == 0 {
        return Err(VocabError::EmptyVocab);
    }
    Ok(count)
}

/// Walk a packed vocabulary's entries, handing each `(token, rank)` to `visit`.
///
/// Shared by the owning and borrowing loaders so the two cannot drift into
/// disagreeing about the format — the only difference between them is what
/// `visit` does with the slice.
fn walk_packed<'a>(
    data: &'a [u8],
    count: usize,
    mut visit: impl FnMut(&'a [u8], u32),
) -> Result<(), VocabError> {
    let mut pos = 12;
    for _ in 0..count {
        let rank = read_varint(data, &mut pos)?;
        let len = read_varint(data, &mut pos)? as usize;
        let end = pos.checked_add(len).ok_or_else(|| {
            VocabError::ParseError("packed vocabulary: token length overflows".to_string())
        })?;
        if end > data.len() {
            return Err(VocabError::ParseError(
                "packed vocabulary: token runs past end of data".to_string(),
            ));
        }
        visit(&data[pos..end], rank);
        pos = end;
    }
    Ok(())
}

/// Read one unsigned LEB128 varint, advancing `pos`.
///
/// Capped at five groups: a `u32` needs at most 32 bits, and without the cap a
/// corrupt file of continuation bytes would shift past the width and loop to the
/// end of the buffer rather than failing.
fn read_varint(data: &[u8], pos: &mut usize) -> Result<u32, VocabError> {
    let mut value: u32 = 0;
    for group in 0..5 {
        let byte = *data.get(*pos).ok_or_else(|| {
            VocabError::ParseError("packed vocabulary: truncated varint".to_string())
        })?;
        *pos += 1;
        value |= u32::from(byte & 0x7F)
            .checked_shl(group * 7)
            .ok_or_else(|| {
                VocabError::ParseError("packed vocabulary: varint too wide".to_string())
            })?;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
    }
    Err(VocabError::ParseError(
        "packed vocabulary: varint too wide".to_string(),
    ))
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
///
/// The bytes are copied once into the table's own buffer, which is two
/// allocations for a whole vocabulary where inverting into a map cost one per
/// token.
pub fn build_decoder(encoder: &Encoder) -> Decoder {
    Decoder::from_encoder(encoder)
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

    /// Build a `.spm` blob from `(piece, score)` pairs listed in id order.
    fn spm_blob(entries: &[(&str, &str)]) -> Vec<u8> {
        let mut out = Vec::new();
        for (piece, score) in entries {
            out.extend_from_slice(STANDARD.encode(piece.as_bytes()).as_bytes());
            out.extend_from_slice(format!(" {score}\n").as_bytes());
        }
        out
    }

    /// The point of the format: pieces arrive spelled exactly as SentencePiece
    /// spells them — byte fallback as `<0xNN>`, word boundaries as `▁` — and the
    /// scores arrive alongside them instead of being inferred from id order.
    #[test]
    fn spm_vocab_keeps_piece_spelling_and_scores() {
        let data = spm_blob(&[
            ("<unk>", "0.0"),
            ("<0x41>", "0.0"),
            ("▁the", "-31.0"),
            ("▁▁", "-1000000000.0"),
        ]);
        let (pieces, scores) = load_spm_vocab(&data).unwrap();

        assert_eq!(pieces, vec!["<unk>", "<0x41>", "▁the", "▁▁"]);
        assert_eq!(scores, vec![0.0, 0.0, -31.0, -1e9]);
    }

    /// The `-1e9` "never merge" sentinel is the whole reason scores are stored:
    /// it must survive the text round-trip bit-for-bit, not land near `-1e9`.
    #[test]
    fn spm_vocab_parses_the_never_merge_sentinel_exactly() {
        let data = spm_blob(&[("▁", "-1000000000.0")]);
        let (_, scores) = load_spm_vocab(&data).unwrap();
        assert_eq!(scores.first().copied(), Some(-1e9f32));
        assert_eq!(
            scores.first().map(|s| s.to_bits()),
            Some((-1e9f32).to_bits())
        );
    }

    /// A trailing newline and CRLF endings carry no id, so they must not shift
    /// every later piece by one slot.
    #[test]
    fn spm_vocab_ignores_blank_and_carriage_return_line_endings() {
        let mut data = spm_blob(&[("a", "-1.0"), ("b", "-2.0")]);
        data.extend_from_slice(b"\n");
        let (pieces, _) = load_spm_vocab(&data).unwrap();
        assert_eq!(pieces, vec!["a", "b"]);

        let crlf = b"YQ== -1.0\r\nYg== -2.0\r\n";
        let (pieces, scores) = load_spm_vocab(crlf).unwrap();
        assert_eq!(pieces, vec!["a", "b"]);
        assert_eq!(scores, vec![-1.0, -2.0]);
    }

    /// A line without a separator has no score, and inventing one would put a
    /// piece at an id whose merge priority is a guess.
    #[test]
    fn spm_vocab_rejects_a_line_without_a_score() {
        assert!(matches!(
            load_spm_vocab(b"YQ== -1.0\nYg==\n"),
            Err(VocabError::SpmMissingScore { id: 1 })
        ));
    }

    /// Bad base64, a non-UTF-8 piece, and an unparseable score are each
    /// reported against the id that carries them.
    #[test]
    fn spm_vocab_reports_malformed_fields_with_their_id() {
        assert!(matches!(
            load_spm_vocab(b"YQ== -1.0\n!!!! -2.0\n"),
            Err(VocabError::SpmBase64 { id: 1, .. })
        ));
        // `loE=` decodes to `96 81` — `▁` (`E2 96 81`) with its lead byte lost,
        // which is not valid UTF-8 and so cannot be a piece.
        assert!(matches!(
            load_spm_vocab(b"YQ== -1.0\nloE= -2.0\n"),
            Err(VocabError::SpmNonUtf8 { id: 1 })
        ));
        assert!(matches!(
            load_spm_vocab(b"YQ== -1.0\nYg== rank\n"),
            Err(VocabError::SpmScore { id: 1, .. })
        ));
    }

    /// A `.tiktoken` file fed to the SentencePiece loader must not be accepted
    /// as if it were one: its raw high bytes are not valid UTF-8 pieces.
    #[test]
    fn spm_vocab_rejects_a_tiktoken_file() {
        let mut data = Vec::new();
        data.extend_from_slice(STANDARD.encode([0x80u8]).as_bytes());
        data.extend_from_slice(b" 0\n");
        assert!(matches!(
            load_spm_vocab(&data),
            Err(VocabError::SpmNonUtf8 { id: 0 })
        ));
    }

    #[test]
    fn spm_vocab_rejects_empty_data() {
        assert!(matches!(load_spm_vocab(b""), Err(VocabError::EmptyVocab)));
    }

    #[test]
    fn test_build_decoder() {
        let mut encoder = Encoder::default();
        encoder.insert(TokenBytes::from(b"Hello".to_vec()), 0);
        encoder.insert(TokenBytes::from(b"World".to_vec()), 1);

        let decoder = build_decoder(&encoder);
        assert_eq!(decoder.get(0), Some(&b"Hello"[..]));
        assert_eq!(decoder.get(1), Some(&b"World"[..]));
    }
}
