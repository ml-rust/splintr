//! Turn a `.tiktoken` rank file into the binary form splintr's loader reads.
//!
//! # Why this exists at all
//!
//! splintr loads a bundled vocabulary from a packed binary rather than from the
//! `base64(token) rank` text, because the text costs roughly twice as long to
//! load: every line needs base64-decoding and a decimal parse, and there are
//! 100k–200k of them.
//!
//! That binary used to be committed next to the text, which meant two files
//! carrying the same ranks, a script to regenerate one from the other, and a
//! test to catch anyone who regenerated only one of them. It also meant every
//! published vocabulary crate was an opaque blob: nothing in a diff, nothing to
//! grep, and no way for a reader to check what they were about to compile in.
//!
//! Building it instead removes all of that. The crates ship the text — which is
//! reviewable, diffable, and the same format `Tokenizer::from_file` accepts —
//! and the packed form is derived at build time, so it cannot disagree with the
//! text it came from.
//!
//! # Format
//!
//! ```text
//! "SPLNTRV1"        magic, 8 bytes
//! count             u32, little-endian, number of entries
//! entry*            count entries, in file order
//!
//! entry = varint(rank) varint(len) token[len]
//! ```
//!
//! `varint` is LEB128: seven bits per byte, low byte first, high bit set on
//! every byte but the last.
//!
//! Ranks are stored, not implied by position. Every vocabulary bundled today is
//! contiguous, so position would encode them correctly and save a couple of
//! bytes per token — but nothing in the tiktoken format promises contiguity, and
//! a vocabulary with a hole in its rank space would be silently renumbered
//! rather than rejected. That is wrong ids with no error, which is the one
//! failure mode a tokenizer must not have.

use base64::engine::general_purpose::STANDARD;
use base64::Engine;

/// Magic at the head of every packed file.
pub const MAGIC: &[u8; 8] = b"SPLNTRV1";

/// Pack `.tiktoken` text into the binary form.
///
/// Blank lines are skipped. A line is `base64(token) rank`, split on the LAST
/// space: cl100k's rank 50256 is an empty token, which is a line that begins
/// with its separator, and splitting on the first space would read the rank as
/// the token.
pub fn pack(text: &[u8]) -> Result<Vec<u8>, String> {
    let mut entries = Vec::with_capacity(text.len() / 2);
    let mut count: u32 = 0;

    for (line_no, line) in text.split(|&b| b == b'\n').enumerate() {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        if line.is_empty() {
            continue;
        }
        let at = line
            .iter()
            .rposition(|&b| b == b' ')
            .ok_or_else(|| format!("line {}: no space between token and rank", line_no + 1))?;
        let token = STANDARD
            .decode(&line[..at])
            .map_err(|e| format!("line {}: {e}", line_no + 1))?;
        let rank: u32 = std::str::from_utf8(&line[at + 1..])
            .map_err(|e| format!("line {}: {e}", line_no + 1))?
            .trim()
            .parse()
            .map_err(|e| format!("line {}: {e}", line_no + 1))?;

        write_varint(&mut entries, rank as u64);
        write_varint(&mut entries, token.len() as u64);
        entries.extend_from_slice(&token);
        count += 1;
    }

    if count == 0 {
        return Err("no entries".to_string());
    }
    let mut out = Vec::with_capacity(MAGIC.len() + 4 + entries.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&count.to_le_bytes());
    out.extend_from_slice(&entries);
    Ok(out)
}

/// Pack `<manifest>/vocabs/<stem>.tiktoken` into `$OUT_DIR/<stem>.splv`, and
/// tell cargo to rerun when the text changes.
///
/// The whole of a vocabulary crate's build script.
pub fn pack_into_out_dir(stem: &str) {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    let out = std::env::var("OUT_DIR").expect("OUT_DIR");
    let src = format!("{manifest}/vocabs/{stem}.tiktoken");
    println!("cargo::rerun-if-changed={src}");

    let text = std::fs::read(&src).unwrap_or_else(|e| panic!("{src}: {e}"));
    let packed = pack(&text).unwrap_or_else(|e| panic!("{src}: {e}"));
    let dst = format!("{out}/{stem}.splv");
    std::fs::write(&dst, packed).unwrap_or_else(|e| panic!("{dst}: {e}"));
}

fn write_varint(out: &mut Vec<u8>, mut n: u64) {
    loop {
        let byte = (n & 0x7F) as u8;
        n >>= 7;
        out.push(byte | if n != 0 { 0x80 } else { 0 });
        if n == 0 {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rank_above_127_takes_two_varint_bytes() {
        // "a" is rank 300, so the varint is 0xAC 0x02 rather than one byte.
        let packed = pack(b"YQ== 300").expect("packs");
        assert_eq!(&packed[..8], MAGIC);
        assert_eq!(&packed[8..12], &1u32.to_le_bytes());
        assert_eq!(&packed[12..], &[0xAC, 0x02, 0x01, b'a']);
    }

    #[test]
    fn the_empty_token_survives() {
        // Whisper's rank 50256 is an empty token: a line that starts with the
        // separator. Splitting on the first space would read "50256" as the
        // token and find no rank.
        let packed = pack(b" 50256").expect("packs");
        assert_eq!(&packed[12..], &[0xD0, 0x88, 0x03, 0x00]);
    }

    #[test]
    fn blank_lines_are_skipped_not_counted() {
        let packed = pack(b"YQ== 0\n\nYg== 1\n").expect("packs");
        assert_eq!(&packed[8..12], &2u32.to_le_bytes());
    }

    #[test]
    fn a_line_without_a_rank_is_an_error() {
        assert!(pack(b"YQ==").is_err());
    }
}
