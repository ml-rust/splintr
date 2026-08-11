//! Derive Llama 2's vocabulary from Code Llama's.
//!
//! Code Llama was built by extending Llama 2's SentencePiece model with 16
//! infill pieces, and it extended it *in place*: ids 0..31,999 keep Llama 2's
//! pieces and Llama 2's scores exactly, verified piece-for-piece and
//! score-for-score against Meta's `tokenizer.model` (md5
//! `eeec4125e9c7560836b4873b6f8e3025`). So the two vocabularies are one file
//! and a length, and committing both would be committing the same 32,000 lines
//! twice — with nothing keeping the copies in step.

use std::io::Write;

/// Llama 2's piece count. Code Llama's ids from here up are its own.
const LLAMA2_PIECES: usize = 32_000;

fn main() {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    let out = std::env::var("OUT_DIR").expect("OUT_DIR");
    let src = format!("{manifest}/vocabs/codellama.spm");
    println!("cargo::rerun-if-changed={src}");

    let text = std::fs::read(&src).unwrap_or_else(|e| panic!("{src}: {e}"));

    // One line per id, so the prefix is taken by counting line ends rather than
    // by splitting: a `.spm` line is `base64(piece) score`, and neither field
    // can contain a newline.
    let mut end = 0;
    let mut lines = 0;
    for (i, &byte) in text.iter().enumerate() {
        if byte == b'\n' {
            lines += 1;
            if lines == LLAMA2_PIECES {
                end = i + 1;
                break;
            }
        }
    }
    assert_eq!(
        lines, LLAMA2_PIECES,
        "{src} has {lines} pieces, fewer than Llama 2's {LLAMA2_PIECES}"
    );

    let dst = format!("{out}/llama2.spm");
    let mut file = std::fs::File::create(&dst).unwrap_or_else(|e| panic!("{dst}: {e}"));
    file.write_all(&text[..end])
        .unwrap_or_else(|e| panic!("{dst}: {e}"));
}
