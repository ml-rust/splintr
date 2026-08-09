//! The packed vocabularies must say exactly what the `.tiktoken` files say.
//!
//! `vocabs/*.splv` is what the published crate embeds; `vocabs/*.tiktoken` is
//! the text form `Tokenizer::from_file` reads, that `docs/vocabularies.md`
//! documents, and that the perf workflow hands to tiktoken-rs and gigatoken so
//! every engine is measured on identical ranks. Two files carrying the same
//! ranks is a duplicated source of truth, and this is what makes the
//! duplication safe: regenerate one without the other and these tests fail.
//!
//! Run `python scripts/pack_vocabs.py` to bring them back into agreement.
//!
//! The text files are not shipped in the crate, so this is a repository test —
//! it reads `vocabs/` relative to `CARGO_MANIFEST_DIR`.

use std::collections::HashMap;
use std::path::PathBuf;

use splintr::core::{load_packed_bpe, load_tiktoken_bpe};

/// Every bundled rank file, by stem. Kept explicit rather than globbed: a
/// vocabulary added to `vocabs/` but not to `pretrained.rs` should show up as a
/// missing entry here, not be silently skipped.
const VOCABS: &[&str] = &[
    "cl100k_base",
    "o200k_base",
    "llama3",
    "deepseek_v3",
    "qwen3",
    "glm4",
    "kimi",
    "mistral_v3_tekken",
    "whisper",
];

fn vocab_path(stem: &str, ext: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("vocabs")
        .join(format!("{stem}.{ext}"))
}

fn read(stem: &str, ext: &str) -> Vec<u8> {
    let path = vocab_path(stem, ext);
    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

#[test]
fn packed_matches_text_for_every_vocabulary() {
    for stem in VOCABS {
        let text = load_tiktoken_bpe(&read(stem, "tiktoken"))
            .unwrap_or_else(|e| panic!("{stem}.tiktoken: {e}"));
        let packed =
            load_packed_bpe(&read(stem, "splv")).unwrap_or_else(|e| panic!("{stem}.splv: {e}"));

        assert_eq!(
            text.len(),
            packed.len(),
            "{stem}: {} tokens in text, {} in packed — regenerate with \
             `python scripts/pack_vocabs.py`",
            text.len(),
            packed.len()
        );

        // Compared entry by entry rather than by whole-map equality so a
        // mismatch names the token, which is the difference between "these
        // files disagree" and a diagnosis.
        for (token, rank) in &text {
            match packed.get(token) {
                Some(packed_rank) => assert_eq!(
                    packed_rank, rank,
                    "{stem}: token {token:?} is rank {rank} in text, {packed_rank} in packed"
                ),
                None => panic!("{stem}: token {token:?} (rank {rank}) missing from packed"),
            }
        }
    }
}

/// The empty token is the case a whitespace-splitting parser loses: whisper's
/// rank 50256 is spelled as a line with nothing before the space. It survived
/// the text parser, so it must survive packing too.
#[test]
fn the_empty_token_survives_packing() {
    let packed = load_packed_bpe(&read("whisper", "splv")).expect("whisper.splv");
    assert_eq!(
        packed.get(&Vec::<u8>::new()),
        Some(&50256),
        "whisper's empty token lost its rank in the packed form"
    );
}

/// Ranks are stored absolutely rather than implied by position in the file.
///
/// Every bundled vocabulary today *is* contiguous — rank equals line number in
/// all nine — so a positional format would encode them correctly and save two
/// or three bytes per token. The format declines that on purpose: nothing in
/// the tiktoken format guarantees contiguity, and a vocabulary with a hole in
/// its rank space would be silently renumbered rather than rejected, which is
/// wrong ids with no error.
///
/// The bundled files cannot demonstrate that property, since none has a gap.
/// This builds a packed buffer with one directly.
#[test]
fn a_gap_in_the_rank_space_is_preserved() {
    let mut buf = b"SPLNTRV1".to_vec();
    buf.extend_from_slice(&2u32.to_le_bytes());
    // rank 0, one byte "a"
    buf.extend_from_slice(&[0x00, 0x01, b'a']);
    // rank 300 (two-byte varint: 0xAC 0x02), one byte "b" — 1..=299 unused.
    buf.extend_from_slice(&[0xAC, 0x02, 0x01, b'b']);

    let packed = load_packed_bpe(&buf).expect("synthetic packed vocabulary");
    let actual: HashMap<Vec<u8>, u32> = packed.into_iter().collect();
    let expected: HashMap<Vec<u8>, u32> = [(b"a".to_vec(), 0), (b"b".to_vec(), 300)].into();
    assert_eq!(
        actual, expected,
        "the gap between rank 0 and rank 300 was lost"
    );
}

/// What the bundled files actually look like, recorded so the assumption above
/// is checked rather than remembered: if a future vocabulary arrives with a
/// gap, this fails and the note in the previous test stops being hypothetical.
#[test]
fn every_bundled_vocabulary_is_contiguous_today() {
    for stem in VOCABS {
        let packed = load_packed_bpe(&read(stem, "splv")).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let max = packed.values().copied().max().expect("non-empty");
        assert_eq!(
            packed.len() as u32,
            max + 1,
            "{stem}: {} tokens but max rank {max} — this vocabulary has a gap, \
             which the packed format handles correctly; update this test",
            packed.len()
        );
    }
}

/// A corrupt or foreign file must be refused, not misread. The magic is the
/// only thing standing between "wrong format" and a map built from noise.
#[test]
fn a_non_packed_file_is_rejected() {
    let text = read("cl100k_base", "tiktoken");
    assert!(
        load_packed_bpe(&text).is_err(),
        "the text form was accepted as packed"
    );
    assert!(load_packed_bpe(b"").is_err(), "empty input was accepted");
    assert!(
        load_packed_bpe(b"SPLNTRV1\x01\x00\x00\x00").is_err(),
        "a header claiming one entry but carrying none was accepted"
    );
}
