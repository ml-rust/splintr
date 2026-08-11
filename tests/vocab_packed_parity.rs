//! The packed vocabularies must say exactly what the `.tiktoken` files say.
//!
//! Each `splintr-vocab-*` crate ships its `.tiktoken` text and packs it into the
//! binary form at build time, so the two cannot drift the way two committed
//! files could. What can still go wrong is the build script packing a different
//! file from the one that ships — a stale `OUT_DIR`, a renamed stem, a crate
//! wired to its neighbour's payload — and none of that is a build error. It
//! shows up here.
//!
//! The text files are shipped but not embedded, so this is a repository test:
//! it reads them relative to `CARGO_MANIFEST_DIR` and compares against the
//! constants the data crates actually expose.

use std::collections::HashMap;
use std::path::PathBuf;

use splintr::core::{load_packed_bpe, load_tiktoken_bpe};

/// Every bundled rank file, as `(data crate, stem, the constant it exposes)`.
/// Kept explicit rather than globbed: a vocabulary added to a crate but not to
/// `pretrained.rs` should show up as a missing entry here, not be silently
/// skipped.
fn bundled() -> Vec<(&'static str, &'static str, &'static [u8])> {
    use splintr::pretrained::{
        CL100K_BASE_VOCAB_PACKED, DEEPSEEK_V3_VOCAB_PACKED, GLM4_VOCAB_PACKED, KIMI_VOCAB_PACKED,
        LLAMA3_VOCAB_PACKED, MISTRAL_V3_VOCAB_PACKED, MODERNBERT_VOCAB_PACKED,
        O200K_BASE_VOCAB_PACKED, QWEN3_VOCAB_PACKED, WHISPER_VOCAB_PACKED,
    };
    vec![
        ("cl100k", "cl100k_base", CL100K_BASE_VOCAB_PACKED),
        ("o200k", "o200k_base", O200K_BASE_VOCAB_PACKED),
        ("llama3", "llama3", LLAMA3_VOCAB_PACKED),
        ("deepseek", "deepseek_v3", DEEPSEEK_V3_VOCAB_PACKED),
        ("qwen", "qwen3", QWEN3_VOCAB_PACKED),
        ("glm", "glm4", GLM4_VOCAB_PACKED),
        ("kimi", "kimi", KIMI_VOCAB_PACKED),
        ("mistral", "mistral_v3_tekken", MISTRAL_V3_VOCAB_PACKED),
        ("modernbert", "modernbert", MODERNBERT_VOCAB_PACKED),
        ("whisper", "whisper", WHISPER_VOCAB_PACKED),
    ]
}

fn text_of(family: &str, stem: &str) -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("crates")
        .join(format!("vocab-{family}"))
        .join("vocabs")
        .join(format!("{stem}.tiktoken"));
    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

#[test]
fn packed_matches_text_for_every_vocabulary() {
    for (family, stem, packed_bytes) in bundled() {
        let text = load_tiktoken_bpe(&text_of(family, stem))
            .unwrap_or_else(|e| panic!("{stem}.tiktoken: {e}"));
        let packed = load_packed_bpe(packed_bytes).unwrap_or_else(|e| panic!("{stem} packed: {e}"));

        assert_eq!(
            text.len(),
            packed.len(),
            "{stem}: {} tokens in the shipped text, {} in what the build script packed",
            text.len(),
            packed.len()
        );

        // Compared entry by entry rather than by whole-map equality so a
        // mismatch names the token, which is the difference between "these
        // disagree" and a diagnosis.
        for (token, rank) in &text {
            match packed.get(token.as_slice()) {
                Some(packed_rank) => assert_eq!(
                    packed_rank, *rank,
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
    let packed =
        load_packed_bpe(splintr::pretrained::WHISPER_VOCAB_PACKED).expect("whisper packed");
    assert_eq!(
        packed.get(b"".as_slice()),
        Some(50256),
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
    let actual: HashMap<Vec<u8>, u32> = packed
        .iter()
        .map(|(token, rank)| (token.to_vec(), rank))
        .collect();
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
    for (_, stem, packed_bytes) in bundled() {
        let packed = load_packed_bpe(packed_bytes).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let max = packed.values().max().expect("non-empty");
        assert_eq!(
            packed.len() as u32,
            max + 1,
            "{stem}: {} tokens but max rank {max} — this vocabulary has a gap, \
             which the packed format handles correctly; update this test",
            packed.len()
        );
    }
}

/// Llama 2's vocabulary is Code Llama's first 32,000 pieces, derived at build
/// time rather than committed a second time.
///
/// Nothing about the two files makes that true by construction — the build
/// script counts lines, and a Code Llama file that ever reordered or reweighted
/// a piece below 32,000 would produce a silently wrong Llama 2. This is where
/// that assumption is checked rather than trusted.
#[test]
fn llama2_is_code_llamas_first_32000_pieces() {
    use splintr::pretrained::{CODELLAMA_SPM_VOCAB, LLAMA2_SPM_VOCAB};

    let llama2 = std::str::from_utf8(LLAMA2_SPM_VOCAB).expect("ascii");
    let codellama = std::str::from_utf8(CODELLAMA_SPM_VOCAB).expect("ascii");

    let pieces = llama2.lines().count();
    assert_eq!(pieces, 32_000, "llama2 has {pieces} pieces, not 32,000");
    assert_eq!(
        codellama.lines().count(),
        32_016,
        "codellama is no longer Llama 2's 32,000 plus 16 infill pieces"
    );

    for (id, (a, b)) in llama2.lines().zip(codellama.lines()).enumerate() {
        assert_eq!(
            a, b,
            "id {id}: llama2 says {a:?}, codellama says {b:?} — the two have \
             diverged and llama2 can no longer be derived from codellama"
        );
    }
}

/// Phi-4 and OLMo-2 ship no payload: both read cl100k_base's rank file and
/// place their own special blocks directly above it, at 100,256.
///
/// That id is a property of the shipped file, not a constant either family
/// declares, so a cl100k_base file that gained or lost a rank would move both
/// of their special blocks on top of live ranks. It is asserted here for the
/// same reason the derivation above is.
#[test]
fn cl100k_holds_exactly_the_100256_ranks_phi4_and_olmo2_build_on() {
    let packed =
        load_packed_bpe(splintr::pretrained::CL100K_BASE_VOCAB_PACKED).expect("cl100k packed");
    assert_eq!(packed.len(), 100_256);
}

/// A corrupt or foreign file must be refused, not misread. The magic is the
/// only thing standing between "wrong format" and a map built from noise.
#[test]
fn a_non_packed_file_is_rejected() {
    let text = text_of("cl100k", "cl100k_base");
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
