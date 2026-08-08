//! Integration tests for the Mistral V1 (SentencePiece) tokenizer.
//!
//! # Ground truth
//!
//! Every expected id list below was produced by the reference implementation —
//! the `sentencepiece` Python package, version 0.2.0 — reading Mistral 7B's own
//! `tokenizer.model` (32,000 pieces, 15 of which carry the `-1e9` "never merge"
//! score sentinel). They are the reference, not a snapshot of splintr's output:
//! a change here means splintr diverged from SentencePiece, never that the
//! numbers need updating.
//!
//! These pin the thing a tiktoken-format vocabulary could not express. Storing
//! the vocabulary as `base64(bytes) rank` threw the scores away, and
//! `SpmTokenizer` fell back to merging in token-id order. That is not an
//! approximation of SentencePiece's order — it inverts it for the 15 whitespace
//! runs (`▁`, `▁▁`, …), which sit at low ids precisely *because* SentencePiece
//! refuses to merge them. `" Hello world"` is the visible symptom.
// Gated on `vocab-mistral` because every test here loads that vocabulary, and the feature is what
// compiles those bytes in. Without it this crate is empty rather than
// a compile error.
#![cfg(feature = "vocab-mistral")]

use splintr::{from_pretrained, AnyTokenizer, Tokenize};
use std::sync::LazyLock;

/// Shared tokenizer instance to avoid expensive re-initialization per test.
static TOKENIZER: LazyLock<AnyTokenizer> =
    LazyLock::new(|| from_pretrained("mistral").expect("mistral loads"));

/// Every reference case in one table, asserted through `encode_raw` — the bare
/// backend output, with no boundary tokens layered on, which is what
/// `sp.encode` returns.
#[test]
fn mistral_v1_matches_sentencepiece_exactly() {
    let cases: &[(&str, &[u32])] = &[
        ("the sourdough", &[272, 18193, 28715, 900]),
        ("sourdough", &[18193, 28715, 900]),
        ("hello world", &[6312, 28709, 1526]),
        ("Hello world", &[22557, 1526]),
        // The case that a score-less, id-order merge got wrong: it produced
        // `▁▁` + `Hello` + `▁world` = [259, 16230, 1526].
        (" Hello world", &[28705, 22557, 1526]),
        ("tokenizer", &[6029, 4024]),
        ("perplexity", &[660, 8899, 472]),
        ("Hello, world!", &[22557, 28725, 1526, 28808]),
        ("你好世界", &[28705, 29383, 29530, 30050, 29822]),
    ];

    for (text, expected) in cases {
        assert_eq!(
            TOKENIZER.encode_raw(text),
            *expected,
            "sentencepiece reference mismatch for {text:?}"
        );
    }
}

/// A leading space is a `▁` of its own (id 28705), and the following word keeps
/// its own boundary — the two must not collapse into the `▁▁` piece.
#[test]
fn mistral_v1_keeps_a_leading_space_separate_from_the_next_word() {
    let ids = TOKENIZER.encode_raw(" Hello world");
    assert_eq!(ids.first().copied(), Some(28705), "leading ▁ in {ids:?}");
    assert_eq!(ids, vec![28705, 22557, 1526]);
}

/// Whatever the ids, the text must come back unchanged.
#[test]
fn mistral_v1_round_trips_every_reference_case() {
    for text in [
        "the sourdough",
        "hello world",
        " Hello world",
        "Hello, world!",
        "你好世界",
        "perplexity",
    ] {
        let ids = TOKENIZER.encode_raw(text);
        let decoded = TOKENIZER.decode(&ids).expect("decodes");
        assert_eq!(decoded, text, "round trip failed for {text:?}");
    }
}

/// The vocabulary file holds 32,000 pieces; splintr's 54 agent tokens sit above
/// it, so ids run 0..=32053.
#[test]
fn mistral_v1_vocab_size_covers_the_agent_tokens() {
    assert_eq!(TOKENIZER.vocab_size(), 32054);
}
