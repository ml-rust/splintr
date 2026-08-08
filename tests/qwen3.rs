//! Integration tests for the Qwen 2/3 tokenizer.
//!
//! The ids pinned here were read from `Qwen/Qwen3-8B`'s own `tokenizer.json`
//! through `tokenizers` 0.22.1, not from splintr's output. Broad agreement with
//! that reference is covered by `tests/fixtures/pretrained/qwen3.json` and
//! `reference_parity`; what this file pins is the handful of facts that fixture
//! cannot state — which pre-tokenizer Qwen gets, and what happens where its own
//! special tokens collide with splintr's agent tokens.
// Gated on `vocab-qwen` because every test here loads that vocabulary, and the feature is what
// compiles those bytes in. Without it this crate is empty rather than
// a compile error.
#![cfg(feature = "vocab-qwen")]

use splintr::pretrained::{from_pretrained, qwen3_special_tokens};
use splintr::{AnyTokenizer, Tokenize};
use std::sync::LazyLock;

static TOKENIZER: LazyLock<AnyTokenizer> =
    LazyLock::new(|| from_pretrained("qwen3").expect("qwen3 is bundled"));

#[test]
fn qwen3_encodes_the_reference_ids() {
    for (text, expected) in [
        ("Hello world", vec![9707u32, 1879]),
        ("Hello, world!", vec![9707, 11, 1879, 0]),
        ("你好世界", vec![108386, 99489]),
    ] {
        assert_eq!(TOKENIZER.encode_raw(text), expected, "ids for {text:?}");
    }
}

/// Qwen splits numbers one digit at a time (`\p{N}`), unlike Llama 3 and GLM,
/// which take runs of up to three. This is the whole difference between the two
/// patterns, so it is what distinguishes a correct pre-tokenizer from a
/// plausible one.
#[test]
fn qwen3_splits_digits_individually() {
    assert_eq!(
        TOKENIZER.encode_raw("1234567890"),
        vec![16u32, 17, 18, 19, 20, 21, 22, 23, 24, 15]
    );
}

#[test]
fn qwen3_round_trips() {
    for text in [
        "Hello world",
        "  leading and trailing  ",
        "混合 mixed スクリプト 123",
        "fn main() { println!(\"hi\"); }",
        "",
    ] {
        let ids = TOKENIZER.encode_raw(text);
        assert_eq!(TOKENIZER.decode(&ids).expect("decodes"), text);
    }
}

/// `<|im_start|>` and `<|im_end|>` are names splintr also uses for agent
/// tokens. Qwen's own ids must win: a chat template encoded against this
/// vocabulary has to produce the ids the checkpoint was trained on, and an
/// agent-token id in their place would be silently wrong rather than an error.
#[test]
fn qwen3_native_special_tokens_outrank_the_agent_tokens_of_the_same_name() {
    let special = qwen3_special_tokens();
    assert_eq!(special.get("<|im_start|>"), Some(&151644));
    assert_eq!(special.get("<|im_end|>"), Some(&151645));
    assert_eq!(special.get("<|endoftext|>"), Some(&151643));
    assert_eq!(special.get("</think>"), Some(&151668));

    // The agent slots those two names would otherwise have taken (base + 3 and
    // base + 4) are left unnamed rather than repacked, so every other agent
    // token keeps the offset it has in every other vocabulary.
    assert_eq!(special.get("<|system|>"), Some(&151669));
    assert_eq!(special.get("<|assistant|>"), Some(&151671));
    assert_eq!(special.get("<|pad|>"), Some(&(151669 + 39)));
    assert!(!special.values().any(|&id| id == 151672 || id == 151673));
}

#[test]
fn qwen3_recognizes_special_tokens_in_text() {
    let ids = TOKENIZER.encode("<|im_start|>user\nhi<|im_end|>");
    assert_eq!(ids.first(), Some(&151644));
    assert_eq!(ids.last(), Some(&151645));
}

/// Baichuan-M2 ships Qwen's tokenizer verbatim — all 151,643 ids identical —
/// so it is an alias rather than a second bundled copy.
#[test]
fn qwen3_aliases_resolve_to_one_vocabulary() {
    let ids = TOKENIZER.encode_raw("Hello world");
    for name in ["qwen", "qwen2", "qwen2.5", "qwen3", "baichuan_m2"] {
        let alias = from_pretrained(name).expect("alias is bundled");
        assert_eq!(alias.encode_raw("Hello world"), ids, "alias {name}");
    }
}

#[test]
fn qwen3_reports_its_base_vocabulary_size() {
    assert_eq!(
        splintr::pretrained::base_vocab_size_by_name("qwen3").unwrap(),
        151669
    );
    assert!(TOKENIZER.vocab_size() > 151669, "agent tokens sit above it");
}
