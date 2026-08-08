//! Integration tests for the GLM-4/4.5 tokenizer.
//!
//! The ids pinned here were read from `zai-org/GLM-4.5`'s own `tokenizer.json`
//! through `tokenizers` 0.22.1. Broad agreement with that reference lives in
//! `tests/fixtures/pretrained/glm4.json`; this file pins what the fixture
//! cannot — the pre-tokenizer GLM gets, and how its native role markers
//! interact with splintr's agent tokens of the same name.

use splintr::pretrained::{from_pretrained, glm4_special_tokens};
use splintr::{AnyTokenizer, Tokenize};
use std::sync::LazyLock;

static TOKENIZER: LazyLock<AnyTokenizer> =
    LazyLock::new(|| from_pretrained("glm4").expect("glm4 is bundled"));

#[test]
fn glm4_encodes_the_reference_ids() {
    for (text, expected) in [
        ("Hello world", vec![9703u32, 1879]),
        ("Hello, world!", vec![9703, 11, 1879, 0]),
        ("你好世界", vec![109377, 99011]),
    ] {
        assert_eq!(TOKENIZER.encode_raw(text), expected, "ids for {text:?}");
    }
}

/// GLM takes digit runs of up to three (`\p{N}{1,3}`), which is Llama 3's split
/// and *not* Qwen's single-digit one — the two vocabularies are the same size
/// and the same era, so this is exactly where a copied pattern would go wrong.
#[test]
fn glm4_splits_digits_in_runs_of_up_to_three() {
    assert_eq!(
        TOKENIZER.encode_raw("1234567890"),
        vec![108714u32, 100461, 21, 100928, 24, 15]
    );
}

#[test]
fn glm4_round_trips() {
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

/// GLM names `<|system|>`, `<|user|>`, `<|assistant|>`, `<|image|>` and
/// `<|video|>` itself — five names splintr also uses for agent tokens. The
/// model's ids win, and the agent slots they would have taken stay unnamed.
#[test]
fn glm4_native_special_tokens_outrank_the_agent_tokens_of_the_same_name() {
    let special = glm4_special_tokens();
    assert_eq!(special.get("<|system|>"), Some(&151335));
    assert_eq!(special.get("<|user|>"), Some(&151336));
    assert_eq!(special.get("<|assistant|>"), Some(&151337));
    assert_eq!(special.get("<|image|>"), Some(&151363));
    assert_eq!(special.get("<|video|>"), Some(&151364));
    assert_eq!(special.get("[gMASK]"), Some(&151331));

    // Agent tokens with no native counterpart keep their usual offsets.
    assert_eq!(special.get("<|im_start|>"), Some(&(151365 + 3)));
    assert_eq!(special.get("<|pad|>"), Some(&(151365 + 39)));
    // Offsets 0, 1, 2, 42 and 46 were vacated, not repacked.
    for vacated in [151365, 151366, 151367, 151365 + 42, 151365 + 46] {
        assert!(
            !special.values().any(|&id| id == vacated),
            "id {vacated} should be reserved and unnamed"
        );
    }
}

#[test]
fn glm4_recognizes_special_tokens_in_text() {
    let ids = TOKENIZER.encode("[gMASK]<sop><|user|>hi");
    assert_eq!(&ids[..3], &[151331, 151333, 151336]);
}

#[test]
fn glm4_aliases_resolve_to_one_vocabulary() {
    let ids = TOKENIZER.encode_raw("Hello world");
    for name in ["glm", "glm4", "glm-4", "glm4.5", "glm-4.5"] {
        let alias = from_pretrained(name).expect("alias is bundled");
        assert_eq!(alias.encode_raw("Hello world"), ids, "alias {name}");
    }
}

#[test]
fn glm4_reports_its_base_vocabulary_size() {
    assert_eq!(
        splintr::pretrained::base_vocab_size_by_name("glm4").unwrap(),
        151365
    );
    assert!(TOKENIZER.vocab_size() > 151365, "agent tokens sit above it");
}
