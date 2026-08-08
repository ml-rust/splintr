//! Integration tests for OpenAI's gpt-oss tokenizer.
//!
//! gpt-oss carries no vocabulary data of its own: its 199,998 ranks are
//! o200k_base's, id for id, and the two differ only in the special-token block
//! above them — o200k_base names two of 199999-200018 and gpt-oss fills the
//! range with the harmony response format's markers. These tests pin both
//! halves of that claim, because it is the claim that justifies shipping
//! gpt-oss without a second 3.4 MB file.

use splintr::pretrained::{from_pretrained, gpt_oss_special_tokens};
use splintr::AnyTokenizer;
use std::sync::LazyLock;

static TOKENIZER: LazyLock<AnyTokenizer> =
    LazyLock::new(|| from_pretrained("gpt-oss").expect("gpt-oss is bundled"));

#[test]
fn gpt_oss_encodes_the_reference_ids() {
    for (text, expected) in [
        ("Hello world", vec![13225u32, 2375]),
        ("Hello, world!", vec![13225, 11, 2375, 0]),
        ("你好世界", vec![177519, 28428]),
        ("1234567890", vec![7633, 19354, 29338, 15]),
    ] {
        assert_eq!(TOKENIZER.encode_raw(text), expected, "ids for {text:?}");
    }
}

/// The shared-ranks claim, tested rather than asserted in a comment: any text
/// with no special token in it must encode identically under both names.
#[test]
fn gpt_oss_and_o200k_base_agree_on_every_ordinary_id() {
    let o200k = from_pretrained("o200k_base").expect("o200k_base is bundled");
    for text in [
        "Hello world",
        "The quick brown fox jumps over the lazy dog.",
        "混合 mixed スクリプト 123",
        "fn main() { println!(\"hi\"); }",
        "    indented\n\tand tabbed\r\n",
        "🌍🌎🌏 emoji run",
    ] {
        assert_eq!(
            TOKENIZER.encode_raw(text),
            o200k.encode_raw(text),
            "ids diverge for {text:?}"
        );
    }
}

/// The other half: the special blocks are *not* the same. o200k_base leaves
/// 200002 unnamed; gpt-oss calls it `<|return|>` and ends its turns with it.
#[test]
fn gpt_oss_carries_the_harmony_special_tokens() {
    let special = gpt_oss_special_tokens();
    assert_eq!(special.get("<|start|>"), Some(&200006));
    assert_eq!(special.get("<|channel|>"), Some(&200005));
    assert_eq!(special.get("<|message|>"), Some(&200008));
    assert_eq!(special.get("<|end|>"), Some(&200007));
    assert_eq!(special.get("<|call|>"), Some(&200012));
    assert_eq!(special.get("<|return|>"), Some(&200002));
    assert_eq!(special.get("<|endofprompt|>"), Some(&200018));

    let o200k = from_pretrained("o200k_base").expect("o200k_base is bundled");
    assert!(
        o200k.special_token_id("<|return|>").is_none(),
        "o200k_base must not know the harmony markers"
    );
    assert_eq!(splintr::pretrained::eos_token_id_by_name("gpt-oss"), 200002);
}

#[test]
fn gpt_oss_recognizes_the_harmony_markers_in_text() {
    let ids = TOKENIZER.encode("<|start|>assistant<|channel|>final<|message|>hi<|return|>");
    assert_eq!(ids.first(), Some(&200006));
    assert_eq!(ids.last(), Some(&200002));
    assert!(ids.contains(&200005) && ids.contains(&200008));
}

#[test]
fn gpt_oss_round_trips() {
    for text in [
        "Hello world",
        "  leading and trailing  ",
        "混合 mixed スクリプト 123",
        "",
    ] {
        let ids = TOKENIZER.encode_raw(text);
        assert_eq!(TOKENIZER.decode(&ids).expect("decodes"), text);
    }
}

/// gpt-oss's added tokens are all `special: true` in its own `tokenizer.json`,
/// so they decode to nothing — where o200k_base's reference (`tiktoken`) has no
/// skip mode at all and renders `<|endoftext|>`. Same ranks, different answer.
#[test]
fn gpt_oss_drops_its_markers_on_decode_where_o200k_renders_them() {
    let o200k = from_pretrained("o200k_base").expect("o200k_base is bundled");
    assert_eq!(TOKENIZER.decode(&[199999]).expect("decodes"), "");
    assert_eq!(o200k.decode(&[199999]).expect("decodes"), "<|endoftext|>");
}

#[test]
fn gpt_oss_aliases_resolve_to_one_vocabulary() {
    let ids = TOKENIZER.encode_raw("Hello world");
    for name in ["gpt-oss", "gpt_oss", "o200k_harmony"] {
        let alias = from_pretrained(name).expect("alias is bundled");
        assert_eq!(alias.encode_raw("Hello world"), ids, "alias {name}");
    }
}
