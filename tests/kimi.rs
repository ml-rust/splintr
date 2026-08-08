//! Integration tests for the Kimi tokenizer (Moonshot AI).
//!
//! The ids pinned here came from a `tiktoken.Encoding` built from Moonshot's own
//! `tiktoken.model` and the `pat_str` in its `tokenization_kimi.py` — the exact
//! construction their tokenizer performs — not from splintr's output. Broad
//! agreement is covered by `tests/fixtures/pretrained/kimi_k2.json` and
//! `kimi_k3.json`; this file pins what those fixtures cannot: the Han branch
//! that makes Kimi's pattern different from o200k's, and the fact that K2 and K3
//! share every merge rank while naming the same ids differently.
// Gated on `vocab-kimi` because every test here loads that vocabulary, and the
// feature is what compiles those bytes in. Without it this crate is empty rather
// than a compile error.
#![cfg(feature = "vocab-kimi")]

use splintr::pretrained::{from_pretrained, kimi_k2_special_tokens, kimi_k3_special_tokens};
use splintr::{AnyTokenizer, SpecialDecode, Tokenize};
use std::sync::LazyLock;

static K2: LazyLock<AnyTokenizer> =
    LazyLock::new(|| from_pretrained("kimi_k2").expect("kimi_k2 is bundled"));
static K3: LazyLock<AnyTokenizer> =
    LazyLock::new(|| from_pretrained("kimi_k3").expect("kimi_k3 is bundled"));

#[test]
fn kimi_encodes_the_reference_ids() {
    for (text, expected) in [
        ("Hello world", vec![19180u32, 2695]),
        ("Hello, world!", vec![19180, 11, 2695, 0]),
        ("你好世界", vec![33845, 2243]),
        ("1234567890", vec![6694, 12972, 16242, 15]),
        ("中文English混合", vec![16717, 44372, 13935]),
    ] {
        assert_eq!(K2.encode_raw(text), expected, "ids for {text:?}");
    }
}

/// The one thing Kimi's pattern does that no other bundled vocabulary's does: a
/// leading `[\p{Han}]+` branch, with Han subtracted from the letter branches so
/// it actually fires. Without the subtraction the letter branch would swallow
/// the Han run and `中文English混合` would not split where it does.
#[test]
fn kimi_splits_han_runs_into_their_own_pre_tokens() {
    let split = |text: &str| K2.pre_tokenize(text).expect("Kimi has a pre-tokenizer");
    assert_eq!(split("中文English混合"), ["中文", "English", "混合"]);
    assert_eq!(split("汉字abc"), ["汉字", "abc"]);
    assert_eq!(split("北京市 Pascal"), ["北京市", " Pascal"]);
}

/// K2 and K3 are one vocabulary below the special block: Moonshot ships a
/// byte-identical `tiktoken.model` for both, so any text without a marker in it
/// must encode the same under either name. This is what justifies one embedded
/// payload rather than two.
#[test]
fn kimi_k2_and_k3_agree_on_every_ordinary_id() {
    for text in [
        "Hello world",
        "The quick brown fox jumps over the lazy dog.",
        "中文和English混合内容。",
        "fn main() { println!(\"hi\"); }",
        "    indented\n\tand tabbed\r\n",
        "🌍🌎🌏 emoji run",
        "日本語のテキストです",
    ] {
        assert_eq!(
            K2.encode_raw(text),
            K3.encode_raw(text),
            "ids diverge for {text:?}"
        );
    }
}

/// The other half: the special blocks are *not* the same. The same id is a
/// different marker in each generation, which is why they are separate names —
/// a K2 chat template encoded against K3 would produce ids K3 never saw.
#[test]
fn kimi_k2_and_k3_name_the_same_ids_differently() {
    let k2 = kimi_k2_special_tokens();
    let k3 = kimi_k3_special_tokens();

    assert_eq!(k2.get("<|im_end|>"), Some(&163586));
    assert_eq!(k3.get("<|end_of_msg|>"), Some(&163586));
    assert_eq!(k2.get("<|im_user|>"), Some(&163587));
    assert_eq!(k3.get("<|open|>"), Some(&163587));

    // Each renders its own name for the shared id.
    assert_eq!(
        K2.decode_with(&[163586], SpecialDecode::Render).unwrap(),
        "<|im_end|>"
    );
    assert_eq!(
        K3.decode_with(&[163586], SpecialDecode::Render).unwrap(),
        "<|end_of_msg|>"
    );

    // What K2 names and K3 does not is a reserved placeholder there, not absent.
    assert_eq!(
        K3.decode_with(&[163594], SpecialDecode::Render).unwrap(),
        "<|reserved_token_163594|>"
    );
    assert_eq!(
        K2.decode_with(&[163594], SpecialDecode::Render).unwrap(),
        "<|im_system|>"
    );

    // Both agree on what they share.
    for (name, id) in [("[BOS]", 163584u32), ("[EOS]", 163585), ("[PAD]", 163839)] {
        assert_eq!(k2.get(name), Some(&id));
        assert_eq!(k3.get(name), Some(&id));
    }
}

/// Moonshot's tokenizer generates a name for every id in the 256-slot reserved
/// block, so all of them decode. Naming only the interesting ones would leave
/// ids splintr can produce but cannot render.
#[test]
fn kimi_reserves_the_whole_256_slot_block() {
    let special = kimi_k2_special_tokens();
    let reserved = (163584..163840).filter(|id| special.values().any(|v| v == id));
    assert_eq!(reserved.count(), 256, "every reserved id must have a name");
    assert_eq!(
        K2.decode_with(&[163700], SpecialDecode::Render).unwrap(),
        "<|reserved_token_163700|>"
    );
}

#[test]
fn kimi_round_trips() {
    for text in [
        "Hello world",
        "  leading and trailing  ",
        "中文和English混合内容。",
        "日本語のテキストです",
        "fn main() { println!(\"hi\"); }",
        "",
    ] {
        let ids = K2.encode_raw(text);
        assert_eq!(K2.decode(&ids).expect("decodes"), text);
    }
}

#[test]
fn kimi_recognizes_markers_in_text() {
    let ids = K2.encode("<|im_user|>hello<|im_end|>");
    assert_eq!(ids.first(), Some(&163587));
    assert_eq!(ids.last(), Some(&163586));
}

/// Bare `kimi` resolves to K2, which covers seven of the published repos to
/// K3's one. Every alias must reach the same vocabulary it names.
#[test]
fn kimi_aliases_resolve_as_documented() {
    let k2_ids = K2.encode_raw("Hello world");
    for name in [
        "kimi",
        "kimi_k2",
        "kimi-k2",
        "kimi_k2.5",
        "kimi-k2.5",
        "kimi_linear",
    ] {
        let alias = from_pretrained(name).expect("alias is bundled");
        assert_eq!(alias.encode_raw("Hello world"), k2_ids, "alias {name}");
        assert_eq!(
            alias.decode_with(&[163586], SpecialDecode::Render).unwrap(),
            "<|im_end|>",
            "alias {name} must carry K2's markers"
        );
    }
    for name in ["kimi_k3", "kimi-k3"] {
        let alias = from_pretrained(name).expect("alias is bundled");
        assert_eq!(
            alias.decode_with(&[163586], SpecialDecode::Render).unwrap(),
            "<|end_of_msg|>",
            "alias {name} must carry K3's markers"
        );
    }
}

#[test]
fn kimi_reports_its_base_vocabulary_size() {
    assert_eq!(
        splintr::pretrained::base_vocab_size_by_name("kimi_k2").unwrap(),
        163840
    );
    assert_eq!(
        splintr::pretrained::base_vocab_size_by_name("kimi_k3").unwrap(),
        163840
    );
    assert!(K2.vocab_size() > 163840, "agent tokens sit above it");
}
