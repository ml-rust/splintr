//! BPE with `model.end_of_word_suffix` — the CLIP shape — loaded through the
//! public `from_json_bytes` entry point.
//!
//! A model that declares a word-final marker appends it to the LAST symbol of
//! every word *before* merging, so the symbol a word ends on (`o</w>`) is a
//! different vocabulary entry from the same character mid-word (`o`), and every
//! merge above it inherits the distinction. That makes the unsuffixed spelling
//! of a whole word — CLIP's `hello`, id 12887 — a mid-word piece no complete
//! word can produce, even though a whole-chunk vocabulary lookup would happily
//! answer with it.
//!
//! Everything here runs off a synthetic `tokenizer.json` embedded in this file:
//! no checkpoint, no network. It is a miniature CLIP — `Lowercase` normalizer,
//! `Split`+`ByteLevel` pre-tokenizer, `end_of_word_suffix: "</w>"` — with a
//! vocabulary small enough to reason about by hand and shaped so the three
//! things that can go wrong are each pinned by a case:
//!
//! - `hello` is in the vocabulary but is named by no merge. It is the answer a
//!   suffix-blind whole-chunk lookup gives, and the wrong one.
//! - `z</w>` is named by no merge either, yet is exactly how the one-character
//!   word `z` is spelled — a marked character is a seed, never unreachable.
//! - the `h e` merge must NOT fire on the word `he`, whose symbols are `h` and
//!   `e</w>`.
//!
//! **Every expected id vector below was produced by the HuggingFace
//! `tokenizers` Python package, version 0.22.1, via `Tokenizer.from_file(...)`
//! on this exact JSON document** (byte-identical to [`SUFFIX_JSON`], written to
//! a scratch file). They are a reference, not a snapshot of splintr's own
//! output.

use splintr::{from_json_bytes, AnyTokenizer, Backend};
use std::sync::LazyLock;

/// A synthetic ByteLevel BPE `tokenizer.json` declaring `end_of_word_suffix`.
///
/// Vocabulary indices are token ids, assigned in sorted-token order so they do
/// NOT follow merge order — merge priority has to come from `merges`, as it does
/// in every real HuggingFace BPE file.
const SUFFIX_JSON: &str = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 6, "content": "<|endoftext|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "normalizer": {"type": "Lowercase"},
  "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
    {"type": "Split", "pattern": {"Regex": "[a-z]+|[0-9]|[^\\s a-z0-9]+"}, "behavior": "Removed", "invert": true},
    {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true}
  ]},
  "post_processor": null,
  "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true},
  "model": {
    "type": "BPE",
    "dropout": null,
    "unk_token": null,
    "continuing_subword_prefix": "",
    "end_of_word_suffix": "</w>",
    "fuse_unk": false,
    "byte_fallback": false,
    "vocab": {
      "!": 0, "!</w>": 1, "1": 2, "1</w>": 3, "2": 4, "2</w>": 5,
      "<|endoftext|>": 6,
      "a": 7, "a</w>": 8, "c": 9, "c</w>": 10, "ca": 11, "cat</w>": 12,
      "e": 13, "e</w>": 14, "h": 15, "h</w>": 16, "he": 17, "hel": 18,
      "hell": 19, "hello": 20, "hello</w>": 21, "i": 22, "i</w>": 23,
      "l": 24, "l</w>": 25, "ll": 26, "o": 27, "o</w>": 28, "s": 29,
      "s</w>": 30, "si": 31, "sit</w>": 32, "t": 33, "t</w>": 34,
      "z": 35, "z</w>": 36
    },
    "merges": ["h e", "l l", "he ll", "hell o</w>", "c a", "ca t</w>", "he l", "s i", "si t</w>"]
  }
}"#;

static TOKENIZER: LazyLock<AnyTokenizer> =
    LazyLock::new(|| from_json_bytes(SUFFIX_JSON.as_bytes()).expect("suffix BPE json loads"));

fn encode(text: &str) -> Vec<u32> {
    TOKENIZER.encode_raw(text)
}

#[test]
fn loads_as_a_bpe_backed_any_tokenizer() {
    assert!(matches!(TOKENIZER.backend(), Backend::Bpe(_)));
}

/// The headline: a whole word is its *marked* spelling. `hello</w>` (21), never
/// `hello` (20).
#[test]
fn a_whole_word_takes_its_marked_spelling() {
    assert_eq!(encode("hello"), vec![21]);
    assert_eq!(encode("cat"), vec![12]);
    assert_eq!(encode("sit"), vec![32]);
}

/// The unsuffixed spelling of a whole word is in the vocabulary and is named by
/// no merge, so BPE can neither produce it nor start from it. It must not be
/// encodable — and must still decode, since a stray id from elsewhere still has
/// a rendering.
#[test]
fn the_unmarked_whole_word_spelling_is_unreachable_but_still_decodes() {
    assert!(!encode("hello").contains(&20));
    assert_eq!(TOKENIZER.decode(&[20]).unwrap(), "hello");
}

/// A merge fires only on the symbols the word actually has. `he` seeds as
/// `h` + `e</w>`, and the `h e` merge names `e`, not `e</w>` — so it cannot
/// apply, and the word is two tokens where a suffix-blind merge would give one.
#[test]
fn a_merge_over_the_bare_character_does_not_reach_the_marked_one() {
    assert_eq!(encode("he"), vec![15, 14]);
    // Same distinction one merge deeper: `hell` is `hel` + `l</w>`, not `hell`.
    assert_eq!(encode("hell"), vec![18, 25]);
}

/// A marked single character is a seed spelling, so it is reachable however the
/// merge list ignores it: `z</w>` is named by no merge at all, yet is the whole
/// of the word `z`. (CLIP has 139 such entries out of its 256.)
#[test]
fn a_marked_character_named_by_no_merge_is_still_reachable() {
    assert_eq!(encode("z"), vec![36]);
    assert_eq!(encode("12"), vec![3, 5]);
}

/// The marker is per word, so each pre-token gets its own — including the
/// punctuation and digit runs the `Split` pattern cuts out separately.
#[test]
fn every_pre_token_is_marked_separately() {
    assert_eq!(encode("hello cat"), vec![21, 12]);
    assert_eq!(encode("hi!"), vec![15, 23, 1]);
    assert_eq!(encode("hello12"), vec![21, 3, 5]);
    assert_eq!(encode("sit hello z"), vec![32, 21, 36]);
}

/// The normalizer runs first, as it does for CLIP, and an added token is still
/// matched ahead of the model.
#[test]
fn normalization_and_added_tokens_still_apply() {
    assert_eq!(encode("HELLO CAT"), vec![21, 12]);
    assert_eq!(encode("hello<|endoftext|>cat"), vec![21, 6, 12]);
}

/// HuggingFace's ByteLevel decoder renders the marker literally — it is
/// ordinary text in the token's spelling, not a flag — so decode needs no
/// knowledge of it. Measured: `decode([21, 12])` is `"hello</w>cat</w>"`.
#[test]
fn decode_renders_the_marker_literally() {
    assert_eq!(TOKENIZER.decode(&[21, 12]).unwrap(), "hello</w>cat</w>");
    assert_eq!(TOKENIZER.decode(&[18, 25]).unwrap(), "hell</w>");
}
