//! BPE with `pre_tokenizer: null` — the Llama 2 shape — loaded through the
//! public `from_json_bytes` entry point.
//!
//! A model declaring no pre-tokenizer does not split: HuggingFace hands the
//! whole text to the BPE model as one word. Llama 2 and Code Llama ship exactly
//! this, with the metaspace transform living in the **normalizer** (`Prepend ▁`
//! then `Replace " " → ▁`) and no stage after it, so a whole document arrives as
//! a single chunk.
//!
//! Merging a document whole and merging its words apart reach the same ids
//! wherever the vocabulary proves the cut free, and splintr therefore cuts such
//! a chunk back apart at marker-run starts (see `RunSplit`). That is an
//! optimization, and these tests exist to pin that it is invisible.
//!
//! # Why the fixture is trained rather than written
//!
//! The vocabulary below was produced by `tokenizers`' own `BpeTrainer` over
//! English text, **with** a `Split("▁")` pre-tokenizer so merges stay inside
//! words — which is how Llama 2's own SentencePiece vocabulary was built — and
//! then serialized with `pre_tokenizer` set to `null`, which is how Llama 2
//! ships. A hand-written merge list does not do: order one by hand and it is
//! easy to name a merge whose result the merge order can never actually reach,
//! which no trainer emits and which tests the wrong thing.
//!
//! It matters that this vocabulary has **no token spanning a marker-run start**
//! (real Llama 2 has none either: its 15 interior-`▁` tokens are all pure runs).
//! Such tokens are `RunSplit` guards, and past a handful of them the cut is
//! abandoned entirely — a fixture carrying them would exercise the fallback
//! rather than the cut, and would pass whether or not the cut works.
//!
//! **Every expected id vector below was produced by the HuggingFace
//! `tokenizers` Python package, version 0.22.1, via `Tokenizer.from_file(...)`
//! on this exact JSON document** (byte-identical to [`NOSPLIT_JSON`], written to
//! a scratch file). They are a reference, not a snapshot of splintr's own output.

use splintr::{from_json_bytes, AnyTokenizer, Backend};
use std::sync::LazyLock;

/// A miniature Llama 2: metaspace in the normalizer, nothing splitting after it.
const NOSPLIT_JSON: &str = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 0, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "normalizer": {"type": "Sequence", "normalizers": [
    {"type": "Prepend", "prepend": "▁"},
    {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}
  ]},
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": {"type": "Sequence", "decoders": [
    {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
    {"type": "Strip", "content": " ", "start": 1, "stop": 0}
  ]},
  "model": {
    "type": "BPE",
    "dropout": null,
    "unk_token": "<unk>",
    "continuing_subword_prefix": null,
    "end_of_word_suffix": null,
    "fuse_unk": false,
    "byte_fallback": false,
    "vocab": {"<unk>": 0, "a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7, "h": 8, "i": 9, "j": 10, "k": 11, "l": 12, "m": 13, "n": 14, "o": 15, "p": 16, "q": 17, "r": 18, "s": 19, "t": 20, "u": 21, "v": 22, "w": 23, "x": 24, "y": 25, "z": 26, "▁": 27, "▁t": 28, "▁a": 29, "in": 30, "▁th": 31, "▁s": 32, "er": 33, "▁o": 34, "▁the": 35, "re": 36, "▁w": 37, "▁c": 38, "on": 39, "en": 40, "▁b": 41, "▁f": 42, "at": 43, "▁p": 44, "▁m": 45, "es": 46, "it": 47, "or": 48, "nd": 49, "is": 50, "▁h": 51, "ing": 52, "ed": 53, "ou": 54, "ar": 55, "▁d": 56, "▁in": 57, "al": 58, "▁to": 59, "an": 60, "▁of": 61, "▁and": 62, "le": 63, "ic": 64, "▁g": 65, "as": 66, "om": 67, "▁n": 68, "ion": 69, "▁re": 70, "▁l": 71, "il": 72, "▁e": 73, "ent": 74, "ve": 75, "ro": 76, "us": 77, "et": 78, "▁i": 79, "ac": 80, "▁y": 81, "ay": 82, "▁be": 83, "▁on": 84, "▁for": 85, "id": 86, "ly": 87, "▁wh": 88, "oo": 89},
    "merges": ["▁ t", "▁ a", "i n", "▁t h", "▁ s", "e r", "▁ o", "▁th e", "r e", "▁ w", "▁ c", "o n", "e n", "▁ b", "▁ f", "a t", "▁ p", "▁ m", "e s", "i t", "o r", "n d", "i s", "▁ h", "in g", "e d", "o u", "a r", "▁ d", "▁ in", "a l", "▁t o", "a n", "▁o f", "▁a nd", "l e", "i c", "▁ g", "a s", "o m", "▁ n", "i on", "▁ re", "▁ l", "i l", "▁ e", "en t", "v e", "r o", "u s", "e t", "▁ i", "a c", "▁ y", "a y", "▁b e", "▁o n", "▁f or", "i d", "l y", "▁w h", "o o"]
  }
}"#;

static TOKENIZER: LazyLock<AnyTokenizer> =
    LazyLock::new(|| from_json_bytes(NOSPLIT_JSON.as_bytes()).expect("no-split BPE json loads"));

fn encode(text: &str) -> Vec<u32> {
    TOKENIZER.encode_raw(text)
}

#[test]
fn loads_as_a_bpe_backed_any_tokenizer() {
    assert!(matches!(TOKENIZER.backend(), Backend::Bpe(_)));
}

/// One word, with no interior marker run to cut at.
#[test]
fn a_single_word_is_unaffected() {
    assert_eq!(encode("the"), vec![35]);
    assert_eq!(encode("and"), vec![62]);
}

/// Several words in one chunk — the case the cut exists for. Cutting at the
/// marker-run starts must reach exactly what merging the whole chunk does.
#[test]
fn many_words_in_one_chunk_agree_with_the_reference() {
    assert_eq!(encode("the cat"), vec![35, 38, 43]);
    assert_eq!(
        encode("the cat sat on the mat"),
        vec![35, 38, 43, 32, 43, 84, 35, 45, 43]
    );
    assert_eq!(encode("and the other"), vec![62, 35, 34, 20, 8, 33]);
}

/// A run of markers is cut only at its start, never through it. Splitting at
/// every marker would be a different segmentation, and on indented text — code,
/// LaTeX — a different set of ids.
#[test]
fn a_marker_run_is_cut_only_at_its_start() {
    assert_eq!(encode("the    cat"), vec![35, 27, 27, 27, 38, 43]);
    assert_eq!(encode("    "), vec![27, 27, 27, 27, 27]);
}

/// A document long enough that merging it whole and merging its words apart are
/// visibly different amounts of work. The ids are the reference's either way.
#[test]
fn a_long_document_agrees_with_the_reference() {
    let text = "the cat sat on the mat and the other one for in of to be";
    assert_eq!(
        encode(text),
        vec![35, 38, 43, 32, 43, 84, 35, 45, 43, 62, 35, 34, 20, 8, 33, 84, 5, 85, 57, 61, 59, 83]
    );
}

/// The cut is invisible in the ids by construction, so nothing above can tell
/// whether it ran — every assertion here passes on a build that merges each
/// document whole. This one cannot: a cut document reaches the chunk cache as
/// one entry per piece, an uncut one as a single entry covering the whole text.
///
/// Without it, a regression that stopped cutting these models would restore the
/// throughput collapse this fixture was written for and no test would notice.
#[test]
fn a_multi_word_document_is_cut_into_pieces() {
    // Its own tokenizer, not the shared one: the suite runs in threads over a
    // single `LazyLock`, so another test's chunks would satisfy this on their own.
    let own = from_json_bytes(NOSPLIT_JSON.as_bytes()).expect("loads");
    let Backend::Bpe(tok) = own.backend() else {
        panic!("fixture is BPE-backed")
    };
    let _ = own.encode_raw("the cat sat on the mat and the other one for in of to be");
    assert!(
        tok.cache_len() > 1,
        "the document reached the merge as {} chunk(s) — it was not cut apart",
        tok.cache_len()
    );
}

/// No marker between the words at all, so no cut is available and the merge runs
/// over the whole chunk — the path the cut falls back to.
#[test]
fn a_chunk_with_no_marker_run_still_merges_whole() {
    assert_eq!(encode("thecatsat"), vec![35, 3, 43, 19, 43]);
}
