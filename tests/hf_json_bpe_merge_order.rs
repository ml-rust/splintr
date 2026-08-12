//! A vocabulary entry the merge list NAMES but the merge order never reaches,
//! loaded through the public `from_json_bytes` entry point.
//!
//! An entry no merge mentions at all is the plain unreachable case, and a scan
//! of the merge list finds it. This file pins the harder one: the merge list
//! names the entry as a result, so that scan calls it reachable, and yet BPE can
//! never produce it — the operands the naming merge needs are consumed by
//! earlier merges and never meet.
//!
//! It is not hypothetical. Gemma 4 has four such entries (`▁yyyy`, `▁YYYY`,
//! `▁diffformul`, `▁::::::::`) and NLLB-200 one (`▁wakakakak`), and before the
//! reachability test became "does merging this surface produce it" every one of
//! them encoded to a single id where HuggingFace gives three or four.
//!
//! The vocabulary below is Gemma 4's case in miniature, and shaped like a real
//! trained one — every multi-character entry is some merge's result, so nothing
//! reaches the merge loop with a base-alphabet rank it would not have in a real
//! file. Walk it by hand on `▁yyyy`:
//!
//! ```text
//! ▁ y y y y            seeded per character
//! ▁y  y y y            `▁ ++ y`  (rank 0), the lowest present
//! ▁y  yy  y            `y ++ y`  (rank 1) beats `▁y ++ y` (rank 2)
//! ▁y  yy  y            `▁y ++ yy` is no merge, and neither is `yy ++ y`
//! ```
//!
//! `▁yy` is now unbuildable inside this word, so `▁yy ++ yy` — the only merge
//! producing `▁yyyy` — can never fire. Three symbols come out, and the entry is
//! reachable by nothing. `▁yy` itself IS reachable, and stays encodable: the
//! test would pass just as well if the fix dropped everything, so it checks both
//! directions.
//!
//! **Every expected id vector below was produced by the HuggingFace
//! `tokenizers` crate (0.23.2-dev) loading this exact JSON document**, and is a
//! reference rather than a snapshot of splintr's own output.

use splintr::{from_json_bytes, AnyTokenizer, Backend};
use std::sync::LazyLock;

/// `▁yyyy` is named by `▁yy ++ yy` and reachable by nothing.
const MERGE_ORDER_JSON: &str = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": null,
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "BPE",
    "dropout": null,
    "unk_token": null,
    "continuing_subword_prefix": "",
    "end_of_word_suffix": "",
    "fuse_unk": false,
    "byte_fallback": false,
    "ignore_merges": false,
    "vocab": {"▁": 0, "y": 1, "▁y": 2, "yy": 3, "▁yy": 4, "▁yyyy": 5},
    "merges": [["▁", "y"], ["y", "y"], ["▁y", "y"], ["▁yy", "yy"]]
  }
}"#;

static TOKENIZER: LazyLock<AnyTokenizer> =
    LazyLock::new(|| from_json_bytes(MERGE_ORDER_JSON.as_bytes()).expect("fixture loads"));

/// The entry the merge order routes around must not be encodable.
///
/// Without the merge-order test this answers `[5]` — the whole-chunk lookup
/// finding an entry BPE cannot produce, one id where HuggingFace gives three.
#[test]
fn an_entry_the_merge_order_routes_around_is_not_encodable() {
    assert_eq!(TOKENIZER.encode_raw("▁yyyy"), vec![2, 3, 1]);
}

/// …while the entries the order does reach keep their ids, including the one
/// that is the routed-around entry's own operand.
#[test]
fn the_entries_the_merge_order_reaches_keep_their_ids() {
    assert_eq!(TOKENIZER.encode_raw("▁yy"), vec![4]);
    assert_eq!(TOKENIZER.encode_raw("▁y"), vec![2]);
    assert_eq!(TOKENIZER.encode_raw("yy"), vec![3]);
    assert_eq!(TOKENIZER.encode_raw("▁yyy"), vec![2, 3]);
}

/// Every id the file states still decodes, unreachable or not: only the encode
/// tables drop them.
#[test]
fn an_unreachable_entry_still_decodes() {
    let Backend::Bpe(bpe) = from_json_bytes(MERGE_ORDER_JSON.as_bytes())
        .expect("fixture loads")
        .into_backend()
    else {
        panic!("expected a BPE backend");
    };
    assert_eq!(bpe.decode(&[5]).expect("decodes"), "▁yyyy");
}
