//! WordPiece (BERT family) backend, loaded through the public
//! `from_json_bytes` entry point.
//!
//! Like the Unigram integration tests, this file is entirely self-contained:
//! the `tokenizer.json` is embedded below, so nothing here reads a checkpoint,
//! a GGUF, or the network. The fixture is a miniature `bert-base-uncased`: a
//! `BertNormalizer` that lowercases (and therefore strips accents), a
//! `BertPreTokenizer`, `##` continuation subwords, an `[UNK]`, a
//! `TemplateProcessing` wrapping content in `[CLS]`/`[SEP]`, and a `WordPiece`
//! decoder with `cleanup`.
//!
//! **Every expected id vector below was produced by the HuggingFace
//! `tokenizers` Python package, version 0.22.1, via `Tokenizer.from_file(...)`
//! on this exact JSON document** (byte-identical to [`WORDPIECE_JSON`], written
//! to a scratch file). They are a reference, not a snapshot of splintr's own
//! output.

use splintr::{from_json_bytes, AnyTokenizer, Backend};
// `AnyTokenizer::encode`/`encode_raw` are inherent; `decode` only arrives
// through the trait.
use splintr::Tokenize;
use std::sync::LazyLock;

/// A synthetic BERT-uncased-shaped `tokenizer.json`.
///
/// The vocabulary is chosen so several words need more than one piece
/// (`tokenizers` -> `token` `##izer` `##s`) and so one word shares a prefix with
/// another entry (`##izer` vs `##ization`), making the greedy longest-match walk
/// observable. `strip_accents` is `null`, which is BERT's own shape: HF then
/// follows `lowercase`, so accents are stripped.
const WORDPIECE_JSON: &str = r###"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 0, "content": "[PAD]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 1, "content": "[UNK]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 2, "content": "[CLS]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 3, "content": "[SEP]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 4, "content": "[MASK]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "normalizer": {"type": "BertNormalizer", "clean_text": true, "handle_chinese_chars": true, "strip_accents": null, "lowercase": true},
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": {
    "type": "TemplateProcessing",
    "single": [
      {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
      {"Sequence": {"id": "A", "type_id": 0}},
      {"SpecialToken": {"id": "[SEP]", "type_id": 0}}
    ],
    "pair": [
      {"SpecialToken": {"id": "[CLS]", "type_id": 0}},
      {"Sequence": {"id": "A", "type_id": 0}},
      {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
      {"Sequence": {"id": "B", "type_id": 1}},
      {"SpecialToken": {"id": "[SEP]", "type_id": 1}}
    ],
    "special_tokens": {
      "[CLS]": {"id": "[CLS]", "ids": [2], "tokens": ["[CLS]"]},
      "[SEP]": {"id": "[SEP]", "ids": [3], "tokens": ["[SEP]"]}
    }
  },
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true},
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 100,
    "vocab": {
      "[PAD]": 0,
      "[UNK]": 1,
      "[CLS]": 2,
      "[SEP]": 3,
      "[MASK]": 4,
      "the": 5,
      "token": 6,
      "##izer": 7,
      "##ization": 8,
      "##s": 9,
      "run": 10,
      "##ning": 11,
      "cafe": 12,
      "un": 13,
      "##able": 14,
      "fast": 15,
      "##er": 16,
      "!": 17,
      ",": 18,
      "na": 19,
      "##ive": 20
    }
  }
}"###;

/// The loaded handle. Its declared type is the assertion that `from_json_bytes`
/// hands back an [`AnyTokenizer`] rather than a backend-specific tokenizer.
static WORDPIECE: LazyLock<AnyTokenizer> =
    LazyLock::new(|| from_json_bytes(WORDPIECE_JSON.as_bytes()).expect("wordpiece fixture loads"));

// =============================================================================
// Loading and dispatch
// =============================================================================

/// The loader must recognize `model.type: "WordPiece"` and dispatch to the
/// WordPiece backend.
#[test]
fn loads_as_a_wordpiece_backed_any_tokenizer() {
    assert_eq!(WORDPIECE.family(), "WordPiece");
    assert!(matches!(WORDPIECE.backend(), Backend::WordPiece(_)));
}

/// The policy reads the boundary tokens out of the same file, so a caller never
/// has to hardcode BERT's ids.
#[test]
fn policy_exposes_cls_and_sep() {
    assert_eq!(WORDPIECE.special_token_id("[CLS]"), Some(2));
    assert_eq!(WORDPIECE.special_token_id("[SEP]"), Some(3));
    // `[SEP]` is the last of the EOS candidates, and the only one this file has.
    assert_eq!(WORDPIECE.eos_token_id(), Some(3));
    assert!(WORDPIECE.is_eos(3));
}

// =============================================================================
// Exact token id tests (reference: `tokenizers` 0.22.1 on this same document)
// =============================================================================

/// `##` continuation subwords: a word that is not in the vocabulary whole is
/// walked longest-match-first, and every piece after the first must carry the
/// continuation prefix. `token` + `##izer` + `##s` also pins that the walk
/// prefers `##izer` over stopping at `##ization`'s shared prefix.
///
/// Reference (`add_special_tokens=False`): `"Tokenizers"` -> `[6, 7, 9]`
/// (`token`, `##izer`, `##s`); `"the tokenization"` -> `[5, 6, 8]`.
#[test]
fn continuation_subwords_are_split_on_the_hash_prefix() {
    assert_eq!(WORDPIECE.encode_raw("Tokenizers"), vec![6, 7, 9]);
    assert_eq!(WORDPIECE.encode_raw("the tokenization"), vec![5, 6, 8]);
    assert_eq!(WORDPIECE.encode_raw("running"), vec![10, 11]);
}

/// The `BertNormalizer` declared in the file must actually run: `lowercase`
/// maps `The` onto `the`, and — because `strip_accents` is `null` and therefore
/// follows `lowercase` — `café` maps onto the unaccented `cafe` entry. Without
/// accent stripping this word would fall to `[UNK]`.
///
/// Reference: `"The café"` -> `[5, 12]` (`the`, `cafe`).
#[test]
fn bert_normalizer_lowercases_and_strips_accents() {
    assert_eq!(WORDPIECE.encode_raw("The café"), vec![5, 12]);
    // The same pair reached without any normalization, for contrast.
    assert_eq!(WORDPIECE.encode_raw("the cafe"), vec![5, 12]);
    // A word whose pieces only exist unaccented after stripping.
    assert_eq!(WORDPIECE.encode_raw("naïve"), vec![19, 20]);
}

/// The `BertPreTokenizer` splits punctuation off as its own words, so `,` and
/// `!` become their own tokens rather than being glued to the neighbouring word
/// (which would make both words unknown).
///
/// Reference: `"unable, faster!"` -> `[13, 14, 18, 15, 16, 17]`
/// (`un`, `##able`, `,`, `fast`, `##er`, `!`).
#[test]
fn punctuation_is_split_into_its_own_tokens() {
    assert_eq!(
        WORDPIECE.encode_raw("unable, faster!"),
        vec![13, 14, 18, 15, 16, 17]
    );
}

/// A word with no viable piece decomposition falls back to a single `[UNK]` for
/// the whole word — not per character, and not silently dropped.
///
/// Reference: `"zqzq"` -> `[1]`.
#[test]
fn out_of_vocabulary_word_becomes_a_single_unk() {
    assert_eq!(WORDPIECE.encode_raw("zqzq"), vec![1]);
}

/// A special token spelled in the input is matched as its id rather than being
/// split into subwords (`[`, `##mask`, …) or collapsing to `[UNK]`.
///
/// Reference: `"the [MASK] run"` -> `[5, 4, 10]`.
#[test]
fn added_token_in_the_input_is_matched() {
    assert_eq!(WORDPIECE.encode_raw("the [MASK] run"), vec![5, 4, 10]);
}

// =============================================================================
// The post-processor template
// =============================================================================

/// `encode` produces the model's real input (`[CLS] … [SEP]`) and `encode_raw`
/// the bare content; the difference is exactly the declared template.
///
/// Reference: `"Tokenizers"` -> `[6, 7, 9]` with `add_special_tokens=False`,
/// `[2, 6, 7, 9, 3]` with `add_special_tokens=True`.
#[test]
fn encode_wraps_with_cls_and_sep_and_encode_raw_does_not() {
    let raw = WORDPIECE.encode_raw("Tokenizers");
    let wrapped = WORDPIECE.encode("Tokenizers");
    assert_eq!(raw, vec![6, 7, 9]);
    assert_eq!(wrapped, vec![2, 6, 7, 9, 3]);
    assert_eq!(&wrapped[1..wrapped.len() - 1], &raw[..]);
}

/// The reranker case: two segments joined as `[CLS] a [SEP] b [SEP]` from the
/// template's `pair` array.
///
/// Reference: `encode("the token", "fast running")` ->
/// `[2, 5, 6, 3, 15, 10, 11, 3]`.
#[test]
fn encode_pair_applies_the_bert_pair_template() {
    assert_eq!(
        WORDPIECE.encode_pair("the token", "fast running").unwrap(),
        vec![2, 5, 6, 3, 15, 10, 11, 3]
    );
}

// =============================================================================
// Decoding through the declared `decoder`
// =============================================================================

/// The declared `WordPiece` decoder must run: continuation pieces rejoin without
/// a space, non-continuation pieces get one, and `cleanup: true` tightens the
/// space before `,` and `!`. A handle that ignored the chain would render
/// `token ##izer ##s`.
///
/// Reference (`skip_special_tokens=True`): `[6, 7, 9]` -> `"tokenizers"`,
/// `[13, 14, 18, 15, 16, 17]` -> `"unable, faster!"`.
#[test]
fn decode_runs_the_declared_wordpiece_decoder() {
    assert_eq!(
        Tokenize::decode(&*WORDPIECE, &[6, 7, 9]).unwrap(),
        "tokenizers"
    );
    assert_eq!(
        Tokenize::decode(&*WORDPIECE, &[13, 14, 18, 15, 16, 17]).unwrap(),
        "unable, faster!"
    );
}

/// `special: true` ids are dropped before the decoder runs (HF's default
/// `skip_special_tokens=true`), so the template's own `[CLS]`/`[SEP]` disappear
/// and an all-`[UNK]` sequence decodes to the empty string.
///
/// Reference: `[2, 6, 7, 9, 3]` -> `"tokenizers"`, `[2, 1, 3]` -> `""`.
#[test]
fn decode_drops_special_ids() {
    assert_eq!(
        Tokenize::decode(&*WORDPIECE, &[2, 6, 7, 9, 3]).unwrap(),
        "tokenizers"
    );
    assert_eq!(Tokenize::decode(&*WORDPIECE, &[2, 1, 3]).unwrap(), "");
}

// =============================================================================
// `strip_accents` is independent of `lowercase`
// =============================================================================

/// A second fixture whose vocabulary keeps every casing/accent variant of the
/// same two words apart (`cafe`/`café`/`Cafe`/`Café`, `naive`/`naïve`), so the
/// ids alone reveal which of the two normalizer flags actually ran. The
/// `__NORMALIZER__` placeholder is filled in per test.
///
/// There is no `post_processor`, so `encode_raw` is directly comparable to
/// `tokenizers`' `encode(..., add_special_tokens=False)`.
const ACCENT_JSON: &str = r###"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 0, "content": "[PAD]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 1, "content": "[UNK]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 2, "content": "[CLS]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 3, "content": "[SEP]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "normalizer": __NORMALIZER__,
  "pre_tokenizer": {"type": "BertPreTokenizer"},
  "post_processor": null,
  "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true},
  "model": {
    "type": "WordPiece",
    "unk_token": "[UNK]",
    "continuing_subword_prefix": "##",
    "max_input_chars_per_word": 100,
    "vocab": {
      "[PAD]": 0,
      "[UNK]": 1,
      "[CLS]": 2,
      "[SEP]": 3,
      "cafe": 4,
      "café": 5,
      "naive": 6,
      "naïve": 7,
      "Cafe": 8,
      "Café": 9,
      "the": 10
    }
  }
}"###;

/// Load [`ACCENT_JSON`] with `normalizer` set to `norm`.
fn accent_tokenizer(norm: &str) -> AnyTokenizer {
    let json = ACCENT_JSON.replace("__NORMALIZER__", norm);
    from_json_bytes(json.as_bytes()).expect("accent fixture loads")
}

/// `strip_accents: null` is the shape BERT itself ships: HuggingFace resolves it
/// to `lowercase`, so accents ARE stripped here.
///
/// Reference (`tokenizers` 0.22.1, `add_special_tokens=False`):
/// `"café"` -> `[4]`, `"Café"` -> `[4]`, `"naïve"` -> `[6]`,
/// `"the café"` -> `[10, 4]`.
#[test]
fn null_strip_accents_follows_lowercase() {
    let tok = accent_tokenizer(
        r#"{"type": "BertNormalizer", "clean_text": true, "handle_chinese_chars": true, "strip_accents": null, "lowercase": true}"#,
    );
    assert_eq!(tok.encode_raw("café"), vec![4]);
    assert_eq!(tok.encode_raw("Café"), vec![4]);
    assert_eq!(tok.encode_raw("naïve"), vec![6]);
    assert_eq!(tok.encode_raw("the café"), vec![10, 4]);
}

/// `strip_accents: false` with `lowercase: true` — the cased-multilingual shape.
/// The explicit `false` wins over the `lowercase` default, so the accented vocab
/// entries are the ones reached; coupling the two flags would wrongly yield
/// `[4]`/`[6]`.
///
/// Reference (`tokenizers` 0.22.1, `add_special_tokens=False`):
/// `"café"` -> `[5]`, `"Café"` -> `[5]`, `"naïve"` -> `[7]`,
/// `"the café"` -> `[10, 5]`.
#[test]
fn explicit_false_strip_accents_keeps_accents_while_lowercasing() {
    let tok = accent_tokenizer(
        r#"{"type": "BertNormalizer", "clean_text": true, "handle_chinese_chars": true, "strip_accents": false, "lowercase": true}"#,
    );
    assert_eq!(tok.encode_raw("café"), vec![5]);
    assert_eq!(tok.encode_raw("Café"), vec![5]);
    assert_eq!(tok.encode_raw("naïve"), vec![7]);
    assert_eq!(tok.encode_raw("the café"), vec![10, 5]);
}

/// `strip_accents: true` with `lowercase: false` — the other direction. Accents
/// go, casing stays, so `"Café"` lands on the capitalized unaccented entry `Cafe`
/// (id 8) rather than `cafe` (id 4).
///
/// Reference (`tokenizers` 0.22.1, `add_special_tokens=False`):
/// `"café"` -> `[4]`, `"Café"` -> `[8]`, `"naïve"` -> `[6]`,
/// `"the café"` -> `[10, 4]`.
#[test]
fn explicit_true_strip_accents_without_lowercasing() {
    let tok = accent_tokenizer(
        r#"{"type": "BertNormalizer", "clean_text": true, "handle_chinese_chars": true, "strip_accents": true, "lowercase": false}"#,
    );
    assert_eq!(tok.encode_raw("café"), vec![4]);
    assert_eq!(tok.encode_raw("Café"), vec![8]);
    assert_eq!(tok.encode_raw("naïve"), vec![6]);
    assert_eq!(tok.encode_raw("the café"), vec![10, 4]);
}

/// A `Lowercase` normalizer lowercases and nothing else — it must not be read as
/// a licence to strip accents. That holds both standing alone and next to a
/// `BertNormalizer` whose own `strip_accents` is `null`: the `null` defaults to
/// *that node's* `lowercase` (here `false`), not to the sequence's.
///
/// Reference (`tokenizers` 0.22.1, `add_special_tokens=False`), for both
/// `Sequence[Lowercase]` and
/// `Sequence[BertNormalizer{lowercase: false, strip_accents: null}, Lowercase]`:
/// `"café"` -> `[5]`, `"Café"` -> `[5]`, `"naïve"` -> `[7]`.
#[test]
fn lowercase_node_in_a_sequence_does_not_strip_accents() {
    for norm in [
        r#"{"type": "Sequence", "normalizers": [{"type": "Lowercase"}]}"#,
        r#"{"type": "Sequence", "normalizers": [
             {"type": "BertNormalizer", "clean_text": true, "handle_chinese_chars": true, "strip_accents": null, "lowercase": false},
             {"type": "Lowercase"}
           ]}"#,
    ] {
        let tok = accent_tokenizer(norm);
        assert_eq!(tok.encode_raw("café"), vec![5], "normalizer {norm}");
        assert_eq!(tok.encode_raw("Café"), vec![5], "normalizer {norm}");
        assert_eq!(tok.encode_raw("naïve"), vec![7], "normalizer {norm}");
    }
}

/// A bare `StripAccents` node is NOT BERT-style accent stripping: HuggingFace's
/// `StripAccents` drops nonspacing marks without decomposing first, so on
/// ordinary (NFC) text it changes nothing. It only bites once an `NFD` has run.
///
/// Reference (`tokenizers` 0.22.1, `add_special_tokens=False`):
/// `Sequence[StripAccents]` gives `"café"` -> `[5]` and `"Café"` -> `[9]`
/// (untouched), while `Sequence[NFD, StripAccents]` gives `"café"` -> `[4]` and
/// `"Café"` -> `[8]`.
#[test]
fn strip_accents_node_only_strips_after_a_decomposition() {
    let bare =
        accent_tokenizer(r#"{"type": "Sequence", "normalizers": [{"type": "StripAccents"}]}"#);
    assert_eq!(bare.encode_raw("café"), vec![5]);
    assert_eq!(bare.encode_raw("Café"), vec![9]);
    assert_eq!(bare.encode_raw("naïve"), vec![7]);

    let decomposed = accent_tokenizer(
        r#"{"type": "Sequence", "normalizers": [{"type": "NFD"}, {"type": "StripAccents"}]}"#,
    );
    assert_eq!(decomposed.encode_raw("café"), vec![4]);
    assert_eq!(decomposed.encode_raw("Café"), vec![8]);
    assert_eq!(decomposed.encode_raw("naïve"), vec![6]);
}

/// Round-trip over the lowercased, normalized form — which is what a BERT-family
/// tokenizer can round-trip at all, since casing and accents are destroyed on
/// the way in.
#[test]
fn reference_cases_round_trip_in_normalized_form() {
    for text in ["tokenizers", "the tokenization", "running", "the cafe"] {
        let ids = WORDPIECE.encode(text);
        assert_eq!(
            Tokenize::decode(&*WORDPIECE, &ids).unwrap(),
            text,
            "round trip for {text:?}"
        );
    }
}
