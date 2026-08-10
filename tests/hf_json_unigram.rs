//! Unigram (SentencePiece) backend, loaded through the public
//! `from_json_bytes` entry point.
//!
//! Everything here runs off a synthetic `tokenizer.json` embedded in this file:
//! no checkpoint, no GGUF, no network. The vocabulary is small enough to reason
//! about by hand but deliberately shaped so the interesting parts of the
//! pipeline are actually exercised — a Metaspace pre-tokenizer, scores whose
//! maximum-score segmentation differs from greedy-longest-match, an added token
//! carrying `lstrip: true`, a `TemplateProcessing` post-processor, and a
//! four-stage `decoder` chain.
//!
//! **Every expected id vector below was produced by the HuggingFace
//! `tokenizers` Python package, version 0.22.1, via
//! `Tokenizer.from_file(...)` on this exact JSON document** (byte-identical to
//! [`UNIGRAM_JSON`], written to a scratch file). They are a reference, not a
//! snapshot of splintr's own output — a mismatch means splintr diverged from
//! HuggingFace, which is the only thing that makes these tests worth having.

use splintr::{from_json_bytes, AnyTokenizer, Backend, FxHashSet, PolicyError, SpecialMode};
// `AnyTokenizer::encode`/`encode_raw`/`encode_with` are inherent; `decode` only
// arrives through the trait.
use splintr::Tokenize;
use std::sync::LazyLock;

/// A synthetic SentencePiece-Unigram `tokenizer.json`.
///
/// Vocabulary indices are token ids. The scores are chosen so `"understanding"`
/// has a non-obvious segmentation: greedy longest-match takes `▁unders`(-8.0) and
/// is forced onto `tand`(-3.5) + `ing`(-1.5) for a total of -13.0, while Viterbi
/// finds `▁under`(-3.0) + `stand`(-2.0) + `ing`(-1.5) = -6.5. The two paths use
/// different ids, so the assertion distinguishes them.
///
/// `<mask>` (id 33) and `<tool>` (id 34) live *outside* `model.vocab` — the
/// XLM-RoBERTa shape, and what makes the `SpecialMode::Ordinary` case below
/// meaningful: with them absent from the model the same text has an ordinary
/// character-level segmentation to fall back to.
const UNIGRAM_JSON: &str = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 0, "content": "<unk>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 1, "content": "<s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 2, "content": "</s>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 3, "content": "<pad>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 33, "content": "<mask>", "single_word": false, "lstrip": true, "rstrip": false, "normalized": false, "special": true},
    {"id": 34, "content": "<tool>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true},
  "post_processor": {
    "type": "TemplateProcessing",
    "single": [
      {"SpecialToken": {"id": "<s>", "type_id": 0}},
      {"Sequence": {"id": "A", "type_id": 0}},
      {"SpecialToken": {"id": "</s>", "type_id": 0}}
    ],
    "pair": [
      {"SpecialToken": {"id": "<s>", "type_id": 0}},
      {"Sequence": {"id": "A", "type_id": 0}},
      {"SpecialToken": {"id": "</s>", "type_id": 0}},
      {"Sequence": {"id": "B", "type_id": 1}},
      {"SpecialToken": {"id": "</s>", "type_id": 1}}
    ],
    "special_tokens": {
      "<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]},
      "</s>": {"id": "</s>", "ids": [2], "tokens": ["</s>"]}
    }
  },
  "decoder": {
    "type": "Sequence",
    "decoders": [
      {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
      {"type": "ByteFallback"},
      {"type": "Fuse"},
      {"type": "Strip", "content": " ", "start": 1, "stop": 0}
    ]
  },
  "model": {
    "type": "Unigram",
    "unk_id": 0,
    "byte_fallback": false,
    "vocab": [
      ["<unk>", 0.0],
      ["<s>", 0.0],
      ["</s>", 0.0],
      ["<pad>", 0.0],
      ["▁", -6.0],
      ["▁un", -4.0],
      ["▁under", -3.0],
      ["▁unders", -8.0],
      ["tand", -3.5],
      ["stand", -2.0],
      ["ing", -1.5],
      ["der", -3.0],
      ["s", -5.0],
      ["▁the", -1.0],
      ["▁a", -4.0],
      ["▁tandem", -2.0],
      ["t", -6.0],
      ["a", -6.0],
      ["n", -6.0],
      ["d", -6.0],
      ["i", -6.0],
      ["g", -6.0],
      ["e", -6.0],
      ["r", -6.0],
      ["u", -6.0],
      ["h", -6.0],
      ["▁s", -6.5],
      ["<", -6.0],
      [">", -6.0],
      ["m", -6.0],
      ["k", -6.0],
      ["o", -6.0],
      ["l", -6.0]
    ]
  }
}"#;

/// The loaded handle. Its declared type is the assertion that `from_json_bytes`
/// hands back an [`AnyTokenizer`] rather than a backend-specific tokenizer.
static UNIGRAM: LazyLock<AnyTokenizer> =
    LazyLock::new(|| from_json_bytes(UNIGRAM_JSON.as_bytes()).expect("unigram fixture loads"));

// =============================================================================
// Loading and dispatch
// =============================================================================

/// The loader must recognize `model.type: "Unigram"` and dispatch to the
/// SentencePiece backend — the family name is what downstream code branches on.
#[test]
fn loads_as_a_unigram_backed_any_tokenizer() {
    assert_eq!(UNIGRAM.family(), "Unigram");
    assert!(matches!(UNIGRAM.backend(), Backend::Unigram(_)));
}

/// The policy is parsed from the same file, so the boundary tokens are
/// answerable without re-reading the model card.
#[test]
fn policy_exposes_the_declared_specials() {
    assert_eq!(UNIGRAM.special_token_id("<s>"), Some(1));
    assert_eq!(UNIGRAM.special_token_id("</s>"), Some(2));
    assert_eq!(UNIGRAM.eos_token_id(), Some(2));
    assert!(UNIGRAM.is_eos(2));
}

// =============================================================================
// Exact token id tests (reference: `tokenizers` 0.22.1 on this same document)
// =============================================================================

/// Viterbi, not greedy longest-match.
///
/// `▁unders` is the longest matching prefix of `▁understanding` but scores
/// -8.0; taking it costs -13.0 overall. The maximum-score path is
/// `▁under`(-3.0) + `stand`(-2.0) + `ing`(-1.5) = -6.5. A greedy implementation
/// would emit `[7, 8, 10]` here.
///
/// Reference (`tokenizers` 0.22.1, `add_special_tokens=False`):
/// `"understanding"` -> `[6, 9, 10]` (`▁under`, `stand`, `ing`).
#[test]
fn viterbi_segmentation_beats_greedy_longest_match() {
    assert_eq!(UNIGRAM.encode_raw("understanding"), vec![6, 9, 10]);
}

/// A longer piece wins when it genuinely scores better: `▁tandem`(-2.0) beats
/// splitting into `t`+`a`+`n`+`d`+`e`+`m` (-36.0). Paired with the test above,
/// this pins that the segmentation is score-driven in both directions rather
/// than a fixed preference for short or long pieces.
///
/// Reference: `"the tandem"` -> `[13, 15]` (`▁the`, `▁tandem`).
#[test]
fn score_maximization_also_picks_the_longer_piece_when_it_wins() {
    assert_eq!(UNIGRAM.encode_raw("the tandem"), vec![13, 15]);
}

/// Metaspace whitespace handling: `prepend_scheme: "always"` gives the first
/// word its own `▁`, and a *run* of spaces is not collapsed — the second space
/// surfaces as the lone `▁` piece (id 4) between the two words.
///
/// Reference: `"the understanding"` -> `[13, 6, 9, 10]`;
/// `"the  understanding"` (two spaces) -> `[13, 4, 6, 9, 10]`.
#[test]
fn metaspace_prepends_and_preserves_whitespace_runs() {
    assert_eq!(UNIGRAM.encode_raw("the understanding"), vec![13, 6, 9, 10]);
    assert_eq!(
        UNIGRAM.encode_raw("the  understanding"),
        vec![13, 4, 6, 9, 10]
    );
}

/// The same fixture behind `Sequence[WhitespaceSplit, Metaspace]` — T5's shape —
/// where whitespace is a separator rather than a piece.
///
/// `WhitespaceSplit` discards what it splits on, so `Metaspace` never sees the
/// run and every input below collapses to the single-space answer, and pure
/// whitespace produces nothing at all. Treating the sequence as plain Metaspace
/// gave `[13, 4, 6, 9, 10]` for the two-space case and `[4]` for `"   "`.
///
/// Reference (`tokenizers` 0.22.1, this document with that `pre_tokenizer`):
/// all three spellings -> `[13, 6, 9, 10]`; `"   "` -> `[]`.
#[test]
fn whitespace_split_drops_the_whitespace_it_splits_on() {
    let json = UNIGRAM_JSON.replace(
        r#""pre_tokenizer": {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true}"#,
        r#""pre_tokenizer": {"type": "Sequence", "pretokenizers": [{"type": "WhitespaceSplit"}, {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true}]}"#,
    );
    let tok = from_json_bytes(json.as_bytes()).expect("whitespace-split fixture loads");
    for text in [
        "the understanding",
        "the  understanding",
        "the\n\nunderstanding",
    ] {
        assert_eq!(tok.encode_raw(text), vec![13, 6, 9, 10], "{text:?}");
    }
    assert_eq!(tok.encode_raw("   "), Vec::<u32>::new());
}

/// A character absent from the vocabulary falls back to `unk_id` (0) for the
/// whole word rather than dropping it or panicking; the Metaspace prefix still
/// becomes its own `▁` piece.
///
/// Reference: `"zqx"` -> `[4, 0]` (`▁`, `<unk>`).
#[test]
fn out_of_vocabulary_text_falls_back_to_unk() {
    assert_eq!(UNIGRAM.encode_raw("zqx"), vec![4, 0]);
}

// =============================================================================
// Added tokens adjacent to text, and the `lstrip` flag
// =============================================================================

/// `lstrip: true` on `<mask>` must absorb the space that precedes it, while
/// `<pad>` — same file, same shape, flag absent — must leave that space alone as
/// its own `▁` piece. This is the XLM-RoBERTa family's shape (bge-m3 and
/// friends), and the flag can only be right if it is carried *per token* from
/// the json through to the matcher; a loader that reads it globally, or not at
/// all, fails exactly one of these two lines.
///
/// Reference (`add_special_tokens=False`):
/// `"understanding <mask>the"` -> `[6, 9, 10, 33, 13]` (no `▁` before `<mask>`),
/// `"understanding <pad>the"`  -> `[6, 9, 10, 4, 3, 13]` (`▁` survives).
#[test]
fn lstrip_added_token_absorbs_the_preceding_space() {
    assert_eq!(
        UNIGRAM.encode_raw("understanding <mask>the"),
        vec![6, 9, 10, 33, 13]
    );
    assert_eq!(
        UNIGRAM.encode_raw("understanding <pad>the"),
        vec![6, 9, 10, 4, 3, 13]
    );
}

/// An added token butted directly against text on both sides is still matched,
/// and each surrounding segment is metaspace-prefixed independently — so `the`
/// after `<mask>` becomes `▁the` (id 13) even though no space was written.
///
/// Reference: `"understanding<mask>the"` -> `[6, 9, 10, 33, 13]`.
#[test]
fn added_token_is_matched_when_adjacent_to_text() {
    assert_eq!(
        UNIGRAM.encode_raw("understanding<mask>the"),
        vec![6, 9, 10, 33, 13]
    );
}

// =============================================================================
// The post-processor template
// =============================================================================

/// `encode` is the model's real input (`<s> … </s>`), `encode_raw` the bare
/// content. The two must differ by exactly the declared template and nothing
/// else — the default being the safe one is the whole point of the split.
///
/// Reference: `"the tandem"` -> `[13, 15]` with `add_special_tokens=False`,
/// `[1, 13, 15, 2]` with `add_special_tokens=True`.
#[test]
fn encode_applies_the_template_and_encode_raw_does_not() {
    let raw = UNIGRAM.encode_raw("the tandem");
    let wrapped = UNIGRAM.encode("the tandem");
    assert_eq!(raw, vec![13, 15]);
    assert_eq!(wrapped, vec![1, 13, 15, 2]);
    // The difference is precisely the template's two boundary tokens.
    assert_eq!(&wrapped[1..wrapped.len() - 1], &raw[..]);
}

/// The `pair` array of the same `TemplateProcessing` joins two segments the way
/// the model was trained, without the caller hand-placing anything.
///
/// Reference: `encode("the tandem", "understanding")` ->
/// `[1, 13, 15, 2, 6, 9, 10, 2]`.
#[test]
fn encode_pair_applies_the_pair_template() {
    assert_eq!(
        UNIGRAM.encode_pair("the tandem", "understanding").unwrap(),
        vec![1, 13, 15, 2, 6, 9, 10, 2]
    );
}

// =============================================================================
// Decoding through the declared `decoder` chain
// =============================================================================

/// Decoding is driven by the file's four-stage `decoder`
/// (Replace `▁`→space, ByteFallback, Fuse, Strip one leading space), not by the
/// backend's built-in rendering. A handle that ignored the chain would return
/// the raw pieces (`▁the▁tandem`).
///
/// Reference (`tokenizers`, `skip_special_tokens=True`):
/// `[13, 15]` -> `"the tandem"`, `[13, 4, 6, 9, 10]` -> `"the  understanding"`
/// (the doubled space survives the chain).
#[test]
fn decode_runs_the_declared_decoder_chain() {
    assert_eq!(
        Tokenize::decode(&*UNIGRAM, &[13, 15]).unwrap(),
        "the tandem"
    );
    assert_eq!(
        Tokenize::decode(&*UNIGRAM, &[13, 4, 6, 9, 10]).unwrap(),
        "the  understanding"
    );
}

/// `special: true` ids are dropped before the decoder chain runs — HF's default
/// `skip_special_tokens=true`. The template's own `<s>`/`</s>` therefore vanish,
/// and so does an `<unk>`.
///
/// Reference: `[1, 13, 6, 9, 10, 2]` -> `"the understanding"`,
/// `[6, 9, 10, 33, 13]` -> `"understanding the"`, `[4, 0]` -> `""`.
#[test]
fn decode_drops_special_ids() {
    assert_eq!(
        Tokenize::decode(&*UNIGRAM, &[1, 13, 6, 9, 10, 2]).unwrap(),
        "the understanding"
    );
    // `<mask>` (id 33, special) is skipped; the `▁the` that followed it still
    // renders with its space.
    assert_eq!(
        Tokenize::decode(&*UNIGRAM, &[6, 9, 10, 33, 13]).unwrap(),
        "understanding the"
    );
    assert_eq!(Tokenize::decode(&*UNIGRAM, &[4, 0]).unwrap(), "");
}

/// Round-trip: encoding then decoding returns the input for text made only of
/// in-vocabulary pieces.
#[test]
fn reference_cases_round_trip() {
    for text in ["the tandem", "understanding", "the understanding"] {
        let ids = UNIGRAM.encode(text);
        assert_eq!(
            Tokenize::decode(&*UNIGRAM, &ids).unwrap(),
            text,
            "round trip for {text:?}"
        );
    }
}

// =============================================================================
// SpecialMode: what a control-token spelling in untrusted text is allowed to do
// =============================================================================

/// Under [`SpecialMode::Ordinary`] the literal spelling `<mask>` must NOT be
/// promoted to the control-token id 33 — it is encoded as ordinary content,
/// character by character. Under [`SpecialMode::All`] (what plain `encode`
/// does) the same text yields id 33. The policy's boundary template applies in
/// both cases: refusing to match a special token *in the content* says nothing
/// about the boundary tokens this tokenizer always wraps a sequence in.
///
/// Reference for the ordinary segmentation (`tokenizers` 0.22.1 on this same
/// document with `added_tokens` emptied and the post-processor removed, so the
/// text has no added token to match and only the model runs):
/// `"understanding<mask>the"` -> `[6, 9, 10, 27, 29, 17, 12, 30, 28, 16, 25, 22]`
/// (`▁under`, `stand`, `ing`, `<`, `m`, `a`, `s`, `k`, `>`, `t`, `h`, `e`).
/// Reference for the matched form (full document, `add_special_tokens=True`):
/// `[1, 6, 9, 10, 33, 13, 2]`.
#[test]
fn ordinary_mode_does_not_promote_a_control_token_spelling() {
    let text = "understanding<mask>the";

    let ordinary = UNIGRAM
        .encode_with(text, &SpecialMode::Ordinary)
        .expect("ordinary mode never refuses");
    assert_eq!(
        ordinary,
        vec![1, 6, 9, 10, 27, 29, 17, 12, 30, 28, 16, 25, 22, 2]
    );
    assert!(!ordinary.contains(&33), "the control id must not appear");

    let matched = UNIGRAM
        .encode_with(text, &SpecialMode::All)
        .expect("all mode never refuses");
    assert_eq!(matched, vec![1, 6, 9, 10, 33, 13, 2]);
    assert_eq!(matched, UNIGRAM.encode(text), "`encode` is the `All` mode");
}

/// [`SpecialMode::Allow`] matches only the named tokens and errors on any other
/// configured special token found in the text — the failure the allow-list
/// exists to produce, rather than silently promoting `<tool>` to id 34.
#[test]
fn allow_mode_errors_on_a_special_token_outside_the_list() {
    let mut allowed = FxHashSet::default();
    allowed.insert("<mask>".to_string());

    // The listed token is matched exactly as under `All`.
    assert_eq!(
        UNIGRAM
            .encode_with("understanding<mask>the", &SpecialMode::Allow(&allowed))
            .unwrap(),
        vec![1, 6, 9, 10, 33, 13, 2]
    );

    // An unlisted one is refused, naming the token that was rejected.
    let err = UNIGRAM.encode_with("understanding<tool>the", &SpecialMode::Allow(&allowed));
    assert!(
        matches!(&err, Err(PolicyError::DisallowedSpecial { token, .. }) if token == "<tool>"),
        "expected DisallowedSpecial for <tool>, got {err:?}"
    );
}
