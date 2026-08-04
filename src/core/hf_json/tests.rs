use super::*;
use crate::core::policy::PolicyError;
// Raw backends expose `encode` only through the trait; `AnyTokenizer`'s is inherent.
use crate::core::tokenize::{Tokenize, TokenizeError};
use crate::core::{AnyTokenizer, Backend};

#[test]
fn dispatches_bpe_byte_level() {
    // Byte-level BPE with a ByteLevel pre_tokenizer and a special token.
    let json = r#"{
        "added_tokens": [{"id": 2, "content": "<|endoftext|>", "special": true}],
        "normalizer": null,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
        "model": {"type": "BPE", "vocab": {"a": 0, "Ġa": 1}, "merges": []}
    }"#;
    let tok = from_json_bytes(json.as_bytes()).expect("bpe ok");
    assert_eq!(tok.family(), "BPE");
    assert!(matches!(tok.backend(), Backend::Bpe(_)));
    // Special token is recognized.
    if let Backend::Bpe(t) = tok.backend() {
        assert_eq!(t.encode_with_special("<|endoftext|>"), vec![2]);
    }
}

#[test]
fn bpe_split_regex_is_read_from_json() {
    // A Sequence pre_tokenizer carrying both a Split regex and ByteLevel.
    let json = r#"{
        "added_tokens": [],
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": "\\w+|[^\\w\\s]+"}, "behavior": "Isolated"},
            {"type": "ByteLevel", "add_prefix_space": false}
        ]},
        "model": {"type": "BPE", "vocab": {"a": 0, "Ġa": 1}, "merges": []}
    }"#;
    let tok = from_json_bytes(json.as_bytes()).expect("bpe ok");
    assert!(matches!(tok.backend(), Backend::Bpe(_)));
}

#[test]
fn bpe_applies_nfc_normalizer_to_content() {
    // With an NFC normalizer (Qwen/GPT-NeoX style), a combining sequence
    // (e + U+0301) must compose to precomposed é before byte-level encoding, so
    // it tokenizes identically to the precomposed input.
    let json = r#"{
        "added_tokens": [],
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
        "model": {"type": "BPE",
            "vocab": {"a": 0, "Ã": 1, "©": 2, "Ã©": 3},
            "merges": ["Ã ©"]}
    }"#;
    let tok = from_json_bytes(json.as_bytes()).expect("bpe ok");
    if let Backend::Bpe(t) = tok.backend() {
        let precomposed = t.encode("é"); // U+00E9
        let combining = t.encode("e\u{301}"); // e + combining acute
        assert_eq!(precomposed, combining);
    } else {
        panic!("expected BPE backend");
    }
}

#[test]
fn rejects_non_byte_level_keys_when_byte_level() {
    let json = r#"{
        "pre_tokenizer": {"type": "ByteLevel"},
        "model": {"type": "BPE", "vocab": {"你好": 0}, "merges": []}
    }"#;
    let err = from_json_bytes(json.as_bytes());
    assert!(matches!(err, Err(HfJsonError::InvalidByteLevel(_))));
}

#[test]
fn bpe_honors_merge_order_independent_of_ids() {
    // Ids do NOT follow merge order: "ab" has the highest priority merge (rank 0)
    // but a large id (5); "bc" is lower priority (rank 1) with a smaller id (4).
    // Correct BPE of "abc" merges "ab" first → ["ab"(5), "c"(2)], proving merges
    // drive the result, not token ids (the RoBERTa case).
    let json = r#"{
        "pre_tokenizer": {"type": "ByteLevel"},
        "model": {"type": "BPE",
            "vocab": {"a":0,"b":1,"c":2,"bc":4,"ab":5},
            "merges": [["a","b"], ["b","c"]]}
    }"#;
    let Backend::Bpe(t) = from_json_bytes(json.as_bytes()).unwrap().into_backend() else {
        panic!("bpe");
    };
    assert_eq!(t.encode("abc"), vec![5, 2]);
}

#[test]
fn bpe_metaspace_pretokenizer_folds_spaces_into_underscore_prefix() {
    // A BPE model (not Unigram) whose pre_tokenizer is a bare `Metaspace` node,
    // the Mistral/Gemma/Llama-SPM shape: the vocab is `▁`-marked and
    // `prepend_scheme: "first"` means a leading space is implied on the first
    // word. Must NOT fall back to per-character/byte-fallback tokens.
    let json = r#"{
        "pre_tokenizer": {"type": "Metaspace", "prepend_scheme": "first"},
        "model": {"type": "BPE",
            "vocab": {"▁hello": 10, "▁world": 11, "hello": 12, "world": 13},
            "merges": []}
    }"#;
    let Backend::Bpe(t) = from_json_bytes(json.as_bytes()).unwrap().into_backend() else {
        panic!("expected BPE backend");
    };
    // Both words get a `▁` prefix: the implied leading space on "hello", and the
    // real space folded onto "world".
    assert_eq!(t.encode("hello world"), vec![10, 11]);
}

#[test]
fn bpe_metaspace_pretokenizer_omits_prefix_when_prepend_scheme_never() {
    // Same vocab and model, but `prepend_scheme: "never"`: no leading `▁` is
    // implied, so the first word matches its bare (non-prefixed) vocab entry
    // while the second, following a real space, still gets `▁`.
    let json = r#"{
        "pre_tokenizer": {"type": "Metaspace", "prepend_scheme": "never"},
        "model": {"type": "BPE",
            "vocab": {"▁hello": 10, "▁world": 11, "hello": 12, "world": 13},
            "merges": []}
    }"#;
    let Backend::Bpe(t) = from_json_bytes(json.as_bytes()).unwrap().into_backend() else {
        panic!("expected BPE backend");
    };
    assert_eq!(t.encode("hello world"), vec![12, 11]);
}

#[test]
fn bpe_metaspace_merges_multibyte_underscore_instead_of_byte_fallback() {
    // Mistral/Llama-SPM shape: a `▁`-marked BPE vocab with `byte_fallback`, and
    // a merge that joins `▁` onto the following word. `▁` is 3 UTF-8 bytes
    // (E2 96 81), and reassembling it from bytes would need a rank for the
    // partial prefix `E2 96`, which is never a vocab entry — so a byte-seeded
    // BPE strands it and emits `<0xE2> <0x96> <0x81>` instead of merging.
    // Character-seeded BPE (what HuggingFace `merges` operate over) merges it.
    let mut vocab = String::new();
    for b in 0..256u32 {
        // `<0x00>`..`<0xFF>` take ids 0..255, so a fallback id IS its byte value.
        vocab.push_str(&format!("\"<0x{b:02X}>\": {b}, "));
    }
    vocab.push_str(
        r#""h": 256, "e": 257, "l": 258, "o": 259, "▁": 260,
           "he": 261, "▁he": 262, "ll": 263, "llo": 264"#,
    );
    let json = format!(
        r#"{{
            "pre_tokenizer": {{"type": "Metaspace", "prepend_scheme": "first"}},
            "model": {{"type": "BPE", "byte_fallback": true,
                "vocab": {{{vocab}}},
                "merges": [["h","e"], ["▁","he"], ["l","l"], ["ll","o"]]}}
        }}"#
    );
    let Backend::Bpe(t) = from_json_bytes(json.as_bytes()).unwrap().into_backend() else {
        panic!("expected BPE backend");
    };

    let ids = t.encode("hello");
    assert_eq!(ids, vec![262, 264], "expected [▁he, llo], got {ids:?}");
    // The regression guard proper: the failure mode is `▁` shattering into its
    // three byte fallbacks, so assert those ids are absent, not just that the
    // value is right.
    for byte in [0xE2u32, 0x96, 0x81] {
        assert!(!ids.contains(&byte), "`▁` shattered into <0x{byte:02X}>");
    }
}

/// A `pre_tokenizer` of `null` (or no `pre_tokenizer` member at all) means the
/// model runs over the **whole** normalized string: HuggingFace only splits
/// when a pre-tokenizer is installed, so substituting a default pattern there
/// invents a split the file never asked for.
///
/// This is the mistral-7b-awq-int4 / mistral-7b-gptq-int4 shape: `pre_tokenizer`
/// is `null` and the metaspace transform lives in the normalizer instead
/// (`Prepend{"▁"}` then `Replace{" " → "▁"}`). Under the GPT-2 default the
/// prepended `▁` — a `\p{S}` character — was cut off the letter run behind it
/// and could never merge.
///
/// The vocabulary is synthetic but the three ids are the real ones from
/// `mistral-7b-awq-int4/tokenizer.json`, and the expectations are measured
/// against `tokenizers` 0.22.1 over that file:
///
/// | text | ids | tokens |
/// |---|---|---|
/// | `"a"`  | `[264]`         | `▁a`      |
/// | `" a"` | `[28705, 264]`  | `▁`, `▁a` |
/// | `"a "` | `[264, 28705]`  | `▁a`, `▁` |
///
/// The old behavior spelled the first of those `[28705, 28708]` (`▁`, `a`).
#[test]
fn bpe_null_pre_tokenizer_runs_the_model_over_the_whole_string() {
    // Both spellings of "there is no pre-tokenizer" must agree.
    for pre_tokenizer_member in [r#""pre_tokenizer": null,"#, ""] {
        let json = format!(
            r#"{{
                "added_tokens": [],
                {pre_tokenizer_member}
                "normalizer": {{"type": "Sequence", "normalizers": [
                    {{"type": "Prepend", "prepend": "▁"}},
                    {{"type": "Replace", "pattern": {{"String": " "}}, "content": "▁"}}
                ]}},
                "model": {{"type": "BPE",
                    "vocab": {{"▁a": 264, "▁": 28705, "a": 28708}},
                    "merges": [["▁", "a"]]}}
            }}"#
        );
        let Backend::Bpe(t) = from_json_bytes(json.as_bytes())
            .expect("a document without a pre_tokenizer loads")
            .into_backend()
        else {
            panic!("expected BPE backend");
        };
        assert_eq!(t.encode("a"), vec![264], "with {pre_tokenizer_member:?}");
        assert_eq!(t.encode(" a"), vec![28705, 264]);
        assert_eq!(t.encode("a "), vec![264, 28705]);
    }
}

/// `Metaspace` prepends its replacement to text that begins with a non-space
/// whitespace character, and suppresses it only on an existing leading
/// **space**. Guarding on whitespace in general dropped the `▁` in front of a
/// leading tab or newline.
///
/// The mistral-7b-v0.3 shape (`prepend_scheme: "first"`, `split: false`) over a
/// synthetic vocabulary carrying that file's real ids. Every row is measured
/// against `tokenizers` 0.22.1 on `mistral-7b-v0.3/tokenizer.json`:
///
/// | text | ids | tokens |
/// |---|---|---|
/// | `"\n\n\n"` | `[29473, 781, 781, 781]` | `▁`, `<0x0A>`×3 |
/// | `"\ta"`    | `[29473, 780, 29476]`    | `▁`, `<0x09>`, `a` |
/// | `"a"`      | `[1032]`                 | `▁a` |
/// | `" a"`     | `[1032]`                 | `▁a` |
/// | `"  a"`    | `[29473, 1032]`          | `▁`, `▁a` |
/// | `"a "`     | `[1032, 29473]`          | `▁a`, `▁` |
///
/// The last four rows are the ones a fix must not trade away: they already
/// agreed, and they are what pins the guard to a literal space rather than
/// making the prefix unconditional.
#[test]
fn bpe_metaspace_prepends_before_leading_non_space_whitespace() {
    let json = r#"{
        "added_tokens": [],
        "pre_tokenizer": {"type": "Metaspace", "replacement": "▁",
            "prepend_scheme": "first", "split": false},
        "model": {"type": "BPE", "byte_fallback": true, "unk_token": "<unk>",
            "vocab": {"<unk>": 0, "▁a": 1032, "<0x09>": 780, "<0x0A>": 781,
                "▁": 29473, "a": 29476},
            "merges": [["▁", "a"]]}
    }"#;
    let Backend::Bpe(t) = from_json_bytes(json.as_bytes())
        .expect("the metaspace document loads")
        .into_backend()
    else {
        panic!("expected BPE backend");
    };
    assert_eq!(t.encode("\n\n\n"), vec![29473, 781, 781, 781]);
    assert_eq!(t.encode("\ta"), vec![29473, 780, 29476]);
    assert_eq!(t.encode("a"), vec![1032]);
    assert_eq!(t.encode(" a"), vec![1032]);
    assert_eq!(t.encode("  a"), vec![29473, 1032]);
    assert_eq!(t.encode("a "), vec![1032, 29473]);
}

#[test]
fn unigram_uses_viterbi_not_greedy() {
    // Tokens: "ab"(-5), "abc"(-1), "c"(-1), plus single chars. Greedy-longest at
    // ▁? no ▁ here — use a vocab with the relevant pieces. For "abc": greedy
    // would take "ab"+? ; Viterbi maximizes total score → "abc" (-1) beats
    // "ab"(-5)+"c"(-1) = -6. Verifies max-score segmentation.
    let json = r#"{
        "pre_tokenizer": {"type": "Metaspace"},
        "model": {"type": "Unigram", "unk_id": 0, "vocab": [
            ["<unk>", 0.0], ["</s>", 0.0],
            ["▁abc", -1.0], ["▁ab", -5.0], ["c", -1.0],
            ["▁a", -3.0], ["b", -3.0]
        ]}
    }"#;
    let Backend::Unigram(t) = from_json_bytes(json.as_bytes()).unwrap().into_backend() else {
        panic!("unigram");
    };
    // "▁abc" as one piece (score -1) is the max-score path.
    assert_eq!(t.encode("abc"), vec![2]);
}

#[test]
fn dispatches_unigram() {
    let json = r#"{
        "added_tokens": [
            {"id": 0, "content": "<unk>", "special": true},
            {"id": 1, "content": "</s>", "special": true}
        ],
        "model": {
            "type": "Unigram",
            "unk_id": 0,
            "vocab": [["<unk>", 0.0], ["</s>", 0.0], ["a", -1.0], ["b", -2.0]]
        }
    }"#;
    let tok = from_json_bytes(json.as_bytes()).expect("unigram ok");
    assert_eq!(tok.family(), "Unigram");
    assert!(matches!(tok.backend(), Backend::Unigram(_)));
}

#[test]
fn unigram_without_eos_loads() {
    // No </s> present: eos falls back to unk_id/0 (it only affects decode
    // skipping), so the tokenizer still loads rather than erroring.
    let json = r#"{
        "added_tokens": [],
        "model": {"type": "Unigram", "unk_id": 0, "vocab": [["<unk>", 0.0], ["a", 0.0]]}
    }"#;
    let tok = from_json_bytes(json.as_bytes()).expect("loads");
    assert_eq!(tok.family(), "Unigram");
}

#[test]
fn dispatches_wordpiece_with_lowercasing() {
    let json = r###"{
        "added_tokens": [],
        "normalizer": {"type": "BertNormalizer", "lowercase": true, "strip_accents": null},
        "pre_tokenizer": {"type": "BertPreTokenizer"},
        "model": {
            "type": "WordPiece",
            "unk_token": "[UNK]",
            "continuing_subword_prefix": "##",
            "max_input_chars_per_word": 100,
            "vocab": {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "hello": 3, "##world": 4, "world": 5}
        }
    }"###;
    let tok = from_json_bytes(json.as_bytes()).expect("wordpiece ok");
    assert_eq!(tok.family(), "WordPiece");
    let Backend::WordPiece(t) = tok.backend() else {
        panic!("expected wordpiece");
    };
    // Lowercasing means "HELLO" maps to the "hello" token.
    assert_eq!(t.encode("HELLO"), vec![3]);
    assert_eq!(t.encode("hello"), vec![3]);
}

#[test]
fn unigram_applies_replace_normalizer_in_order() {
    // Replace '' -> " must run before tokenization (albert's case). With "x" the
    // replacement target in the vocab, encoding "''" yields the "x" token, not
    // the raw quotes.
    let json = r#"{
        "normalizer": {"type": "Sequence", "normalizers": [
            {"type": "Replace", "pattern": {"String": "''"}, "content": "x"}
        ]},
        "pre_tokenizer": {"type": "Metaspace"},
        "added_tokens": [{"id": 1, "content": "</s>", "special": true}],
        "model": {"type": "Unigram", "unk_id": 0, "vocab": [
            ["<unk>", 0.0], ["</s>", 0.0], ["▁x", -1.0], ["▁", -2.0], ["x", -3.0]
        ]}
    }"#;
    let Backend::Unigram(t) = from_json_bytes(json.as_bytes()).unwrap().into_backend() else {
        panic!("unigram");
    };
    // "''" normalizes to "x" → "▁x" piece (id 2). Without the Replace it would be
    // unknown quotes. Confirms the normalizer ran.
    assert_eq!(t.encode("''"), vec![2]);
}

#[test]
fn wordpiece_custom_continuation_prefix() {
    // A model whose continuation prefix is "@@" rather than "##".
    let json = r###"{
        "model": {"type": "WordPiece", "unk_token": "[UNK]",
            "continuing_subword_prefix": "@@", "max_input_chars_per_word": 100,
            "vocab": {"[UNK]": 0, "foo": 1, "@@bar": 2}}
    }"###;
    let Backend::WordPiece(t) = from_json_bytes(json.as_bytes()).unwrap().into_backend() else {
        panic!("wordpiece");
    };
    // "foobar" → "foo" + "@@bar"
    assert_eq!(t.encode("foobar"), vec![1, 2]);
}

#[test]
fn bpe_matches_added_tokens_in_encode() {
    // A non-special added token must be recognized in the input during `encode`
    // (HF always matches added tokens), not BPE'd into its characters.
    let json = r#"{
        "added_tokens": [{"id": 2, "content": "<sp>", "special": false}],
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
        "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "<sp>": 2}, "merges": []}
    }"#;
    let tok = from_json_bytes(json.as_bytes()).unwrap();
    // "a<sp>b" → a(0), matched <sp>(2), b(1).
    assert_eq!(tok.encode("a<sp>b"), vec![0, 2, 1]);
}

#[test]
fn wordpiece_matches_added_tokens_in_input() {
    // `[SEP]` in the input must be recognized as its id, not split into subwords.
    let json = r###"{
        "added_tokens": [{"id": 2, "content": "[SEP]", "special": true}],
        "model": {"type": "WordPiece", "unk_token": "[UNK]",
            "continuing_subword_prefix": "##", "max_input_chars_per_word": 100,
            "vocab": {"[UNK]": 0, "a": 1, "[SEP]": 2, "b": 3}}
    }"###;
    let tok = from_json_bytes(json.as_bytes()).unwrap();
    assert_eq!(tok.encode("a [SEP] b"), vec![1, 2, 3]);
}

#[test]
fn unigram_matches_added_tokens_in_input() {
    // `</s>` in the input is matched as its id (not Viterbi-segmented).
    let json = r#"{
        "added_tokens": [{"id": 1, "content": "</s>", "special": true}],
        "pre_tokenizer": {"type": "Metaspace"},
        "model": {"type": "Unigram", "unk_id": 0, "vocab": [
            ["<unk>", 0.0], ["</s>", 0.0], ["▁a", -1.0], ["▁b", -1.0]
        ]}
    }"#;
    let tok = from_json_bytes(json.as_bytes()).unwrap();
    // "a</s>b" → ▁a(2), </s>(1), ▁b(3)
    assert_eq!(tok.encode("a</s>b"), vec![2, 1, 3]);
}

/// A BERT-style tokenizer: `encode` is the model's real input (`[CLS] hi
/// [SEP]`), and only `encode_raw` gives the bare content tokens. The default is
/// the safe one — forgetting to wrap can no longer happen by omission.
#[test]
fn post_processor_wraps_with_special_tokens() {
    let tok = from_json_bytes(BERT_PAIR_JSON.as_bytes()).unwrap();
    assert_eq!(tok.encode_raw("hi"), vec![3]); // content only
    assert_eq!(tok.encode("hi"), vec![1, 3, 2]); // [CLS] hi [SEP]
}

/// A BERT-style `tokenizer.json` whose post-processor defines no `pair` array —
/// the pair template is synthesized from the cls/sep ids.
const BERT_PAIR_JSON: &str = r###"{
    "post_processor": {"type": "BertProcessing", "cls": ["[CLS]", 1], "sep": ["[SEP]", 2]},
    "model": {"type": "WordPiece", "unk_token": "[UNK]",
        "continuing_subword_prefix": "##", "max_input_chars_per_word": 100,
        "vocab": {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "hi": 3, "yo": 4}}
}"###;

/// The reranker case: two segments joined the way the model was trained,
/// `[CLS] a [SEP] b [SEP]`, without the caller hand-placing anything.
#[test]
fn encode_pair_applies_the_bert_pair_template() {
    let tok = from_json_bytes(BERT_PAIR_JSON.as_bytes()).unwrap();
    assert_eq!(tok.encode_pair("hi", "yo").unwrap(), vec![1, 3, 2, 4, 2]);
}

/// Without a pair template there is no sound way to join two sequences, so
/// `encode_pair` errors rather than concatenating them separator-less.
#[test]
fn encode_pair_without_a_template_errors() {
    let json = r#"{
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
        "model": {"type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": []}
    }"#;
    let tok = from_json_bytes(json.as_bytes()).unwrap();
    assert!(matches!(
        tok.encode_pair("a", "b"),
        Err(PolicyError::NoPairTemplate)
    ));
}

/// The policy answers "what is EOS" and "what id is this special token" so
/// downstream consumers stop re-deriving them from the model card.
#[test]
fn policy_exposes_named_specials_and_eos() {
    let tok = from_json_bytes(
        r###"{
            "added_tokens": [
                {"id": 1, "content": "[CLS]", "special": true},
                {"id": 2, "content": "[SEP]", "special": true}
            ],
            "model": {"type": "WordPiece", "unk_token": "[UNK]",
                "continuing_subword_prefix": "##", "max_input_chars_per_word": 100,
                "vocab": {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "hi": 3}}
        }"###
            .as_bytes(),
    )
    .unwrap();
    assert_eq!(tok.special_token_id("[CLS]"), Some(1));
    assert_eq!(tok.eos_token_id(), Some(2));
    assert!(tok.is_eos(2));
}

#[test]
fn decode_skips_special_keeps_nonspecial_added_tokens() {
    // Byte-level BPE with one special added token (id 2) and one non-special
    // added token (id 3). Decode drops the special, keeps the non-special.
    let json = r#"{
        "added_tokens": [
            {"id": 2, "content": "<|end|>", "special": true},
            {"id": 3, "content": "<sp>", "special": false}
        ],
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
        "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "<|end|>": 2, "<sp>": 3}, "merges": []}
    }"#;
    let tok = from_json_bytes(json.as_bytes()).unwrap();
    let ids = tok.encode("a<|end|><sp>b"); // [0, 2, 3, 1]
    assert_eq!(ids, vec![0, 2, 3, 1]);
    // Decode drops <|end|> (special) but keeps <sp> (non-special).
    assert_eq!(Tokenize::decode(&tok, &ids).unwrap(), "a<sp>b");
}

#[test]
fn unsupported_model_type_errors() {
    let json = r#"{"model": {"type": "Phantom", "vocab": {}}}"#;
    let err = from_json_bytes(json.as_bytes());
    assert!(matches!(err, Err(HfJsonError::UnsupportedModelType(t)) if t == "Phantom"));
}

#[test]
fn missing_model_errors() {
    let err = from_json_bytes(b"{}");
    assert!(matches!(err, Err(HfJsonError::MissingField("model"))));
}

// Real-world tokenizer.json files frequently omit `model.type`; the family must
// be inferred from the model's shape.
#[test]
fn infers_unigram_without_model_type() {
    let json = r#"{
        "added_tokens": [{"id": 1, "content": "</s>", "special": true}],
        "model": {"unk_id": 0, "vocab": [["<unk>", 0.0], ["</s>", 0.0], ["x", -1.0]]}
    }"#;
    let tok = from_json_bytes(json.as_bytes()).expect("inferred unigram");
    assert_eq!(tok.family(), "Unigram");
}

#[test]
fn infers_wordpiece_without_model_type() {
    let json = r#"{
        "model": {
            "unk_token": "[UNK]", "continuing_subword_prefix": "@@",
            "max_input_chars_per_word": 100, "vocab": {"[UNK]": 0, "hi": 1}
        }
    }"#;
    let tok = from_json_bytes(json.as_bytes()).expect("inferred wordpiece");
    assert_eq!(tok.family(), "WordPiece");
}

#[test]
fn infers_bpe_without_model_type() {
    let json = r#"{
        "pre_tokenizer": {"type": "ByteLevel"},
        "model": {"vocab": {"a": 0, "Ġa": 1}, "merges": []}
    }"#;
    let tok = from_json_bytes(json.as_bytes()).expect("inferred bpe");
    assert_eq!(tok.family(), "BPE");
}

#[test]
fn unknown_normalizer_errors_not_silently_dropped() {
    // An unrecognized normalizer step would change the tokens if dropped, so the
    // loader must refuse rather than silently skip it.
    let json = r#"{
        "normalizer": {"type": "Sequence", "normalizers": [
            {"type": "NFC"},
            {"type": "SomeFutureNormalizer"}
        ]},
        "pre_tokenizer": {"type": "ByteLevel"},
        "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []}
    }"#;
    let err = from_json_bytes(json.as_bytes());
    assert!(
        matches!(&err, Err(HfJsonError::UnsupportedNormalizer(t)) if t.contains("SomeFutureNormalizer"))
    );
}

#[test]
fn uncompilable_replace_regex_errors_not_literal() {
    // A Replace regex that won't compile must error, not fall back to a literal
    // replacement of the pattern source (which would mis-normalize).
    let json = r#"{
        "normalizer": {"type": "Replace", "pattern": {"Regex": "(?P<"}, "content": "x"},
        "pre_tokenizer": {"type": "ByteLevel"},
        "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []}
    }"#;
    let err = from_json_bytes(json.as_bytes());
    assert!(matches!(&err, Err(HfJsonError::InvalidNormalizerRegex(_))));
}

#[test]
fn unknown_pretokenizer_without_recognized_split_errors() {
    // No ByteLevel/Metaspace/Split to anchor the split, only an unknown type:
    // defaulting to the GPT-2 pattern would be a silent guess, so refuse.
    let json = r#"{
        "pre_tokenizer": {"type": "UnicodeScripts"},
        "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []}
    }"#;
    let err = from_json_bytes(json.as_bytes());
    assert!(
        matches!(&err, Err(HfJsonError::UnsupportedPreTokenizer(t)) if t.contains("UnicodeScripts"))
    );
}

#[test]
fn engine_handled_pretokenizer_without_bytelevel_still_loads() {
    // A `Digits` pre-tokenizer is handled by the multi-stage engine even though
    // the simple distiller doesn't "anchor" it (no ByteLevel/Metaspace/Split).
    // The guess guard must NOT reject it (it would be a false rejection).
    let json = r#"{
        "pre_tokenizer": {"type": "Digits", "individual_digits": true},
        "model": {"type": "BPE", "vocab": {"a": 0, "1": 1}, "merges": []}
    }"#;
    let tok = from_json_bytes(json.as_bytes()).expect("digits pre-tokenizer loads");
    assert_eq!(tok.family(), "BPE");
}

/// A *declared but empty* pipeline is not a reason to guess the GPT-2 default
/// either: HuggingFace runs the model over the whole normalized string when no
/// stage is installed, and an empty `Sequence` installs no stage.
///
/// Ground truth from `tokenizers` 0.22.1 on this exact document. `"ab1"` is the
/// discriminating input: the GPT-2 pattern cuts between the letters and the
/// digit, so `ab1` (id 5) can only form if nothing splits.
///
/// | `pre_tokenizer` | reference |
/// |---|---|
/// | `null` | `['ab1']` = `[5]` |
/// | `{"type":"Sequence","pretokenizers":[]}` | `['ab1']` = `[5]` |
/// | one nested inside another | `['ab1']` = `[5]` |
///
/// splintr used to answer `[4, 3]` (`ab`, `1`) for the latter two.
#[test]
fn declared_but_empty_pretokenizer_sequence_does_not_split() {
    for pre in [
        "null",
        r#"{"type": "Sequence", "pretokenizers": []}"#,
        r#"{"type": "Sequence", "pretokenizers": [
            {"type": "Sequence", "pretokenizers": []}
        ]}"#,
    ] {
        let json = format!(
            r#"{{
                "pre_tokenizer": {pre},
                "model": {{"type": "BPE", "unk_token": "<unk>",
                    "vocab": {{"<unk>": 0, "a": 1, "b": 2, "1": 3, "ab": 4, "ab1": 5}},
                    "merges": ["a b", "ab 1"]}}
            }}"#
        );
        let tok = from_json_bytes(json.as_bytes()).expect("an empty pipeline is loadable");
        assert_eq!(tok.encode("ab1ab1"), vec![5, 5], "with pre_tokenizer {pre}");
    }
}

/// The other declared-but-inert shapes must NOT become a silent no-split — the
/// distinction is "no stages to run" versus "a stage this loader cannot read".
///
/// Every shape below is a hard load failure in `tokenizers` 0.22.1 (measured: a
/// `Sequence` missing `pretokenizers` and a `Split` missing `pattern` both fail
/// with `missing field`, and a node with no `type` or an unknown `type` fails to
/// match any pre-tokenizer variant), so refusing them is what agrees with the
/// reference. Two of the four already refused; the `Split`-without-`pattern` and
/// the typeless node used to reach the GPT-2 default silently.
#[test]
fn declared_but_unreadable_pretokenizers_are_refused_not_treated_as_empty() {
    for pre in [
        r#"{"type": "Sequence"}"#,
        r#"{"type": "Split", "behavior": "Isolated"}"#,
        r#"{"foo": "bar"}"#,
        r#"{"type": "SomeFuturePreTokenizer"}"#,
        r#"{"type": "Sequence", "pretokenizers": [{"type": "Split", "behavior": "Isolated"}]}"#,
    ] {
        let json = format!(
            r#"{{
                "pre_tokenizer": {pre},
                "model": {{"type": "BPE", "vocab": {{"a": 0, "b": 1, "ab": 2}},
                    "merges": ["a b"]}}
            }}"#
        );
        assert!(
            matches!(
                from_json_bytes(json.as_bytes()),
                Err(HfJsonError::UnsupportedPreTokenizer(_))
            ),
            "pre_tokenizer {pre} must be refused, not guessed at"
        );
    }
}

#[test]
fn unknown_pretokenizer_is_ok_when_split_is_anchored() {
    // The same unknown type alongside a ByteLevel (which fixes the split) is
    // harmless and must still load.
    let json = r#"{
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "UnicodeScripts"},
            {"type": "ByteLevel", "add_prefix_space": false}
        ]},
        "model": {"type": "BPE", "vocab": {"a": 0, "Ġa": 1}, "merges": []}
    }"#;
    let tok = from_json_bytes(json.as_bytes()).expect("loads with anchored split");
    assert_eq!(tok.family(), "BPE");
}

#[test]
fn added_token_lstrip_is_read_from_json_and_reaches_the_matcher() {
    // `<mask>` declares `lstrip: true` and `<pad>` does not — the shape of every
    // XLM-RoBERTa-family vocabulary (bge-m3 and friends). The space before
    // `<mask>` must be absorbed into it, while the one before `<pad>` survives
    // as its own piece, so the flags can only be right if they are carried per
    // token from the json all the way into the shared matcher.
    //
    // Reference (`tokenizers` 0.22.1, bge-m3, add_special_tokens=False):
    // "end. <mask>x" -> [3564, 5, 250001, 1022]; splintr used to emit the lone
    // `▁` piece (id 6) between the two.
    let json = r#"{
        "added_tokens": [
            {"id": 10, "content": "<mask>", "special": true, "lstrip": true, "rstrip": false},
            {"id": 11, "content": "<pad>", "special": true, "lstrip": false, "rstrip": false}
        ],
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
        "model": {"type": "BPE", "vocab": {"a": 0, "Ġ": 1, "b": 2}, "merges": []}
    }"#;
    let tok = from_json_bytes(json.as_bytes()).expect("bpe ok");
    let Backend::Bpe(t) = tok.backend() else {
        panic!("expected BPE backend");
    };
    // lstrip: the space between "a" and <mask> never becomes token 1.
    assert_eq!(t.encode("a <mask>b"), vec![0, 10, 2]);
    // Same input shape, unflagged token: the space stays.
    assert_eq!(t.encode("a <pad>b"), vec![0, 1, 11, 2]);
}

/// A `model.vocab` entry that is ALSO an `added_tokens` entry is spelled
/// literally, so it must be taken literally rather than byte-level-decoded.
///
/// This is DeepSeek V3's shape: its `tokenizer.json` declares 818 added tokens
/// and 3 of them (`<｜begin▁of▁sentence｜>`, `<｜end▁of▁sentence｜>`,
/// `<｜▁pad▁｜>`, ids 0/1/2) also occupy a vocab slot. `｜` is U+FF5C, outside
/// the byte-level alphabet, so byte-level-decoding those entries fails and the
/// whole 128000-entry file used to be unloadable.
///
/// Reference (`tokenizers` 0.22.1, deepseek-v3 `tokenizer.json`,
/// `add_special_tokens=False`): `"Hello <｜begin▁of▁sentence｜> world"` ->
/// `[19923, 223, 0, 2058]`, and decoding that back with
/// `skip_special_tokens=False` returns the literal string — the added token is
/// never run through the ByteLevel decoder.
#[test]
fn vocab_entry_that_is_also_an_added_token_is_taken_literally() {
    let json = r#"{
        "added_tokens": [
            {"id": 3, "content": "<｜eos｜>", "special": true},
            {"id": 4, "content": "<｜tool｜>", "special": false}
        ],
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
        "decoder": {"type": "ByteLevel"},
        "model": {"type": "BPE",
            "vocab": {"a": 0, "Ġ": 1, "b": 2, "<｜eos｜>": 3, "<｜tool｜>": 4},
            "merges": []}
    }"#;
    let tok = from_json_bytes(json.as_bytes()).expect("loads despite literal vocab entries");

    // Encoding keeps the id `model.vocab` gives the entry.
    assert_eq!(tok.encode_raw("a<｜tool｜>b"), vec![0, 4, 2]);
    assert_eq!(tok.encode_raw("<｜eos｜>"), vec![3]);

    // Decoding round-trips to the literal spelling, not byte-level garbage: the
    // ByteLevel decoder passes a non-byte-level surface through unchanged.
    assert_eq!(
        Tokenize::decode(&tok, &[0, 4, 2]).expect("decodes"),
        "a<｜tool｜>b"
    );
    // `special: true` is still dropped on decode (HF's default
    // `skip_special_tokens=true`; `tokenizers` returns "" for deepseek's [0,1,2]).
    assert_eq!(Tokenize::decode(&tok, &[3]).expect("decodes"), "");
}

/// The literal-spelling exemption is per entry, keyed on membership in
/// `added_tokens` — never a blanket "ignore byte-level decode failures". A vocab
/// entry that is NOT an added token and fails to decode means a corrupt
/// vocabulary, so it must still be a hard error even in a file that does declare
/// added tokens.
#[test]
fn non_added_token_vocab_entry_that_is_not_byte_level_still_errors() {
    let json = r#"{
        "added_tokens": [{"id": 3, "content": "<｜eos｜>", "special": true}],
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
        "model": {"type": "BPE",
            "vocab": {"a": 0, "<｜eos｜>": 3, "你好": 5},
            "merges": []}
    }"#;
    let err = from_json_bytes(json.as_bytes());
    assert!(matches!(&err, Err(HfJsonError::InvalidByteLevel(t)) if t == "你好"));
}

/// When both sections claim a token but give it different ids, neither can win:
/// the added-token matcher would emit the `added_tokens` id while BPE and the
/// decode tables use the `model.vocab` id, so the tokenizer's encode and decode
/// would contradict each other on that token. Report it instead of choosing.
///
/// (No local `tokenizer.json` — deepseek-v3, llama-3.2, mistral-7b-v0.3,
/// embeddinggemma, whisper — actually disagrees, so this rejects only genuinely
/// inconsistent files.)
#[test]
fn added_token_id_disagreeing_with_the_vocab_id_errors() {
    let json = r#"{
        "added_tokens": [{"id": 7, "content": "<｜eos｜>", "special": true}],
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
        "model": {"type": "BPE", "vocab": {"a": 0, "<｜eos｜>": 3}, "merges": []}
    }"#;
    let err = from_json_bytes(json.as_bytes());
    assert!(matches!(
        &err,
        Err(HfJsonError::AddedTokenIdConflict {
            vocab_id: 3,
            added_id: 7,
            ..
        })
    ));
}

/// Builds the 256 `"<0xNN>": id` vocab entries (uppercase hex, ids starting at
/// `start_id`) that spell a *complete* byte-fallback set. A complete set is no
/// longer required — `byte_fallback_from_encoder` keeps a partial one — but it
/// is what the real full-coverage vocabularies (mistral-7b, embeddinggemma)
/// declare, so the tests that pin their behavior need all 256 spelled exactly
/// right, which is easier to get right here than in a literal JSON blob.
fn byte_fallback_vocab_entries(start_id: u32) -> String {
    (0u32..256)
        .map(|b| format!(r#""<0x{b:02X}>": {}"#, start_id + b))
        .collect::<Vec<_>>()
        .join(", ")
}

/// `model.byte_fallback: true` with a complete `<0xNN>` table: a byte the raw
/// (non-byte-level) BPE vocabulary cannot represent — here `b` (0x62), absent
/// from the ordinary vocab entries — is emitted through the derived table
/// instead of being dropped, in the correct position among the surrounding
/// resolved tokens.
#[test]
fn byte_fallback_true_emits_the_fallback_id_for_an_unrepresented_byte() {
    let json = format!(
        r#"{{
            "model": {{"type": "BPE", "byte_fallback": true,
                "vocab": {{"a": 0, "c": 1, {}}},
                "merges": []}}
        }}"#,
        byte_fallback_vocab_entries(2)
    );
    let Backend::Bpe(t) = from_json_bytes(json.as_bytes())
        .expect("loads with a complete byte_fallback table")
        .into_backend()
    else {
        panic!("expected BPE backend");
    };
    // 'b' is 0x62; its fallback id is 2 (start_id) + 0x62 (98) = 100.
    assert_eq!(t.encode("abc"), vec![0, 100, 1]);
}

/// Same vocab and text as above, but `byte_fallback` is `false` (and, in a
/// second case, entirely absent): the unrepresented byte is dropped, pinning
/// that the `<0xNN>` branch is opt-in and this file's defaults are unaffected.
///
/// The document declares no `model.unk_token`, which is what makes the *drop*
/// the answer rather than an unk — the unk branch is not gated on the flag (see
/// `unk_fallback_is_not_gated_on_the_byte_fallback_flag`), so this test pins the
/// flag-off half only in the absence of an unk. `tokenizers` 0.22.1 agrees:
/// with neither an unk nor the flag, `encode('abc')` → `['a', 'c']`.
#[test]
fn byte_fallback_false_or_absent_still_drops_the_unrepresented_byte() {
    for model_fragment in [
        r#""byte_fallback": false, "vocab""#,
        // omitted entirely
        r#""vocab""#,
    ] {
        let json = format!(
            r#"{{
                "model": {{"type": "BPE", {model_fragment}: {{"a": 0, "c": 1, {}}},
                    "merges": []}}
            }}"#,
            byte_fallback_vocab_entries(2)
        );
        let Backend::Bpe(t) = from_json_bytes(json.as_bytes())
            .expect("loads")
            .into_backend()
        else {
            panic!("expected BPE backend");
        };
        // 'b' (0x62) has no vocab entry, no `<0xNN>` table (the flag is off)
        // and no unk to fall back to, so it is silently dropped: today's
        // pre-existing behavior. With nothing on either half, no fallback is
        // configured at all.
        assert!(!t.has_byte_fallback());
        assert_eq!(t.encode("abc"), vec![0, 1]);
    }
}

/// The unk branch is NOT gated on `model.byte_fallback`. A file declaring a
/// resolvable `model.unk_token` renders an unrepresentable piece as that unk
/// whether the flag is `true`, `false`, or absent — so all three spellings of
/// this document must agree.
///
/// Ground truth from `tokenizers` 0.22.1, not from splintr's own output: on a
/// `{"<unk>": 0, "▁": 1, "▁hello": 2}` vocab with `unk_token: "<unk>"`, the two
/// flag settings encode the same input to the *identical* token sequence
/// (`['▁', '<unk>', '<unk>', '<unk>', '<unk>', '<unk>', '▁', '<unk>']` both
/// times), while dropping `unk_token` from the same document collapses it to
/// `['▁', '▁']`. Here that rule shows as `encode('abc')` → `['a', '<unk>',
/// 'c']` = `[1, 0, 2]` in all three cases — before this was fixed, the
/// flag-off cases dropped the `b` entirely.
#[test]
fn unk_fallback_is_not_gated_on_the_byte_fallback_flag() {
    for model_fragment in [
        r#""byte_fallback": true,"#,
        r#""byte_fallback": false,"#,
        // omitted entirely
        "",
    ] {
        let json = format!(
            r#"{{
                "model": {{"type": "BPE", {model_fragment} "unk_token": "<unk>",
                    "vocab": {{"<unk>": 0, "a": 1, "c": 2}},
                    "merges": []}}
            }}"#
        );
        let Backend::Bpe(t) = from_json_bytes(json.as_bytes())
            .expect("loads")
            .into_backend()
        else {
            panic!("expected BPE backend");
        };
        assert!(t.has_byte_fallback());
        assert_eq!(t.encode("abc"), vec![1, 0, 2]);
    }
}

/// `model.fuse_unk` collapses a RUN of unk-resolved characters into a single
/// unk id. It was never read, so a file declaring it emitted one unk per
/// unrepresentable character where HuggingFace emits one per run.
///
/// Ground truth from `tokenizers` 0.22.1 on this exact document —
/// `{"<unk>": 0, "a": 1, "<0x7A>": 2, "b": 3, "ab": 4}`, `unk_token: "<unk>"`,
/// `byte_fallback: true`, no `pre_tokenizer` — so `z` (0x7A) is the one
/// unrepresentable character with a `<0xNN>` entry and `x`/`y` have none:
///
/// | input | `fuse_unk: false` | `fuse_unk: true` |
/// |---|---|---|
/// | `"xxzxx"` | `[0, 2, 0, 0, 0]` | `[2, 0]` |
/// | `"xzx"`   | `[2, 0, 0]`       | `[2, 0]` |
/// | `"axyzb"` | `[1, 0, 2, 0, 3]` | `[1, 2, 0, 3]` |
/// | `"ééé"`   | `[0, 0, 0]`       | `[0]` |
/// | `"xax"`   | `[0, 1, 0]`       | `[0, 1, 0]` |
///
/// The last row is the load-bearing one: fusing does **not** cross a
/// *vocabulary* hit, so `a` still ends the run — while the `<0x7A>` rows show it
/// does cross a `<0xNN>` hit, which is the deliberately-reproduced HuggingFace
/// quirk (a pending unk is flushed by a vocabulary hit and never by a byte one)
/// surviving intact under the flag.
#[test]
fn fuse_unk_collapses_a_run_of_unks_into_one() {
    let cases = [
        ("xxzxx", vec![0, 2, 0, 0, 0], vec![2, 0]),
        ("xzx", vec![2, 0, 0], vec![2, 0]),
        ("axyzb", vec![1, 0, 2, 0, 3], vec![1, 2, 0, 3]),
        ("ééé", vec![0, 0, 0], vec![0]),
        ("xax", vec![0, 1, 0], vec![0, 1, 0]),
    ];
    // The third spelling omits the field: `tokenizers` defaults it to false
    // (measured — an omitting file encodes `"xyz"` as three `<unk>`s), so it
    // must behave as the explicit `false`.
    for (fragment, fused) in [
        (r#""fuse_unk": true,"#, true),
        (r#""fuse_unk": false,"#, false),
        ("", false),
    ] {
        let json = format!(
            r#"{{
                "model": {{"type": "BPE", "byte_fallback": true, {fragment}
                    "unk_token": "<unk>",
                    "vocab": {{"<unk>": 0, "a": 1, "<0x7A>": 2, "b": 3, "ab": 4}},
                    "merges": ["a b"]}}
            }}"#
        );
        let Backend::Bpe(t) = from_json_bytes(json.as_bytes())
            .expect("loads")
            .into_backend()
        else {
            panic!("expected BPE backend");
        };
        for (text, unfused_ids, fused_ids) in &cases {
            let expected = if fused { fused_ids } else { unfused_ids };
            assert_eq!(&t.encode(text), expected, "{text:?} with {fragment:?}");
        }
    }
}

/// A ByteLevel BPE model (GPT-2 style) with a resolvable `model.unk_token`
/// must NOT report a fallback: `Tokenizer::bpe` discards `byte_fallback`
/// outright whenever `use_byte_level` is true (the `<0xNN>` table is keyed by
/// RAW byte value, the wrong space once input has been byte-level-encoded),
/// so a `Some` here would be dead weight that `has_byte_fallback()` reports as
/// live despite never firing. Pins that `build_bpe` skips constructing it for
/// this shape rather than only relying on the encode-time gate.
#[test]
fn byte_level_bpe_never_reports_a_fallback_even_with_a_resolvable_unk() {
    let json = r#"{
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
        "model": {"type": "BPE", "unk_token": "<|endoftext|>",
            "vocab": {"<|endoftext|>": 0, "a": 1, "Ġa": 2},
            "merges": []}
    }"#;
    let Backend::Bpe(t) = from_json_bytes(json.as_bytes())
        .expect("loads")
        .into_backend()
    else {
        panic!("expected BPE backend");
    };
    assert!(!t.has_byte_fallback());
}

/// The `<0xNN>` branch, by contrast, IS gated on the flag — and the gate is on
/// the flag alone, not on whether the vocabulary happens to spell `<0xNN>`
/// tokens. With `<0x7A>` present in the vocab, `byte_fallback: false` still
/// resolves `z` to the *unk* id; only `byte_fallback: true` reaches the byte
/// token. Both directions are asserted here, in one place, so the flag cannot
/// be ignored either way.
///
/// Ground truth from `tokenizers` 0.22.1, not from splintr's own output: on the
/// `{"<unk>": 0, "a": 1, "<0x7A>": 2}` vocab below with `unk_token: "<unk>"`,
/// `encode("az")` → `['a', '<0x7A>']` = `[1, 2]` with the flag true, and
/// `['a', '<unk>']` = `[1, 0]` with the flag false.
#[test]
fn byte_fallback_flag_gates_only_the_byte_token_branch() {
    for (flag, expected) in [(true, vec![1, 2]), (false, vec![1, 0])] {
        let json = format!(
            r#"{{
                "model": {{"type": "BPE", "byte_fallback": {flag}, "unk_token": "<unk>",
                    "vocab": {{"<unk>": 0, "a": 1, "<0x7A>": 2}},
                    "merges": []}}
            }}"#
        );
        let Backend::Bpe(t) = from_json_bytes(json.as_bytes())
            .expect("loads")
            .into_backend()
        else {
            panic!("expected BPE backend");
        };
        assert_eq!(t.encode("az"), expected, "byte_fallback: {flag}");
    }
}

/// `model.byte_fallback: true` with only *some* of the 256 `<0xNN>` entries is
/// a valid file, not a malformed one: HuggingFace resolves fallback per
/// character, so it loads and each byte is resolved on its own — `<0x78>` for
/// `x`, which is declared, and `model.unk_token`'s id for `b`, which is not.
///
/// Ground truth from `tokenizers` 0.22.1 on this same document:
/// `encode('abxbc')` → `['a', '<0x78>', '<unk>', '<unk>', 'c']` = `[1, 3, 0, 0,
/// 2]`. (An earlier revision rejected the whole file here with
/// `MissingSpecial("byte_fallback")`, which refused files HuggingFace accepts.)
#[test]
fn byte_fallback_true_with_a_partial_table_loads_and_resolves_per_byte() {
    let json = r#"{
        "model": {"type": "BPE", "byte_fallback": true, "unk_token": "<unk>",
            "vocab": {"<unk>": 0, "a": 1, "c": 2, "<0x78>": 3},
            "merges": []}
    }"#;
    let Backend::Bpe(t) = from_json_bytes(json.as_bytes())
        .expect("a partial byte_fallback set is not a load failure")
        .into_backend()
    else {
        panic!("expected BPE backend");
    };
    assert!(t.has_byte_fallback());
    assert_eq!(t.encode("abxbc"), vec![1, 3, 0, 0, 2]);
}

/// The same partial document with `<0x62>` added: that one byte flips from the
/// unk id to its own `<0xNN>` id and nothing else moves. This is what pins
/// per-byte resolution as opposed to an all-or-nothing table — a 256-entry set
/// is just the case where the `<0xNN>` branch always wins.
///
/// Ground truth from `tokenizers` 0.22.1: `encode('abxbc')` → `['a', '<0x62>',
/// '<0x78>', '<0x62>', 'c']` = `[1, 4, 3, 4, 2]`.
#[test]
fn declaring_one_more_byte_token_flips_only_that_byte() {
    let json = r#"{
        "model": {"type": "BPE", "byte_fallback": true, "unk_token": "<unk>",
            "vocab": {"<unk>": 0, "a": 1, "c": 2, "<0x78>": 3, "<0x62>": 4},
            "merges": []}
    }"#;
    let Backend::Bpe(t) = from_json_bytes(json.as_bytes())
        .expect("loads")
        .into_backend()
    else {
        panic!("expected BPE backend");
    };
    assert_eq!(t.encode("abxbc"), vec![1, 4, 3, 4, 2]);
}

/// `model.byte_fallback: true` with neither any `<0xNN>` entry nor a resolvable
/// `model.unk_token`: there is nothing to fall back to, so no fallback is
/// configured at all and the unrepresentable byte is dropped — the documented
/// no-fallback contract rather than a silently wrong id. `tokenizers` 0.22.1
/// agrees: `encode('abc')` → `['a', 'c']`.
#[test]
fn byte_fallback_true_without_byte_tokens_or_unk_drops_the_byte() {
    let json = r#"{
        "model": {"type": "BPE", "byte_fallback": true,
            "vocab": {"a": 1, "c": 2}, "merges": []}
    }"#;
    let Backend::Bpe(t) = from_json_bytes(json.as_bytes())
        .expect("loads")
        .into_backend()
    else {
        panic!("expected BPE backend");
    };
    assert!(!t.has_byte_fallback());
    assert_eq!(t.encode("abc"), vec![1, 2]);
}

/// D18: `build_bpe` reads `model.unk_token`, so a declared unk that is NOT the
/// conventional `<unk>` spelling still resolves — the fallback id follows the
/// declaration rather than a hardcoded name.
#[test]
fn byte_fallback_honors_a_non_default_unk_token_spelling() {
    let json = r#"{
        "model": {"type": "BPE", "byte_fallback": true, "unk_token": "[MISSING]",
            "vocab": {"[MISSING]": 9, "a": 1, "c": 2},
            "merges": []}
    }"#;
    let Backend::Bpe(t) = from_json_bytes(json.as_bytes())
        .expect("loads")
        .into_backend()
    else {
        panic!("expected BPE backend");
    };
    assert_eq!(t.encode("abc"), vec![1, 9, 2]);
}

/// D23: a `<0xNN>` id decodes to the byte it denotes, so the bare BPE backend
/// agrees with the declared `ByteFallback` decoder op instead of rendering the
/// token's literal vocabulary spelling.
///
/// The two paths are genuinely independent: `AnyTokenizer::decode` runs the
/// json's declared decoder chain over token *surfaces*, while the backend's own
/// `decode` resolves the id through the tokenizer's encode-side `<0xNN>` table.
/// Only the second was broken. Measured on
/// `mistral-7b-v0.3/tokenizer.json` before this fix, encoding `𐍈` gives ids
/// whose surfaces are `["▁", "<0xF0>", "<0x90>", "<0x8D>", "<0x88>"]` and
/// decoding them back gave `Ok("𐍈")` through `AnyTokenizer::decode` but
/// `Ok(" <0xF0><0x90><0x8D><0x88>")` through the bare backend. Those model
/// files live outside the repository, so the agreement is pinned here on a
/// synthetic vocabulary of the same shape.
///
/// (The leading space of the mistral output comes from that file's declared
/// `Strip{start: 1}` op and is a separate concern; this document declares no
/// such op, so the two paths agree exactly.)
#[test]
fn byte_fallback_ids_decode_to_bytes_agreeing_with_the_declared_decoder() {
    let json = r#"{
        "decoder": {"type": "ByteFallback"},
        "model": {"type": "BPE", "byte_fallback": true, "unk_token": "<unk>",
            "vocab": {"<unk>": 0, "a": 1, "c": 2,
                "<0xF0>": 3, "<0x90>": 4, "<0x8D>": 5, "<0x88>": 6},
            "merges": []}
    }"#;
    let tok = from_json_bytes(json.as_bytes()).expect("loads");
    assert!(tok.declares_decoder());

    let Backend::Bpe(bpe) = tok.backend() else {
        panic!("expected BPE backend");
    };
    // `𐍈` (U+10348) is 4 UTF-8 bytes, none of which the merge vocabulary can
    // represent, so it encodes as its four `<0xNN>` ids between `a` and `c`.
    let ids = bpe.encode("a𐍈c");
    assert_eq!(ids, vec![1, 3, 4, 5, 6, 2]);

    let bare = bpe.decode(&ids).expect("the bare backend decodes");
    assert_eq!(bare, "a𐍈c");
    assert_eq!(
        tok.decode(&ids).expect("the declared pipeline decodes"),
        bare
    );
}

// =============================================================================
// `AnyTokenizer::streaming_decoder` — the same decision `decode` makes
// =============================================================================

/// Drive a loaded tokenizer's streaming decoder over `ids` in chunks of
/// `chunk`, concatenating every emission plus the final flush.
fn stream_in_chunks(tok: &AnyTokenizer, ids: &[u32], chunk: usize) -> String {
    let mut decoder = tok.streaming_decoder().expect("this document streams");
    let mut out = String::new();
    for group in ids.chunks(chunk.max(1)) {
        let emitted = decoder.add_tokens(group).expect("the ids are all known");
        out.push_str(&emitted.unwrap_or_default());
    }
    out.push_str(&decoder.flush());
    out
}

/// The unit's proof obligation: a streamed drive is `AnyTokenizer::decode`, at
/// every chunking of the id list. Returns what both produced, so a caller can
/// also pin the exact text.
fn streams_like_decode(json: &str, ids: &[u32]) -> String {
    let tok = from_json_bytes(json.as_bytes()).expect("the document loads");
    let expected = tok.decode(ids).expect("whole-sequence decode succeeds");
    for chunk in 1..=ids.len().max(1) {
        assert_eq!(
            stream_in_chunks(&tok, ids, chunk),
            expected,
            "streamed in chunks of {chunk} over {ids:?}"
        );
    }
    expected
}

/// A byte-level BPE document declaring the `ByteLevel` decoder — GPT-2's shape.
/// `é` is spelled by two separate byte-level tokens, so the character only
/// exists once both have arrived.
const BYTE_LEVEL_JSON: &str = r#"{
    "added_tokens": [{"id": 5, "content": "<|end|>", "special": true}],
    "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false},
    "decoder": {"type": "ByteLevel"},
    "model": {"type": "BPE",
        "vocab": {"a": 0, "Ġ": 1, "Ã": 2, "©": 3, "b": 4},
        "merges": []}
}"#;

/// A `▁`-marked BPE document declaring the `Metaspace` decoder.
const METASPACE_JSON: &str = r#"{
    "added_tokens": [{"id": 9, "content": "<s>", "special": true}],
    "pre_tokenizer": {"type": "Metaspace", "prepend_scheme": "always"},
    "decoder": {"type": "Metaspace", "prepend_scheme": "always"},
    "model": {"type": "BPE",
        "vocab": {"▁Hello": 10, "▁world": 11, "▁": 12},
        "merges": []}
}"#;

/// A BERT-family document declaring the `WordPiece` decoder, cleanup included.
const WORDPIECE_JSON: &str = r###"{
    "added_tokens": [
        {"id": 1, "content": "[CLS]", "special": true},
        {"id": 2, "content": "[SEP]", "special": true}
    ],
    "normalizer": {"type": "BertNormalizer", "lowercase": false, "strip_accents": null},
    "pre_tokenizer": {"type": "BertPreTokenizer"},
    "decoder": {"type": "WordPiece", "prefix": "##", "cleanup": true},
    "model": {"type": "WordPiece", "unk_token": "[UNK]",
        "continuing_subword_prefix": "##", "max_input_chars_per_word": 100,
        "vocab": {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2,
            "hello": 3, "##world": 4, ",": 5, "world": 6}}
}"###;

/// The Llama/Mistral SentencePiece shape:
/// `Sequence[Replace, ByteFallback, Fuse, Strip]` over a `▁`-marked,
/// byte-fallback BPE vocabulary. Four of the shipping `tokenizer.json` files
/// declare exactly this chain.
const MISTRAL_JSON: &str = r#"{
    "added_tokens": [{"id": 1, "content": "<s>", "special": true}],
    "pre_tokenizer": {"type": "Metaspace", "prepend_scheme": "first"},
    "decoder": {"type": "Sequence", "decoders": [
        {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
        {"type": "ByteFallback"},
        {"type": "Fuse"},
        {"type": "Strip", "content": " ", "start": 1, "stop": 0}
    ]},
    "model": {"type": "BPE", "byte_fallback": true, "unk_token": "<unk>",
        "vocab": {"<unk>": 0, "<s>": 1, "▁Hi": 2,
            "<0xE2>": 3, "<0x82>": 4, "<0xAC>": 5,
            "▁a": 6, "<0x80>": 7, "▁b": 8, "▁": 9},
        "merges": []}
}"#;

/// Every shipping declared pipeline, streamed at every chunking, must come out
/// as `AnyTokenizer::decode` does — the whole point of the factory: a caller
/// who streams and a caller who decodes the finished sequence read the same
/// text.
#[test]
fn declared_pipelines_stream_exactly_as_they_decode() {
    // ByteLevel: `é` is ids 2 and 3, so a chunk boundary can fall inside the
    // character, and the `special = true` id 5 is dropped either way.
    assert_eq!(streams_like_decode(BYTE_LEVEL_JSON, &[0, 1, 2, 3]), "a é");
    assert_eq!(
        streams_like_decode(BYTE_LEVEL_JSON, &[0, 5, 2, 3, 4]),
        "aéb"
    );

    // Metaspace: the leading `▁` of the first token becomes the dummy prefix
    // that `prepend_scheme` strips, and the strip must be spent on the first
    // emission whichever chunk carries it.
    assert_eq!(
        streams_like_decode(METASPACE_JSON, &[10, 11]),
        "Hello world"
    );
    assert_eq!(
        streams_like_decode(METASPACE_JSON, &[9, 10, 11]),
        "Hello world"
    );
    // A token that renders nothing but the space it takes still spends the
    // strip.
    assert_eq!(streams_like_decode(METASPACE_JSON, &[12, 10]), " Hello");

    // WordPiece: `##` glues, anything else starts a word, and the per-token
    // cleanup pulls the comma back onto the previous word — across a chunk
    // boundary too, since the space it eats was emitted with the token before.
    assert_eq!(streams_like_decode(WORDPIECE_JSON, &[3, 4]), "helloworld");
    assert_eq!(
        streams_like_decode(WORDPIECE_JSON, &[1, 3, 5, 6, 2]),
        "hello, world"
    );

    // Mistral: a `<0xNN>` run reassembles into one character, an invalid run
    // becomes one U+FFFD per byte, and the leading-space strip is spent once.
    assert_eq!(streams_like_decode(MISTRAL_JSON, &[1, 2, 3, 4, 5]), "Hi€");
    assert_eq!(streams_like_decode(MISTRAL_JSON, &[6, 7, 8]), "a\u{fffd} b");
    assert_eq!(streams_like_decode(MISTRAL_JSON, &[9, 6]), " a");
}

/// A declared pipeline that cannot be evaluated incrementally is refused, and
/// the refusal names the step. Silently falling back to the backend's own
/// decode would stream `hello</w>world</w>` while `decode` returned
/// `hello world` — exactly the drift the shared machinery exists to eliminate.
#[test]
fn a_pipeline_that_cannot_stream_is_refused_rather_than_approximated() {
    let json = r#"{
        "decoder": {"type": "BPEDecoder", "suffix": "</w>"},
        "model": {"type": "BPE", "vocab": {"hello</w>": 0, "world</w>": 1}, "merges": []}
    }"#;
    let tok = from_json_bytes(json.as_bytes()).expect("the document loads");
    assert!(tok.declares_decoder());

    let err = tok
        .streaming_decoder()
        .err()
        .expect("BPEDecoder cannot stream");
    assert!(
        matches!(err, TokenizeError::UnstreamableDecoder("BPEDecoder")),
        "unexpected error: {err}"
    );
    // ...and whole-sequence decoding still handles it.
    assert_eq!(tok.decode(&[0, 1]).expect("decodes"), "hello world");
}

/// With no declared pipeline the factory delegates to the backend's own, which
/// is the same delegation `decode` makes — so the two still agree.
#[test]
fn a_document_with_no_declared_decoder_delegates_to_the_backend() {
    // The WordPiece document above with its `decoder` removed.
    let json = r###"{
        "added_tokens": [
            {"id": 1, "content": "[CLS]", "special": true},
            {"id": 2, "content": "[SEP]", "special": true}
        ],
        "pre_tokenizer": {"type": "BertPreTokenizer"},
        "model": {"type": "WordPiece", "unk_token": "[UNK]",
            "continuing_subword_prefix": "##", "max_input_chars_per_word": 100,
            "vocab": {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2,
                "hello": 3, "##world": 4, ",": 5, "world": 6}}
    }"###;
    let tok = from_json_bytes(json.as_bytes()).expect("the document loads");
    assert!(!tok.declares_decoder());

    assert_eq!(streams_like_decode(json, &[3, 4]), "helloworld");
    assert_eq!(streams_like_decode(json, &[1, 3, 5, 6, 2]), "hello, world");
}
