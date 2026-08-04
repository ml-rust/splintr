use super::*;
use crate::core::policy::PolicyError;
// Raw backends expose `encode` only through the trait; `AnyTokenizer`'s is inherent.
use crate::core::tokenize::Tokenize;
use crate::core::Backend;

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
/// `start_id`) that `build_bpe` requires, in full, for `model.byte_fallback:
/// true` to derive a complete table. Written as a helper rather than a literal
/// 256-entry JSON blob: `byte_fallback_ids_from_encoder` (src/core/hf_json's
/// consumer) is all-or-nothing, so every test that wants a *complete* table
/// needs all 256 spelled exactly right.
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
/// that byte fallback is opt-in and this file's defaults are unaffected.
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
        // 'b' (0x62) has no vocab entry and no fallback table configured, so
        // it is silently dropped: today's pre-existing behavior.
        assert_eq!(t.encode("abc"), vec![0, 1]);
    }
}

/// `model.byte_fallback: true` but the `<0xNN>` set in `model.vocab` is
/// incomplete (only 3 of the 256 required entries): loading must fail with
/// `MissingSpecial("byte_fallback")` rather than silently produce a
/// tokenizer that would drop the missing bytes instead of falling back.
#[test]
fn byte_fallback_true_with_an_incomplete_table_errors() {
    let json = r#"{
        "model": {"type": "BPE", "byte_fallback": true,
            "vocab": {"a": 0, "c": 1, "<0x00>": 2, "<0x01>": 3, "<0x02>": 4},
            "merges": []}
    }"#;
    let err = from_json_bytes(json.as_bytes());
    assert!(matches!(
        err,
        Err(HfJsonError::MissingSpecial("byte_fallback"))
    ));
}
