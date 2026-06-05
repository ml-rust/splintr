use super::*;

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
    use crate::core::Tokenize;
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
    use crate::core::Tokenize;
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
    use crate::core::Tokenize;
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
    use crate::core::Tokenize;
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

#[test]
fn post_processor_wraps_with_special_tokens() {
    use crate::core::Tokenize;
    // BERT-style post_processor: [CLS] $A [SEP]. `encode` gives content tokens;
    // `encode_with_special_tokens` wraps them.
    let json = r###"{
        "post_processor": {"type": "BertProcessing", "cls": ["[CLS]", 1], "sep": ["[SEP]", 2]},
        "model": {"type": "WordPiece", "unk_token": "[UNK]",
            "continuing_subword_prefix": "##", "max_input_chars_per_word": 100,
            "vocab": {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "hi": 3}}
    }"###;
    let tok = from_json_bytes(json.as_bytes()).unwrap();
    assert_eq!(tok.encode("hi"), vec![3]); // content only
    assert_eq!(tok.encode_with_special_tokens("hi"), vec![1, 3, 2]); // [CLS] hi [SEP]
}

#[test]
fn decode_skips_special_keeps_nonspecial_added_tokens() {
    use crate::core::Tokenize;
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
