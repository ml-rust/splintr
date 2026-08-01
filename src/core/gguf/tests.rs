//! Vocabulary normalization, merge-rank construction and dialect dispatch.

use rustc_hash::FxHashMap;

use super::loader::{
    build_merge_ranks, byte_level_pattern, find_special_token_id, normalize_wordpiece_vocab,
    unigram_prefix_space,
};
use super::{from_gguf_vocab, GgufVocab, GgufVocabError};

fn v(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| (*s).to_owned()).collect()
}

// ── WordPiece vocab normalization ────────────────────────────────────────────

/// The `nomic-embed-text-v1.5` shape: `▁`-marked word-initial pieces, bare
/// continuations, bracketed specials.
#[test]
fn sentencepiece_marked_bert_vocab_is_converted_to_wordpiece() {
    let got = normalize_wordpiece_vocab(v(&[
        "[PAD]", "[CLS]", "[SEP]", "[UNK]", "▁the", "▁hello", "s", "ing", "▁!", "▁1",
    ]));
    assert_eq!(
        got,
        v(&["[PAD]", "[CLS]", "[SEP]", "[UNK]", "the", "hello", "##s", "##ing", "!", "1",]),
        "▁X must become X, bare X must become ##X, specials must be untouched"
    );
}

/// A vocab already in WordPiece convention must be returned byte-identical —
/// otherwise fixing one model's tokenizer would break every other BERT GGUF.
#[test]
fn already_wordpiece_vocab_is_left_untouched() {
    let original = v(&["[PAD]", "[CLS]", "the", "##s", "hello", "!"]);
    assert_eq!(normalize_wordpiece_vocab(original.clone()), original);
}

/// Mixed marking (some `▁`, some `##`) means the file is already using the
/// WordPiece continuation marker, so rewriting would corrupt it.
#[test]
fn mixed_marking_is_left_untouched() {
    let original = v(&["▁the", "##s", "hello"]);
    assert_eq!(normalize_wordpiece_vocab(original.clone()), original);
}

/// No `▁` anywhere → nothing to convert.
#[test]
fn unmarked_vocab_is_left_untouched() {
    let original = v(&["the", "hello", "world"]);
    assert_eq!(normalize_wordpiece_vocab(original.clone()), original);
}

// ── Byte-level BPE merge ranks ───────────────────────────────────────────────

fn rank(map: &FxHashMap<Vec<u8>, u32>, token: &str) -> u32 {
    match map.get(token.as_bytes()) {
        Some(rank) => *rank,
        None => panic!("{token:?} has no merge rank"),
    }
}

/// Merge priority comes from the `merges` list order, not from token id — the
/// two disagree in real vocabularies, and using ids silently changes every
/// tokenization.
#[test]
fn merge_priority_follows_list_order_not_token_id() {
    // Ids put "lo" before "he", but the merges list puts "he" first.
    let tokens = v(&["h", "e", "l", "o", "lo", "he", "hel", "hello"]);
    let ranks = build_merge_ranks(&v(&["h e", "he l", "l o", "hel lo"]), &tokens);

    assert!(
        rank(&ranks, "he") < rank(&ranks, "lo"),
        "\"he\" is earlier in the merges list, so it must merge first regardless \
         of \"lo\" having the lower token id"
    );
    assert!(rank(&ranks, "he") < rank(&ranks, "hel"));
    assert!(rank(&ranks, "hel") < rank(&ranks, "hello"));
}

/// Single characters are never a merge result, so they must all rank below every
/// merge — multi-byte UTF-8 has to coalesce before any real merge runs.
#[test]
fn the_base_alphabet_outranks_every_merge() {
    let tokens = v(&["a", "b", "c", "ab", "abc"]);
    let ranks = build_merge_ranks(&v(&["a b", "ab c"]), &tokens);

    let base_max = ["a", "b", "c"]
        .iter()
        .map(|t| rank(&ranks, t))
        .max()
        .unwrap_or_default();
    let merge_min = ["ab", "abc"]
        .iter()
        .map(|t| rank(&ranks, t))
        .min()
        .unwrap_or_default();
    assert!(
        base_max < merge_min,
        "every base-alphabet token must rank below every merge"
    );
}

/// Byte-level tokens spell real spaces as `Ġ`, so only the first space in a
/// merge entry is the separator — splitting on all of them would corrupt any
/// merge involving a space token.
#[test]
fn only_the_first_space_separates_a_merge_entry() {
    let tokens = v(&["Ġ", "a", "Ġa"]);
    let ranks = build_merge_ranks(&v(&["Ġ a"]), &tokens);
    assert!(
        ranks.contains_key("Ġa".as_bytes()),
        "the merge result must be the concatenation \"Ġa\""
    );
}

/// A merge naming a token that is not in the vocab must not displace or
/// renumber the ranks of tokens that are.
#[test]
fn merges_referencing_absent_tokens_do_not_disturb_the_rest() {
    let tokens = v(&["a", "b", "ab"]);
    let ranks = build_merge_ranks(&v(&["a b", "z z"]), &tokens);
    assert!(rank(&ranks, "a") < rank(&ranks, "ab"));
    assert!(rank(&ranks, "b") < rank(&ranks, "ab"));
}

// ── Pre-tokenizer selection ──────────────────────────────────────────────────

/// The `pre` name selects the split pattern, and the three families must not
/// collapse onto one another.
#[test]
fn pre_tokenizer_names_select_distinct_patterns() {
    let gpt2 = byte_level_pattern(None).expect("absent `pre` is llama.cpp's GPT-2 default");
    assert_eq!(
        byte_level_pattern(Some("default")).expect("default"),
        gpt2,
        "`default` is the same GPT-2 split as an absent key"
    );
    assert_eq!(
        byte_level_pattern(Some("jina-v2-code")).expect("jina"),
        gpt2
    );

    let qwen = byte_level_pattern(Some("qwen2")).expect("qwen2");
    let llama = byte_level_pattern(Some("llama-bpe")).expect("llama-bpe");
    assert_ne!(qwen, gpt2);
    assert_ne!(llama, gpt2);
    assert_ne!(qwen, llama);
}

/// An unrecognised pre-tokenizer is refused, never defaulted: a wrong split is
/// invisible downstream because every id it produces is still in range.
#[test]
fn unknown_pre_tokenizer_is_refused_not_guessed() {
    assert!(matches!(
        byte_level_pattern(Some("some-future-pre")),
        Err(GgufVocabError::UnsupportedPreTokenizer(name)) if name == "some-future-pre"
    ));
}

// ── Flag resolution ──────────────────────────────────────────────────────────

/// `jina-embeddings-v3`'s shape: `add_space_prefix = false` with
/// `remove_extra_whitespaces = true` still marks the first word, because
/// llama.cpp's Unigram normalizer ORs the two flags.
#[test]
fn unigram_prefix_space_ors_the_two_flags() {
    let with = |space: Option<bool>, extra: Option<bool>| {
        unigram_prefix_space(&GgufVocab {
            add_space_prefix: space,
            remove_extra_whitespaces: extra,
            ..GgufVocab::default()
        })
    };
    assert!(with(None, None), "add_space_prefix defaults to true");
    assert!(
        with(Some(false), Some(true)),
        "remove_extra_whitespaces alone must still mark the first word"
    );
    assert!(with(Some(true), Some(false)));
    assert!(
        !with(Some(false), Some(false)),
        "neither flag set means the first word stays unmarked"
    );
    assert!(
        !with(Some(false), None),
        "remove_extra_whitespaces defaults to false"
    );
}

/// The vocabulary's own string is ground truth: a file whose `[UNK]` sits at a
/// different id than `unknown_token_id` claims would otherwise emit an id that
/// decodes to some other token.
#[test]
fn special_token_lookup_prefers_the_vocab_over_the_metadata() {
    let tokens = v(&["[PAD]", "[UNK]", "the"]);
    let vocab = GgufVocab {
        unknown_token_id: Some(99),
        ..GgufVocab::default()
    };
    assert_eq!(find_special_token_id(&tokens, &vocab, "[UNK]", 0), 1);
}

/// With no matching string, the declared id is used — and with neither, the
/// caller's default.
#[test]
fn special_token_lookup_falls_back_to_metadata_then_default() {
    let tokens = v(&["a", "b"]);
    let declared = GgufVocab {
        unknown_token_id: Some(7),
        ..GgufVocab::default()
    };
    assert_eq!(find_special_token_id(&tokens, &declared, "[UNK]", 0), 7);
    assert_eq!(
        find_special_token_id(&tokens, &GgufVocab::default(), "[UNK]", 3),
        3
    );
}

// ── Dialect dispatch ─────────────────────────────────────────────────────────

/// A vocabulary whose algorithm we do not implement is refused rather than run
/// through whichever backend happens to accept its data.
#[test]
fn unsupported_model_is_refused() {
    let vocab = GgufVocab {
        model: "rwkv".to_owned(),
        tokens: v(&["a"]),
        ..GgufVocab::default()
    };
    assert!(matches!(
        from_gguf_vocab(vocab),
        Err(GgufVocabError::UnsupportedModel(name)) if name == "rwkv"
    ));
}

#[test]
fn empty_vocabulary_is_refused() {
    assert!(matches!(
        from_gguf_vocab(GgufVocab {
            model: "llama".to_owned(),
            ..GgufVocab::default()
        }),
        Err(GgufVocabError::EmptyVocab)
    ));
}

/// byte-level BPE *is* its merge list, so a `gpt2` vocabulary without one cannot
/// be reconstructed and must not be approximated from the vocabulary.
#[test]
fn gpt2_without_merges_is_refused() {
    let vocab = GgufVocab {
        model: "gpt2".to_owned(),
        tokens: v(&["a", "b", "ab"]),
        ..GgufVocab::default()
    };
    assert!(matches!(
        from_gguf_vocab(vocab),
        Err(GgufVocabError::MissingMerges)
    ));
}

fn llama_vocab() -> GgufVocab {
    GgufVocab {
        model: "llama".to_owned(),
        tokens: v(&["<unk>", "<s>", "</s>", "▁hello", "▁world"]),
        bos_token_id: Some(1),
        eos_token_id: Some(2),
        ..GgufVocab::default()
    }
}

/// llama.cpp's defaults for a SentencePiece BPE vocabulary: BOS is prepended,
/// EOS is not appended. The policy owns both — the backend was built with
/// neither id, so nothing else can insert them.
#[test]
fn llama_prepends_bos_and_omits_eos_by_default() {
    let tok = from_gguf_vocab(llama_vocab()).expect("builds");
    assert_eq!(tok.family(), "Spm");

    let ids = tok.encode("hello world");
    assert_eq!(ids.first(), Some(&1), "add_bos_token defaults to true");
    assert_ne!(ids.last(), Some(&2), "add_eos_token defaults to false");
    assert_eq!(
        tok.encode_raw("hello world").as_slice(),
        &ids[1..],
        "the boundary token must come from the policy, not the backend"
    );
    assert_eq!(tok.eos_token_id(), Some(2));
    assert!(tok.is_eos(2));
}

/// The file's own flags win over the defaults, in both directions.
#[test]
fn llama_honours_the_declared_boundary_flags() {
    let tok = from_gguf_vocab(GgufVocab {
        add_bos_token: Some(false),
        add_eos_token: Some(true),
        ..llama_vocab()
    })
    .expect("builds");

    let ids = tok.encode("hello");
    assert_ne!(ids.first(), Some(&1));
    assert_eq!(ids.last(), Some(&2));
}

/// A flag asking for a boundary token the file never gives an id for adds
/// nothing — there is no id to add.
#[test]
fn a_boundary_flag_without_an_id_adds_nothing() {
    let tok = from_gguf_vocab(GgufVocab {
        bos_token_id: None,
        ..llama_vocab()
    })
    .expect("builds");
    assert_eq!(tok.encode("hello"), tok.encode_raw("hello"));
}

/// `t5` is the one dialect whose defaults ask for both boundaries.
#[test]
fn t5_wraps_with_both_boundaries_by_default() {
    let tok = from_gguf_vocab(GgufVocab {
        model: "t5".to_owned(),
        tokens: v(&["<unk>", "<s>", "</s>", "▁hi"]),
        bos_token_id: Some(1),
        eos_token_id: Some(2),
        ..GgufVocab::default()
    })
    .expect("builds");

    assert_eq!(tok.family(), "Unigram");
    let ids = tok.encode("hi");
    assert_eq!(ids.first(), Some(&1));
    assert_eq!(ids.last(), Some(&2));
}

/// BERT wraps with `[CLS]`/`[SEP]` through its vocabulary, so the policy must
/// add no boundary tokens at all — but must still resolve the ids by name.
#[test]
fn bert_gets_no_boundary_template_but_keeps_the_named_ids() {
    let tok = from_gguf_vocab(GgufVocab {
        model: "bert".to_owned(),
        tokens: v(&["[PAD]", "[UNK]", "[CLS]", "[SEP]", "the"]),
        // Set even though BERT ignores them: they must not leak into the ids.
        add_bos_token: Some(true),
        add_eos_token: Some(true),
        bos_token_id: Some(2),
        eos_token_id: Some(3),
        ..GgufVocab::default()
    })
    .expect("builds");

    assert_eq!(tok.family(), "WordPiece");
    assert_eq!(
        tok.encode("the"),
        tok.encode_raw("the"),
        "BERT's boundaries come from [CLS]/[SEP], not from a boundary template"
    );
    assert_eq!(tok.special_token_id("[CLS]"), Some(2));
    assert_eq!(tok.special_token_id("[SEP]"), Some(3));
    assert_eq!(tok.special_token_id("[UNK]"), Some(1));
    assert_eq!(tok.special_token_id("[PAD]"), Some(0));
}

/// CONTROL-flagged tokens are the special tokens of a `gpt2` vocabulary, and
/// they must be reachable by name as well as matched in the input.
#[test]
fn gpt2_control_tokens_become_named_specials() {
    let tok = from_gguf_vocab(GgufVocab {
        model: "gpt2".to_owned(),
        tokens: v(&["a", "b", "ab", "<|endoftext|>"]),
        merges: Some(v(&["a b"])),
        token_type: Some(vec![1, 1, 1, 3]),
        eos_token_id: Some(3),
        ..GgufVocab::default()
    })
    .expect("builds");

    assert_eq!(tok.family(), "BPE");
    assert_eq!(tok.special_token_id("<|endoftext|>"), Some(3));
    assert_eq!(
        tok.special_token_id("ab"),
        None,
        "only CONTROL-flagged tokens are special"
    );
    assert_eq!(tok.eos_token_id(), Some(3));
    assert_eq!(
        tok.encode_raw("ab<|endoftext|>"),
        vec![2, 3],
        "a control token in the text stays whole"
    );
}

// ── Control tokens across every dialect ──────────────────────────────────────
//
// A chat template is assembled by splicing markers like `<start_of_turn>` into
// the prompt string. If the backend does not match them, they shatter into
// content pieces; if the policy does not name them, the caller cannot splice the
// id instead. Both failures are invisible — the ids stay in range and decode
// back to the original string — so every dialect is pinned here.

/// `llama` (SPM-BPE): the dialect that had no matcher at all, so a Gemma-style
/// chat marker was silently ground into fragments.
///
/// The vocabulary carries the pieces a real SPM file would (`hi` before `▁hi`),
/// so the gap after the marker has to complete a merge chain rather than land on
/// a single entry.
#[test]
fn llama_control_tokens_are_matched_and_named() {
    let tok = from_gguf_vocab(GgufVocab {
        tokens: v(&[
            "<unk>",
            "<s>",
            "</s>",
            "<start_of_turn>",
            "▁",
            "h",
            "i",
            "hi",
            "▁hi",
        ]),
        token_type: Some(vec![3, 3, 3, 3, 1, 1, 1, 1, 1]),
        ..llama_vocab()
    })
    .expect("builds");

    assert_eq!(tok.family(), "Spm");
    assert_eq!(tok.special_token_id("<start_of_turn>"), Some(3));
    assert_eq!(
        tok.encode_raw("<start_of_turn>hi"),
        vec![3, 8],
        "the marker is one id, and the text after it still merges to a whole word"
    );
    assert_eq!(
        tok.special_token_id("▁hi"),
        None,
        "only CONTROL-flagged tokens are special"
    );
}

/// `t5` (Unigram): the map was empty, so nothing resolved by name.
#[test]
fn t5_control_tokens_are_matched_and_named() {
    let tok = from_gguf_vocab(GgufVocab {
        model: "t5".to_owned(),
        tokens: v(&["<unk>", "<s>", "</s>", "▁hi", "<start_of_turn>"]),
        scores: Some(vec![-10.0, -10.0, -10.0, -1.0, -10.0]),
        token_type: Some(vec![3, 3, 3, 1, 3]),
        bos_token_id: Some(1),
        eos_token_id: Some(2),
        ..GgufVocab::default()
    })
    .expect("builds");

    assert_eq!(tok.family(), "Unigram");
    assert_eq!(tok.special_token_id("<start_of_turn>"), Some(4));
    assert_eq!(tok.encode_raw("<start_of_turn>hi"), vec![4, 3]);
}

/// `bert` (WordPiece): the control map must be merged into the `[UNK]`/`[CLS]`/
/// `[SEP]` lookups, never replace them.
#[test]
fn bert_control_tokens_are_matched_without_losing_the_bracketed_ids() {
    let tok = from_gguf_vocab(GgufVocab {
        model: "bert".to_owned(),
        tokens: v(&["[PAD]", "[UNK]", "[CLS]", "[SEP]", "the", "<start_of_turn>"]),
        token_type: Some(vec![3, 3, 3, 3, 1, 3]),
        ..GgufVocab::default()
    })
    .expect("builds");

    assert_eq!(tok.family(), "WordPiece");
    assert_eq!(tok.special_token_id("<start_of_turn>"), Some(5));
    assert_eq!(tok.encode_raw("<start_of_turn>the"), vec![5, 4]);

    // The pre-existing lookups must survive the merge.
    assert_eq!(tok.special_token_id("[UNK]"), Some(1));
    assert_eq!(tok.special_token_id("[CLS]"), Some(2));
    assert_eq!(tok.special_token_id("[SEP]"), Some(3));
    assert_eq!(tok.special_token_id("[PAD]"), Some(0));
}

/// A file with no `token_type` array names no control tokens at all, so no
/// matcher is attached and tokenization is exactly what it was before.
#[test]
fn a_vocabulary_without_token_types_gets_no_specials() {
    let tok = from_gguf_vocab(llama_vocab()).expect("builds");
    assert_eq!(tok.special_token_id("<s>"), None);
    assert!(
        !tok.encode_raw("<s>hello").contains(&1),
        "nothing declared the token special, so it is ordinary text"
    );
}
