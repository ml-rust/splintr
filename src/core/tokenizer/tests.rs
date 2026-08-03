use super::*;
use crate::core::added::AddedTokenSet;
use crate::core::normalizer::{NormOp, Normalizer};
use crate::core::policy::SpecialMode;
use rustc_hash::FxHashMap;

fn make_test_tokenizer() -> Tokenizer {
    let mut encoder = FxHashMap::default();
    for b in 32u8..=126 {
        encoder.insert(vec![b], b as u32);
    }
    encoder.insert(b"Hello".to_vec(), 200);
    encoder.insert(b"World".to_vec(), 201);
    encoder.insert(b" World".to_vec(), 202);

    let mut special_tokens = FxHashMap::default();
    special_tokens.insert("<|endoftext|>".to_string(), 50256);

    let pattern = r"\S+|\s+";
    Tokenizer::new(encoder, special_tokens, pattern).unwrap()
}

#[test]
fn test_encode_decode() {
    let tokenizer = make_test_tokenizer();
    let text = "Hello World";
    let tokens = tokenizer.encode(text);
    let decoded = tokenizer.decode(&tokens).unwrap();
    assert_eq!(decoded, text);
}

/// D3 regression: an id absent from the vocab, the special-tokens decoder,
/// and the `special=true` skip set must error, not silently render as `""`.
#[test]
fn decode_of_unknown_id_errors_with_invalid_token_id() {
    let tokenizer = make_test_tokenizer();
    let err = tokenizer.decode(&[7_000_000]).unwrap_err();
    assert!(matches!(err, TokenizerError::InvalidTokenId(7_000_000)));
}

/// `decode_lossy` stays infallible: unknown ids are skipped, and the
/// recognised ids around them still decode normally.
#[test]
fn decode_lossy_skips_unknown_ids() {
    let tokenizer = make_test_tokenizer();
    let mut tokens = tokenizer.encode("Hello");
    tokens.push(7_000_000);
    tokens.extend(tokenizer.encode(" World"));
    let decoded = tokenizer.decode_lossy(&tokens);
    assert_eq!(decoded, "Hello World");
}

/// `decode_batch` must propagate the error when any list in the batch
/// contains an unknown id, not just the offending list.
#[test]
fn decode_batch_propagates_invalid_token_id() {
    let tokenizer = make_test_tokenizer();
    let good = tokenizer.encode("Hello");
    let bad = vec![7_000_000u32];
    let err = tokenizer.decode_batch(&[good, bad]).unwrap_err();
    assert!(matches!(err, TokenizerError::InvalidTokenId(7_000_000)));
}

/// Guard against over-strictness: special-token ids and ordinary byte-level
/// ids from a normal round trip must never be treated as unknown.
#[test]
fn decode_encode_round_trip_does_not_misclassify_known_ids() {
    let tokenizer = make_test_tokenizer();
    let text = "Hello<|endoftext|>World";
    let tokens = tokenizer.encode_with_special(text);
    let decoded = tokenizer.decode(&tokens).unwrap();
    assert_eq!(decoded, text);
}

#[test]
fn test_encode_with_special() {
    let tokenizer = make_test_tokenizer();
    let text = "Hello<|endoftext|>World";
    let tokens = tokenizer.encode_with_special(text);
    assert!(tokens.contains(&50256));
}

/// `SpecialMode::All` and `SpecialMode::Ordinary` must diverge on text
/// containing a special token's literal spelling, and `Ordinary` must
/// never promote it — the literal text round-trips through decode.
#[test]
fn encode_with_all_vs_ordinary_diverge_on_a_special_token() {
    let tokenizer = make_test_tokenizer().with_added_token_matching(true);
    let text = "Hello<|endoftext|>World";

    let all_ids = tokenizer.encode_with(text, &SpecialMode::All).unwrap();
    let ordinary_ids = tokenizer.encode_with(text, &SpecialMode::Ordinary).unwrap();

    assert_ne!(all_ids, ordinary_ids);
    assert!(all_ids.contains(&50256));
    assert!(!ordinary_ids.contains(&50256));

    let decoded = tokenizer.decode(&ordinary_ids).unwrap();
    assert_eq!(decoded, text);
}

#[test]
fn test_batch_encode() {
    let tokenizer = make_test_tokenizer();
    let texts = vec!["Hello".to_string(), "World".to_string()];
    let batch_tokens = tokenizer.encode_batch(&texts);
    assert_eq!(batch_tokens.len(), 2);
}

#[test]
fn test_vocab_size() {
    let tokenizer = make_test_tokenizer();
    assert!(tokenizer.vocab_size() > 0);
}

#[test]
fn test_cache_works() {
    let tokenizer = make_test_tokenizer();
    let text = "HelloWorld";
    let tokens1 = tokenizer.encode(text);
    let tokens2 = tokenizer.encode(text);
    assert_eq!(tokens1, tokens2);
    assert!(tokenizer.cache_len() > 0);
}

#[test]
fn test_clear_cache() {
    let tokenizer = make_test_tokenizer();
    tokenizer.encode("HelloWorld");
    assert!(tokenizer.cache_len() > 0);
    tokenizer.clear_cache();
    assert_eq!(tokenizer.cache_len(), 0);
}

/// A cache hit must return the ids for the chunk that was actually queried,
/// never another chunk's ids (guards against the old bare-hash key, where a
/// collision would silently return a different chunk's tokens).
#[test]
fn cache_hit_returns_ids_for_the_queried_chunk_not_a_different_one() {
    let tokenizer = make_test_tokenizer();
    let texts = ["abc", "abcd", "xyz", "Hello World", "foobar", "zzz"];

    // First pass populates the cache.
    let first_pass: Vec<Vec<u32>> = texts.iter().map(|t| tokenizer.encode(t)).collect();

    // Second pass should hit the cache; ids must be unchanged.
    let second_pass: Vec<Vec<u32>> = texts.iter().map(|t| tokenizer.encode(t)).collect();
    assert_eq!(first_pass, second_pass);

    // And must match a fresh tokenizer (empty cache) encoding the same text,
    // so a cache hit can never be substituting a different chunk's result.
    for (text, ids) in texts.iter().zip(first_pass.iter()) {
        let fresh = make_test_tokenizer();
        assert_eq!(&fresh.encode(text), ids, "mismatch for {text:?}");
    }
}

/// A chunk whose bytes are a strict prefix of another chunk's bytes must get
/// its own cache entry — guards against any length-insensitive keying.
#[test]
fn prefix_chunk_gets_its_own_cache_entry() {
    let tokenizer = make_test_tokenizer();

    let short = tokenizer.encode("abc");
    let len_after_short = tokenizer.cache_len();

    let long = tokenizer.encode("abcd");
    assert!(tokenizer.cache_len() > len_after_short);
    assert_ne!(short, long);

    // Re-encoding the short chunk must still return the short result, not
    // whatever got cached for the longer chunk that starts with it.
    assert_eq!(tokenizer.encode("abc"), short);
    assert_eq!(tokenizer.encode("abcd"), long);
}

#[cfg(feature = "pcre2")]
#[test]
fn test_pcre2_backend() {
    let tokenizer = make_test_tokenizer().pcre2(true).unwrap();
    let text = "Hello World";
    let tokens = tokenizer.encode(text);
    let decoded = tokenizer.decode(&tokens).unwrap();
    assert_eq!(decoded, text);
}

#[cfg(not(feature = "pcre2"))]
#[test]
fn test_pcre2_not_enabled() {
    let tokenizer = make_test_tokenizer();
    let result = tokenizer.pcre2(true);
    assert!(result.is_err());
}

#[test]
fn test_jit_disable() {
    let tokenizer = make_test_tokenizer().jit(false).unwrap();
    let text = "Hello World";
    let tokens = tokenizer.encode(text);
    let decoded = tokenizer.decode(&tokens).unwrap();
    assert_eq!(decoded, text);
}

#[test]
fn test_jit_enable() {
    let tokenizer = make_test_tokenizer().jit(true).unwrap();
    let text = "Hello World";
    let tokens = tokenizer.encode(text);
    let decoded = tokenizer.decode(&tokens).unwrap();
    assert_eq!(decoded, text);
}

#[cfg(feature = "pcre2")]
#[test]
fn test_pcre2_switch_back_to_regexr() {
    // Start with regexr, switch to pcre2, then back to regexr
    let tokenizer = make_test_tokenizer()
        .pcre2(true)
        .unwrap()
        .pcre2(false)
        .unwrap();
    let text = "Hello World";
    let tokens = tokenizer.encode(text);
    let decoded = tokenizer.decode(&tokens).unwrap();
    assert_eq!(decoded, text);
}

#[cfg(feature = "pcre2")]
#[test]
fn test_pcre2_with_jit_disabled() {
    let tokenizer = make_test_tokenizer()
        .jit(false)
        .unwrap()
        .pcre2(true)
        .unwrap();
    let text = "Hello World";
    let tokens = tokenizer.encode(text);
    let decoded = tokenizer.decode(&tokens).unwrap();
    assert_eq!(decoded, text);
}

// ── Multi-pass pre-tokenizer (llama.cpp `unicode_regex_split`) ───────────

/// Build a tokenizer over `patterns` and report the pieces it splits `text`
/// into, so a pass composition can be asserted as text rather than ids.
fn pieces(patterns: &[&str], text: &str) -> Vec<String> {
    let tokenizer =
        Tokenizer::new_byte_level_chain(FxHashMap::default(), AddedTokenSet::new(), patterns)
            .expect("patterns compile");
    tokenizer
        .split_chunks(text)
        .into_iter()
        .filter_map(|(s, e)| text.get(s..e).map(str::to_owned))
        .collect()
}

/// A one-expression list must take the single-regex path and behave exactly
/// like the plain constructor — matches only, unmatched text dropped.
#[test]
fn single_expression_list_keeps_the_original_split() {
    let one = Tokenizer::new_byte_level_chain(
        FxHashMap::default(),
        AddedTokenSet::new(),
        &[GPT2_PATTERN],
    )
    .expect("compiles");
    assert!(
        one.chain.is_empty(),
        "a one-expression list must not engage the chained path"
    );

    let plain = Tokenizer::new_byte_level(FxHashMap::default(), AddedTokenSet::new(), GPT2_PATTERN)
        .expect("compiles");
    let text = "Hello, world! 1234\n\n  trailing";
    assert_eq!(one.split_chunks(text), plain.split_chunks(text));
}

/// The defining property: a later pass only subdivides what an earlier pass
/// produced. `\p{N}` first cuts every digit apart, so the GPT-2 split's
/// ` ?\p{N}+` can no longer take `123` as one piece — which is precisely why
/// `starcoder` is not the GPT-2 pre-tokenizer.
#[test]
fn later_pass_subdivides_earlier_pieces_and_cannot_re_merge() {
    // One expression: ` ?\p{N}+` takes the whole digit run with its space.
    assert_eq!(pieces(&[GPT2_PATTERN], "abc 123"), vec!["abc", " 123"]);
    // Two: `\p{N}` has already cut the digits apart AND left `"abc "` as a
    // gap, so pass 2 can only split that gap — it can never reunite the
    // space with a digit.
    assert_eq!(
        pieces(&[r"\p{N}", GPT2_PATTERN], "abc 123"),
        vec!["abc", " ", "1", "2", "3"],
    );
}

/// Text a pass leaves unmatched is kept as a piece of its own rather than
/// dropped, and stays eligible for the passes that follow.
#[test]
fn unmatched_gaps_are_kept_and_still_subdivided() {
    // Pass 1 matches only the digits; the letters survive as gaps. Pass 2
    // then cuts those gaps on the letter/space boundary.
    assert_eq!(
        pieces(&[r"\p{N}+", r"\p{L}+"], "ab12cd"),
        vec!["ab", "12", "cd"],
    );
    // With no second pass the same gaps are still pieces, not losses.
    assert_eq!(
        pieces(&[r"\p{N}+", r"\p{N}+"], "ab12cd"),
        vec!["ab", "12", "cd"]
    );
}

/// Each pass sees one span in isolation, so an anchor or lookahead resolves
/// against the span's edges — llama.cpp matches over `[start, start+offset)`
/// only (unicode.cpp:487). Here pass 1 isolates the digits, and `^.` in
/// pass 2 therefore fires inside EVERY resulting span, not once per text.
#[test]
fn each_pass_matches_within_a_span_not_across_the_text() {
    assert_eq!(
        pieces(&[r"\p{N}+", r"^."], "ab12cd"),
        vec!["a", "b", "1", "2", "c", "d"],
    );
}

/// Falcon's three passes compose: punctuation runs first, then the GPT-2
/// split inside the remaining pieces, then digit runs chopped into threes
/// from the left of each piece pass 2 produced.
#[test]
fn falcon_three_pass_composition() {
    let falcon = [r"[\p{P}\$\+<=>\^~\|`]+", GPT2_PATTERN, r"[0-9][0-9][0-9]"];
    assert_eq!(pieces(&falcon, "a=1234"), vec!["a", "=", "123", "4"]);
    // The alternation of the same three expressions cannot do this: it takes
    // `1234` whole via ` ?\p{N}+` and never revisits it.
    assert_eq!(
        pieces(&[r"[\p{P}\$\+<=>\^~\|`]+|'s| ?\p{L}+| ?\p{N}+"], "a=1234"),
        vec!["a", "=", "1234"],
    );
}

/// An empty list has no first expression to compile and is refused rather
/// than silently becoming a no-op split.
#[test]
fn empty_pattern_list_is_refused() {
    assert!(matches!(
        Tokenizer::new_byte_level_chain(FxHashMap::default(), AddedTokenSet::new(), &[]),
        Err(TokenizerError::EmptyPatternList)
    ));
}

/// Switching JIT recompiles the later passes too, so the split is unchanged.
#[test]
fn toggling_jit_preserves_a_chained_split() {
    let patterns = [r"\p{N}", GPT2_PATTERN];
    let tokenizer =
        Tokenizer::new_byte_level_chain(FxHashMap::default(), AddedTokenSet::new(), &patterns)
            .expect("compiles");
    let text = "abc 123";
    let before = tokenizer.split_chunks(text);
    let tokenizer = tokenizer.jit(false).expect("recompiles");
    assert_eq!(tokenizer.chain.len(), 1);
    assert_eq!(tokenizer.split_chunks(text), before);
}

/// Cloning shares the compiled passes and keeps the split identical.
#[test]
fn cloning_preserves_a_chained_split() {
    let patterns = [r"\p{N}", GPT2_PATTERN];
    let tokenizer =
        Tokenizer::new_byte_level_chain(FxHashMap::default(), AddedTokenSet::new(), &patterns)
            .expect("compiles");
    let text = "abc 123";
    assert_eq!(
        tokenizer.clone().split_chunks(text),
        tokenizer.split_chunks(text)
    );
}

const _: () = {
    assert!(super::cl100k_agent_tokens::SYSTEM > 100276);
    assert!(super::cl100k_agent_tokens::SUMMARY_END == 100330);
    assert!(super::o200k_agent_tokens::SYSTEM > 200018);
    assert!(super::o200k_agent_tokens::SUMMARY_END == 200072);
    assert!(super::cl100k_agent_tokens::USER == super::cl100k_agent_tokens::SYSTEM + 1);
    assert!(super::o200k_agent_tokens::USER == super::o200k_agent_tokens::SYSTEM + 1);
};

// ── `encode_rayon` must agree with `encode` ───────────────────────────────
//
// `encode_rayon` is a separate dispatch path from `encode` (see
// `Tokenizer::encode_rayon` in `encode.rs`) that historically skipped the
// normalizer and added-token dispatch. These tests pin `encode_rayon(x) ==
// encode(x)` across every stage that path must go through.

/// A tokenizer whose encoder covers every byte 0..=255 as its own single-byte
/// token, so BPE never silently drops a chunk to an empty result — useful for
/// tests that want to exercise non-ASCII text meaningfully.
fn make_full_byte_tokenizer() -> Tokenizer {
    let mut encoder = FxHashMap::default();
    for b in 0u16..=255 {
        encoder.insert(vec![b as u8], b as u32);
    }
    let pattern = r"\S+|\s+";
    Tokenizer::new(encoder, FxHashMap::default(), pattern).unwrap()
}

/// Direct repro of the bug report: an added token in the input must be
/// recognized by `encode_rayon` exactly as `encode` recognizes it, not
/// shredded into punctuation.
#[test]
fn encode_rayon_matches_encode_with_added_tokens_in_input() {
    let mut encoder = FxHashMap::default();
    for b in 32u8..=126 {
        encoder.insert(vec![b], b as u32);
    }
    let mut special_tokens = FxHashMap::default();
    special_tokens.insert("<|s|>".to_string(), 1000);
    let tokenizer = Tokenizer::new(encoder, special_tokens, r"\S+|\s+")
        .unwrap()
        .with_added_token_matching(true);

    let text = "a<|s|>b";
    let expected = vec![97u32, 1000, 98];
    assert_eq!(tokenizer.encode(text), expected);
    assert_eq!(tokenizer.encode_rayon(text), expected);
}

/// A normalizer attached via `with_normalizer` must run on the `encode_rayon`
/// path too, not just `encode`. NFC-normalizes `"e" + U+0301` (combining
/// acute) into the precomposed `"é"` before splitting/BPE.
#[test]
fn encode_rayon_matches_encode_with_normalizer() {
    let tokenizer = make_full_byte_tokenizer().with_normalizer(Normalizer::new(vec![NormOp::Nfc]));

    let decomposed = "e\u{0301}";
    let precomposed = "\u{e9}";

    // Sanity: the normalizer actually changes the encoding — `encode` on the
    // decomposed form must match `encode` on the already-normalized form.
    assert_eq!(tokenizer.encode(decomposed), tokenizer.encode(precomposed));

    assert_eq!(
        tokenizer.encode_rayon(decomposed),
        tokenizer.encode(decomposed)
    );
}

/// Metaspace-decoder tokenizers run `encode_content` sequentially regardless
/// of the `parallel` flag (state is a left-to-right fold), but `encode_rayon`
/// must still reach that path and produce identical ids.
#[test]
fn encode_rayon_matches_encode_for_metaspace_tokenizer() {
    let mut encoder = FxHashMap::default();
    for b in 0u16..=255 {
        encoder.insert(vec![b as u8], b as u32);
    }
    let tokenizer =
        Tokenizer::new_with_metaspace_decoder(encoder, FxHashMap::default(), r"\S+|\s+").unwrap();

    let text = "  hello   world\tfoo bar  ";
    assert_eq!(tokenizer.encode_rayon(text), tokenizer.encode(text));
}

/// A large (>1MB) input actually exercises the `par_iter` branch of
/// `map_chunks` under the `rayon` feature, not just the trivial single-chunk
/// case.
#[test]
fn encode_rayon_matches_encode_for_large_input() {
    let tokenizer = make_full_byte_tokenizer();
    let sentence = "Hello World, this is a test of the rayon parallel encoding path. ";
    let repeats = 1 + (1_048_576 / sentence.len());
    let text = sentence.repeat(repeats);
    assert!(text.len() > 1_048_576);

    assert_eq!(tokenizer.encode_rayon(&text), tokenizer.encode(&text));
}

/// The already-working case: a plain tokenizer with no normalizer and no
/// added-token matching, across empty, whitespace-only, CJK, and emoji input.
#[test]
fn encode_rayon_matches_encode_for_plain_tokenizer() {
    let tokenizer = make_full_byte_tokenizer();
    for text in ["", "   ", "你好世界", "😀🎉", "Hello World"] {
        assert_eq!(
            tokenizer.encode_rayon(text),
            tokenizer.encode(text),
            "mismatch for {text:?}"
        );
    }
}
