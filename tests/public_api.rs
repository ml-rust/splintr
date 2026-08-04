//! Reachability guards for splintr's public API surface.
//!
//! Integration tests compile as a **separate crate**, so everything here is
//! resolved exactly as a downstream user would resolve it. That is the whole
//! point of this file: an in-crate test can name an item through `crate::`
//! whether it is `pub` or merely `pub(crate)`, so only a test that lives
//! outside the crate can tell "public" apart from "public in name only".
//!
//! A builder method whose parameter type cannot be named by a caller is
//! uncallable no matter what its own visibility says. Each test below
//! therefore *constructs* the argument through crate-root paths and then
//! actually calls the method — compiling is most of the assertion.
use splintr::{
    ByteFallback, NormOp, Normalizer, PreTokStage, PreTokenizer, SentencePieceTokenizer,
    SplitBehavior, SplitPattern, SpmTokenizer, StreamingDecoder, Tokenizer,
};

/// A vocabulary that tells the two splits apart: as one chunk `"a12"` BPEs into
/// `"a" + "12"`, but a per-digit pre-tokenizer forces `"a" + "1" + "2"`.
fn digit_encoder() -> splintr::FxHashMap<Vec<u8>, u32> {
    [
        (b"a".to_vec(), 0u32),
        (b"1".to_vec(), 1u32),
        (b"2".to_vec(), 2u32),
        (b"12".to_vec(), 3u32),
    ]
    .into_iter()
    .collect()
}

fn digit_tokenizer() -> Tokenizer {
    Tokenizer::new(digit_encoder(), splintr::FxHashMap::default(), r"\S+|\s+")
        .expect("tokenizer construction")
}

/// The normalizer pipeline is nameable and constructible from outside the
/// crate, and [`Tokenizer::with_normalizer`] accepts what it produces.
#[test]
fn normalizer_can_be_built_and_attached_from_outside_the_crate() {
    let normalizer = Normalizer::new(vec![NormOp::Lowercase]);
    assert!(!normalizer.is_empty());
    assert_eq!(normalizer.normalize("MiXeD"), "mixed");

    let encoder = [(b"hello".to_vec(), 0u32), (b"HELLO".to_vec(), 1u32)]
        .into_iter()
        .collect();
    let tokenizer = Tokenizer::new(encoder, splintr::FxHashMap::default(), r"\S+|\s+")
        .expect("tokenizer construction")
        .with_normalizer(normalizer);

    // Lowercasing happens before the vocabulary is consulted, so the uppercase
    // spelling must resolve to the lowercase entry's id.
    assert_eq!(tokenizer.encode("HELLO"), vec![0]);
}

/// [`Normalizer::new`] takes the ops by value in order, and an empty pipeline
/// is a valid no-op rather than an error — both are part of the contract a
/// downstream caller depends on.
#[test]
fn empty_normalizer_is_a_valid_no_op() {
    let normalizer = Normalizer::new(vec![]);
    assert!(normalizer.is_empty());
    assert_eq!(normalizer.normalize("Unchanged"), "Unchanged");
}

/// [`NormOp::replace_regex`] is the only way to build [`NormOp::ReplaceRegex`]
/// from outside the crate — the variant holds a `Box<regexr::Regex>` and
/// `regexr` is not re-exported — so the constructor has to stay public.
#[test]
fn normalizer_regex_op_is_constructible_from_outside_the_crate() {
    let op = NormOp::replace_regex(r"\s+", "_".to_string()).expect("regex builds");
    assert_eq!(Normalizer::new(vec![op]).normalize("a   b"), "a_b");
    // An uncompilable pattern is reported rather than silently degraded to a
    // literal replacement.
    assert!(NormOp::replace_regex("(", "_".to_string()).is_none());
}

/// The pre-tokenizer pipeline is nameable and constructible from outside the
/// crate, and [`Tokenizer::with_pre_tokenizer`] accepts what it produces.
#[test]
fn pre_tokenizer_can_be_built_and_attached_from_outside_the_crate() {
    let pt =
        PreTokenizer::new(vec![PreTokStage::Digits { individual: true }]).expect("pipeline builds");
    assert!(!pt.is_empty());
    assert!(!pt.byte_level());
    assert_eq!(pt.stages(), [PreTokStage::Digits { individual: true }]);
    assert_eq!(pt.split("a12"), vec!["a", "1", "2"]);

    // Unattached, "a12" is a single chunk and BPE prefers the "12" token.
    assert_eq!(digit_tokenizer().encode("a12"), vec![0, 3]);
    // Attached, each digit is its own pre-token, so "12" can never form.
    assert_eq!(
        digit_tokenizer().with_pre_tokenizer(pt).encode("a12"),
        vec![0, 1, 2]
    );
}

/// [`ByteFallback`] is nameable and constructible from outside the crate via
/// [`ByteFallback::new`] — its fields are private — and
/// [`Tokenizer::with_byte_fallback`] accepts what it produces.
#[test]
fn byte_fallback_can_be_built_and_attached_from_outside_the_crate() {
    let mut byte_ids = [None; 256];
    byte_ids[0x62] = Some(999);
    let fallback = ByteFallback::new(byte_ids, None);

    let mut encoder = splintr::FxHashMap::default();
    encoder.insert(b"a".to_vec(), 1u32);
    encoder.insert(b"c".to_vec(), 2u32);
    // `b` (0x62) is deliberately absent: BPE cannot represent it at all.
    let tokenizer = Tokenizer::new(encoder, splintr::FxHashMap::default(), r"\S+|\s+")
        .expect("tokenizer construction")
        .with_byte_fallback(Some(fallback));

    assert!(tokenizer.has_byte_fallback());
    assert_eq!(tokenizer.encode("abc"), vec![1, 999, 2]);
}

/// A `Split` pattern that does not compile is reported through the crate-root
/// [`TokenizerError`](splintr::TokenizerError) — which therefore has to be
/// nameable downstream too — instead of being dropped, which would silently
/// change the split and the ids.
#[test]
fn split_stage_reports_an_invalid_pattern() {
    // Only a regex pattern can fail to compile: a literal is escaped before
    // compiling, so an uncompilable literal is impossible by construction.
    let err: splintr::TokenizerError = PreTokenizer::new(vec![PreTokStage::Split {
        pattern: SplitPattern::Regex("(".to_string()),
        behavior: SplitBehavior::Isolated,
        invert: false,
    }])
    .expect_err("unbalanced group must not compile");
    assert!(matches!(err, splintr::TokenizerError::RegexrError(_)));
}

/// An empty pipeline is a valid no-op rather than an error, and attaching one
/// leaves encoding exactly as it was — an empty engine must not hijack the split
/// and hand the whole text to BPE as one chunk.
#[test]
fn empty_pre_tokenizer_is_a_valid_no_op() {
    let pt = PreTokenizer::new(vec![]).expect("empty pipeline builds");
    assert!(pt.is_empty());
    assert!(pt.stages().is_empty());

    let baseline = digit_tokenizer().encode("a12");
    assert_eq!(
        digit_tokenizer().with_pre_tokenizer(pt).encode("a12"),
        baseline
    );
}

/// All five [`SplitBehavior`] variants are nameable from outside the crate and
/// each produces a distinct split, so the public set cannot silently shrink.
#[test]
fn every_split_behavior_is_nameable() {
    let split = |behavior| {
        PreTokenizer::new(vec![PreTokStage::Split {
            pattern: SplitPattern::Regex(r"\s".to_string()),
            behavior,
            invert: false,
        }])
        .expect("pipeline builds")
        .split("a  b")
    };
    assert_eq!(split(SplitBehavior::Isolated), vec!["a", " ", " ", "b"]);
    assert_eq!(split(SplitBehavior::Removed), vec!["a", "b"]);
    assert_eq!(split(SplitBehavior::MergedWithPrevious), vec!["a  ", "b"]);
    assert_eq!(split(SplitBehavior::MergedWithNext), vec!["a", "  b"]);
    assert_eq!(split(SplitBehavior::Contiguous), vec!["a", "  ", "b"]);
    // `Isolated` is the default, matching HuggingFace's absent-behavior case.
    assert_eq!(SplitBehavior::default(), SplitBehavior::Isolated);
}

/// [`StreamingDecoder`] is nameable from the crate root — a caller has to be
/// able to write the type down to store one in a struct — and
/// [`Tokenizer::streaming_decoder`] is the *only* way to obtain one: the type
/// has no public constructor, and no other public method returns it. That is
/// what makes "streaming with the decoder that does not match the vocabulary"
/// unrepresentable rather than merely discouraged, so it is asserted from
/// outside the crate, where `pub(crate)` items are genuinely unreachable.
///
/// The binding is annotated deliberately: it fails to compile if the type ever
/// grows a lifetime parameter, which would stop callers owning a decoder past
/// the tokenizer's scope.
#[test]
fn streaming_decoder_is_nameable_and_only_reachable_through_the_factory() {
    let mut decoder: StreamingDecoder = digit_tokenizer().streaming_decoder();

    let ids = digit_tokenizer().encode("a12");
    let mut streamed = String::new();
    for id in &ids {
        if let Some(text) = decoder.add_token(*id).expect("ids come from encode") {
            streamed.push_str(&text);
        }
    }
    streamed.push_str(&decoder.flush());

    // The whole point of the type: the stream says what `decode` says.
    assert_eq!(streamed, "a12");
    assert_eq!(
        streamed,
        digit_tokenizer()
            .decode(&ids)
            .expect("ids come from encode")
    );

    // An id in no table is reported, exactly as `decode` reports it.
    assert!(matches!(
        decoder.add_token(9_999),
        Err(splintr::TokenizeError::InvalidTokenId(9_999))
    ));
    // ...and the lossy twin skips it instead, exactly as `decode_lossy` does.
    decoder.reset();
    assert_eq!(decoder.add_token_lossy(9_999), None);
    assert!(!decoder.has_pending());
    assert_eq!(decoder.pending_bytes(), 0);
}

/// A four-piece SentencePiece-BPE vocabulary: `▁` is the word boundary, so
/// `["▁hello", "▁world"]` decodes to `"hello world"` once the dummy prefix
/// comes off.
fn spm_tokenizer() -> SpmTokenizer {
    let tokens = ["<unk>", "▁", "▁hello", "▁world"]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    // Empty scores: id order is the merge order, which is all this needs.
    SpmTokenizer::new(tokens, vec![], None, None).expect("vocabulary builds")
}

/// [`SpmTokenizer::streaming_decoder`] is callable from outside the crate and
/// hands back the *same* [`StreamingDecoder`] — so the SPM backend widens who
/// may hand one out without widening how one can be built: the type still has
/// no public constructor, and the annotated binding still fails to compile if
/// it ever grows a lifetime parameter.
#[test]
fn spm_streaming_decoder_is_reachable_and_agrees_with_decode() {
    let spm = spm_tokenizer();
    let ids = [2u32, 3];

    let mut decoder: StreamingDecoder = spm.streaming_decoder();
    let mut streamed = String::new();
    for id in ids {
        if let Some(text) = decoder.add_token(id).expect("ids are in the vocabulary") {
            streamed.push_str(&text);
        }
    }
    streamed.push_str(&decoder.flush());

    // The dummy prefix comes off exactly once, on the stream as on `decode`.
    assert_eq!(streamed, "hello world");
    assert_eq!(
        streamed,
        spm.decode(&ids).expect("ids are in the vocabulary")
    );
}

/// The SPM decoder owns its configuration too, so it outlives the tokenizer
/// that built it — and a leading skipped id must not spend the dummy-prefix
/// strip, which is only observable from a caller driving the stream by hand.
#[test]
fn spm_streaming_decoder_outlives_its_tokenizer_and_keeps_the_prefix_strip() {
    let mut decoder = {
        let spm = spm_tokenizer();
        spm.streaming_decoder()
    };

    // `<unk>` (0) is skipped, renders nothing, and therefore leaves the strip
    // armed for the first character that actually arrives.
    assert_eq!(decoder.add_token(0).expect("known id"), None);
    assert_eq!(
        decoder.add_tokens(&[2, 3]).expect("known ids"),
        Some("hello world".to_string())
    );
}

/// A five-piece Unigram vocabulary: `▁` is the word boundary, so
/// `["▁hello", "▁world"]` decodes to `"hello world"` once the metaspace prefix
/// comes off, and `<0x21>` is a byte-fallback token for `!`.
fn unigram_tokenizer() -> SentencePieceTokenizer {
    let tokens = ["<unk>", "</s>", "▁hello", "▁world", "<0x21>"]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    // Empty scores: uniform, which is all a decode-side guard needs.
    SentencePieceTokenizer::new(tokens, vec![], None, 1).expect("vocabulary builds")
}

/// [`SentencePieceTokenizer::streaming_decoder`] is callable from outside the
/// crate and hands back the *same* [`StreamingDecoder`] — so the Unigram
/// backend widens who may hand one out without widening how one can be built:
/// the type still has no public constructor, and the annotated binding still
/// fails to compile if it ever grows a lifetime parameter.
#[test]
fn unigram_streaming_decoder_is_reachable_and_agrees_with_decode() {
    let unigram = unigram_tokenizer();
    let ids = [2u32, 3, 4];

    let mut decoder: StreamingDecoder = unigram.streaming_decoder();
    let mut streamed = String::new();
    for id in ids {
        if let Some(text) = decoder.add_token(id).expect("ids are in the vocabulary") {
            streamed.push_str(&text);
        }
    }
    streamed.push_str(&decoder.flush());

    // The metaspace prefix comes off exactly once, on the stream as on `decode`,
    // and the byte-fallback id is its byte rather than its spelling.
    assert_eq!(streamed, "hello world!");
    assert_eq!(
        streamed,
        unigram.decode(&ids).expect("ids are in the vocabulary")
    );
}

/// The Unigram decoder owns its configuration too, so it outlives the tokenizer
/// that built it — and a leading skipped id must not spend the metaspace-prefix
/// strip, which is only observable from a caller driving the stream by hand.
#[test]
fn unigram_streaming_decoder_outlives_its_tokenizer_and_keeps_the_prefix_strip() {
    let mut decoder = {
        let unigram = unigram_tokenizer();
        unigram.streaming_decoder()
    };

    // `<unk>` (0) is skipped, renders nothing, and therefore leaves the strip
    // armed for the first character that actually arrives.
    assert_eq!(decoder.add_token(0).expect("known id"), None);
    assert_eq!(
        decoder.add_tokens(&[2, 3]).expect("known ids"),
        Some("hello world".to_string())
    );
}

/// A decoder owns its configuration, so it outlives the tokenizer that built
/// it. This test compiles only because [`StreamingDecoder`] carries no
/// lifetime — the property that lets a caller move one into a generation task.
#[test]
fn streaming_decoder_outlives_the_tokenizer_that_built_it() {
    let mut decoder = {
        let tokenizer = digit_tokenizer();
        tokenizer.streaming_decoder()
    };

    assert_eq!(
        decoder.add_tokens(&[0, 3]).expect("known ids"),
        Some("a12".to_string())
    );
}

/// Both [`SplitPattern`] variants are nameable from outside the crate and mean
/// what HuggingFace means by them: a literal matches its characters verbatim,
/// a regex is compiled. Describing a `Split` stage is impossible without this
/// type, so it has to be reachable downstream.
#[test]
fn both_split_patterns_are_nameable() {
    let split = |pattern| {
        PreTokenizer::new(vec![PreTokStage::Split {
            pattern,
            behavior: SplitBehavior::Removed,
            invert: false,
        }])
        .expect("pipeline builds")
        .split("a.b c")
    };
    assert_eq!(
        split(SplitPattern::Literal(".".to_string())),
        vec!["a", "b c"]
    );
    assert!(split(SplitPattern::Regex(".".to_string())).is_empty());
}
