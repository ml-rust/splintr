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
use splintr::{NormOp, Normalizer, PreTokStage, PreTokenizer, SplitBehavior, Tokenizer};

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

/// A `Split` pattern that does not compile is reported through the crate-root
/// [`TokenizerError`](splintr::TokenizerError) — which therefore has to be
/// nameable downstream too — instead of being dropped, which would silently
/// change the split and the ids.
#[test]
fn split_stage_reports_an_invalid_pattern() {
    let err: splintr::TokenizerError = PreTokenizer::new(vec![PreTokStage::Split {
        pattern: "(".to_string(),
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
            pattern: r"\s".to_string(),
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
