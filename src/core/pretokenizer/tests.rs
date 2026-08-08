use super::parse::{gpt2_regex, parse};
use super::pipeline::PreTokenizer;
use super::spec::{Behavior, PreTokStage, SplitBehavior, SplitPattern};
use super::split::{split_digits, split_punctuation, split_regex};
use super::stage::SplitMatcher;
use serde_json::Value;

/// Run a splitter and collect its borrowed pieces as owned strings, so the
/// assertions below can keep comparing against `vec!["a", "b"]` literals.
fn split<'p>(piece: &'p str, f: impl Fn(&'p str, &mut Vec<&'p str>)) -> Vec<String> {
    let mut out = Vec::new();
    f(piece, &mut out);
    out.into_iter().map(str::to_owned).collect()
}

#[test]
fn digits_grouped_and_individual() {
    assert_eq!(
        split("abc123def", |p, o| split_digits(p, false, o)),
        vec!["abc", "123", "def"]
    );
    assert_eq!(
        split("abc123", |p, o| split_digits(p, true, o)),
        vec!["abc", "1", "2", "3"]
    );
    // Unicode numerics (superscripts/fractions) count as digits, like HF.
    assert_eq!(
        split("a²½b", |p, o| split_digits(p, true, o)),
        vec!["a", "²", "½", "b"]
    );
}

#[test]
fn punctuation_isolated_and_contiguous() {
    assert_eq!(
        split("a,b!", |p, o| split_punctuation(p, Behavior::Isolated, o)),
        vec!["a", ",", "b", "!"]
    );
    assert_eq!(
        split("a)=b", |p, o| split_punctuation(p, Behavior::Contiguous, o)),
        vec!["a", ")=", "b"]
    );
    assert_eq!(
        split("a,b", |p, o| split_punctuation(p, Behavior::Removed, o)),
        vec!["a", "b"]
    );
}

#[test]
fn split_merge_behaviors() {
    let re = SplitMatcher::compile(r"\s+").unwrap();
    let go = |b| split("a b c", |p, o| split_regex(p, &re, b, false, o));
    assert_eq!(go(Behavior::Isolated), vec!["a", " ", "b", " ", "c"]);
    assert_eq!(go(Behavior::Removed), vec!["a", "b", "c"]);
    assert_eq!(go(Behavior::MergedWithPrevious), vec!["a ", "b ", "c"]);
    assert_eq!(go(Behavior::MergedWithNext), vec!["a", " b", " c"]);
    // Adjacent delimiters merge under Contiguous.
    let re2 = SplitMatcher::compile(r"\s").unwrap();
    assert_eq!(
        split("a  b", |p, o| split_regex(
            p,
            &re2,
            Behavior::Contiguous,
            false,
            o
        )),
        vec!["a", "  ", "b"]
    );
}

#[test]
fn pipeline_digits_then_byte_level() {
    // Sequence[Digits(individual), ByteLevel] like starcoder2: "a 12" → each
    // digit isolated, then byte-encoded (space → Ġ).
    let json = serde_json::json!({
        "type": "Sequence",
        "pretokenizers": [
            {"type": "Digits", "individual_digits": true},
            {"type": "ByteLevel", "add_prefix_space": false, "use_regex": true}
        ]
    });
    let pt = parse(Some(&json)).expect("parses").expect("pipeline");
    assert!(pt.byte_level());
    // "a 12" → ["a"," 1","2"]? Digits first: ["a ","1","2"]; then ByteLevel
    // GPT2-splits "a " → "a"+" "? then byte-encodes. Just assert digits split.
    let pieces = pt.split("a12");
    assert_eq!(pieces, vec!["a", "1", "2"]);
}

/// `invert` swaps which side is the delimiter: the matches become the content
/// and the spans between them the delimiters.
///
/// The contiguous case is the one that matters and the one that was wrong:
/// a pre-tokenizer pattern like `\w+|\s+` covers the whole input, so there are
/// no gaps between its matches. Deriving the content by complementing the gaps
/// therefore yields one span covering everything, and the text is never split —
/// which is what every `tokenizer.json` carrying `"invert": true` used to get,
/// the Xenova GPT-4 and GPT-4o exports among them.
#[test]
fn split_invert_keeps_each_match_when_matches_are_contiguous() {
    let split_with = |pattern: &str, behavior, text: &'static str| {
        let re = SplitMatcher::compile(pattern).expect("compiles");
        let mut out = Vec::new();
        split_regex(text, &re, behavior, true, &mut out);
        out.into_iter().map(str::to_owned).collect::<Vec<_>>()
    };

    // Contiguous matches: every piece survives on its own.
    assert_eq!(
        split_with(r"\w+|\s+", Behavior::Removed, "ab cd"),
        vec!["ab", " ", "cd"]
    );
    // Gaps between matches are the delimiters, so `Removed` drops them.
    assert_eq!(
        split_with(r"\w+", Behavior::Removed, "ab cd"),
        vec!["ab", "cd"]
    );
    // and `Isolated` keeps them as their own pieces.
    assert_eq!(
        split_with(r"\w+", Behavior::Isolated, "ab cd"),
        vec!["ab", " ", "cd"]
    );
}

#[test]
fn split_invert_partitions_the_whole_piece() {
    let re = gpt2_regex().expect("GPT2_PATTERN compiles");
    let mut out = Vec::new();
    split_regex(
        "def f(x):\n    return x",
        &re,
        Behavior::Isolated,
        true,
        &mut out,
    );
    assert_eq!(out.concat(), "def f(x):\n    return x");
    assert!(out.len() > 1, "a GPT-2 pattern must split this into pieces");
}

/// HuggingFace's `Split` takes either a literal string or a regex, and they
/// are not interchangeable. Reference (`tokenizers` package, behavior
/// `removed`, input `"a.b c"`): `Split(pattern=".")` yields
/// `[('a', (0,1)), ('b c', (2,5))]`, while `Split(pattern=Regex("."))`
/// matches every character and yields nothing.
#[test]
fn literal_and_regex_split_patterns_are_not_interchangeable() {
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

#[test]
fn literal_split_pattern_matches_metacharacters_verbatim() {
    let split = |pattern, text: &str| {
        PreTokenizer::new(vec![PreTokStage::Split {
            pattern,
            behavior: SplitBehavior::Removed,
            invert: false,
        }])
        .expect("pipeline builds")
        .split(text)
    };
    // As a regex `a+b` would need one-or-more `a`; as a literal it is the
    // three characters, which appear only in the middle here.
    assert_eq!(
        split(SplitPattern::Literal("a+b".to_string()), "xa+by"),
        vec!["x", "y"]
    );
    // As a regex `|` is an empty alternation matching everywhere.
    assert_eq!(
        split(SplitPattern::Literal("|".to_string()), "a|b"),
        vec!["a", "b"]
    );
}

#[test]
fn split_with_uncompilable_pattern_is_an_error() {
    // Previously such a stage was silently dropped, which changed the split
    // (and so the ids) with nothing to point at.
    let json = serde_json::json!({
        "type": "Split",
        "pattern": {"Regex": "("},
        "behavior": "Isolated"
    });
    assert!(parse(Some(&json)).is_err());
}

/// Everything the loader builds from JSON must be expressible through the
/// public [`PreTokStage`] builder: `parse` reports the spec it used, and
/// rebuilding from that spec must pre-tokenize identically. Adding a `Stage`
/// without a `PreTokStage` counterpart fails here.
#[test]
fn parsed_stages_round_trip_through_the_public_builder() {
    let probe = "Hello, wörld 42 items!";
    let mut cases: Vec<(Value, Vec<PreTokStage>)> = vec![
        // Nested Sequence, ByteLevel with use_regex:false + add_prefix_space.
        (
            serde_json::json!({
                "type": "Sequence",
                "pretokenizers": [
                    {"type": "Sequence", "pretokenizers": [
                        {"type": "Punctuation", "behavior": "Contiguous"},
                        {"type": "Digits", "individual_digits": true}
                    ]},
                    {"type": "ByteLevel", "use_regex": false, "add_prefix_space": true}
                ]
            }),
            vec![
                PreTokStage::Punctuation {
                    behavior: SplitBehavior::Contiguous,
                },
                PreTokStage::Digits { individual: true },
                PreTokStage::ByteLevel {
                    use_regex: false,
                    add_prefix_space: true,
                },
            ],
        ),
        // Bare ByteLevel: use_regex defaults to true, add_prefix_space to false.
        (
            serde_json::json!({"type": "ByteLevel"}),
            vec![PreTokStage::ByteLevel {
                use_regex: true,
                add_prefix_space: false,
            }],
        ),
        (
            serde_json::json!({"type": "Whitespace"}),
            vec![PreTokStage::Whitespace],
        ),
        (
            serde_json::json!({"type": "WhitespaceSplit"}),
            vec![PreTokStage::WhitespaceSplit],
        ),
        // A `String` pattern is a literal, a `Regex` pattern is a regex —
        // conflating them changes the split for any metacharacter.
        (
            serde_json::json!({
                "type": "Split",
                "pattern": {"String": ","},
                "behavior": "Removed"
            }),
            vec![PreTokStage::Split {
                pattern: SplitPattern::Literal(",".to_string()),
                behavior: SplitBehavior::Removed,
                invert: false,
            }],
        ),
        // A `String` whose text is regex-significant still matches literally.
        (
            serde_json::json!({
                "type": "Split",
                "pattern": {"String": "."},
                "behavior": "Removed"
            }),
            vec![PreTokStage::Split {
                pattern: SplitPattern::Literal(".".to_string()),
                behavior: SplitBehavior::Removed,
                invert: false,
            }],
        ),
        (
            serde_json::json!({
                "type": "Split",
                "pattern": {"Regex": r"\w+"},
                "invert": true
            }),
            vec![PreTokStage::Split {
                pattern: SplitPattern::Regex(r"\w+".to_string()),
                behavior: SplitBehavior::Isolated,
                invert: true,
            }],
        ),
    ];
    // Every delimiter behavior, spelled as HuggingFace spells it.
    for (name, behavior) in [
        ("Isolated", SplitBehavior::Isolated),
        ("Removed", SplitBehavior::Removed),
        ("MergedWithPrevious", SplitBehavior::MergedWithPrevious),
        ("MergedWithNext", SplitBehavior::MergedWithNext),
        ("Contiguous", SplitBehavior::Contiguous),
    ] {
        cases.push((
            serde_json::json!({
                "type": "Split",
                "pattern": {"Regex": r"\s+"},
                "behavior": name
            }),
            vec![PreTokStage::Split {
                pattern: SplitPattern::Regex(r"\s+".to_string()),
                behavior,
                invert: false,
            }],
        ));
    }

    for (json, expected) in cases {
        let parsed = parse(Some(&json)).expect("parses").expect("pipeline");
        assert_eq!(parsed.stages(), expected.as_slice(), "spec for {json}");
        let built = PreTokenizer::new(expected).expect("builds");
        assert_eq!(built.byte_level(), parsed.byte_level(), "byte_level {json}");
        assert_eq!(built.split(probe), parsed.split(probe), "split for {json}");
    }
}
