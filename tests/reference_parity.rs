//! Diff splintr's bundled pretrained vocabularies against independent
//! reference-tokenizer output, automatically, in `cargo test`.
//!
//! The fixtures this test reads (`tests/fixtures/pretrained/*.json`) are
//! generated out of band by `scripts/extract_reference_cases.py` -- see that
//! script's module docstring for which reference tool is authoritative for
//! which vocabulary (`tiktoken`, `tokenizers` or `sentencepiece`), how each
//! fixture is produced, and how it is gated against the bundled vocabulary it
//! pairs with. This test performs no network access and requires no Python at
//! test time; it only reads the already-committed JSON.
//!
//! # Which encode entry point?
//!
//! Fixtures are generated with the reference's *untemplated* encode. This
//! mirrors the mode-selection logic in `examples/verify_pretrained.rs`: both
//! [`AnyTokenizer::encode_raw`] (no boundary template, matching the
//! untemplated reference) and [`AnyTokenizer::encode`] (the policy's
//! template) are tried for every case, and whichever produces fewer
//! mismatches is treated as the authoritative comparison -- so this test and
//! that example can never silently disagree about what "correct" means for a
//! given vocabulary.
//!
//! # Why decode is checked too
//!
//! Encode and decode are separate pipelines, and pinning only ids leaves
//! byte-level unmapping, byte fallback, the SentencePiece dummy-prefix strip
//! and special-token skipping entirely unpinned -- historically where a large
//! share of this crate's real divergences have been. So every case also
//! carries the reference's own decode of the ids it produced, and
//! [`AnyTokenizer::decode`] is asserted against it. That check runs over the
//! *reference's* ids rather than splintr's, so a decode failure is reported as
//! a decode failure even when encode already disagrees.
//!
//! # Why the pre-tokenizer split is checked too
//!
//! Ids and decoded text pin the two ends of the pipeline but say nothing
//! directly about the stage between them. A pre-tokenizer pattern transcribed
//! from a `tokenizer.json` can drift from it — a lowered quantifier, a dropped
//! alternation branch — and stay completely invisible until the drift happens
//! to move a token id on some input nobody tested. So every case whose
//! reference *has* a pre-tokenizer also carries that reference's own split,
//! piece for piece, asserted against [`AnyTokenizer::pre_tokenize`].
//!
//! Both columns are optional and skipped when absent: `pieces` is missing for
//! the SentencePiece-backed vocabularies, which have no pre-tokenizer stage at
//! all, and for individual cases that split into nothing. `normalized`, when
//! present, is the reference's own normalization stage's output.
//!
//! # Why the normalization stage is checked too
//!
//! `normalized` does double duty, and the two uses are the same fact read from
//! two sides rather than two meanings of one column:
//!
//! * It is the *input* to the split comparison — the text the reference's own
//!   split ran on, and therefore what [`AnyTokenizer::pre_tokenize`] (which
//!   does not normalize; see its docs) must be driven with instead of `input`.
//! * It is the *expected output* of [`AnyTokenizer::normalize`], which is the
//!   stage that produces exactly that text.
//!
//! So a fixture stating both columns pins the two stages and their hand-off in
//! one go, and a disagreement is reported against whichever of the two is
//! actually wrong instead of surfacing only as a confusing split diff.
//!
//! This is what covers the SentencePiece-backed vocabularies' front end, which
//! is otherwise pinned by ids alone: they have no pre-tokenizer to report, and
//! their `▁` escaping and dummy prefix are the whole of the stage between input
//! and merge loop. `sentencepiece` exposes precisely that as
//! `SentencePieceProcessor.normalize`.
//!
//! This file intentionally duplicates a small amount of fixture-parsing and
//! mode-selection logic from `examples/verify_pretrained.rs` rather than
//! sharing a module: the example is a standalone diagnostic binary and this
//! is a `cargo test` integration test, and factoring out ~15 lines shared
//! between them was not worth the coupling.

use serde_json::Value;
use splintr::pretrained::from_pretrained;
use splintr::AnyTokenizer;
use std::fs;
use std::path::Path;

/// How a case's text is turned into ids -- see the module docs above.
#[derive(Clone, Copy)]
enum Mode {
    /// `encode_raw` -- backend output, no boundary template.
    Raw,
    /// `encode` -- the policy's single-sequence template applied on top.
    Template,
}

impl Mode {
    const ALL: [Mode; 2] = [Mode::Raw, Mode::Template];

    fn name(self) -> &'static str {
        match self {
            Mode::Raw => "encode_raw",
            Mode::Template => "encode",
        }
    }

    fn run(self, tokenizer: &AnyTokenizer, text: &str) -> Vec<u32> {
        match self {
            Mode::Raw => tokenizer.encode_raw(text),
            Mode::Template => tokenizer.encode(text),
        }
    }
}

/// One reference case: input text, the reference tokenizer's ids, the
/// reference tokenizer's own decode of exactly those ids, and — when that
/// reference has a pre-tokenizer — its own split of the input.
struct Case {
    input: String,
    expected: Vec<u32>,
    decoded: String,
    /// The reference's pre-tokenizer pieces, as raw text. `None` for a
    /// reference with no pre-tokenizer stage, and for a case that splits into
    /// nothing — see the module docs.
    pieces: Option<Vec<String>>,
    /// The reference's normalization-stage output, recorded only where it
    /// differs from `input`; that is both the text `pieces` is a split of and
    /// what `AnyTokenizer::normalize` has to produce — see the module docs.
    normalized: Option<String>,
}

/// A whole fixture: the bundled vocabulary name plus every case for it.
struct Fixture {
    vocab: String,
    cases: Vec<Case>,
}

fn load_fixture(path: &Path) -> Fixture {
    let text = fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("reference_parity: failed to read {}: {e}", path.display()));
    let json: Value = serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("reference_parity: failed to parse {}: {e}", path.display()));

    let vocab = json
        .get("vocab")
        .and_then(Value::as_str)
        .unwrap_or_else(|| {
            panic!(
                "reference_parity: {}: missing `vocab` string",
                path.display()
            )
        })
        .to_owned();

    let array = json
        .get("cases")
        .and_then(Value::as_array)
        .unwrap_or_else(|| {
            panic!(
                "reference_parity: {}: missing `cases` array",
                path.display()
            )
        });

    let cases = array
        .iter()
        .enumerate()
        .map(|(index, entry)| {
            let input = entry
                .get("input")
                .and_then(Value::as_str)
                .unwrap_or_else(|| {
                    panic!(
                        "reference_parity: {}: case {index}: `input` missing or not a string",
                        path.display()
                    )
                })
                .to_owned();
            let expected: Vec<u32> = entry
                .get("expected")
                .and_then(Value::as_array)
                .unwrap_or_else(|| {
                    panic!(
                        "reference_parity: {}: case {index}: `expected` missing or not an array",
                        path.display()
                    )
                })
                .iter()
                .enumerate()
                .map(|(id_index, v)| {
                    v.as_u64()
                        .and_then(|n| u32::try_from(n).ok())
                        .unwrap_or_else(|| {
                            panic!(
                                "reference_parity: {}: case {index}: expected[{id_index}] is not a valid token id",
                                path.display()
                            )
                        })
                })
                .collect();
            // Required, not optional: a fixture without it would silently
            // check half of what this test exists to check.
            let decoded = entry
                .get("decoded")
                .and_then(Value::as_str)
                .unwrap_or_else(|| {
                    panic!(
                        "reference_parity: {}: case {index}: `decoded` missing or not a string -- \
                         regenerate this fixture with scripts/extract_reference_cases.py",
                        path.display()
                    )
                })
                .to_owned();
            // Optional, unlike `decoded`: a reference with no pre-tokenizer
            // stage has nothing to state here, and an empty array would assert
            // "splits into zero pieces" instead. A present-but-malformed one is
            // still a hard error.
            let pieces = entry.get("pieces").map(|value| {
                value
                    .as_array()
                    .unwrap_or_else(|| {
                        panic!(
                            "reference_parity: {}: case {index}: `pieces` is not an array",
                            path.display()
                        )
                    })
                    .iter()
                    .enumerate()
                    .map(|(piece_index, v)| {
                        v.as_str()
                            .unwrap_or_else(|| {
                                panic!(
                                    "reference_parity: {}: case {index}: pieces[{piece_index}] is not a string",
                                    path.display()
                                )
                            })
                            .to_owned()
                    })
                    .collect()
            });
            let normalized = entry.get("normalized").map(|value| {
                value
                    .as_str()
                    .unwrap_or_else(|| {
                        panic!(
                            "reference_parity: {}: case {index}: `normalized` is not a string",
                            path.display()
                        )
                    })
                    .to_owned()
            });
            Case {
                input,
                expected,
                decoded,
                pieces,
                normalized,
            }
        })
        .collect();

    Fixture { vocab, cases }
}

/// The first index at which two sequences disagree, length included.
fn first_diff<T: PartialEq>(expected: &[T], actual: &[T]) -> Option<usize> {
    let shared = expected.len().min(actual.len());
    for i in 0..shared {
        if expected[i] != actual[i] {
            return Some(i);
        }
    }
    if expected.len() == actual.len() {
        None
    } else {
        Some(shared)
    }
}

fn at_or_end(ids: &[u32], index: usize) -> String {
    match ids.get(index) {
        Some(id) => id.to_string(),
        None => "<end>".to_owned(),
    }
}

fn format_ids(ids: &[u32]) -> String {
    ids.iter().map(u32::to_string).collect::<Vec<_>>().join(" ")
}

/// [`at_or_end`] for pre-tokenizer pieces, quoted and whitespace-escaped —
/// they differ from one another almost exclusively in leading spaces.
fn piece_at_or_end(pieces: &[String], index: usize) -> String {
    match pieces.get(index) {
        Some(piece) => format!("\"{}\"", escape(piece)),
        None => "<end>".to_owned(),
    }
}

/// [`format_ids`] for pre-tokenizer pieces.
fn format_pieces(pieces: &[String]) -> String {
    pieces
        .iter()
        .map(|piece| format!("\"{}\"", escape(piece)))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Render text with whitespace visible -- most reference cases differ only
/// in leading spaces, tabs and newlines, which are otherwise invisible in a
/// panic message.
fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            ' ' => out.push('\u{2423}'),
            c if c.is_control() => out.push_str(&format!("\\u{{{:04x}}}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Diff every bundled-vocabulary fixture in `tests/fixtures/pretrained/`
/// against the independent reference-tokenizer output it was generated from
/// (see `scripts/extract_reference_cases.py`), on ids, decoded text and — where
/// the fixture states them — the normalization stage and pre-tokenizer split.
///
/// Fails loudly (rather than vacuously passing) if the fixtures directory is
/// missing or contains no `.json` files, so an accidental deletion of the
/// fixtures cannot silently turn this into a no-op.
#[test]
fn pretrained_vocabularies_match_reference_tokenizers() {
    let fixtures_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pretrained");

    assert!(
        fixtures_dir.is_dir(),
        "reference_parity: fixtures directory {} does not exist -- \
         run scripts/extract_reference_cases.py to (re)generate it",
        fixtures_dir.display()
    );

    let mut fixture_paths: Vec<_> = fs::read_dir(&fixtures_dir)
        .unwrap_or_else(|e| {
            panic!(
                "reference_parity: failed to read {}: {e}",
                fixtures_dir.display()
            )
        })
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .collect();
    fixture_paths.sort();

    assert!(
        !fixture_paths.is_empty(),
        "reference_parity: no .json fixtures found in {} -- \
         this test must never silently pass with zero fixtures",
        fixtures_dir.display()
    );

    let mut failure_reports: Vec<String> = Vec::new();

    for path in &fixture_paths {
        let fixture = load_fixture(path);
        let tokenizer = from_pretrained(&fixture.vocab).unwrap_or_else(|e| {
            panic!(
                "reference_parity: {}: from_pretrained({:?}) failed: {e}",
                path.display(),
                fixture.vocab
            )
        });

        // Score every mode across all cases before deciding which one to
        // report against, mirroring examples/verify_pretrained.rs: the
        // reported mismatches are always the ones under whichever mode
        // comes closest, not a mode chosen up front.
        let mut per_mode: Vec<(Mode, Vec<usize>)> = Vec::new();
        for mode in Mode::ALL {
            let failures: Vec<usize> = fixture
                .cases
                .iter()
                .enumerate()
                .filter(|(_, case)| mode.run(&tokenizer, &case.input) != case.expected)
                .map(|(index, _)| index)
                .collect();
            per_mode.push((mode, failures));
        }

        let (best_mode, best_failures) = per_mode
            .into_iter()
            .min_by_key(|(_, failures)| failures.len())
            .expect("Mode::ALL is non-empty");

        // Decode is scored independently of the encode mode: it is driven from
        // the reference's own ids, so it says something even when encode
        // already disagrees.
        let decode_failures: Vec<(usize, Result<String, String>)> = fixture
            .cases
            .iter()
            .enumerate()
            .filter_map(|(index, case)| match tokenizer.decode(&case.expected) {
                Ok(text) if text == case.decoded => None,
                Ok(text) => Some((index, Ok(text))),
                Err(e) => Some((index, Err(e.to_string()))),
            })
            .collect();

        if !decode_failures.is_empty() {
            let mut detail = format!(
                "vocab {:?} ({}): {}/{} cases decoded differently from the reference:",
                fixture.vocab,
                path.display(),
                decode_failures.len(),
                fixture.cases.len(),
            );
            for (index, outcome) in &decode_failures {
                let case = &fixture.cases[*index];
                let actual = match outcome {
                    Ok(text) => format!("\"{}\"", escape(text)),
                    Err(message) => format!("<error: {message}>"),
                };
                detail.push_str(&format!(
                    "\n  [case {index}] input: \"{}\"\n    ids: {}\
                     \n    expected: \"{}\"\n    actual:   {actual}",
                    escape(&case.input),
                    format_ids(&case.expected),
                    escape(&case.decoded),
                ));
            }
            failure_reports.push(detail);
        }

        // The normalization stage, scored independently of the encode mode for
        // the same reason decode is. Cases with no `normalized` are skipped,
        // not failed: the column is written only where the reference's
        // normalization changed the input, so its absence asserts nothing --
        // and asserting the identity everywhere else would turn "this reference
        // does not normalize" into a claim about splintr's backend.
        let normalize_failures: Vec<(usize, Option<String>)> = fixture
            .cases
            .iter()
            .enumerate()
            .filter_map(|(index, case)| {
                let expected = case.normalized.as_ref()?;
                match tokenizer.normalize(&case.input) {
                    Some(actual) if actual == *expected => None,
                    outcome => Some((index, outcome)),
                }
            })
            .collect();

        if !normalize_failures.is_empty() {
            let mut detail = format!(
                "vocab {:?} ({}): {}/{} cases normalized differently from the reference:",
                fixture.vocab,
                path.display(),
                normalize_failures.len(),
                fixture.cases.len(),
            );
            for (index, outcome) in &normalize_failures {
                let case = &fixture.cases[*index];
                let actual = match outcome {
                    Some(text) => format!("\"{}\"", escape(text)),
                    // The fixture states a normalization this backend cannot
                    // report at all -- a wrong reference pairing, not a wrong
                    // pipeline.
                    None => "<this backend exposes no normalization stage>".to_owned(),
                };
                detail.push_str(&format!(
                    "\n  [case {index}] input: \"{}\"\n    expected: \"{}\"\n    actual:   {actual}",
                    escape(&case.input),
                    escape(case.normalized.as_deref().unwrap_or_default()),
                ));
            }
            failure_reports.push(detail);
        }

        // The pre-tokenizer split, scored independently of the encode mode for
        // the same reason decode is: it is a stage the boundary template never
        // touches. Cases with no `pieces` are skipped, not failed.
        let piece_failures: Vec<(usize, Option<Vec<String>>)> = fixture
            .cases
            .iter()
            .enumerate()
            .filter_map(|(index, case)| {
                let expected = case.pieces.as_ref()?;
                let text = case.normalized.as_deref().unwrap_or(&case.input);
                match tokenizer.pre_tokenize(text) {
                    Some(actual) if actual == *expected => None,
                    outcome => Some((index, outcome)),
                }
            })
            .collect();

        if !piece_failures.is_empty() {
            let mut detail = format!(
                "vocab {:?} ({}): {}/{} cases pre-tokenized differently from the reference:",
                fixture.vocab,
                path.display(),
                piece_failures.len(),
                fixture.cases.len(),
            );
            for (index, outcome) in &piece_failures {
                let case = &fixture.cases[*index];
                let expected = case.pieces.as_deref().unwrap_or(&[]);
                let text = case.normalized.as_deref().unwrap_or(&case.input);
                detail.push_str(&format!(
                    "\n  [case {index}] input: \"{}\"\n    expected ({:>3}): {}",
                    escape(text),
                    expected.len(),
                    format_pieces(expected),
                ));
                match outcome {
                    Some(actual) => {
                        detail.push_str(&format!(
                            "\n    actual   ({:>3}): {}",
                            actual.len(),
                            format_pieces(actual),
                        ));
                        match first_diff(expected, actual) {
                            Some(at) => detail.push_str(&format!(
                                "\n    first differs at index {at}: expected {}, got {}",
                                piece_at_or_end(expected, at),
                                piece_at_or_end(actual, at)
                            )),
                            None => {
                                detail.push_str("\n    (sequences compare equal -- unreachable)")
                            }
                        }
                    }
                    // The fixture states a split this backend cannot report at
                    // all -- a wrong reference pairing, not a wrong pattern.
                    None => detail
                        .push_str("\n    actual: <this backend exposes no pre-tokenizer split>"),
                }
            }
            failure_reports.push(detail);
        }

        if best_failures.is_empty() {
            continue;
        }

        let mut detail = format!(
            "vocab {:?} ({}): {}/{} cases mismatched under best mode `{}`:",
            fixture.vocab,
            path.display(),
            best_failures.len(),
            fixture.cases.len(),
            best_mode.name(),
        );
        for &index in &best_failures {
            let case = &fixture.cases[index];
            let actual = best_mode.run(&tokenizer, &case.input);
            let diff_at = first_diff(&case.expected, &actual);
            detail.push_str(&format!(
                "\n  [case {index}] input: \"{}\"\n    expected ({:>3}): {}\n    actual   ({:>3}): {}",
                escape(&case.input),
                case.expected.len(),
                format_ids(&case.expected),
                actual.len(),
                format_ids(&actual),
            ));
            match diff_at {
                Some(at) => detail.push_str(&format!(
                    "\n    first differs at index {at}: expected {}, got {}",
                    at_or_end(&case.expected, at),
                    at_or_end(&actual, at)
                )),
                None => detail.push_str("\n    (sequences compare equal -- unreachable)"),
            }
        }
        failure_reports.push(detail);
    }

    assert!(
        failure_reports.is_empty(),
        "reference_parity: splintr disagrees with the reference tokenizer:\n\n{}",
        failure_reports.join("\n\n")
    );
}
