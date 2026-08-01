//! Verify splintr's GGUF loader against llama.cpp's own tokenizer fixtures.
//!
//! llama.cpp ships, next to each `models/ggml-vocab-*.gguf`, an `.inp` file of
//! test inputs and an `.out` file of the token ids its tokenizer produces. Those
//! files are the only independent ground truth we have for the GGUF dialects, so
//! this example diffs splintr against them.
//!
//! `research/` (where those vocabularies live) is not committed, so the GGUF
//! files are turned into self-contained JSON once, out of band:
//!
//! ```text
//! python3 scripts/extract_gguf_vocab.py <llama.cpp>/models --out-dir /tmp/gguf --require-cases
//! cargo run --example verify_gguf -- /tmp/gguf
//! ```
//!
//! It is an example rather than a test precisely because it needs those
//! generated inputs: it is committed, it never runs in `cargo test`, and it
//! exits non-zero when any case disagrees.
//!
//! # Which encode entry point?
//!
//! llama.cpp's `tests/test-tokenizer-0.cpp` calls
//! `common_tokenize(ctx, text, add_special, parse_special)` with
//! `add_special = false` and `parse_special = false`: no BOS/EOS is added, and
//! special tokens are *not* matched inside the input text. The corresponding
//! splintr call is [`AnyTokenizer::encode_raw`], which is what
//! [`Mode::Raw`] uses. [`Mode::Template`] runs [`AnyTokenizer::encode`] — the
//! policy's boundary template — so a mismatch that is only about a leading BOS
//! is visible as such instead of being reported as 46 broken cases. Both modes
//! are always run and the per-vocabulary summary names the one that matched.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use serde_json::Value;
use splintr::{from_gguf_vocab, AnyTokenizer, GgufVocab};

/// How the harness turns a case's text into ids.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// `encode_raw` — backend output with no boundary template. Matches
    /// llama.cpp's `add_special = false`.
    Raw,
    /// `encode` — the policy's single-sequence template applied on top.
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

/// One `.inp` case paired with the ids llama.cpp produced for it.
struct Case {
    input: String,
    expected: Vec<u32>,
}

/// A whole fixture: the vocabulary metadata plus every case for it.
struct Fixture {
    name: String,
    vocab: GgufVocab,
    cases: Vec<Case>,
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: cargo run --example verify_gguf -- <fixture.json | dir> [...]");
        eprintln!("       fixtures come from scripts/extract_gguf_vocab.py");
        return ExitCode::from(2);
    }

    let mut paths = Vec::new();
    for arg in &args {
        if let Err(err) = collect(Path::new(arg), &mut paths) {
            eprintln!("error: {arg}: {err}");
            return ExitCode::from(2);
        }
    }
    paths.sort();

    if paths.is_empty() {
        eprintln!("error: no .json fixtures found in {args:?}");
        return ExitCode::from(2);
    }

    let mut failed = false;
    // name -> one-line verdict, printed together at the end so a 13-vocabulary
    // run is readable without scrolling back through the failure detail.
    let mut summary: BTreeMap<String, String> = BTreeMap::new();

    for path in &paths {
        let fixture = match load_fixture(path) {
            Ok(fixture) => fixture,
            Err(err) => {
                failed = true;
                let name = path.file_stem().map_or_else(
                    || path.display().to_string(),
                    |s| s.to_string_lossy().into_owned(),
                );
                println!("\n=== {name} ===\n  FIXTURE ERROR: {err}");
                summary.insert(name, format!("FIXTURE ERROR: {err}"));
                continue;
            }
        };

        match verify(&fixture) {
            Ok(line) => {
                if line.starts_with("FAIL") {
                    failed = true;
                }
                summary.insert(fixture.name.clone(), line);
            }
            Err(err) => {
                failed = true;
                println!("\n=== {} ===\n  LOAD ERROR: {err}", fixture.name);
                summary.insert(fixture.name.clone(), format!("LOAD ERROR: {err}"));
            }
        }
    }

    println!("\n=== summary ===");
    for (name, line) in &summary {
        println!("  {name:<28} {line}");
    }

    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

/// Run every case in both modes and print the detail for the better one.
///
/// Returns the one-line verdict, or the loader's error when the vocabulary
/// cannot be built at all.
fn verify(fixture: &Fixture) -> Result<String, String> {
    let model = fixture.vocab.model.clone();
    let pre = fixture.vocab.pre.clone().unwrap_or_else(|| "-".to_owned());
    let tokenizer = from_gguf_vocab(fixture.vocab.clone()).map_err(|e| e.to_string())?;

    println!(
        "\n=== {} ===\n  model={model} pre={pre} family={} cases={}",
        fixture.name,
        tokenizer.family(),
        fixture.cases.len()
    );

    // Every mode is scored before anything is printed in detail, so the failure
    // dump below describes the mode that actually comes closest rather than a
    // mode chosen up front.
    let mut results: Vec<(Mode, Vec<usize>)> = Vec::new();
    for mode in Mode::ALL {
        let mut failures = Vec::new();
        for (index, case) in fixture.cases.iter().enumerate() {
            if mode.run(&tokenizer, &case.input) != case.expected {
                failures.push(index);
            }
        }
        results.push((mode, failures));
    }

    for (mode, failures) in &results {
        let passed = fixture.cases.len() - failures.len();
        println!(
            "  {:<11} {passed}/{} passed",
            mode.name(),
            fixture.cases.len()
        );
    }

    let Some((best_mode, best_failures)) = results.iter().min_by_key(|(_, f)| f.len()) else {
        return Err("no modes were run".to_owned());
    };

    if best_failures.is_empty() {
        return Ok(format!(
            "ok   ({}/{} via {})",
            fixture.cases.len(),
            fixture.cases.len(),
            best_mode.name()
        ));
    }

    println!(
        "\n  --- failures under {} ({} of {}) ---",
        best_mode.name(),
        best_failures.len(),
        fixture.cases.len()
    );
    for &index in best_failures {
        let case = &fixture.cases[index];
        let actual = best_mode.run(&tokenizer, &case.input);
        println!("  [case {index}] input: \"{}\"", escape(&case.input));
        println!(
            "    expected ({:>3}): {}",
            case.expected.len(),
            ids(&case.expected)
        );
        println!("    actual   ({:>3}): {}", actual.len(), ids(&actual));
        match first_diff(&case.expected, &actual) {
            Some(at) => println!(
                "    first differs at index {at}: expected {}, got {}",
                at_or_end(&case.expected, at),
                at_or_end(&actual, at)
            ),
            // Unreachable while the vectors differ, but stated rather than
            // asserted so the example never panics.
            None => println!("    (sequences compare equal — nothing to report)"),
        }
    }

    Ok(format!(
        "FAIL ({}/{} via {})",
        fixture.cases.len() - best_failures.len(),
        fixture.cases.len(),
        best_mode.name()
    ))
}

/// The first index at which two id sequences disagree, length included.
fn first_diff(expected: &[u32], actual: &[u32]) -> Option<usize> {
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

fn ids(ids: &[u32]) -> String {
    ids.iter().map(u32::to_string).collect::<Vec<_>>().join(" ")
}

/// Render a case's text with whitespace visible — most of these cases differ
/// only in leading spaces, tabs and newlines.
fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            ' ' => out.push('␣'),
            c if c.is_control() => out.push_str(&format!("\\u{{{:04x}}}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Gather `.json` fixtures from a file or a directory of them.
fn collect(path: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    if path.is_dir() {
        let entries = fs::read_dir(path).map_err(|e| e.to_string())?;
        for entry in entries {
            let entry = entry.map_err(|e| e.to_string())?;
            let candidate = entry.path();
            if candidate.extension().is_some_and(|e| e == "json") {
                out.push(candidate);
            }
        }
        Ok(())
    } else if path.is_file() {
        out.push(path.to_path_buf());
        Ok(())
    } else {
        Err("no such file or directory".to_owned())
    }
}

// ---------------------------------------------------------------------------
// JSON -> GgufVocab
//
// Built field by field rather than through a `Deserialize` derive: deriving on
// `GgufVocab` would mean a `serde` dependency in the library itself, and the
// whole point of that struct is that it costs a consumer nothing. `serde_json`
// is already a dependency of splintr (the HuggingFace json loader uses it), so
// the example adds no dependency at all.
// ---------------------------------------------------------------------------

fn load_fixture(path: &Path) -> Result<Fixture, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("read: {e}"))?;
    let json: Value = serde_json::from_str(&text).map_err(|e| format!("parse: {e}"))?;

    let name = match json.get("name").and_then(Value::as_str) {
        Some(name) => name.to_owned(),
        None => path.file_stem().map_or_else(
            || path.display().to_string(),
            |s| s.to_string_lossy().into_owned(),
        ),
    };

    let raw_vocab = json
        .get("vocab")
        .ok_or_else(|| "missing `vocab` object".to_owned())?;

    let vocab = GgufVocab {
        model: raw_vocab
            .get("model")
            .and_then(Value::as_str)
            .ok_or_else(|| "vocab.model missing or not a string".to_owned())?
            .to_owned(),
        tokens: string_vec(raw_vocab, "tokens")
            .ok_or_else(|| "vocab.tokens missing or not an array of strings".to_owned())?,
        scores: f32_vec(raw_vocab, "scores"),
        merges: string_vec(raw_vocab, "merges"),
        token_type: u32_vec(raw_vocab, "token_type"),
        add_space_prefix: opt_bool(raw_vocab, "add_space_prefix"),
        remove_extra_whitespaces: opt_bool(raw_vocab, "remove_extra_whitespaces"),
        add_bos_token: opt_bool(raw_vocab, "add_bos_token"),
        add_eos_token: opt_bool(raw_vocab, "add_eos_token"),
        bos_token_id: opt_u32(raw_vocab, "bos_token_id"),
        eos_token_id: opt_u32(raw_vocab, "eos_token_id"),
        unknown_token_id: opt_u32(raw_vocab, "unknown_token_id"),
        padding_token_id: opt_u32(raw_vocab, "padding_token_id"),
        cls_token_id: opt_u32(raw_vocab, "cls_token_id"),
        sep_token_id: opt_u32(raw_vocab, "sep_token_id"),
        pre: raw_vocab
            .get("pre")
            .and_then(Value::as_str)
            .map(str::to_owned),
    };

    let mut cases = Vec::new();
    if let Some(array) = json.get("cases").and_then(Value::as_array) {
        for (index, entry) in array.iter().enumerate() {
            let input = entry
                .get("input")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("case {index}: `input` missing or not a string"))?
                .to_owned();
            let expected = u32_vec(entry, "expected")
                .ok_or_else(|| format!("case {index}: `expected` missing or not an id array"))?;
            cases.push(Case { input, expected });
        }
    }

    Ok(Fixture { name, vocab, cases })
}

fn opt_bool(value: &Value, key: &str) -> Option<bool> {
    value.get(key).and_then(Value::as_bool)
}

fn opt_u32(value: &Value, key: &str) -> Option<u32> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|n| u32::try_from(n).ok())
}

fn string_vec(value: &Value, key: &str) -> Option<Vec<String>> {
    let array = value.get(key)?.as_array()?;
    array
        .iter()
        .map(|v| v.as_str().map(str::to_owned))
        .collect()
}

fn f32_vec(value: &Value, key: &str) -> Option<Vec<f32>> {
    let array = value.get(key)?.as_array()?;
    array.iter().map(|v| v.as_f64().map(|f| f as f32)).collect()
}

fn u32_vec(value: &Value, key: &str) -> Option<Vec<u32>> {
    let array = value.get(key)?.as_array()?;
    array
        .iter()
        .map(|v| v.as_u64().and_then(|n| u32::try_from(n).ok()))
        .collect()
}
