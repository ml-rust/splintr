//! Are splintr's two SentencePiece-BPE implementations the same tokenizer?
//!
//! splintr ships two paths that both claim to be SentencePiece BPE:
//!
//! * [`SpmTokenizer`] — piece-level (`Vec<String>`), llama.cpp's
//!   `llm_tokenizer_spm` merge-by-score loop, byte fallback through the
//!   `<0xNN>` pieces. This is what a GGUF `model=llama` vocabulary loads into,
//!   and it reproduces llama.cpp's ids for `ggml-vocab-llama-spm` exactly.
//! * [`Tokenizer`] in SentencePiece mode (`from_bytes_sentencepiece`) —
//!   byte-level (`Vec<u8>`), tiktoken-style pairwise merges over a regex-chunked
//!   input. This is what the bundled MistralV1/V2 vocabularies use.
//!
//! If the byte-level path can reproduce llama.cpp's ids for the *same*
//! vocabulary, the two are equivalent on real input and one of them is
//! redundant. If it cannot, they are different algorithms — and the bundled
//! Mistral vocabularies, which are SentencePiece vocabularies on the byte-level
//! path, are on the wrong one.
//!
//! This example runs llama.cpp's own 46 `ggml-vocab-llama-spm` cases through
//! both paths and prints them side by side against llama.cpp's expected ids.
//!
//! ```text
//! python3 scripts/extract_gguf_vocab.py \
//!     research/llama.cpp-master/models/ggml-vocab-llama-spm.gguf \
//!     --out-dir /tmp/gguf --tiktoken-dir /tmp/gguf
//! cargo run --example verify_spm_vs_bpe -- \
//!     /tmp/gguf/ggml-vocab-llama-spm.json /tmp/gguf/ggml-vocab-llama-spm.tiktoken
//! ```
//!
//! # Which byte-level constructor, and why
//!
//! [`Tokenizer::from_bytes_sentencepiece`] — not
//! `from_bytes_sentencepiece_with_decoder`. The two differ only in how they
//! resolve byte sequences that appear at more than one id, and this vocabulary
//! has 95 of them: every `<0xNN>` byte-fallback piece whose byte is also a real
//! single-character piece (`<0x21>` at id 36 vs `!` at id 29991). `_with_decoder`
//! keeps the *lowest* id for the encoder, which would make every `!` encode as
//! the byte-fallback token 36; llama.cpp emits 29991, because in SPM byte
//! fallback applies only to text with no piece at all. Plain
//! `from_bytes_sentencepiece` keeps the *last* line for a repeated byte
//! sequence, and since the file is written in id order that is the real piece —
//! the id llama.cpp actually produces. Its decoder loses the duplicate ids, but
//! this experiment only compares encode output.
//!
//! # Prefix space
//!
//! llama.cpp's SPM applies `add_dummy_prefix` (GGUF `add_space_prefix`,
//! defaulting to true): `ied 4 ½ months` tokenizes as `▁i|ed|…`. The byte-level
//! `Tokenizer` has no notion of that flag; the closest equivalent a caller can
//! reach without touching library code is [`Tokenizer::with_prefix_space`],
//! which prepends a literal space that SentencePiece mode then turns into `▁`.
//! Both variants are scored, so the prefix is never the thing that decides the
//! verdict.

use std::fs;
use std::path::Path;
use std::process::ExitCode;

use base64::{engine::general_purpose::STANDARD, Engine};
use rustc_hash::FxHashMap;
use serde_json::Value;
use splintr::{from_gguf_vocab, AnyTokenizer, GgufVocab, Tokenizer, SENTENCEPIECE_PATTERN};

/// One `.inp` case paired with the ids llama.cpp produced for it.
struct Case {
    input: String,
    expected: Vec<u32>,
}

/// The fixture as `scripts/extract_gguf_vocab.py` writes it.
struct Fixture {
    name: String,
    vocab: GgufVocab,
    cases: Vec<Case>,
}

/// An encode entry point under test: text in, token ids out.
type EncodeFn<'a> = Box<dyn Fn(&str) -> Vec<u32> + 'a>;

/// One tokenizer under test plus its running score.
struct Candidate<'a> {
    label: &'static str,
    run: EncodeFn<'a>,
    passed: usize,
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [fixture_arg, tiktoken_arg] = match args.as_slice() {
        [a, b] => [a.clone(), b.clone()],
        _ => {
            eprintln!(
                "usage: cargo run --example verify_spm_vs_bpe -- <fixture.json> <vocab.tiktoken>"
            );
            eprintln!("       both come from scripts/extract_gguf_vocab.py (--tiktoken-dir)");
            return ExitCode::from(2);
        }
    };

    let fixture = match load_fixture(Path::new(&fixture_arg)) {
        Ok(fixture) => fixture,
        Err(err) => {
            eprintln!("error: {fixture_arg}: {err}");
            return ExitCode::from(2);
        }
    };
    let tiktoken = match fs::read(&tiktoken_arg) {
        Ok(bytes) => bytes,
        Err(err) => {
            eprintln!("error: {tiktoken_arg}: {err}");
            return ExitCode::from(2);
        }
    };

    println!(
        "=== {} ===\n  model={} pieces={} cases={}",
        fixture.name,
        fixture.vocab.model,
        fixture.vocab.tokens.len(),
        fixture.cases.len()
    );

    // The conversion is the one thing that could fake a tokenizer difference,
    // so it is re-derived here from the fixture and diffed against the file.
    if !check_conversion(&fixture, &tiktoken) {
        eprintln!("\nerror: the tiktoken rendering does not match the fixture — the");
        eprintln!("       comparison below would be measuring the conversion, not the");
        eprintln!("       tokenizers. Fix the conversion first.");
        return ExitCode::from(2);
    }

    let spm: AnyTokenizer = match from_gguf_vocab(fixture.vocab.clone()) {
        Ok(tokenizer) => tokenizer,
        Err(err) => {
            eprintln!("error: from_gguf_vocab: {err}");
            return ExitCode::from(2);
        }
    };
    println!("  SpmTokenizer family={}", spm.family());

    // No special tokens: llama.cpp's test harness runs with `parse_special =
    // false`, so nothing in the input text is matched as special.
    let specials: FxHashMap<String, u32> = FxHashMap::default();
    let bpe = match Tokenizer::from_bytes_sentencepiece(
        &tiktoken,
        SENTENCEPIECE_PATTERN,
        specials.clone(),
    ) {
        Ok(tokenizer) => tokenizer,
        Err(err) => {
            eprintln!("error: from_bytes_sentencepiece: {err}");
            return ExitCode::from(2);
        }
    };
    let bpe_prefixed =
        match Tokenizer::from_bytes_sentencepiece(&tiktoken, SENTENCEPIECE_PATTERN, specials) {
            Ok(tokenizer) => tokenizer.with_prefix_space(true),
            Err(err) => {
                eprintln!("error: from_bytes_sentencepiece: {err}");
                return ExitCode::from(2);
            }
        };

    let mut paths = vec![
        Candidate {
            // llama.cpp's `common_tokenize(..., add_special = false)`.
            label: "spm.encode_raw",
            run: Box::new(|text: &str| spm.encode_raw(text)),
            passed: 0,
        },
        Candidate {
            label: "bpe.encode",
            run: Box::new(|text: &str| bpe.encode(text)),
            passed: 0,
        },
        Candidate {
            label: "bpe.encode+prefix",
            run: Box::new(|text: &str| bpe_prefixed.encode(text)),
            passed: 0,
        },
    ];

    println!("\n--- per case (llama.cpp expected first) ---");
    for (index, case) in fixture.cases.iter().enumerate() {
        println!("\n[case {index}] input: \"{}\"", escape(&case.input));
        println!(
            "  {:<18} {:>3}  {}",
            "llama.cpp",
            case.expected.len(),
            ids(&case.expected)
        );
        for path in paths.iter_mut() {
            let actual = (path.run)(&case.input);
            let ok = actual == case.expected;
            if ok {
                path.passed += 1;
            }
            println!(
                "  {:<18} {:>3}  {}   {}",
                path.label,
                actual.len(),
                ids(&actual),
                if ok {
                    "ok".to_owned()
                } else {
                    match first_diff(&case.expected, &actual) {
                        Some(at) => format!(
                            "DIFF @{at}: expected {}, got {}",
                            at_or_end(&case.expected, at),
                            at_or_end(&actual, at)
                        ),
                        None => "DIFF".to_owned(),
                    }
                }
            );
        }
    }

    let total = fixture.cases.len();
    println!("\n=== summary ({total} cases from {}) ===", fixture.name);
    for candidate in &paths {
        println!(
            "  {:<18} {}/{} match llama.cpp",
            candidate.label, candidate.passed, total
        );
    }

    let spm_score = paths[0].passed;
    let bpe_score = paths[1..].iter().map(|p| p.passed).max().unwrap_or(0);
    println!(
        "\n  piece-level SpmTokenizer: {spm_score}/{total}\
         \n  byte-level Tokenizer (best of both prefix settings): {bpe_score}/{total}"
    );
    println!(
        "  {}",
        if bpe_score == spm_score {
            "the two paths agree with llama.cpp equally — equivalent on these cases"
        } else {
            "the two paths are NOT equivalent on these cases"
        }
    );

    // The exit code reports the health of the experiment, not its finding: a
    // byte-level score of 0 is a result, but a regressed SpmTokenizer means the
    // reference this comparison rests on is broken.
    if spm_score < total {
        eprintln!("\nerror: SpmTokenizer no longer reproduces every case — the reference");
        eprintln!("       leg of this comparison is broken, so the finding is unusable.");
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

// ---------------------------------------------------------------------------
// Conversion sanity checks
// ---------------------------------------------------------------------------

/// Re-derive the piece → bytes mapping in Rust and diff it against the file.
///
/// Returns false only when the *conversion* is broken (line count, id order, or
/// byte content disagree). Duplicate byte sequences are reported, not rejected:
/// they are inherent to this vocabulary, and which id survives is a documented
/// property of the loader the comparison is about to exercise.
fn check_conversion(fixture: &Fixture, tiktoken: &[u8]) -> bool {
    let mut lines: Vec<(Vec<u8>, u32)> = Vec::new();
    for (line_no, line) in tiktoken.split(|&b| b == b'\n').enumerate() {
        if line.is_empty() {
            continue;
        }
        let Some(space) = line.iter().rposition(|&b| b == b' ') else {
            println!("  CONVERSION: line {line_no} has no space separator");
            return false;
        };
        let Ok(bytes) = STANDARD.decode(&line[..space]) else {
            println!("  CONVERSION: line {line_no} is not valid base64");
            return false;
        };
        let rank: u32 = match std::str::from_utf8(&line[space + 1..])
            .ok()
            .and_then(|s| s.trim().parse().ok())
        {
            Some(rank) => rank,
            None => {
                println!("  CONVERSION: line {line_no} has no integer rank");
                return false;
            }
        };
        lines.push((bytes, rank));
    }

    let tokens = &fixture.vocab.tokens;
    if lines.len() != tokens.len() {
        println!(
            "  CONVERSION: {} lines but {} pieces",
            lines.len(),
            tokens.len()
        );
        return false;
    }

    let mut bad = 0usize;
    for (id, ((bytes, rank), piece)) in lines.iter().zip(tokens.iter()).enumerate() {
        let id = id as u32;
        if *rank != id {
            println!("  CONVERSION: line {id} carries rank {rank}");
            return false;
        }
        let want = piece_to_bytes(piece);
        if *bytes != want {
            if bad < 5 {
                println!("  CONVERSION: id {id} piece {piece:?} -> {bytes:?}, expected {want:?}");
            }
            bad += 1;
        }
    }
    if bad > 0 {
        println!("  CONVERSION: {bad} pieces render differently than expected");
        return false;
    }

    // Duplicates: expected, and the reason the constructor choice matters.
    let mut by_bytes: FxHashMap<&[u8], Vec<u32>> = FxHashMap::default();
    for (id, (bytes, _)) in lines.iter().enumerate() {
        by_bytes
            .entry(bytes.as_slice())
            .or_default()
            .push(id as u32);
    }
    let mut dups: Vec<(&[u8], &Vec<u32>)> = by_bytes
        .iter()
        .filter(|(_, hits)| hits.len() > 1)
        .map(|(bytes, hits)| (*bytes, hits))
        .collect();
    dups.sort_by_key(|(_, hits)| hits[0]);
    println!(
        "  conversion: {} lines, {} distinct byte sequences, {} collisions",
        lines.len(),
        by_bytes.len(),
        dups.len()
    );
    for (bytes, hits) in dups.iter().take(3) {
        let named: Vec<String> = hits
            .iter()
            .map(|&id| format!("{id}={:?}", tokens[id as usize]))
            .collect();
        println!(
            "    {bytes:?} at {} — from_bytes_sentencepiece keeps the last ({})",
            named.join(", "),
            hits.last().copied().unwrap_or_default()
        );
    }

    // Probes: pieces whose id llama.cpp's expected output pins down.
    for probe in ["\u{2581}Hello", "\u{2581}", "\u{2581}months", "!"] {
        let want = piece_to_bytes(probe);
        match by_bytes.get(want.as_slice()) {
            Some(hits) => println!(
                "    probe {probe:?} -> ids {hits:?} (encoder uses {})",
                hits.last().copied().unwrap_or_default()
            ),
            None => println!("    probe {probe:?} -> ABSENT from the rendering"),
        }
    }

    true
}

/// The Rust twin of `piece_to_bytes` in `scripts/extract_gguf_vocab.py`.
fn piece_to_bytes(piece: &str) -> Vec<u8> {
    let bytes = piece.as_bytes();
    if bytes.len() == 6 && bytes.starts_with(b"<0x") && bytes[5] == b'>' {
        let hex = &piece[3..5];
        if let Ok(byte) = u8::from_str_radix(hex, 16) {
            return vec![byte];
        }
    }
    bytes.to_vec()
}

// ---------------------------------------------------------------------------
// Reporting helpers
// ---------------------------------------------------------------------------

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

/// Render a case's text with whitespace visible — most cases differ only in
/// leading spaces, tabs and newlines.
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

// ---------------------------------------------------------------------------
// JSON -> Fixture (same shape as examples/verify_gguf.rs reads)
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

    let raw = json
        .get("vocab")
        .ok_or_else(|| "missing `vocab` object".to_owned())?;

    let vocab = GgufVocab {
        model: raw
            .get("model")
            .and_then(Value::as_str)
            .ok_or_else(|| "vocab.model missing or not a string".to_owned())?
            .to_owned(),
        tokens: string_vec(raw, "tokens")
            .ok_or_else(|| "vocab.tokens missing or not an array of strings".to_owned())?,
        scores: f32_vec(raw, "scores"),
        merges: string_vec(raw, "merges"),
        token_type: u32_vec(raw, "token_type"),
        add_space_prefix: opt_bool(raw, "add_space_prefix"),
        remove_extra_whitespaces: opt_bool(raw, "remove_extra_whitespaces"),
        add_bos_token: opt_bool(raw, "add_bos_token"),
        add_eos_token: opt_bool(raw, "add_eos_token"),
        bos_token_id: opt_u32(raw, "bos_token_id"),
        eos_token_id: opt_u32(raw, "eos_token_id"),
        unknown_token_id: opt_u32(raw, "unknown_token_id"),
        padding_token_id: opt_u32(raw, "padding_token_id"),
        cls_token_id: opt_u32(raw, "cls_token_id"),
        sep_token_id: opt_u32(raw, "sep_token_id"),
        pre: raw.get("pre").and_then(Value::as_str).map(str::to_owned),
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
