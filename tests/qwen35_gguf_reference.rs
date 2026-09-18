//! Reference parity for the `qwen35` GGUF vocabulary path against the
//! PrismML fork's own `llama-tokenize`, on a real checkpoint too large to
//! commit as a fixture.
//!
//! # Fixtures
//!
//! `tests/fixtures/qwen35/corpus.txt` -- 17 UTF-8 lines (ASCII, accented
//! Latin, Vietnamese, Devanagari, Arabic, CJK, emoji including a ZWJ
//! sequence and a flag, digits, punctuation, leading/trailing/embedded
//! whitespace and tabs, blank lines, Rust source, contractions, and a Zalgo
//! line with stacked combining marks), file ends with a trailing newline.
//!
//! `tests/fixtures/qwen35/corpus.ids.txt` -- 218 space-separated token ids,
//! produced by:
//!
//! ```text
//! llama-tokenize --ids --no-bos --log-disable \
//!     -m Ternary-Bonsai-2-27B-PQ2_0.gguf -f corpus.txt
//! ```
//!
//! against the PrismML fork, `prism` branch @ `5d80cff`. That command
//! tokenizes `corpus.txt` as one string, newlines included, with no BOS and
//! no special tokens -- exactly what `AnyTokenizer::encode_raw` produces.
//!
//! # Fixture file (the model)
//!
//! `BOOSTR_BONSAI2_DIR` must hold `Ternary-Bonsai-2-27B-PQ2_0.gguf`. Unset or
//! missing, this test prints one `skip:` line and passes -- the same
//! contract `boostr/tests/qwen35_gguf_real_file.rs` and
//! `blazr/tests/qwen35_generate.rs` use.
//!
//! ```bash
//! BOOSTR_BONSAI2_DIR=/path/to/dir cargo test --test qwen35_gguf_reference \
//!     -- --nocapture
//! ```
//!
//! # Why this file parses the GGUF header itself
//!
//! splintr never opens a GGUF container (see `src/core/gguf/vocab.rs`): the
//! caller parses the file and hands a filled-in [`splintr::GgufVocab`] to
//! [`splintr::from_gguf_vocab`]. In the rest of the monorepo that caller is
//! `boostr::format::Gguf` + `boostr::format::extract_gguf_vocab`, but boostr
//! depends on splintr (`boostr/Cargo.toml`: `splintr = "0.20"`), so splintr
//! cannot take a dev-dependency on boostr without a cycle. This file reads
//! just the GGUF header and `tokenizer.ggml.*` metadata keys itself -- the
//! same binary layout `boostr/src/format/gguf/io.rs` and
//! `boostr/src/format/gguf_vocab.rs` read, trimmed to metadata only, since
//! this test never touches a tensor.
//!
//! # Why there is no second, per-line concatenation test
//!
//! [`QWEN35_PATTERN`](splintr::QWEN35_PATTERN) includes the alternative
//! `\s*[\r\n]+`, which pre-tokenizes a run of trailing whitespace *together
//! with* the newline that follows it as one piece. Line 7 of the corpus ends
//! in trailing spaces before its newline specifically to exercise this: in
//! the real corpus those spaces and the newline are one pre-token, so BPE
//! merges them as a unit, where `encode_raw("   ")` (the line alone, spaces
//! at end of string) and `encode_raw("\n")` are two separate pre-tokens with
//! no merge between them. Concatenating per-line ids with a `"\n"`-token id
//! spliced between them therefore does not reproduce the whole-file ids, so
//! that test is not written.
//!
//! No `#![cfg(feature = ...)]` gate: `GgufVocab`/`from_gguf_vocab`/
//! `AnyTokenizer` are unconditional exports (`src/lib.rs`) -- this test
//! bundles no `vocab-*` payload, it reads the real GGUF file's own vocabulary
//! at runtime, gated by `BOOSTR_BONSAI2_DIR` instead.

use splintr::{from_gguf_vocab, AnyTokenizer, GgufVocab};
use std::fs;
use std::io::{Cursor, Read};
use std::path::PathBuf;

const ENV_DIR: &str = "BOOSTR_BONSAI2_DIR";
const PQ2_0_FILE: &str = "Ternary-Bonsai-2-27B-PQ2_0.gguf";

const CORPUS_TXT: &str = include_str!("fixtures/qwen35/corpus.txt");
const CORPUS_IDS: &str = include_str!("fixtures/qwen35/corpus.ids.txt");

/// The PQ2_0 file, or `None` after one `skip:` line.
fn require_file() -> Option<PathBuf> {
    let Some(dir) = std::env::var(ENV_DIR).ok().map(PathBuf::from) else {
        println!("skip: {ENV_DIR} not set");
        return None;
    };
    let path = dir.join(PQ2_0_FILE);
    if !path.is_file() {
        println!("skip: {} not found", path.display());
        return None;
    }
    Some(path)
}

fn reference_ids() -> Vec<u32> {
    CORPUS_IDS
        .split_whitespace()
        .map(|tok| {
            tok.parse::<u32>()
                .unwrap_or_else(|e| panic!("bad id {tok:?}: {e}"))
        })
        .collect()
}

// ── Minimal GGUF header/metadata reader ────────────────────────────
//
// Mirrors `boostr/src/format/gguf/io.rs`'s value-type ids and read order,
// trimmed to the header + key-value block: this test never reads tensor
// info or tensor bytes, only `tokenizer.ggml.*` metadata.

// Every GGUF value type is decoded so the KV block parses; the test reads
// only the string and array payloads.
#[allow(dead_code)]
#[derive(Debug, Clone)]
enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    fn as_u32(&self) -> Option<u32> {
        match self {
            Value::U8(v) => Some(*v as u32),
            Value::U16(v) => Some(*v as u32),
            Value::U32(v) => Some(*v),
            Value::I32(v) => Some(*v as u32),
            _ => None,
        }
    }

    fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }
}

fn read_u8<R: Read>(r: &mut R) -> u8 {
    let mut b = [0u8; 1];
    r.read_exact(&mut b).expect("read u8");
    b[0]
}

fn read_u16<R: Read>(r: &mut R) -> u16 {
    let mut b = [0u8; 2];
    r.read_exact(&mut b).expect("read u16");
    u16::from_le_bytes(b)
}

fn read_u32<R: Read>(r: &mut R) -> u32 {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).expect("read u32");
    u32::from_le_bytes(b)
}

fn read_i32<R: Read>(r: &mut R) -> i32 {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).expect("read i32");
    i32::from_le_bytes(b)
}

fn read_u64<R: Read>(r: &mut R) -> u64 {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).expect("read u64");
    u64::from_le_bytes(b)
}

fn read_i64<R: Read>(r: &mut R) -> i64 {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).expect("read i64");
    i64::from_le_bytes(b)
}

fn read_f32<R: Read>(r: &mut R) -> f32 {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).expect("read f32");
    f32::from_le_bytes(b)
}

fn read_f64<R: Read>(r: &mut R) -> f64 {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).expect("read f64");
    f64::from_le_bytes(b)
}

fn read_string<R: Read>(r: &mut R) -> String {
    let len = read_u64(r) as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).expect("read string bytes");
    String::from_utf8(buf).expect("string is UTF-8")
}

fn read_value<R: Read>(r: &mut R, value_type: u32) -> Value {
    match value_type {
        0 => Value::U8(read_u8(r)),
        1 => Value::I8(read_u8(r) as i8),
        2 => Value::U16(read_u16(r)),
        3 => Value::I16(read_u16(r) as i16),
        4 => Value::U32(read_u32(r)),
        5 => Value::I32(read_i32(r)),
        6 => Value::F32(read_f32(r)),
        7 => Value::Bool(read_u8(r) != 0),
        8 => Value::String(read_string(r)),
        9 => {
            let elem_type = read_u32(r);
            let len = read_u64(r) as usize;
            let mut arr = Vec::with_capacity(len);
            for _ in 0..len {
                arr.push(read_value(r, elem_type));
            }
            Value::Array(arr)
        }
        10 => Value::U64(read_u64(r)),
        11 => Value::I64(read_i64(r)),
        12 => Value::F64(read_f64(r)),
        other => panic!("unknown GGUF value type {other}"),
    }
}

/// Read the GGUF header and key-value metadata block of `path` into a plain
/// map, stopping before the tensor-info table (this test needs no tensor).
fn read_gguf_metadata(path: &std::path::Path) -> std::collections::HashMap<String, Value> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let mut r = Cursor::new(bytes);

    let magic = read_u32(&mut r);
    assert_eq!(magic, 0x4655_4747, "not a GGUF file: {}", path.display());

    let version = read_u32(&mut r);
    assert!(
        (1..=3).contains(&version),
        "unsupported GGUF version {version}"
    );

    let _tensor_count = read_u64(&mut r);
    let kv_count = read_u64(&mut r);

    let mut kv = std::collections::HashMap::with_capacity(kv_count as usize);
    for _ in 0..kv_count {
        let key = read_string(&mut r);
        let value_type = read_u32(&mut r);
        let value = read_value(&mut r, value_type);
        kv.insert(key, value);
    }
    kv
}

fn build_tokenizer() -> Option<AnyTokenizer> {
    let path = require_file()?;
    let kv = read_gguf_metadata(&path);

    let get = |k: &str| kv.get(k);
    let string_array = |k: &str| -> Option<Vec<String>> {
        get(k)?
            .as_array()?
            .iter()
            .map(|v| v.as_str().map(str::to_owned))
            .collect()
    };
    let u32_array =
        |k: &str| -> Option<Vec<u32>> { get(k)?.as_array()?.iter().map(Value::as_u32).collect() };

    let vocab = GgufVocab {
        model: get("tokenizer.ggml.model")
            .and_then(Value::as_str)
            .unwrap_or("llama")
            .to_owned(),
        tokens: string_array("tokenizer.ggml.tokens").expect("tokenizer.ggml.tokens present"),
        scores: None,
        merges: string_array("tokenizer.ggml.merges"),
        token_type: u32_array("tokenizer.ggml.token_type"),
        add_space_prefix: get("tokenizer.ggml.add_space_prefix").and_then(Value::as_bool),
        remove_extra_whitespaces: get("tokenizer.ggml.remove_extra_whitespaces")
            .and_then(Value::as_bool),
        add_bos_token: get("tokenizer.ggml.add_bos_token").and_then(Value::as_bool),
        add_eos_token: get("tokenizer.ggml.add_eos_token").and_then(Value::as_bool),
        bos_token_id: get("tokenizer.ggml.bos_token_id").and_then(Value::as_u32),
        eos_token_id: get("tokenizer.ggml.eos_token_id").and_then(Value::as_u32),
        unknown_token_id: get("tokenizer.ggml.unknown_token_id").and_then(Value::as_u32),
        padding_token_id: get("tokenizer.ggml.padding_token_id").and_then(Value::as_u32),
        cls_token_id: get("tokenizer.ggml.cls_token_id").and_then(Value::as_u32),
        sep_token_id: get("tokenizer.ggml.sep_token_id").and_then(Value::as_u32),
        pre: get("tokenizer.ggml.pre")
            .and_then(Value::as_str)
            .map(str::to_owned),
        precompiled_charsmap: None,
    };

    assert_eq!(
        vocab.pre.as_deref(),
        Some("qwen35"),
        "expected the qwen35 pre-tokenizer, GGUF declared {:?}",
        vocab.pre
    );

    Some(from_gguf_vocab(vocab).unwrap_or_else(|e| panic!("from_gguf_vocab: {e}")))
}

/// Report the first index where `actual` and `expected` diverge, with a
/// 5-id window either side of the mismatch and the decoded text of both
/// windows.
fn assert_ids_match(tok: &AnyTokenizer, actual: &[u32], expected: &[u32]) {
    if actual == expected {
        return;
    }
    let common = actual.len().min(expected.len());
    let first_diff = (0..common)
        .find(|&i| actual[i] != expected[i])
        .unwrap_or(common);

    let lo = first_diff.saturating_sub(5);
    let got_hi = (first_diff + 5).min(actual.len());
    let want_hi = (first_diff + 5).min(expected.len());
    let got_window = &actual[lo..got_hi];
    let want_window = &expected[lo..want_hi];

    panic!(
        "qwen35 GGUF encode diverges from the reference at index {first_diff} \
         (got {} ids, want {} ids)\n  \
         got[{lo}..{got_hi}]  = {got_window:?} -> {:?}\n  \
         want[{lo}..{want_hi}] = {want_window:?} -> {:?}",
        actual.len(),
        expected.len(),
        tok.decode(got_window)
            .unwrap_or_else(|e| format!("<decode error: {e}>")),
        tok.decode(want_window)
            .unwrap_or_else(|e| format!("<decode error: {e}>")),
    );
}

#[test]
fn qwen35_gguf_encodes_the_reference_ids() {
    let Some(tok) = build_tokenizer() else {
        return;
    };

    let expected = reference_ids();
    let actual = tok.encode_raw(CORPUS_TXT);
    assert_ids_match(&tok, &actual, &expected);

    let decoded = tok
        .decode(&expected)
        .unwrap_or_else(|e| panic!("decode reference ids: {e}"));
    assert_eq!(
        decoded, CORPUS_TXT,
        "decoding the reference ids must round-trip to the corpus"
    );
}
