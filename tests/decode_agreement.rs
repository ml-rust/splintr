//! Streaming decode must agree with whole-sequence decode, for every bundled
//! vocabulary, every backend reachable from it, and every chunking.
//!
//! [`StreamingDecoder`](splintr::StreamingDecoder) documents the contract this
//! file exists to hold: concatenating every emission plus the final `flush`
//! equals `decode_lossy` of the same ids, and equals `decode` whenever that
//! succeeds; chunk boundaries affect only *when* text is emitted, never
//! *what*. That is a property of splintr alone -- internal consistency between
//! two of its own code paths -- so unlike `tests/reference_parity.rs` it needs
//! no reference tokenizer, no model files and no Python, and runs
//! unconditionally in CI.
//!
//! # What drives it
//!
//! The corpus is the *inputs* of the committed reference fixtures
//! (`tests/fixtures/pretrained/*.json`), so the ids under test are the ids
//! real text actually produces rather than hand-picked ones, plus the shapes
//! that broke streaming in practice and that a prose corpus cannot reach:
//! multi-byte characters split across token boundaries, byte-fallback runs,
//! id sequences truncated mid-character (so `flush` has to resolve an
//! incomplete UTF-8 sequence), and special tokens pressed against text.
//!
//! # What it does not replace
//!
//! Bundled vocabularies only. A `tokenizer.json` is megabytes and cannot be
//! vendored, so agreement against externally-published models -- and the
//! declared-`decoder`-pipeline streaming path that only a loaded
//! `tokenizer.json` reaches -- is verified out of band by
//! `scripts/verify_external_models.py` and `scripts/fuzz_reference.py`.

use serde_json::Value;
use splintr::pretrained::from_pretrained;
use splintr::{AnyTokenizer, Backend, PretrainedVocab, SpecialDecode, Tokenize, TokenizeError};
use std::fs;
use std::path::Path;

/// Ids that no corpus of prose reaches, appended to every vocabulary's id
/// lists. Each entry is a distinct phenomenon, not a variation on another.
const EXTRA_TEXTS: [&str; 18] = [
    // multi-byte characters that BPE routinely splits mid-sequence
    "日本語のテキスト",
    "한국어와中文混合",
    "Здравствуй, мир",
    "café résumé naïve",
    // emoji, including sequences held together by joiners and selectors
    "😀🎉🚀",
    "👨‍👩‍👧‍👦",
    "🏳️‍🌈",
    "emoji😀letterRun日本語",
    // rare scripts and symbols, which fall through to byte fallback in the
    // SentencePiece vocabularies and to long byte runs in the BPE ones
    "𐍈𐌰𐌱",
    "🜁🜂🜃🜄",
    "\u{10FFFF}",
    "\u{FFFD}",
    // whitespace and control characters
    "",
    " ",
    "\t\n\r\n",
    "   \t\n   ",
    // text pressed directly against punctuation, no whitespace to split on
    "config.getValue()#hashtagWord",
    "a\tb\nc d",
];

/// Special-token spellings to interleave with text. Only the ones a given
/// vocabulary actually names are used -- `special_token_id` answering `None`
/// means this vocabulary does not carry that token, which is not a failure.
const SPECIAL_SPELLINGS: [&str; 8] = [
    "<|endoftext|>",
    "<|end_of_text|>",
    "</s>",
    "<s>",
    "[INST]",
    "<|im_start|>",
    "<|pad|>",
    "<|think|>",
];

/// The text a special token is pressed against, on both sides.
const SPECIAL_NEIGHBOURS: [(&str, &str); 3] = [("abc", "def"), ("", "日本語"), ("🎉", "")];

/// Render text with whitespace and non-printables visible -- a decode
/// disagreement is usually one space or one replacement character, which is
/// otherwise invisible in a panic message.
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

fn format_ids(ids: &[u32]) -> String {
    ids.iter().map(u32::to_string).collect::<Vec<_>>().join(" ")
}

/// Every `input` string across every committed reference fixture, deduplicated
/// and in a stable order.
///
/// Fails loudly rather than falling back to the built-in texts if the fixtures
/// are missing: the point of driving this from them is that the ids are the
/// ones real text produces, and quietly substituting a smaller corpus would
/// turn a deleted fixture into a weaker test that still passes.
fn fixture_inputs() -> Vec<String> {
    let fixtures_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pretrained");
    assert!(
        fixtures_dir.is_dir(),
        "decode_agreement: fixtures directory {} does not exist -- \
         run scripts/extract_reference_cases.py to (re)generate it",
        fixtures_dir.display()
    );

    let mut paths: Vec<_> = fs::read_dir(&fixtures_dir)
        .unwrap_or_else(|e| {
            panic!(
                "decode_agreement: failed to read {}: {e}",
                fixtures_dir.display()
            )
        })
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .collect();
    paths.sort();

    assert!(
        !paths.is_empty(),
        "decode_agreement: no .json fixtures found in {} -- \
         this test must never silently run on a smaller corpus",
        fixtures_dir.display()
    );

    let mut inputs: Vec<String> = Vec::new();
    for path in &paths {
        let text = fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("decode_agreement: failed to read {}: {e}", path.display()));
        let json: Value = serde_json::from_str(&text).unwrap_or_else(|e| {
            panic!("decode_agreement: failed to parse {}: {e}", path.display())
        });
        let cases = json
            .get("cases")
            .and_then(Value::as_array)
            .unwrap_or_else(|| panic!("decode_agreement: {}: missing `cases`", path.display()));
        for case in cases {
            let input = case
                .get("input")
                .and_then(Value::as_str)
                .unwrap_or_else(|| {
                    panic!("decode_agreement: {}: case without `input`", path.display())
                });
            if !inputs.iter().any(|seen| seen == input) {
                inputs.push(input.to_owned());
            }
        }
    }
    inputs
}

/// The id sequences one vocabulary is exercised over, each labelled by where
/// it came from so a failure names the shape rather than just the ids.
fn id_lists(tokenizer: &AnyTokenizer, corpus: &[String]) -> Vec<(String, Vec<u32>)> {
    let mut lists: Vec<(String, Vec<u32>)> = Vec::new();

    for text in corpus.iter().map(String::as_str).chain(EXTRA_TEXTS) {
        lists.push((format!("text \"{}\"", escape(text)), tokenizer.encode(text)));
    }

    // Every prefix of a few multi-byte encodings: a truncated id sequence is
    // the one shape where `flush` has to resolve an incomplete UTF-8 sequence
    // rather than a complete one, and it is exactly what a generation loop
    // stopped mid-word hands the decoder.
    for text in ["日本語です。", "👨‍👩‍👧‍👦", "𐍈𐌰𐌱"] {
        let ids = tokenizer.encode(text);
        for len in 0..ids.len() {
            lists.push((
                format!("prefix {len} of \"{}\"", escape(text)),
                ids[..len].to_vec(),
            ));
        }
    }

    // Special tokens pressed against text on both sides, as single ids rather
    // than as their spelling: decode has to drop or render them consistently
    // whether it sees the whole sequence or one token at a time.
    for spelling in SPECIAL_SPELLINGS {
        let Some(id) = tokenizer.special_token_id(spelling) else {
            continue;
        };
        for (before, after) in SPECIAL_NEIGHBOURS {
            let mut ids = tokenizer.encode(before);
            ids.push(id);
            ids.extend(tokenizer.encode(after));
            lists.push((
                format!("special {spelling:?} between {before:?} and {after:?}"),
                ids,
            ));
        }
    }

    lists
}

/// Feed `ids` in `chunk`-sized groups through a fresh strict decoder.
fn stream_strict<T: Tokenize>(
    tokenizer: &T,
    ids: &[u32],
    chunk: usize,
) -> Result<String, TokenizeError> {
    let mut decoder = Tokenize::streaming_decoder(tokenizer)?;
    let mut out = String::new();
    for part in ids.chunks(chunk.max(1)) {
        if let Some(text) = decoder.add_tokens(part)? {
            out.push_str(&text);
        }
    }
    out.push_str(&decoder.flush());
    Ok(out)
}

/// Feed `ids` in `chunk`-sized groups through a fresh lossy decoder.
fn stream_lossy<T: Tokenize>(
    tokenizer: &T,
    ids: &[u32],
    chunk: usize,
) -> Result<String, TokenizeError> {
    let mut decoder = Tokenize::streaming_decoder(tokenizer)?;
    let mut out = String::new();
    for part in ids.chunks(chunk.max(1)) {
        if let Some(text) = decoder.add_tokens_lossy(part) {
            out.push_str(&text);
        }
    }
    out.push_str(&decoder.flush());
    Ok(out)
}

/// Feed `ids` in `chunk`-sized groups through a fresh strict decoder that
/// *renders* the declared-special ids -- [`SpecialDecode::Render`].
fn stream_rendering_specials<T: Tokenize>(
    tokenizer: &T,
    ids: &[u32],
    chunk: usize,
) -> Result<String, TokenizeError> {
    let mut decoder = Tokenize::streaming_decoder_with(tokenizer, SpecialDecode::Render)?;
    let mut out = String::new();
    for part in ids.chunks(chunk.max(1)) {
        if let Some(text) = decoder.add_tokens(part)? {
            out.push_str(&text);
        }
    }
    out.push_str(&decoder.flush());
    Ok(out)
}

/// Assert the streaming contract for one backend over one id sequence, at
/// every chunk size, and append a report per violation.
fn check_one<T: Tokenize>(
    label: &str,
    origin: &str,
    tokenizer: &T,
    ids: &[u32],
    failures: &mut Vec<String>,
) {
    let whole = Tokenize::decode(tokenizer, ids);
    let whole_lossy = tokenizer.decode_lossy(ids);
    // The same sequence under `SpecialDecode::Render`, decoded once. `Err` here
    // is the U+FFFD divergence handled below for the default mode, and means
    // there is nothing for the rendering stream to be compared against.
    let whole_rendered = Tokenize::decode_with(tokenizer, ids, SpecialDecode::Render).ok();

    // `chunks` needs a non-zero size and yields nothing for an empty slice, so
    // the empty sequence is still exercised once -- at chunk size 1, feeding no
    // tokens at all and flushing. That case matters: a decoder must render the
    // empty stream as the empty string, not as a stray replacement character.
    for chunk in 1..=ids.len().max(1) {
        let strict = match stream_strict(tokenizer, ids, chunk) {
            Ok(text) => text,
            Err(e) => {
                failures.push(format!(
                    "{label}: {origin}: strict stream at chunk {chunk} failed: {e}\n    ids: {}",
                    format_ids(ids)
                ));
                continue;
            }
        };

        // A stream cannot see the future, so bytes still invalid at `flush`
        // become U+FFFD where strict whole-sequence decoding reports an error
        // instead. That is the documented single divergence, and in that case
        // the lossy rendering is what the stream must equal -- not a licence
        // to skip the case.
        let expected_strict = match &whole {
            Ok(text) => text,
            Err(_) => &whole_lossy,
        };
        if &strict != expected_strict {
            failures.push(format!(
                "{label}: {origin}: strict stream at chunk {chunk} disagrees with decode\
                 \n    ids:      {}\n    expected: \"{}\"\n    streamed: \"{}\"",
                format_ids(ids),
                escape(expected_strict),
                escape(&strict),
            ));
        }

        let lossy = match stream_lossy(tokenizer, ids, chunk) {
            Ok(text) => text,
            Err(e) => {
                failures.push(format!(
                    "{label}: {origin}: lossy stream at chunk {chunk} failed: {e}\n    ids: {}",
                    format_ids(ids)
                ));
                continue;
            }
        };
        if lossy != whole_lossy {
            failures.push(format!(
                "{label}: {origin}: lossy stream at chunk {chunk} disagrees with decode_lossy\
                 \n    ids:      {}\n    expected: \"{}\"\n    streamed: \"{}\"",
                format_ids(ids),
                escape(&whole_lossy),
                escape(&lossy),
            ));
        }

        // The same contract under `SpecialDecode::Render`. It is a separate
        // drive, not a variation on the one above: the mode changes the *skip
        // set*, and a skipped id is precisely the thing a cursor has to handle
        // without spending a leading separator or a first-token strip on it. So
        // whether a stream still equals whole-sequence decode when those ids
        // start rendering is a distinct question, and one this file is the only
        // place that can answer for every bundled vocabulary at every chunking.
        let Some(whole_rendered) = &whole_rendered else {
            continue;
        };
        match stream_rendering_specials(tokenizer, ids, chunk) {
            Ok(rendered) if &rendered == whole_rendered => {}
            Ok(rendered) => failures.push(format!(
                "{label}: {origin}: rendering-specials stream at chunk {chunk} disagrees with \
                 decode_with(Render)\n    ids:      {}\n    expected: \"{}\"\n    streamed: \"{}\"",
                format_ids(ids),
                escape(whole_rendered),
                escape(&rendered),
            )),
            Err(e) => failures.push(format!(
                "{label}: {origin}: rendering-specials stream at chunk {chunk} failed: {e}\
                 \n    ids: {}",
                format_ids(ids)
            )),
        }
    }
}

/// Assert that `reset` leaves a decoder indistinguishable from a fresh one.
///
/// The interesting state to discard is a half-finished character, so the
/// decoder is deliberately stopped at every prefix of `ids` before being reset
/// -- resetting a decoder that happened to be on a character boundary would
/// prove almost nothing.
fn check_reset<T: Tokenize>(
    label: &str,
    origin: &str,
    tokenizer: &T,
    ids: &[u32],
    failures: &mut Vec<String>,
) {
    let Ok(fresh) = stream_lossy(tokenizer, ids, 1) else {
        failures.push(format!(
            "{label}: {origin}: could not build a streaming decoder"
        ));
        return;
    };

    for prefix in 0..=ids.len() {
        let mut decoder = match Tokenize::streaming_decoder(tokenizer) {
            Ok(decoder) => decoder,
            Err(e) => {
                failures.push(format!("{label}: {origin}: streaming_decoder failed: {e}"));
                return;
            }
        };
        for &id in &ids[..prefix] {
            // The emissions are deliberately discarded: this half of the run
            // exists only to put the decoder into a state `reset` has to clear.
            let _ = decoder.add_token_lossy(id);
        }
        decoder.reset();

        if decoder.has_pending() || decoder.pending_bytes() != 0 {
            failures.push(format!(
                "{label}: {origin}: reset after {prefix} id(s) left {} pending byte(s)",
                decoder.pending_bytes()
            ));
        }

        let mut reused = String::new();
        for &id in ids {
            if let Some(text) = decoder.add_token_lossy(id) {
                reused.push_str(&text);
            }
        }
        reused.push_str(&decoder.flush());

        if reused != fresh {
            failures.push(format!(
                "{label}: {origin}: reset after {prefix} id(s) does not match a fresh decoder\
                 \n    ids:   {}\n    fresh: \"{}\"\n    reset: \"{}\"",
                format_ids(ids),
                escape(&fresh),
                escape(&reused),
            ));
        }
    }
}

/// Streaming and whole-sequence decoding agree for every bundled vocabulary,
/// every backend reachable from it, and every chunk size.
#[test]
fn streaming_decode_agrees_with_whole_sequence_decode() {
    let corpus = fixture_inputs();

    // One entry per distinct `PretrainedVocab`, not per accepted name: the
    // aliases resolve to the same vocabulary, and running all sixteen would
    // multiply the work without covering anything new. Resolving each name is
    // itself checked, so an alias that stopped resolving still fails here.
    let mut vocabs: Vec<(&str, PretrainedVocab)> = Vec::new();
    for &name in PretrainedVocab::supported_names() {
        let vocab = PretrainedVocab::from_name(name).unwrap_or_else(|| {
            panic!(
                "decode_agreement: PretrainedVocab::from_name({name:?}) \
                 returned None despite being listed as supported"
            )
        });
        if !vocabs.iter().any(|(_, seen)| *seen == vocab) {
            vocabs.push((name, vocab));
        }
    }

    let mut failures: Vec<String> = Vec::new();

    for &(name, _) in &vocabs {
        let handle = from_pretrained(name)
            .unwrap_or_else(|e| panic!("decode_agreement: from_pretrained({name:?}) failed: {e}"));
        let lists = id_lists(&handle, &corpus);

        // The universal handle first, then the concrete backend underneath it.
        // Both are reachable by a caller, both own a `streaming_decoder`, and
        // for a bundled vocabulary the handle delegates to the backend -- so
        // checking both is what pins that delegation rather than assuming it.
        let handle_label = format!("{name}/AnyTokenizer");
        for (origin, ids) in &lists {
            check_one(&handle_label, origin, &handle, ids, &mut failures);
            check_reset(&handle_label, origin, &handle, ids, &mut failures);
        }

        let backend = from_pretrained(name)
            .unwrap_or_else(|e| panic!("decode_agreement: from_pretrained({name:?}) failed: {e}"))
            .into_backend();
        let backend_label = format!("{name}/{}", backend_name(&backend));
        for (origin, ids) in &lists {
            match &backend {
                Backend::Bpe(t) => {
                    check_one(&backend_label, origin, t, ids, &mut failures);
                    check_reset(&backend_label, origin, t, ids, &mut failures);
                }
                Backend::Spm(t) => {
                    check_one(&backend_label, origin, t, ids, &mut failures);
                    check_reset(&backend_label, origin, t, ids, &mut failures);
                }
                Backend::Unigram(t) => {
                    check_one(&backend_label, origin, t, ids, &mut failures);
                    check_reset(&backend_label, origin, t, ids, &mut failures);
                }
                Backend::WordPiece(t) => {
                    check_one(&backend_label, origin, t, ids, &mut failures);
                    check_reset(&backend_label, origin, t, ids, &mut failures);
                }
            }
        }
    }

    // Truncated on purpose: a systematic streaming bug produces thousands of
    // these, and the first few name it as precisely as the last few would.
    let shown = failures.len().min(20);
    assert!(
        failures.is_empty(),
        "decode_agreement: {} streaming/decode disagreement(s), first {shown}:\n\n{}",
        failures.len(),
        failures[..shown].join("\n\n")
    );
}

/// The backend variant's name, for failure labels.
fn backend_name(backend: &Backend) -> &'static str {
    match backend {
        Backend::Bpe(_) => "Bpe",
        Backend::Spm(_) => "Spm",
        Backend::Unigram(_) => "Unigram",
        Backend::WordPiece(_) => "WordPiece",
    }
}
