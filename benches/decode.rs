//! Decode-path benchmarks.
//!
//! `encode.rs` gates the encode path; this file gates decoding, which had no
//! measurement at all. Decoding is what a serving loop runs per generated
//! token, so its constants are paid per token rather than per request.
//!
//! The groups are the regimes the path actually has:
//!
//! - `whole_sequence` — one `decode` over a full id list, the request shape.
//! - `streaming` — the same ids one at a time, the serving shape. The ratio to
//!   `whole_sequence` is the cost of incrementality, and is what matters here.
//! - `lossy_vs_strict` — the two drives differ only in what an undecodable byte
//!   becomes, so a gap means one grew a slow path the other did not.
//! - `batch_scaling` — `decode_batch` against the same work done sequentially.
//!
//! llama3 is the plain id-keyed BPE shape `RenderRules::plain_by_id`
//! specializes; the `unicode` fixtures split characters across tokens, which is
//! the work no specialized loop can skip.

// A `#[flux::synthetic]` struct is registered by the macro and never
// constructed by name, which the dead-code lint cannot see. Matches the
// convention in the other benchmark suites in this workspace.
#![allow(dead_code)]

use fluxbench::{flux, Bencher, TrackingAllocator};
use std::hint::black_box;

use splintr::pretrained::{llama3_special_tokens, LLAMA3_VOCAB_PACKED};
use splintr::{Tokenizer, LLAMA3_PATTERN};

/// Allocation counts are deterministic where timings on a loaded machine are
/// not, and the questions here — what does incrementality cost per token, does
/// the batch path pay setup per item — are about allocation first.
#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// Deterministic LCG, so every run benchmarks byte-identical inputs.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as usize
    }
}

fn tokenizer() -> Tokenizer {
    Tokenizer::from_packed_chain(
        LLAMA3_VOCAB_PACKED,
        &[LLAMA3_PATTERN],
        llama3_special_tokens(),
    )
    .expect("bundled llama3 vocabulary must load")
}

/// Ordinary ASCII prose, the dominant decode workload.
fn ascii_text(words: usize, seed: u64) -> String {
    const WORDS: [&str; 12] = [
        "the",
        "tokenizer",
        "decodes",
        "these",
        "identifiers",
        "back",
        "into",
        "text",
        "again",
        "quickly",
        "and",
        "correctly",
    ];
    let mut rng = Rng(seed | 1);
    let mut out = String::new();
    for _ in 0..words {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(WORDS[rng.next() % WORDS.len()]);
    }
    out
}

/// Text whose characters are mostly multi-byte, so tokens routinely split a
/// character and the UTF-8 buffer has to hold a partial sequence across ids.
/// This is the work no specialized rendering loop can skip.
fn unicode_text(words: usize, seed: u64) -> String {
    const WORDS: [&str; 12] = [
        "こんにちは",
        "世界",
        "токенизатор",
        "δοκιμή",
        "مرحبا",
        "עולם",
        "字节",
        "编码",
        "καλημέρα",
        "здравствуй",
        "テキスト",
        "変換",
    ];
    let mut rng = Rng(seed | 1);
    let mut out = String::new();
    for _ in 0..words {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(WORDS[rng.next() % WORDS.len()]);
    }
    out
}

/// The ids for a text, which is what every benchmark below actually drives.
/// Produced through `encode_ordinary` so the id list is one a real encode would
/// hand back rather than a synthetic range.
fn ids(tok: &Tokenizer, text: &str) -> Vec<u32> {
    tok.encode_ordinary(text)
}

// ---------------------------------------------------------------------------
// Whole sequence: the request-shaped call
// ---------------------------------------------------------------------------

#[flux::bench(group = "whole_sequence", args = [64, 256, 1024])]
fn decode_ascii(b: &mut Bencher, words: usize) {
    let tok = tokenizer();
    let tokens = ids(&tok, &ascii_text(words, 0x51D3));
    b.iter(|| black_box(tok.decode(black_box(&tokens)).unwrap()));
}

#[flux::bench(group = "whole_sequence", args = [64, 256, 1024])]
fn decode_unicode(b: &mut Bencher, words: usize) {
    let tok = tokenizer();
    let tokens = ids(&tok, &unicode_text(words, 0xA71E));
    b.iter(|| black_box(tok.decode(black_box(&tokens)).unwrap()));
}

/// `decode_bytes` skips the UTF-8 reassembly entirely — it renders through the
/// rules directly and never builds a `String`. The gap to `decode_ascii@1024`
/// is therefore the price of producing text rather than bytes, and it is the
/// headroom any change to the buffer drain is competing for.
#[flux::bench(group = "whole_sequence", args = [1024])]
fn decode_bytes_ascii(b: &mut Bencher, words: usize) {
    let tok = tokenizer();
    let tokens = ids(&tok, &ascii_text(words, 0x51D3));
    b.iter(|| black_box(tok.decode_bytes(black_box(&tokens)).unwrap()));
}

/// What producing text costs over producing bytes. A value near 1 means the
/// buffer drain is close to free; the further above 1, the more a decode is
/// paying to copy text it already had.
#[flux::synthetic(
    id = "text_over_bytes_1024",
    formula = "decode_ascii@1024 / decode_bytes_ascii@1024",
    unit = "x"
)]
struct TextOverBytes1024;

// ---------------------------------------------------------------------------
// Streaming: the serving shape
// ---------------------------------------------------------------------------

/// One id at a time through a `StreamingDecoder`, which is what a generation
/// loop does. Every emission is consumed so the work cannot be optimized away.
#[flux::bench(group = "streaming", args = [1024])]
fn stream_ascii(b: &mut Bencher, words: usize) {
    let tok = tokenizer();
    let tokens = ids(&tok, &ascii_text(words, 0x51D3));
    b.iter(|| {
        let mut decoder = tok.streaming_decoder();
        let mut len = 0usize;
        for &id in &tokens {
            if let Some(text) = decoder.add_token(black_box(id)).unwrap() {
                len += text.len();
            }
        }
        len += decoder.flush().len();
        black_box(len)
    });
}

/// The same ids in one call, so the group's ratio is a like-for-like
/// comparison rather than two different workloads.
#[flux::bench(group = "streaming", args = [1024])]
fn stream_whole(b: &mut Bencher, words: usize) {
    let tok = tokenizer();
    let tokens = ids(&tok, &ascii_text(words, 0x51D3));
    b.iter(|| black_box(tok.decode(black_box(&tokens)).unwrap().len()));
}

/// Multi-byte characters split across tokens, which is the case that actually
/// exercises the held partial sequence rather than emitting on every id.
#[flux::bench(group = "streaming", args = [1024])]
fn stream_unicode(b: &mut Bencher, words: usize) {
    let tok = tokenizer();
    let tokens = ids(&tok, &unicode_text(words, 0xA71E));
    b.iter(|| {
        let mut decoder = tok.streaming_decoder();
        let mut len = 0usize;
        for &id in &tokens {
            if let Some(text) = decoder.add_token(black_box(id)).unwrap() {
                len += text.len();
            }
        }
        len += decoder.flush().len();
        black_box(len)
    });
}

/// The cost of incrementality: how much more the same ids cost fed one at a
/// time than fed at once. This is the number a serving stack pays per token,
/// and the one to watch — the absolute times in this group are dominated by
/// how many ids the fixture happens to produce.
#[flux::synthetic(
    id = "streaming_overhead_1024",
    formula = "stream_ascii@1024 / stream_whole@1024",
    unit = "x"
)]
struct StreamingOverhead1024;

// ---------------------------------------------------------------------------
// Lossy against strict: two drives that should cost the same
// ---------------------------------------------------------------------------

/// `decode` and `decode_lossy` differ only in what becomes of a byte that can
/// never be valid UTF-8 — a case neither fixture contains. They share the
/// rendering loop and the buffer scan, so they should measure the same; a gap
/// is a slow path one of them grew alone.
#[flux::bench(group = "lossy_vs_strict", args = [1024])]
fn decode_strict(b: &mut Bencher, words: usize) {
    let tok = tokenizer();
    let tokens = ids(&tok, &unicode_text(words, 0xA71E));
    b.iter(|| black_box(tok.decode(black_box(&tokens)).unwrap()));
}

#[flux::bench(group = "lossy_vs_strict", args = [1024])]
fn decode_lossy(b: &mut Bencher, words: usize) {
    let tok = tokenizer();
    let tokens = ids(&tok, &unicode_text(words, 0xA71E));
    b.iter(|| black_box(tok.decode_lossy(black_box(&tokens))));
}

#[flux::synthetic(
    id = "lossy_over_strict_1024",
    formula = "decode_lossy@1024 / decode_strict@1024",
    unit = "x"
)]
struct LossyOverStrict1024;

// ---------------------------------------------------------------------------
// Batch scaling: the parallel path against its own sequential fallback
// ---------------------------------------------------------------------------

/// Short sequences on purpose. `decode_batch` builds a fresh `DecodeState` per
/// item, so the per-call setup is amortized over less work the shorter the
/// sequences are — which is the serving shape, and the regime where that setup
/// is worth measuring.
fn corpus(tok: &Tokenizer, texts: usize) -> Vec<Vec<u32>> {
    (0..texts)
        .map(|i| ids(tok, &ascii_text(24, 0x9E37 ^ i as u64)))
        .collect()
}

#[flux::bench(group = "batch_scaling", args = [512])]
fn batch_sequential(b: &mut Bencher, texts: usize) {
    let tok = tokenizer();
    let batch = corpus(&tok, texts);
    b.iter(|| {
        for tokens in &batch {
            black_box(tok.decode(tokens).unwrap());
        }
    });
}

#[flux::bench(group = "batch_scaling", args = [512])]
fn batch_parallel(b: &mut Bencher, texts: usize) {
    let tok = tokenizer();
    let batch = corpus(&tok, texts);
    b.iter(|| black_box(tok.decode_batch(black_box(&batch)).unwrap()));
}

/// How much faster the batch API is than decoding one sequence at a time. A
/// value near 1 means the workers are serializing on shared state rather than
/// overlapping — the same question `encode.rs` asks of its own batch path.
#[flux::synthetic(
    id = "batch_speedup_512",
    formula = "batch_sequential@512 / batch_parallel@512",
    unit = "x"
)]
struct BatchSpeedup512;

fn main() {
    fluxbench::run().unwrap();
}
