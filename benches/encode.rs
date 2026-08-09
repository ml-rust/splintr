//! Encode-path benchmarks.
//!
//! These exist because the crate had no executable performance gate. Token ids
//! were pinned exhaustively; what they cost to produce was not, so a change
//! could keep every id byte-identical, keep every test green, and still ship a
//! materially slower tokenizer — which is exactly what happened once.
//!
//! `tests/encode_cost.rs` guards the *shapes* that regression had (per-merge
//! allocation, quadratic selection, a serialized parallel path) and fails the
//! build when they come back. This file measures the constants those guards
//! deliberately do not, so a change that is merely slower — without breaking a
//! shape — is visible before release rather than after.
//!
//! The groups map onto the regimes the encode path actually has, because they
//! are not interchangeable and a win in one can be a loss in another:
//!
//! - `short_chunks` — novel words, the dominant real workload. A pre-tokenizer
//!   emits pieces of a handful of symbols and this is where nearly all time
//!   goes on ordinary text.
//! - `cached_chunks` — repeated words, where the chunk cache answers instead of
//!   the merge loop. Real prose sits between this and the above.
//! - `unsplit_piece` — one long piece with no pre-tokenizer splitting, the
//!   shape a `tokenizer.json` with `pre_tokenizer: null` produces (Mistral's
//!   AWQ and GPTQ files). This is the regime that needs the heap; watch it for
//!   super-linear growth.
//! - `batch_scaling` — `encode_batch` against the same work done sequentially.
//!   The ratio is the real question, not the absolute number.

// A `#[flux::synthetic]` struct is registered by the macro and never
// constructed by name, which the dead-code lint cannot see. Matches the
// convention in the other benchmark suites in this workspace.
#![allow(dead_code)]

use fluxbench::{flux, Bencher};
use std::hint::black_box;

use splintr::pretrained::{llama3_special_tokens, LLAMA3_VOCAB_PACKED};
use splintr::{Tokenizer, LLAMA3_PATTERN, NO_SPLIT_PATTERN};

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

fn unsplit_tokenizer() -> Tokenizer {
    Tokenizer::from_packed_chain(
        LLAMA3_VOCAB_PACKED,
        &[NO_SPLIT_PATTERN],
        llama3_special_tokens(),
    )
    .expect("bundled llama3 vocabulary must load under a no-split pattern")
}

/// Novel pseudo-words, so the chunk cache and the whole-chunk vocabulary lookup
/// both miss and the merge loop actually runs.
fn novel_text(words: usize, seed: u64) -> String {
    let alphabet = b"abcdefghijklmnopqrstuvwxyz";
    let mut rng = Rng(seed | 1);
    let mut out = String::new();
    for _ in 0..words {
        if !out.is_empty() {
            out.push(' ');
        }
        let len = 4 + rng.next() % 9;
        for _ in 0..len {
            out.push(alphabet[rng.next() % alphabet.len()] as char);
        }
    }
    out
}

/// A small recurring vocabulary, so nearly every chunk is a cache hit.
fn repetitive_text(words: usize, seed: u64) -> String {
    const WORDS: [&str; 12] = [
        "the",
        "tokenizer",
        "encodes",
        "text",
        "into",
        "tokens",
        "and",
        "then",
        "decodes",
        "them",
        "again",
        "quickly",
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

// ---------------------------------------------------------------------------
// Short chunks: the dominant real workload
// ---------------------------------------------------------------------------

#[flux::bench(group = "short_chunks", args = [64, 256, 1024])]
fn encode_novel(b: &mut Bencher, words: usize) {
    let tok = tokenizer();
    let text = novel_text(words, 0x51D3);
    b.iter(|| black_box(tok.encode_ordinary(black_box(&text))));
}

// ---------------------------------------------------------------------------
// Cached chunks: the same path when the cache answers
// ---------------------------------------------------------------------------

#[flux::bench(group = "cached_chunks", args = [64, 256, 1024])]
fn encode_repetitive(b: &mut Bencher, words: usize) {
    let tok = tokenizer();
    let text = repetitive_text(words, 0xA71E);
    // Warm the cache so this measures the hit path, not the fill.
    black_box(tok.encode_ordinary(&text));
    b.iter(|| black_box(tok.encode_ordinary(black_box(&text))));
}

// ---------------------------------------------------------------------------
// Unsplit piece: the regime that needs the heap
// ---------------------------------------------------------------------------

/// Watch this group for growth that outpaces its argument. Cost should rise
/// close to linearly in the piece length; a quadratic selection strategy shows
/// up as roughly 4x the time for 2x the input.
#[flux::bench(group = "unsplit_piece", args = [500, 1000, 2000, 4000, 8000])]
fn encode_one_long_piece(b: &mut Bencher, chars: usize) {
    let tok = unsplit_tokenizer();
    let text: String = novel_text(4000, 0x3311).chars().take(chars).collect();
    b.iter(|| black_box(tok.encode_ordinary(black_box(&text))));
}

// ---------------------------------------------------------------------------
// Batch scaling: the parallel path against its own sequential fallback
// ---------------------------------------------------------------------------

#[flux::bench(group = "batch_scaling", args = [512])]
fn batch_sequential(b: &mut Bencher, texts: usize) {
    let tok = tokenizer();
    let corpus: Vec<String> = (0..texts)
        .map(|i| novel_text(24, 0x9E37 ^ i as u64))
        .collect();
    b.iter(|| {
        for text in &corpus {
            black_box(tok.encode(text));
        }
    });
}

#[flux::bench(group = "batch_scaling", args = [512])]
fn batch_parallel(b: &mut Bencher, texts: usize) {
    let tok = tokenizer();
    let corpus: Vec<String> = (0..texts)
        .map(|i| novel_text(24, 0x9E37 ^ i as u64))
        .collect();
    b.iter(|| black_box(tok.encode_batch(black_box(&corpus))));
}

/// The number that matters for the parallel path: how much faster the batch API
/// is than doing the same work one text at a time. A value near 1 means the
/// workers are serializing on shared state rather than overlapping.
#[flux::synthetic(
    id = "batch_speedup_512",
    formula = "batch_sequential@512 / batch_parallel@512",
    unit = "x"
)]
struct BatchSpeedup512;

fn main() {
    fluxbench::run().unwrap();
}
