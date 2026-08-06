//! Cost guarantees of the encode hot path.
//!
//! Token ids are pinned exhaustively elsewhere (`reference_parity.rs`,
//! `llama3.rs`, …). Nothing in this crate pins what those ids *cost* to
//! produce, and the encode path is the crate's entire reason to exist: it is
//! sold on being an order of magnitude faster than the reference tokenizers it
//! agrees with. A change can therefore keep every id byte-identical, keep every
//! test green, and still hand users a materially slower tokenizer.
//!
//! The three properties below are the ones a merge-selection strategy can break
//! without changing a single id:
//!
//! 1. **Per-chunk work is allocation-flat.** Merging a piece must not allocate
//!    once per merge. A selection structure that grows as merges are queued
//!    turns every word into a handful of `malloc` calls, which is most of what
//!    a short chunk costs — and short chunks are what a pre-tokenizer produces,
//!    so this is the dominant cost of ordinary text.
//! 2. **Long unsplit pieces stay sub-quadratic.** A pre-tokenizer is not
//!    required to chunk anything (`ByteLevel { use_regex: false }` with no
//!    `Split` stage hands the whole input to the merge loop as one piece), so
//!    selection cost must not be linear-scan-per-merge either.
//!
//!    (1) and (2) pull in opposite directions and are stated together
//!    deliberately: satisfying either one alone is easy and has been shipped in
//!    both directions. Only a strategy that satisfies both is correct.
//! 3. **Batch encoding uses the cores it asks for.** `encode_batch` exists to
//!    be parallel. If shared per-chunk state serializes the workers, it is
//!    slower than the sequential loop it replaces, and the parallelism is a
//!    liability rather than a feature.
//!
//! Timing-based assertions here are deliberately coarse: they separate
//! *complexity classes* and *sign of a difference*, not percentages, so they
//! carry order-of-magnitude headroom against a loaded or slow machine.

use splintr::core::byte_pair_encode;
use splintr::pretrained::{llama3_special_tokens, LLAMA3_VOCAB};
use splintr::{FxHashMap, Tokenizer, LLAMA3_PATTERN, NO_SPLIT_PATTERN};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::LazyLock;
use std::time::Instant;

// =============================================================================
// Allocation counting
// =============================================================================

thread_local! {
    /// Allocations made by the current thread while `COUNTING` is on.
    static ALLOCATIONS: Cell<u64> = const { Cell::new(0) };
    static COUNTING: Cell<bool> = const { Cell::new(false) };
}

/// Passthrough allocator that tallies allocations per thread.
///
/// Per-thread rather than global so the tests in this binary, which cargo runs
/// concurrently, cannot contaminate each other's counts.
struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // `try_with` because the thread-local is itself torn down at thread
        // exit, after which touching it would panic inside the allocator.
        let _ = COUNTING.try_with(|on| {
            if on.get() {
                let _ = ALLOCATIONS.try_with(|n| n.set(n.get() + 1));
            }
        });
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // A realloc is a growth event, which is exactly what an
        // incrementally-grown selection structure does, so it counts.
        let _ = COUNTING.try_with(|on| {
            if on.get() {
                let _ = ALLOCATIONS.try_with(|n| n.set(n.get() + 1));
            }
        });
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Run `f`, returning how many allocations it made on this thread.
fn allocations_of<T>(f: impl FnOnce() -> T) -> u64 {
    ALLOCATIONS.with(|n| n.set(0));
    COUNTING.with(|on| on.set(true));
    let out = f();
    COUNTING.with(|on| on.set(false));
    drop(out);
    ALLOCATIONS.with(Cell::get)
}

// =============================================================================
// Fixtures
// =============================================================================

static TOKENIZER: LazyLock<Tokenizer> = LazyLock::new(|| {
    Tokenizer::from_bytes_chain(LLAMA3_VOCAB, &[LLAMA3_PATTERN], llama3_special_tokens())
        .expect("bundled llama3 vocabulary must load")
});

/// A tokenizer whose pre-tokenizer never splits, so the merge loop sees the
/// whole input as ONE piece.
///
/// This is not a contrived shape. It is what a `tokenizer.json` with
/// `pre_tokenizer: null` loads as — HuggingFace splits only where an installed
/// stage says to, and with none the model runs over the entire normalized
/// string as one word. Mistral's AWQ and GPTQ `tokenizer.json` files are
/// exactly that, which is why the crate carries [`NO_SPLIT_PATTERN`] for them.
static UNSPLIT: LazyLock<Tokenizer> = LazyLock::new(|| {
    Tokenizer::from_bytes_chain(LLAMA3_VOCAB, &[NO_SPLIT_PATTERN], llama3_special_tokens())
        .expect("bundled llama3 vocabulary must load under a no-split pattern")
});

/// Deterministic novel lowercase words, so every piece misses both the
/// whole-chunk vocabulary lookup and the LRU cache and actually runs merges.
fn novel_words(count: usize, len: usize, seed: u64) -> Vec<String> {
    let alphabet = b"abcdefghijklmnopqrstuvwxyz";
    let mut state = seed | 1;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as usize
    };
    (0..count)
        .map(|_| {
            (0..len)
                .map(|_| alphabet[next() % alphabet.len()] as char)
                .collect()
        })
        .collect()
}

// =============================================================================
// 1. Per-chunk work is allocation-flat
// =============================================================================

/// A vocabulary over a dense 4-letter alphabet whose merges chain all the way
/// up, so a piece of any length keeps merging instead of stalling early.
///
/// Hand-built rather than taken from a bundled vocabulary so the merge count is
/// a function of the input length alone.
fn chaining_encoder() -> FxHashMap<Vec<u8>, u32> {
    let mut encoder = FxHashMap::default();
    let alphabet = *b"abcd";
    for (i, &b) in alphabet.iter().enumerate() {
        encoder.insert(vec![b], i as u32);
    }
    // Every 2-, 4-, 8- and 16-byte run of each letter, so merges cascade.
    let mut next = alphabet.len() as u32;
    for &b in &alphabet {
        for width in [2usize, 4, 8, 16, 32] {
            encoder.insert(vec![b; width], next);
            next += 1;
        }
    }
    // Mixed pairs, so a merge is available at every position, not only inside runs.
    for &l in &alphabet {
        for &r in &alphabet {
            encoder.insert(vec![l, r], next);
            next += 1;
        }
    }
    encoder
}

/// Merging a long piece must not allocate more times than merging a short one.
///
/// This measures [`splintr::core::byte_pair_encode`] directly rather than a
/// whole `encode` call, and that is the point: the surrounding pipeline has
/// legitimate costs that DO scale with input length (the regex engine allocates
/// a scratch buffer proportional to the text it scans, and grows it
/// geometrically), and folding those in would make this assertion untrue for
/// reasons that have nothing to do with the property being pinned.
///
/// Isolated, the invariant is exact and the constant is fully explainable: the
/// node list and the output vector, both sized up front, and nothing else. Any
/// growth at all means merge selection allocates as it queues candidates —
/// a `malloc` per merge, on every word of every document.
#[test]
fn merge_allocation_count_does_not_grow_with_piece_length() {
    let encoder = chaining_encoder();
    // Both lengths sit below the threshold where selection switches strategy,
    // so this compares like with like; the crossover itself is covered by the
    // property tests in the bpe module.
    let short = b"abcdab".to_vec();
    let long = b"abcdabcdabcdabcdabcdabcdabcdabcdabcdabcd".to_vec();

    // Warm any one-time allocation out of the measured region.
    let _ = byte_pair_encode(b"aa", &encoder);

    let short_allocs = allocations_of(|| byte_pair_encode(&short, &encoder));
    let long_allocs = allocations_of(|| byte_pair_encode(&long, &encoder));

    assert_eq!(
        long_allocs,
        short_allocs,
        "merging a {}-byte piece allocated {long_allocs} times against {short_allocs} for a \
         {}-byte piece: selection allocates as it queues candidates, so every additional merge \
         costs a malloc",
        long.len(),
        short.len()
    );
}

/// The same property from the other side: the absolute count must stay tiny.
///
/// Flatness alone is satisfiable by a strategy that allocates a large constant
/// number of times per piece, which would be just as slow. Three is not a tuned
/// threshold — it is exactly what the algorithm structurally needs (the node
/// list and the output vector, plus one for slack) with nothing left over for a
/// per-merge selection structure.
#[test]
fn merging_one_piece_allocates_a_small_constant_number_of_times() {
    let encoder = chaining_encoder();
    let piece = b"abcdabcdabcdabcdabcdabcd".to_vec();
    let _ = byte_pair_encode(b"aa", &encoder);

    let allocs = allocations_of(|| byte_pair_encode(&piece, &encoder));

    assert!(
        allocs <= 3,
        "merging one {}-byte piece allocated {allocs} times; the algorithm needs the node list \
         and the output vector, so anything beyond that is per-merge allocation",
        piece.len()
    );
}

// =============================================================================
// 2. Long unsplit pieces stay sub-quadratic
// =============================================================================

/// Quadrupling the length of a single unsplit piece must not multiply the time
/// by ~16.
///
/// This is the guard on the OTHER side of the trade-off in this file: a
/// selection strategy tuned purely for the short chunks a pre-tokenizer emits
/// (a scan of the live list for each merge) is quadratic, and a `tokenizer.json`
/// with no `Split` stage feeds it the entire document. The threshold sits at 8×
/// for a 4× input — comfortably above linear-with-overhead, comfortably below
/// quadratic — so it separates the complexity classes rather than measuring a
/// constant factor.
#[test]
fn unsplit_input_encodes_sub_quadratically() {
    let tokenizer = &*UNSPLIT;
    let base: String = novel_words(400, 5, 0x3311).join(" ");
    let small: String = base.chars().take(1000).collect();
    let large: String = base.chars().take(4000).collect();

    // Warm the code paths, but not these inputs (the LRU cache would answer).
    tokenizer.encode_ordinary("warmup");

    let time = |text: &str| {
        let start = Instant::now();
        std::hint::black_box(tokenizer.encode_ordinary(text));
        start.elapsed().as_secs_f64()
    };
    let small_secs = time(&small);
    let large_secs = time(&large);

    let ratio = large_secs / small_secs;
    assert!(
        ratio < 8.0,
        "4x the input took {ratio:.1}x the time (1000 chars: {:.3}ms, 4000 chars: {:.3}ms); \
         merge selection is quadratic in the length of an unsplit piece",
        small_secs * 1000.0,
        large_secs * 1000.0
    );
}

// =============================================================================
// 3. Batch encoding uses the cores it asks for
// =============================================================================

/// `encode_batch` must get real speedup from the cores it uses.
///
/// The bar is 1.3x on a machine with at least four of them. That number is not
/// a target — it is the midpoint of a gap. A batch path serialized on shared
/// per-chunk state cannot exceed ~1x however many cores it is given, because it
/// does the sequential loop's work plus lock contention and thread hand-off;
/// the regression that motivated this test measured 0.61x. A working one clears
/// it comfortably: 1.79x on a 4-core shared CI runner, far more on real
/// hardware. So 1.3x separates the two states with margin on both sides while
/// staying immune to how loaded a shared runner happens to be.
///
/// Deliberately NOT set near what healthy code actually achieves. This test
/// asks "is the parallel path parallel", and the answer is binary; how *fast*
/// it is belongs to `benches/encode.rs`, which measures it without having to
/// pass or fail on a machine it does not control.
///
/// The texts are novel, so they miss the shared cache — which is the case that
/// exposes contention. Repetitive text hides it completely, because the cache
/// hit path is short enough that the lock is barely held.
///
/// Skipped below four usable cores, where parallel speedup is not a property
/// the code can have.
#[test]
fn batch_encoding_scales_with_available_cores() {
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    if cores < 4 {
        eprintln!("skipped: needs >= 4 cores, found {cores}");
        return;
    }

    let tokenizer = &*TOKENIZER;
    // Enough texts that thread-pool startup is not the dominant term.
    let texts: Vec<String> = (0..512)
        .map(|i| novel_words(24, 7, 0x9E37 ^ (i as u64)).join(" "))
        .collect();

    // One untimed pass so neither measurement pays first-touch costs. It also
    // fills the cache identically for both, so the comparison is like-for-like.
    std::hint::black_box(tokenizer.encode_batch(&texts));

    // Best of five each. A test binary runs its tests concurrently and CI
    // machines are shared, so a single round can be stolen from by unrelated
    // work; the fastest round of each is the one least polluted by it.
    let best = |mut f: Box<dyn FnMut()>| {
        (0..5)
            .map(|_| {
                let start = Instant::now();
                f();
                start.elapsed().as_secs_f64()
            })
            .fold(f64::INFINITY, f64::min)
    };

    let sequential = best(Box::new(|| {
        for text in &texts {
            std::hint::black_box(tokenizer.encode(text));
        }
    }));
    let parallel = best(Box::new(|| {
        std::hint::black_box(tokenizer.encode_batch(&texts));
    }));

    let speedup = sequential / parallel;
    assert!(
        speedup >= 1.3,
        "encode_batch was only {speedup:.2}x the sequential loop on {cores} cores ({:.3}ms \
         against {:.3}ms): the parallel path is contending on shared per-chunk state instead \
         of overlapping work",
        parallel * 1000.0,
        sequential * 1000.0
    );
}
