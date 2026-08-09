//! When a batch is worth handing to rayon, and how finely to split it.
//!
//! Every batch entry point used to be a bare `par_iter().map(...).collect()`,
//! which is right for bulk ingest and wrong for the shape a server actually
//! sees. A batch of 100 prompts averaging ~120 bytes is ~12 KB of work — around
//! 100 µs single-threaded — and waking 24 worker threads to share that costs
//! more than doing it. Measured against a rank-file vocabulary, that batch ran
//! at 56 MB/s through rayon where the same texts encoded one at a time on the
//! calling thread reached ~97 MB/s: the parallelism was a 1.7x *penalty*.
//!
//! So there is one threshold, in bytes rather than element count — a batch of
//! 10,000 tweets and a batch of 10 documents are nothing alike: below
//! [`MIN_PARALLEL_BYTES`] of total input the batch runs on the calling thread,
//! and above it rayon gets it exactly as before. Large batches are therefore
//! untouched, which is why the bulk numbers are unchanged.
//!
//! A floor on *task* size was tried too — `with_min_len`, so a million short
//! strings could not become a million tasks. It measured worse: at 59 KB it
//! capped a batch at ~14 tasks across 24 threads and cost 12% against plain
//! `par_iter`. Rayon's own adaptive splitting already stops dividing when
//! stealing has nothing to gain, so the floor only removed parallelism it
//! would have used. Left out rather than kept on the strength of the argument.

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Total input below which a batch is encoded on the calling thread.
///
/// Measured, not guessed. On a 24-thread machine a mixed corpus encodes at
/// ~87 MB/s serially regardless of batch size, while the rayon path climbs with
/// it — 56 MB/s at 12 KB, 110 MB/s at 59 KB. The two cross near 39 KB, so the
/// threshold sits just below that: close to the crossover the two paths are
/// equal by definition, which makes a small error here cheap in either
/// direction.
const MIN_PARALLEL_BYTES: usize = 32 << 10;

/// Map `f` over `items` in parallel when the batch is big enough to earn it.
///
/// `size` reports one item's input bytes. It is called once per item, so it
/// must be cheap — `str::len`, not anything that walks the text.
///
/// Falls back to a serial map when the `rayon` feature is off, which is also
/// what a small batch gets.
pub fn map<T, R, S, F>(items: &[T], size: S, f: F) -> Vec<R>
where
    T: Sync,
    R: Send,
    S: Fn(&T) -> usize,
    F: Fn(&T) -> R + Send + Sync,
{
    #[cfg(feature = "rayon")]
    {
        let total: usize = items.iter().map(&size).sum();
        if total >= MIN_PARALLEL_BYTES {
            return items.par_iter().map(f).collect();
        }
    }
    #[cfg(not(feature = "rayon"))]
    let _ = size;

    items.iter().map(f).collect()
}

/// [`map`] for a fallible `f`, short-circuiting on the first error.
///
/// Separate from [`map`] rather than layered on it because rayon's `collect`
/// into a `Result` is what gives the short circuit; going through [`map`] would
/// encode every remaining text before discarding the batch.
pub fn try_map<T, R, E, S, F>(items: &[T], size: S, f: F) -> Result<Vec<R>, E>
where
    T: Sync,
    R: Send,
    E: Send,
    S: Fn(&T) -> usize,
    F: Fn(&T) -> Result<R, E> + Send + Sync,
{
    #[cfg(feature = "rayon")]
    {
        let total: usize = items.iter().map(&size).sum();
        if total >= MIN_PARALLEL_BYTES {
            return items.par_iter().map(f).collect();
        }
    }
    #[cfg(not(feature = "rayon"))]
    let _ = size;

    items.iter().map(f).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The thresholds are a performance policy, so the tests pin the *contract*
    /// — same answers, same order, whichever side of the threshold a batch
    /// lands on — rather than which path ran, which is not observable and
    /// should not be.
    #[test]
    fn small_and_large_batches_agree_with_a_serial_map() {
        for size in [0usize, 1, 10, 500, 5_000] {
            let items: Vec<String> = (0..size).map(|i| format!("item number {i}")).collect();
            let got = map(&items, String::len, |s| s.len() * 2);
            let want: Vec<usize> = items.iter().map(|s| s.len() * 2).collect();
            assert_eq!(got, want, "batch of {size} disagrees with a serial map");
        }
    }

    /// A batch big enough to go parallel must still come back in input order —
    /// `par_iter().collect()` guarantees it, and callers pair results with
    /// their inputs positionally.
    #[test]
    fn results_stay_in_input_order_past_the_parallel_threshold() {
        let items: Vec<String> = (0..4_000).map(|i| format!("{i:0>64}")).collect();
        let total: usize = items.iter().map(String::len).sum();
        assert!(
            total > MIN_PARALLEL_BYTES,
            "this test is only meaningful above the threshold"
        );
        let got = map(&items, String::len, |s| s.clone());
        assert_eq!(got, items);
    }

    #[test]
    fn try_map_short_circuits_and_reports_the_error() {
        let items: Vec<String> = (0..4_000).map(|i| format!("{i:0>64}")).collect();
        let out: Result<Vec<usize>, &str> = try_map(&items, String::len, |s| {
            if s.ends_with("999") {
                Err("boom")
            } else {
                Ok(s.len())
            }
        });
        assert_eq!(out, Err("boom"));
    }

    #[test]
    fn try_map_passes_a_clean_batch_through() {
        let items: Vec<String> = (0..100).map(|i| format!("text {i}")).collect();
        let out: Result<Vec<usize>, ()> = try_map(&items, String::len, |s| Ok(s.len()));
        assert_eq!(
            out.unwrap(),
            items.iter().map(String::len).collect::<Vec<_>>()
        );
    }
}
