//! Per-thread scratch buffers for the encode path.
//!
//! Several buffers on the hot path are sized correctly, used once, and thrown
//! away — a span list is built, walked, and dropped within a single `encode`.
//! Sizing them is what the surrounding commits did; it removes the *regrows*
//! but not the allocation itself, and one malloc/free pair per call is still
//! worth roughly 4% of a short encode on macOS, whose allocator does far more
//! per call than glibc's thread cache.
//!
//! Removing it needs storage that outlives one `encode`, and the only such
//! storage a `&self` shared across rayon workers can offer without a lock is
//! thread-local. Hence this module.
//!
//! Only spans live here. The pre-tokenizer's *piece* buffers hold `&'p str`
//! borrowed from the caller's text, and parking those in a `thread_local`
//! would mean transmuting a lifetime — this crate contains no `unsafe` and
//! this optimization is not worth becoming the exception. Spans are plain
//! `usize` pairs and carry no such problem.

use std::cell::RefCell;

thread_local! {
    /// Reused by [`with_spans`]. `const` init so the first access on a thread
    /// is not a lazy-initialization check.
    static SPANS: RefCell<Vec<(usize, usize)>> = const { RefCell::new(Vec::new()) };
}

/// Run `f` with a cleared span buffer that survives between calls on this
/// thread, so a caller that would allocate one per `encode` allocates once per
/// thread instead.
///
/// The buffer is cleared on entry rather than on exit: an `f` that panics
/// leaves its spans behind, and clearing first means the next caller cannot
/// observe them. `f` is handed `&mut Vec`, so it may grow the buffer — that
/// growth is the point, since it is what later calls reuse.
///
/// **Reentrancy.** `f` may end up back here — a chained pre-tokenizer
/// subdivides spans by re-running the matcher over each piece — and the
/// buffer is already borrowed at that point. The nested call gets a fresh
/// `Vec` rather than a panicking `borrow_mut` or, worse, the outer caller's
/// spans overwritten underneath it. That costs the nested call exactly what it
/// paid before this module existed.
pub(crate) fn with_spans<R>(f: impl FnOnce(&mut Vec<(usize, usize)>) -> R) -> R {
    SPANS.with(|cell| match cell.try_borrow_mut() {
        Ok(mut buf) => {
            buf.clear();
            f(&mut buf)
        }
        Err(_) => f(&mut Vec::new()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The contract is that `f` always starts from an empty buffer, whether or
    /// not anything ran before it.
    #[test]
    fn the_buffer_arrives_empty_however_it_was_left() {
        with_spans(|spans| spans.extend([(0, 1), (1, 2)]));
        with_spans(|spans| assert!(spans.is_empty(), "spans leaked between calls"));
    }

    /// A nested call must not see, or clobber, the outer call's spans — this is
    /// the chained-pre-tokenizer shape.
    #[test]
    fn a_nested_call_gets_its_own_buffer() {
        with_spans(|outer| {
            outer.extend([(0, 3), (3, 6)]);
            with_spans(|inner| {
                assert!(inner.is_empty());
                inner.push((9, 9));
            });
            assert_eq!(
                outer,
                &[(0, 3), (3, 6)],
                "a nested call overwrote the outer buffer"
            );
        });
    }

    /// Capacity is what makes this worth having, so it must actually persist.
    #[test]
    fn capacity_survives_between_calls() {
        with_spans(|spans| spans.extend((0..64).map(|i| (i, i + 1))));
        with_spans(|spans| {
            assert!(
                spans.capacity() >= 64,
                "buffer was reallocated instead of reused"
            )
        });
    }
}
