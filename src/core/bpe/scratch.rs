//! Per-thread scratch for the long-piece merge path.
//!
//! A piece above [`SCAN_SYMBOL_LIMIT`](super::merge::SCAN_SYMBOL_LIMIT) leaves
//! the stack-resident scan and needs two heap structures, the node list and the
//! candidate heap. Crossing that threshold is rare in English but routine in
//! any script whose characters are multi-byte: symbols are counted in ByteLevel
//! space, where one CJK character is several of them, so those chunks were
//! paying both allocations continuously.
//!
//! Same shape and same reasoning as [`crate::core::scratch`] — thread-local
//! because the storage has to outlive one `encode` and `&self` is shared across
//! rayon workers without a lock.

use std::cell::RefCell;

use super::merge::QueueScratch;
use super::nodes::Node;

/// The buffers the queue-selection merge needs.
#[derive(Default)]
pub(super) struct MergeScratch {
    pub(super) nodes: Vec<Node>,
    pub(super) queue: QueueScratch,
}

thread_local! {
    /// Reused by [`with_merge_scratch`]. Not `const`-initialized: `BinaryHeap`
    /// has no const constructor, and the one lazy check per thread is paid on a
    /// path that was allocating twice per chunk before this existed.
    static MERGE: RefCell<MergeScratch> = RefCell::new(MergeScratch::default());
}

/// Run `f` with cleared merge buffers that survive between calls on this
/// thread.
///
/// Cleared on entry, not exit, so a panicking `f` cannot leave its nodes
/// visible to the next caller. Capacity is retained, which is the entire point.
///
/// **Reentrancy.** A nested call gets fresh buffers rather than a panicking
/// `borrow_mut` or the outer caller's nodes overwritten underneath it. Nothing
/// on the merge path re-enters today; this costs a nested caller exactly what
/// it paid before this module existed, which is the same bargain
/// [`crate::core::scratch::with_spans`] makes.
pub(super) fn with_merge_scratch<R>(f: impl FnOnce(&mut MergeScratch) -> R) -> R {
    MERGE.with(|cell| match cell.try_borrow_mut() {
        Ok(mut s) => {
            s.nodes.clear();
            s.queue.clear();
            f(&mut s)
        }
        Err(_) => f(&mut MergeScratch::default()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_buffers_arrive_empty_however_they_were_left() {
        with_merge_scratch(|s| {
            s.nodes.push(Node::PLACEHOLDER);
            s.queue.cold.push(1);
            s.queue.ranks.push(0);
        });
        with_merge_scratch(|s| {
            assert!(s.nodes.is_empty(), "nodes leaked between calls");
            assert!(s.queue.is_empty(), "queue leaked between calls");
        });
    }

    #[test]
    fn a_nested_call_gets_its_own_buffers() {
        with_merge_scratch(|outer| {
            outer.nodes.extend([Node::PLACEHOLDER; 3]);
            with_merge_scratch(|inner| {
                assert!(inner.nodes.is_empty());
                inner.nodes.push(Node::PLACEHOLDER);
            });
            assert_eq!(
                outer.nodes.len(),
                3,
                "a nested call clobbered the outer buffer"
            );
        });
    }

    /// Retained capacity is what makes this worth having.
    #[test]
    fn capacity_survives_between_calls() {
        with_merge_scratch(|s| s.nodes.extend([Node::PLACEHOLDER; 256]));
        with_merge_scratch(|s| {
            assert!(
                s.nodes.capacity() >= 256,
                "nodes were reallocated instead of reused"
            )
        });
    }
}
