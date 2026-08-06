/// A node in the doubly-linked list used for BPE merging.
///
/// This approach avoids the O(N) memory shifting of vector-based approaches
/// when merging pairs. Instead of removing elements, we simply update pointers.
#[derive(Debug, Clone, Copy)]
pub(super) struct Node {
    /// Index of previous node (usize::MAX if head)
    pub(super) prev: usize,
    /// Index of next node (usize::MAX if tail)
    pub(super) next: usize,
    /// Starting index in the original byte slice
    pub(super) start: usize,
    /// Length of this piece in bytes. Zero marks a node absorbed by a merge
    /// (a tombstone); it is no longer reachable through the list.
    pub(super) len: usize,
}

impl Node {
    /// Filler for a node buffer before seeding writes the real spans.
    ///
    /// Every field is overwritten before use — `start`/`len` by the seeding
    /// pass and `prev`/`next` by the linking pass — so the values here are
    /// arbitrary. It exists only so a fixed-size array can be declared without
    /// requiring `Node: Default`.
    pub(super) const PLACEHOLDER: Self = Self {
        prev: 0,
        next: 0,
        start: 0,
        len: 0,
    };
}

/// A candidate merge of two adjacent nodes, queued in the selection heap.
///
/// Node indices are assigned in initial list order (by byte offset, or by
/// character offset under character seeding) and never reassigned — a
/// merge keeps the LEFT node and tombstones the right — so `left` is a sound
/// proxy for position in the list, and ordering by it orders left-to-right.
#[derive(Debug, Clone, Copy)]
pub(super) struct Merge {
    pub(super) left: usize,
    pub(super) right: usize,
    /// Merge rank of the pair. Lower merges first.
    pub(super) rank: u32,
    /// Byte length of the merged piece at push time, used to detect a stale
    /// queue entry.
    pub(super) len: usize,
}

impl PartialEq for Merge {
    fn eq(&self, other: &Self) -> bool {
        self.rank == other.rank && self.left == other.left
    }
}
impl Eq for Merge {}

impl Ord for Merge {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // `BinaryHeap` pops the MAXIMUM, but BPE wants the LOWEST rank first,
        // so the rank comparison is reversed.
        other
            .rank
            .cmp(&self.rank)
            // Equal ranks must resolve LEFTMOST to stay bit-exact with
            // tiktoken (and with this crate's own prior linear scan, which
            // took the first minimum in list order). Reversed for the same
            // reason: lowest left index has to pop first. Do NOT "simplify"
            // this tiebreak away — `test_tiebreak_leftmost_wins` locks it in.
            .then_with(|| other.left.cmp(&self.left))
    }
}
impl PartialOrd for Merge {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
