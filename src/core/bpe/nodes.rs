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
    /// The id of the token this node's surface is, on the id-keyed merge path;
    /// `u32::MAX` and unread on the byte-keyed one, which resolves ids from the
    /// surface after merging instead.
    pub(super) id: u32,
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
        id: u32::MAX,
    };
}
