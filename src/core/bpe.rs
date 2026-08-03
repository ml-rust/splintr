//! Byte-pair encoding (BPE) algorithm using a linked-list approach.
//!
//! This module implements the core BPE algorithm used by modern tokenizers
//! like tiktoken. The key innovation is using a doubly-linked list instead
//! of a vector for merge operations.
//!
//! # Why Linked List?
//!
//! Traditional vector-based BPE implementations suffer from O(N) memory
//! movement on each merge operation (removing an element requires shifting
//! all subsequent elements). With M merges on N bytes, this leads to
//! O(N × M) worst-case complexity.
//!
//! The linked list makes the splice itself O(1): a merge absorbs the right
//! node into the left one and rewires two pointers. Nothing moves. But the
//! splice is only half the problem — *selecting* which pair to merge next has
//! to be cheap too, or the scan for the minimum rank reintroduces the O(N)
//! per merge the list just removed. That selection is done by a binary heap
//! of candidate pairs with lazy deletion: superseded entries are left in the
//! heap and discarded on pop (see `Merge`), so no entry ever has to be found
//! and removed. Each merge pushes at most two new candidates, so the heap
//! holds O(N) entries over the whole run and every push/pop is O(log N).
//!
//! # Complexity Analysis
//!
//! - **Time**: O(N log N) where N is text length
//!   - Initialization: O(N) to create nodes, O(N log N) to seed the heap
//!   - Each merge: O(1) splice + O(log N) to pop and to push ≤ 2 candidates
//!   - Total merges: O(M) where M ≤ N-1
//!   - Lazy deletion means pops are bounded by pushes, which are O(N)
//!
//! - **Space**: O(N) for the node list and O(N) for the heap
//!
//! # Algorithm Steps
//!
//! 1. Initialize linked list with one node per byte
//! 2. Push every adjacent pair that has a merge rank onto the heap
//! 3. Pop the best candidate: lowest rank, leftmost position on a tie
//! 4. Skip it if stale (its nodes have since changed), otherwise merge by
//!    updating pointers (O(1) operation)
//! 5. Push the two pairs the merge created, around the merged node
//! 6. Repeat until the heap is empty

use rustc_hash::FxHashMap;
use std::collections::BinaryHeap;

/// A node in the doubly-linked list used for BPE merging.
///
/// This approach avoids the O(N) memory shifting of vector-based approaches
/// when merging pairs. Instead of removing elements, we simply update pointers.
#[derive(Debug, Clone, Copy)]
struct Node {
    /// Index of previous node (usize::MAX if head)
    prev: usize,
    /// Index of next node (usize::MAX if tail)
    next: usize,
    /// Starting index in the original byte slice
    start: usize,
    /// Length of this piece in bytes. Zero marks a node absorbed by a merge
    /// (a tombstone); it is no longer reachable through the list.
    len: usize,
}

/// A candidate merge of two adjacent nodes, queued in the selection heap.
///
/// Node indices are assigned by initial byte offset and never reassigned — a
/// merge keeps the LEFT node and tombstones the right — so `left` is a sound
/// proxy for position in the list, and ordering by it orders left-to-right.
#[derive(Debug, Clone, Copy)]
struct Merge {
    left: usize,
    right: usize,
    /// Merge rank of the pair. Lower merges first.
    rank: u32,
    /// Byte length of the merged piece at push time, used to detect a stale
    /// queue entry.
    len: usize,
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

/// Perform byte-pair encoding on a piece of text using a linked-list approach.
///
/// This is the core BPE algorithm that:
/// 1. Initializes a linked list with one node per byte
/// 2. Queues every adjacent pair in a binary heap, ordered by merge rank
/// 3. Pops the pair with the lowest rank (leftmost on a tie), skipping
///    entries the list has since invalidated
/// 4. Merges that pair by updating linked list pointers
/// 5. Queues the pairs the merge created
/// 6. Continues until no more merges are possible
///
/// The linked list makes each splice O(1), and the heap makes each selection
/// O(log N), for O(N log N) overall.
pub fn byte_pair_encode(piece: &[u8], encoder: &FxHashMap<Vec<u8>, u32>) -> Vec<u32> {
    // tiktoken-style: the token id doubles as its merge rank.
    byte_pair_encode_with_ranks(piece, encoder, encoder)
}

/// Byte-pair encoding with **separate** merge-rank and output-id maps.
///
/// HuggingFace BPE models define merge priority via a `merges` list that is
/// independent of token ids (e.g. RoBERTa orders ids differently from merges).
/// `merge_ranks` maps a merged token's bytes → its merge priority (lower =
/// merged first); `id_encoder` maps token bytes → output id. For tiktoken-style
/// vocabs the two maps are identical, which is what [`byte_pair_encode`] passes.
pub fn byte_pair_encode_with_ranks(
    piece: &[u8],
    merge_ranks: &FxHashMap<Vec<u8>, u32>,
    id_encoder: &FxHashMap<Vec<u8>, u32>,
) -> Vec<u32> {
    byte_pair_encode_pieces(piece, merge_ranks, id_encoder)
        .into_iter()
        .filter_map(|p| match p {
            Piece::Token(id) => Some(id),
            Piece::Unresolved { .. } => None,
        })
        .collect()
}

/// One output of the merge phase: either a resolved token, or a run of bytes
/// the vocabulary could not represent at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Piece {
    Token(u32),
    /// Byte range `[start, start + len)` within the input piece that resolved
    /// to no token — neither as a whole nor byte-by-byte.
    Unresolved {
        start: usize,
        len: usize,
    },
}

/// Same algorithm as [`byte_pair_encode_with_ranks`], but reports byte spans
/// the vocabulary could not resolve instead of silently dropping them.
///
/// [`byte_pair_encode_with_ranks`] is defined in terms of this function and
/// simply filters out the [`Piece::Unresolved`] spans, preserving its
/// documented "drops what it cannot represent" contract exactly.
pub(super) fn byte_pair_encode_pieces(
    piece: &[u8],
    merge_ranks: &FxHashMap<Vec<u8>, u32>,
    id_encoder: &FxHashMap<Vec<u8>, u32>,
) -> Vec<Piece> {
    if piece.is_empty() {
        return vec![];
    }

    // Fast path: single byte
    if piece.len() == 1 {
        return match id_encoder.get(piece) {
            Some(&r) => vec![Piece::Token(r)],
            None => vec![Piece::Unresolved { start: 0, len: 1 }],
        };
    }

    // Fast path: entire piece is a single token
    if let Some(&id) = id_encoder.get(piece) {
        return vec![Piece::Token(id)];
    }

    // Initialize linked list - one node per byte
    let mut nodes: Vec<Node> = Vec::with_capacity(piece.len());
    for i in 0..piece.len() {
        nodes.push(Node {
            prev: if i == 0 { usize::MAX } else { i - 1 },
            next: if i == piece.len() - 1 {
                usize::MAX
            } else {
                i + 1
            },
            start: i,
            len: 1,
        });
    }

    // Queue a pair as a merge candidate, if the vocabulary can merge it.
    // A pair with no rank can never be selected, so it is simply never pushed.
    // `u32::MAX` is the "unmergeable" sentinel, so a vocabulary that maps a
    // token to it is not queued either — that is what the previous linear
    // scan did, and this stays bit-exact with it.
    let push = |queue: &mut BinaryHeap<Merge>, left: usize, right: usize, nodes: &[Node]| {
        if left == usize::MAX || right == usize::MAX {
            return;
        }
        let (l, r) = (&nodes[left], &nodes[right]);
        let len = l.len + r.len;
        let slice = &piece[l.start..l.start + len];
        if let Some(rank) = merge_ranks.get(slice).copied().filter(|&r| r != u32::MAX) {
            queue.push(Merge {
                left,
                right,
                rank,
                len,
            });
        }
    };

    // Seed the heap with every adjacent pair.
    let mut queue: BinaryHeap<Merge> = BinaryHeap::new();
    for i in 0..nodes.len() - 1 {
        push(&mut queue, i, i + 1, &nodes);
    }

    // Main merge loop: best candidate first, stale ones dropped on the floor.
    while let Some(candidate) = queue.pop() {
        let (li, ri) = (candidate.left, candidate.right);

        // Staleness test, exact rather than heuristic. `left.len` only ever
        // changes by absorbing rightward, and that also changes `left.next` —
        // so `next == right` proves the left side is untouched since the push,
        // and the length sum then proves the right side is too. A tombstoned
        // left node (len 0) has been absorbed by its own predecessor.
        let (left, right) = (nodes[li], nodes[ri]);
        if left.len == 0 || left.next != ri || left.len + right.len != candidate.len {
            continue;
        }

        // Absorb the right node into the left one and unlink it.
        nodes[li].len = left.len + right.len;
        nodes[ri].len = 0;

        let new_next = right.next;
        nodes[li].next = new_next;
        if new_next != usize::MAX {
            nodes[new_next].prev = li;
        }

        // The merge created at most two new adjacencies, around the merged node.
        push(&mut queue, nodes[li].prev, li, &nodes);
        push(&mut queue, li, new_next, &nodes);
    }

    // Collect final pieces by traversing the linked list. Node 0 is always the
    // head: a merge keeps the left node, so the first node is never absorbed.
    // Consecutive unresolved bytes coalesce into a single `Unresolved` span:
    // an open run is tracked and only flushed when a resolved byte or token
    // breaks it (or the list ends), rather than pushing one span per byte.
    // Upper bound: at most one `Piece` per input byte (worst case: no merges
    // and nothing resolves, so every byte is its own unresolved run — the
    // coalescing below only ever shrinks that).
    let mut result: Vec<Piece> = Vec::with_capacity(piece.len());
    let mut curr = 0;
    let mut unresolved_run: Option<(usize, usize)> = None; // (start, len)

    while curr != usize::MAX {
        let node = &nodes[curr];
        let slice = &piece[node.start..node.start + node.len];

        if let Some(&id) = id_encoder.get(slice) {
            if let Some((start, len)) = unresolved_run.take() {
                result.push(Piece::Unresolved { start, len });
            }
            result.push(Piece::Token(id));
        } else {
            // Fallback: if somehow we have an unknown token, try to encode bytes individually
            // This shouldn't happen with a proper BPE vocabulary that covers all bytes
            for (offset, &byte) in slice.iter().enumerate() {
                if let Some(&id) = id_encoder.get(&[byte][..]) {
                    if let Some((start, len)) = unresolved_run.take() {
                        result.push(Piece::Unresolved { start, len });
                    }
                    result.push(Piece::Token(id));
                } else {
                    let byte_start = node.start + offset;
                    match &mut unresolved_run {
                        Some((start, len)) if *start + *len == byte_start => *len += 1,
                        // Covers two cases: no run open yet (`None`), and a
                        // run open but not adjacent to `byte_start`. The
                        // latter can't actually happen: traversal visits
                        // nodes in list order and nodes partition
                        // `0..piece.len()` with no gaps (a merge only ever
                        // absorbs a node into its immediate left neighbour,
                        // never creating a hole), and within one node's
                        // fallback loop `offset` is strictly increasing. So
                        // whenever a run is open, `byte_start` is always its
                        // next byte. Kept as `_` rather than narrowed to
                        // `None` — it's a correct, cheap defensive fallback
                        // for that unreachable case, not dead code to prune.
                        _ => {
                            if let Some((start, len)) = unresolved_run.take() {
                                result.push(Piece::Unresolved { start, len });
                            }
                            unresolved_run = Some((byte_start, 1));
                        }
                    }
                }
            }
        }
        curr = nodes[curr].next;
    }

    if let Some((start, len)) = unresolved_run.take() {
        result.push(Piece::Unresolved { start, len });
    }

    result
}

/// Build a bytes → merge-rank map (lower rank = merged first) from a model's
/// ordered merge list and the vocabulary it was built over.
///
/// Merge priority is independent of token id (RoBERTa orders its merges
/// differently from GPT-2, and GGUF vocabularies disagree with their own id
/// order), so the ranks come from the list, not from the ids. The map covers two
/// groups, ranked so the first always wins:
///
/// 1. **Base alphabet** — vocabulary entries that are never a merge *result*
///    (the byte-level single chars). Their multi-byte UTF-8 has to coalesce
///    before any real merge runs, so they take the lowest ranks `0..b`.
/// 2. **Merges** — each merged token (`a ++ b`) at rank `b + merge_index`.
///
/// `merged` holds the already-concatenated result of each merge, in list order.
/// `vocab_in_id_order` yields every vocabulary token, lowest id first, so the
/// base ranks are deterministic.
pub(super) fn merge_ranks<'a>(
    merged: &[String],
    vocab_in_id_order: impl Iterator<Item = &'a str>,
) -> FxHashMap<Vec<u8>, u32> {
    let merge_set: std::collections::HashSet<&str> = merged.iter().map(String::as_str).collect();
    let mut ranks: FxHashMap<Vec<u8>, u32> = FxHashMap::default();

    // Base alphabet first, in id order for determinism.
    for token in vocab_in_id_order.filter(|t| !merge_set.contains(t)) {
        let next = ranks.len() as u32;
        ranks.entry(token.as_bytes().to_vec()).or_insert(next);
    }

    // Then the merges, preserving list priority.
    let base_count = ranks.len() as u32;
    for (i, token) in merged.iter().enumerate() {
        ranks
            .entry(token.as_bytes().to_vec())
            .or_insert(base_count + i as u32);
    }
    ranks
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// A node in the reference implementation's linked list.
    ///
    /// Identical to [`Node`] plus the `rank` field the linear scan needs.
    #[derive(Debug, Clone, Copy)]
    struct RefNode {
        prev: usize,
        next: usize,
        rank: u32,
        start: usize,
        len: usize,
    }

    /// The pre-heap implementation, kept verbatim as a correctness oracle.
    ///
    /// It rescans the whole list for the minimum rank on every merge, which is
    /// O(N × M) and unusable in production — but it is obviously correct, and
    /// [`byte_pair_encode_with_ranks`] must not diverge from it on ANY input.
    /// In particular it resolves equal ranks LEFTMOST, because the scan keeps
    /// the first strict minimum in list order.
    fn byte_pair_encode_reference(
        piece: &[u8],
        merge_ranks: &FxHashMap<Vec<u8>, u32>,
        id_encoder: &FxHashMap<Vec<u8>, u32>,
    ) -> Vec<u32> {
        if piece.is_empty() {
            return vec![];
        }

        // Fast path: single byte
        if piece.len() == 1 {
            return id_encoder.get(piece).copied().map_or(vec![], |r| vec![r]);
        }

        // Fast path: entire piece is a single token
        if let Some(&id) = id_encoder.get(piece) {
            return vec![id];
        }

        // Initialize linked list - one node per byte
        let mut nodes: Vec<RefNode> = Vec::with_capacity(piece.len());
        for i in 0..piece.len() {
            nodes.push(RefNode {
                prev: if i == 0 { usize::MAX } else { i - 1 },
                next: if i == piece.len() - 1 {
                    usize::MAX
                } else {
                    i + 1
                },
                rank: u32::MAX,
                start: i,
                len: 1,
            });
        }

        // Helper closure to compute the merge rank of a pair
        let get_rank = |left_idx: usize, right_idx: usize, nodes: &[RefNode]| -> u32 {
            if left_idx == usize::MAX || right_idx == usize::MAX {
                return u32::MAX;
            }
            let left = &nodes[left_idx];
            let right = &nodes[right_idx];

            let start = left.start;
            let len = left.len + right.len;
            let slice = &piece[start..start + len];

            merge_ranks.get(slice).copied().unwrap_or(u32::MAX)
        };

        // Initial rank calculation for all adjacent pairs
        for i in 0..nodes.len() - 1 {
            nodes[i].rank = get_rank(i, nodes[i].next, &nodes);
        }

        // Main merge loop
        loop {
            // Find the pair with minimum rank (highest priority merge)
            let mut min_rank = u32::MAX;
            let mut min_idx = usize::MAX;

            let mut curr = 0;
            // Find the head of the list (in case we started from a deleted node)
            while nodes[curr].prev != usize::MAX {
                curr = nodes[curr].prev;
            }

            // Linear scan through the linked list
            while curr != usize::MAX {
                let r = nodes[curr].rank;
                if r < min_rank {
                    min_rank = r;
                    min_idx = curr;
                }
                curr = nodes[curr].next;
            }

            // No more merges possible
            if min_rank == u32::MAX {
                break;
            }

            // Merge min_idx with its next node
            let next_idx = nodes[min_idx].next;

            // Update the merged node's length
            nodes[min_idx].len += nodes[next_idx].len;

            // Update linked list pointers (skip over next_idx)
            let new_next = nodes[next_idx].next;
            nodes[min_idx].next = new_next;
            if new_next != usize::MAX {
                nodes[new_next].prev = min_idx;
            }

            // Update ranks for affected pairs:
            // 1. The pair (prev, min_idx) if prev exists
            if nodes[min_idx].prev != usize::MAX {
                let prev = nodes[min_idx].prev;
                nodes[prev].rank = get_rank(prev, min_idx, &nodes);
            }

            // 2. The pair (min_idx, new_next)
            nodes[min_idx].rank = get_rank(min_idx, nodes[min_idx].next, &nodes);
        }

        // Collect final tokens by traversing the linked list
        let mut result = Vec::new();

        // Find head
        let mut curr = 0;
        while nodes[curr].prev != usize::MAX {
            curr = nodes[curr].prev;
        }

        while curr != usize::MAX {
            let node = &nodes[curr];
            let slice = &piece[node.start..node.start + node.len];

            if let Some(&id) = id_encoder.get(slice) {
                result.push(id);
            } else {
                // Fallback: if somehow we have an unknown token, try to encode bytes individually
                // This shouldn't happen with a proper BPE vocabulary that covers all bytes
                for &byte in slice {
                    if let Some(&id) = id_encoder.get(&[byte][..]) {
                        result.push(id);
                    }
                }
            }
            curr = nodes[curr].next;
        }

        result
    }

    fn make_encoder() -> FxHashMap<Vec<u8>, u32> {
        let mut encoder = FxHashMap::default();
        // Individual bytes
        encoder.insert(b"a".to_vec(), 0);
        encoder.insert(b"b".to_vec(), 1);
        encoder.insert(b"c".to_vec(), 2);
        // Merged pairs (lower rank = higher priority)
        encoder.insert(b"ab".to_vec(), 3);
        encoder.insert(b"bc".to_vec(), 4);
        encoder.insert(b"abc".to_vec(), 5);
        encoder
    }

    #[test]
    fn test_single_byte() {
        let encoder = make_encoder();
        assert_eq!(byte_pair_encode(b"a", &encoder), vec![0]);
    }

    #[test]
    fn test_simple_merge() {
        let encoder = make_encoder();
        // "ab" should merge to token 3
        assert_eq!(byte_pair_encode(b"ab", &encoder), vec![3]);
    }

    #[test]
    fn test_chain_merge() {
        let encoder = make_encoder();
        // "abc" should merge to token 5
        // First "ab" (rank 3) or "bc" (rank 4)? "ab" has lower rank, so:
        // a b c -> ab c -> abc
        assert_eq!(byte_pair_encode(b"abc", &encoder), vec![5]);
    }

    #[test]
    fn test_empty() {
        let encoder = make_encoder();
        let empty: Vec<u32> = vec![];
        assert_eq!(byte_pair_encode(b"", &encoder), empty);
    }

    #[test]
    fn test_no_merge_possible() {
        let encoder = make_encoder();
        // "ac" has no merge entry, so stays as [a, c]
        assert_eq!(byte_pair_encode(b"ac", &encoder), vec![0, 2]);
    }

    /// A vocabulary in which every pair of a repeated character ties.
    fn tie_encoder() -> FxHashMap<Vec<u8>, u32> {
        let mut encoder = FxHashMap::default();
        encoder.insert(b"a".to_vec(), 0);
        encoder.insert(b"aa".to_vec(), 1);
        encoder
    }

    /// Equal-rank pairs MUST resolve leftmost.
    ///
    /// In `"aaa"` both `(0,1)` and `(1,2)` spell `"aa"` at rank 1. Taking the
    /// left one yields `[aa][a]` = `[1, 0]`; taking the right one yields
    /// `[a][aa]` = `[0, 1]`. tiktoken produces the former, so we must too —
    /// and repeated-character runs are exactly the pathological input where
    /// this is reachable, not some exotic corner.
    ///
    /// This is the test that fails if someone "simplifies" `Merge`'s `Ord` to
    /// compare rank alone: a rank-only heap picks an arbitrary one of the tied
    /// pairs and silently changes token output.
    #[test]
    fn test_tiebreak_leftmost_wins() {
        let encoder = tie_encoder();
        assert_eq!(byte_pair_encode(b"aaa", &encoder), vec![1, 0]);
        assert_eq!(byte_pair_encode(b"aaaaa", &encoder), vec![1, 1, 0]);

        // And the oracle agrees, which is where the invariant comes from.
        assert_eq!(
            byte_pair_encode_reference(b"aaa", &encoder, &encoder),
            vec![1, 0]
        );
        assert_eq!(
            byte_pair_encode_reference(b"aaaaa", &encoder, &encoder),
            vec![1, 1, 0]
        );
    }

    /// tiktoken-style vocabulary: the id doubles as the merge rank.
    fn prop_encoder() -> FxHashMap<Vec<u8>, u32> {
        let mut encoder = FxHashMap::default();
        let tokens: [&[u8]; 18] = [
            b"a", b"b", b"c", b"d", b"aa", b"ab", b"ba", b"bb", b"cd", b"dc", b"cc", b"aaa",
            b"aab", b"abab", b"aaaa", b"bcd", b"abcd", b"abc",
        ];
        for (i, token) in tokens.iter().enumerate() {
            encoder.insert(token.to_vec(), i as u32);
        }
        encoder
    }

    /// HuggingFace-style vocabulary: merge priority is independent of the id,
    /// and several merges deliberately share a rank so the leftmost tiebreak
    /// is exercised rather than accidentally avoided by unique ranks.
    fn prop_two_maps() -> (FxHashMap<Vec<u8>, u32>, FxHashMap<Vec<u8>, u32>) {
        let ranked: [(&[u8], u32); 13] = [
            (b"aa", 1),
            (b"ab", 1),
            (b"bb", 1),
            (b"ba", 2),
            (b"cc", 2),
            (b"cd", 3),
            (b"dc", 3),
            (b"aaa", 4),
            (b"aab", 4),
            (b"abb", 4),
            (b"abab", 5),
            (b"aaaa", 5),
            (b"abcd", 6),
        ];
        let mut merge_ranks = FxHashMap::default();
        for (token, rank) in ranked {
            merge_ranks.insert(token.to_vec(), rank);
        }

        // Ids in an order that has nothing to do with the merge ranks, so a
        // mix-up between the two maps cannot pass unnoticed.
        let ids: [&[u8]; 17] = [
            b"abcd", b"aaaa", b"abab", b"abb", b"aab", b"aaa", b"dc", b"cd", b"cc", b"ba", b"bb",
            b"ab", b"aa", b"d", b"c", b"b", b"a",
        ];
        let mut id_encoder = FxHashMap::default();
        for (i, token) in ids.iter().enumerate() {
            id_encoder.insert(token.to_vec(), i as u32);
        }
        (merge_ranks, id_encoder)
    }

    #[test]
    fn test_long_single_char_run() {
        let encoder = tie_encoder();
        let piece = vec![b'a'; 4096];
        let expected = vec![1u32; 2048];
        assert_eq!(byte_pair_encode(&piece, &encoder), expected);
        assert_eq!(
            byte_pair_encode_reference(&piece, &encoder, &encoder),
            expected
        );
    }

    #[test]
    fn test_repeated_ab() {
        let encoder = prop_encoder();
        let piece = b"ab".repeat(512);
        assert_eq!(
            byte_pair_encode(&piece, &encoder),
            byte_pair_encode_reference(&piece, &encoder, &encoder)
        );
    }

    #[test]
    fn test_whole_piece_is_one_token() {
        let encoder = prop_encoder();
        // "abcd" is in the vocabulary, so the fast path returns its id alone.
        assert_eq!(byte_pair_encode(b"abcd", &encoder), vec![16]);
    }

    #[test]
    fn test_single_byte_and_empty_agree_with_reference() {
        let encoder = prop_encoder();
        let pieces: [&[u8]; 3] = [b"", b"a", b"z"];
        for piece in pieces {
            assert_eq!(
                byte_pair_encode(piece, &encoder),
                byte_pair_encode_reference(piece, &encoder, &encoder),
                "piece {piece:?}"
            );
        }
    }

    #[test]
    fn test_bytes_absent_from_vocab() {
        let encoder = prop_encoder();
        // 'z' and 0xFF are not in the vocabulary at all; the fallback path
        // drops them, which the new implementation must reproduce exactly.
        let pieces: [&[u8]; 4] = [b"azb", b"\xff\xfe", b"abzcd", b"zzz"];
        for piece in pieces {
            assert_eq!(
                byte_pair_encode(piece, &encoder),
                byte_pair_encode_reference(piece, &encoder, &encoder),
                "piece {piece:?}"
            );
        }
    }

    /// A vocabulary covering `a` and `c` (and nothing else, no merges) — used
    /// to exercise `byte_pair_encode_pieces`'s `Unresolved` reporting.
    fn ac_only_encoder() -> FxHashMap<Vec<u8>, u32> {
        let mut encoder = FxHashMap::default();
        encoder.insert(b"a".to_vec(), 0);
        encoder.insert(b"c".to_vec(), 1);
        encoder
    }

    #[test]
    fn test_pieces_reports_unresolved_span() {
        let encoder = ac_only_encoder();
        assert_eq!(
            byte_pair_encode_pieces(b"abc", &encoder, &encoder),
            vec![
                Piece::Token(0),
                Piece::Unresolved { start: 1, len: 1 },
                Piece::Token(1),
            ]
        );
        // The preserved primitive still just drops the gap.
        assert_eq!(byte_pair_encode(b"abc", &encoder), vec![0, 1]);
    }

    #[test]
    fn test_pieces_coalesce_consecutive_unresolved() {
        // Vocabulary missing both `b` and `c`.
        let mut encoder = FxHashMap::default();
        encoder.insert(b"a".to_vec(), 0);
        encoder.insert(b"d".to_vec(), 1);
        assert_eq!(
            byte_pair_encode_pieces(b"abcd", &encoder, &encoder),
            vec![
                Piece::Token(0),
                Piece::Unresolved { start: 1, len: 2 },
                Piece::Token(1),
            ]
        );
    }

    #[test]
    fn test_pieces_unresolved_at_start_and_end() {
        let encoder = ac_only_encoder();
        // "bab" style: missing byte at the very start and the very end.
        assert_eq!(
            byte_pair_encode_pieces(b"bab", &encoder, &encoder),
            vec![
                Piece::Unresolved { start: 0, len: 1 },
                Piece::Token(0),
                Piece::Unresolved { start: 2, len: 1 },
            ]
        );
    }

    #[test]
    fn test_pieces_full_coverage_has_no_unresolved() {
        let encoder = prop_encoder();
        for piece in [&b""[..], b"a", b"abcd", b"aabbccdd", b"dcbadcba"] {
            let pieces = byte_pair_encode_pieces(piece, &encoder, &encoder);
            assert!(
                pieces.iter().all(|p| matches!(p, Piece::Token(_))),
                "piece {piece:?} produced {pieces:?}"
            );
        }
    }

    #[test]
    fn test_pieces_single_byte_and_empty_fast_paths() {
        let encoder = ac_only_encoder();
        let empty: Vec<Piece> = vec![];
        assert_eq!(byte_pair_encode_pieces(b"", &encoder, &encoder), empty);
        assert_eq!(
            byte_pair_encode_pieces(b"a", &encoder, &encoder),
            vec![Piece::Token(0)]
        );
        assert_eq!(
            byte_pair_encode_pieces(b"b", &encoder, &encoder),
            vec![Piece::Unresolved { start: 0, len: 1 }]
        );
    }

    proptest! {
        /// Single-map (tiktoken-style) path over a dense small alphabet, where
        /// merges actually chain.
        #[test]
        fn prop_matches_reference_single_map(
            piece in prop::collection::vec(prop::sample::select(vec![b'a', b'b', b'c', b'd']), 0..64)
        ) {
            let encoder = prop_encoder();
            prop_assert_eq!(
                byte_pair_encode(&piece, &encoder),
                byte_pair_encode_reference(&piece, &encoder, &encoder)
            );
        }

        /// Same path, but over arbitrary bytes so the unknown-byte fallback and
        /// the no-merge-possible cases are covered too.
        #[test]
        fn prop_matches_reference_arbitrary_bytes(
            piece in prop::collection::vec(any::<u8>(), 0..48)
        ) {
            let encoder = prop_encoder();
            prop_assert_eq!(
                byte_pair_encode(&piece, &encoder),
                byte_pair_encode_reference(&piece, &encoder, &encoder)
            );
        }

        /// Two-map (HuggingFace-style) path: merge ranks tie and disagree with
        /// the ids, so both the tiebreak and the map separation are exercised.
        #[test]
        fn prop_matches_reference_two_maps(
            piece in prop::collection::vec(prop::sample::select(vec![b'a', b'b', b'c', b'd', b'z']), 0..64)
        ) {
            let (merge_ranks, id_encoder) = prop_two_maps();
            prop_assert_eq!(
                byte_pair_encode_with_ranks(&piece, &merge_ranks, &id_encoder),
                byte_pair_encode_reference(&piece, &merge_ranks, &id_encoder)
            );
        }

        /// `byte_pair_encode_pieces` filtered down to its `Token` ids must
        /// agree with `byte_pair_encode_with_ranks` exactly — the latter is
        /// now defined in terms of the former, but this pins the equivalence
        /// as an independent property over arbitrary bytes (so the
        /// `Unresolved`-reporting fallback path is exercised too).
        #[test]
        fn prop_pieces_tokens_match_with_ranks(
            piece in prop::collection::vec(any::<u8>(), 0..48)
        ) {
            let encoder = prop_encoder();
            let tokens_only: Vec<u32> = byte_pair_encode_pieces(&piece, &encoder, &encoder)
                .into_iter()
                .filter_map(|p| match p {
                    Piece::Token(id) => Some(id),
                    Piece::Unresolved { .. } => None,
                })
                .collect();
            prop_assert_eq!(
                tokens_only,
                byte_pair_encode_with_ranks(&piece, &encoder, &encoder)
            );
        }
    }
}
