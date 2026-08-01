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
//! The linked-list approach achieves O(1) per merge by simply updating
//! pointers, giving O(N + M) total complexity for the merge phase.
//!
//! # Complexity Analysis
//!
//! - **Time**: O(N × M) where N is text length and M is number of merges
//!   - Initialization: O(N) to create nodes
//!   - Each merge: O(N) to find minimum rank (linear scan)
//!   - Total merges: O(M) where M ≤ N-1
//!   - Average case with good vocabularies: O(N log N)
//!
//! - **Space**: O(N) for the node list
//!
//! # Algorithm Steps
//!
//! 1. Initialize linked list with one node per byte
//! 2. Compute initial ranks for all adjacent pairs
//! 3. Find pair with minimum rank (highest priority)
//! 4. Merge by updating pointers (O(1) operation)
//! 5. Update affected neighbor ranks
//! 6. Repeat until no merges possible

use rustc_hash::FxHashMap;

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
    /// Rank of the pair (this node, next node). MAX if no merge possible.
    rank: u32,
    /// Starting index in the original byte slice
    start: usize,
    /// Length of this piece in bytes
    len: usize,
}

/// Perform byte-pair encoding on a piece of text using a linked-list approach.
///
/// This is the core BPE algorithm that:
/// 1. Initializes a linked list with one node per byte
/// 2. Repeatedly finds the pair with the lowest rank (highest priority merge)
/// 3. Merges that pair by updating linked list pointers
/// 4. Updates ranks of affected neighbors
/// 5. Continues until no more merges are possible
///
/// The linked-list approach has O(N) complexity per merge instead of O(N)
/// memory copying that vector-based approaches require.
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
    let mut nodes: Vec<Node> = Vec::with_capacity(piece.len());
    for i in 0..piece.len() {
        nodes.push(Node {
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
    let get_rank = |left_idx: usize, right_idx: usize, nodes: &[Node]| -> u32 {
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
}
