use crate::core::encoder::Encoder;
use std::collections::BinaryHeap;

use super::encode::{Piece, Seed};
use super::nodes::{Merge, Node};
use super::ranks::RankLookup;

/// Symbol count at or below which merges are selected by scanning a
/// stack-resident rank table instead of a heap.
///
/// The two strategies are asymptotically opposite and there is no single right
/// answer, so the threshold is measured rather than guessed. Selection cost per
/// merge is O(symbols) for the scan and O(log symbols) for the heap, but the
/// heap pays a heap *allocation* that grows as candidates are queued, and at
/// small symbol counts that allocation dominates everything else. Measured on
/// llama-3 over novel words, the scan wins up to roughly 100 symbols and the
/// heap wins above it, diverging fast in both directions (at 8000 symbols the
/// heap is ~46x ahead; at 16 symbols the scan is ~25% ahead).
///
/// 64 sits deliberately on the safe side of that crossover. It also covers the
/// entire realistic range of the dominant workload: a pre-tokenizer emits
/// chunks of a handful of symbols, so ordinary text never reaches the heap at
/// all, while a `tokenizer.json` with no splitting stage (`pre_tokenizer:
/// null`, which is how Mistral's AWQ and GPTQ files are shaped) hands over the
/// whole document and needs the heap's asymptote to stay usable.
///
/// The table is `[u32; 64]` — 256 bytes, four cache lines, no allocation.
pub(super) const SCAN_SYMBOL_LIMIT: usize = 64;

/// Merge rank of the pair `(left, right)`, or `u32::MAX` when the vocabulary
/// cannot merge it.
///
/// Both selection strategies go through here, which is what keeps them
/// bit-exact with each other. The `u32::MAX` sentinel and its handling live in
/// [`RankLookup::get`].
#[inline]
fn rank_of(
    piece: &[u8],
    nodes: &[Node],
    left: usize,
    right: usize,
    merge_ranks: RankLookup<'_>,
) -> u32 {
    if left == usize::MAX || right == usize::MAX {
        return u32::MAX;
    }
    let (l, r) = (&nodes[left], &nodes[right]);
    let len = l.len + r.len;
    merge_ranks.get(&piece[l.start..l.start + len])
}

/// Absorb `right` into `left` and unlink it, returning the node that follows.
#[inline]
fn absorb(nodes: &mut [Node], left: usize, right: usize) -> usize {
    nodes[left].len += nodes[right].len;
    nodes[right].len = 0;

    let new_next = nodes[right].next;
    nodes[left].next = new_next;
    if new_next != usize::MAX {
        nodes[new_next].prev = left;
    }
    new_next
}

/// Merge selection for short pieces: a rank per node in a stack array, rescanned
/// for the minimum on every merge.
///
/// The scan runs over the flat array rather than walking the list, because the
/// two are equivalent here and the array is contiguous: node indices are
/// assigned in list order and never reassigned, an absorbed node is tombstoned
/// to `u32::MAX`, and so the lowest-ranked live index is exactly the leftmost
/// lowest-ranked pair in the list — the tie-break tiktoken requires.
fn merge_by_scan(piece: &[u8], nodes: &mut [Node], merge_ranks: RankLookup<'_>) -> usize {
    let count = nodes.len();
    let mut ranks = [u32::MAX; SCAN_SYMBOL_LIMIT];
    for (i, rank) in ranks.iter_mut().enumerate().take(count.saturating_sub(1)) {
        *rank = rank_of(piece, nodes, i, i + 1, merge_ranks);
    }

    let mut live = count;
    loop {
        let mut best_rank = u32::MAX;
        let mut best = usize::MAX;
        for (i, &rank) in ranks.iter().enumerate().take(count) {
            if rank < best_rank {
                best_rank = rank;
                best = i;
            }
        }
        if best_rank == u32::MAX {
            return live;
        }

        let right = nodes[best].next;
        let new_next = absorb(nodes, best, right);
        live -= 1;
        // The absorbed node can never be selected again.
        ranks[right] = u32::MAX;

        // Only the two adjacencies around the merged node changed.
        let prev = nodes[best].prev;
        if prev != usize::MAX {
            ranks[prev] = rank_of(piece, nodes, prev, best, merge_ranks);
        }
        ranks[best] = rank_of(piece, nodes, best, new_next, merge_ranks);
    }
}

/// Merge selection for long pieces: a binary heap of candidates with lazy
/// deletion, so selection is O(log N) instead of O(N) per merge.
///
/// Superseded entries are left in the heap and discarded on pop, so no entry
/// ever has to be found and removed. Each merge pushes at most two new
/// candidates, so the heap holds O(N) entries over the whole run.
fn merge_by_heap(piece: &[u8], nodes: &mut [Node], merge_ranks: RankLookup<'_>) -> usize {
    let count = nodes.len();

    // Queue a pair as a merge candidate, if the vocabulary can merge it. A pair
    // with no rank can never be selected, so it is simply never pushed.
    let push = |queue: &mut BinaryHeap<Merge>, left: usize, right: usize, nodes: &[Node]| {
        let rank = rank_of(piece, nodes, left, right, merge_ranks);
        if rank != u32::MAX {
            queue.push(Merge {
                left,
                right,
                rank,
                len: nodes[left].len + nodes[right].len,
            });
        }
    };

    // Seed the heap with every adjacent pair. `saturating_sub` because the
    // callers guarantee a non-empty list through their own guards, and neither
    // may become an underflow if a third ever appears.
    let mut queue: BinaryHeap<Merge> = BinaryHeap::with_capacity(count * 2);
    for i in 0..count.saturating_sub(1) {
        push(&mut queue, i, i + 1, nodes);
    }

    let mut live = count;
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

        let new_next = absorb(nodes, li, ri);
        live -= 1;

        // The merge created at most two new adjacencies, around the merged node.
        let prev = nodes[li].prev;
        if prev != usize::MAX {
            push(&mut queue, prev, li, nodes);
        }
        if new_next != usize::MAX {
            push(&mut queue, li, new_next, nodes);
        }
    }
    live
}

/// Link `nodes` into a list and run the merge loop over them, returning how
/// many nodes are still live.
///
/// `nodes` arrives with `start`/`len` set and `prev`/`next` ignored.
fn link_and_merge(piece: &[u8], nodes: &mut [Node], merge_ranks: RankLookup<'_>) -> usize {
    let count = nodes.len();
    for (i, node) in nodes.iter_mut().enumerate() {
        node.prev = if i == 0 { usize::MAX } else { i - 1 };
        node.next = if i + 1 == count { usize::MAX } else { i + 1 };
    }

    if count <= SCAN_SYMBOL_LIMIT {
        merge_by_scan(piece, nodes, merge_ranks)
    } else {
        merge_by_heap(piece, nodes, merge_ranks)
    }
}

/// Resolve one node's surface to an id, honoring a presegmented seed's own id.
///
/// A presegmented seed that no merge absorbed (its length is still the one it
/// was seeded with) keeps the id the caller already resolved it to. That id is
/// the authority: the caller may have resolved it from a table other than
/// `id_encoder` — a `<0xNN>` byte-fallback table, say — and re-deriving it from
/// the surface would answer a different question. A merged node has no seed id
/// and resolves as usual.
#[inline]
fn resolve(
    slice: &[u8],
    node: &Node,
    index: usize,
    id_encoder: &Encoder,
    seeds: Option<&[Seed]>,
) -> Option<u32> {
    seeds
        .and_then(|seeds| seeds.get(index))
        .filter(|seed| seed.len == node.len)
        .and_then(|seed| seed.id)
        .or_else(|| id_encoder.get(slice))
}

/// Link, merge, and collect the result as [`Piece`]s, reporting byte spans the
/// vocabulary could not resolve.
///
/// `seeds`, when present, is the segmentation `nodes` was built from, and
/// supplies the id of a symbol no merge absorbed (see [`Seed::id`]); `None` is
/// the ordinary path, where every node resolves through `id_encoder` alone.
pub(super) fn merge_and_collect(
    piece: &[u8],
    mut nodes: Vec<Node>,
    merge_ranks: RankLookup<'_>,
    id_encoder: &Encoder,
    seeds: Option<&[Seed]>,
) -> Vec<Piece> {
    link_and_merge(piece, &mut nodes, merge_ranks);

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

        if let Some(id) = resolve(slice, node, curr, id_encoder, seeds) {
            if let Some((start, len)) = unresolved_run.take() {
                result.push(Piece::Unresolved { start, len });
            }
            result.push(Piece::Token(id));
        } else {
            // Fallback: if somehow we have an unknown token, try to encode bytes individually
            // This shouldn't happen with a proper BPE vocabulary that covers all bytes
            for (offset, &byte) in slice.iter().enumerate() {
                if let Some(id) = id_encoder.get(&[byte][..]) {
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

/// Link, merge, and collect the result directly as token ids, dropping what the
/// vocabulary cannot represent.
///
/// This is [`merge_and_collect`] with the [`Piece`] layer removed rather than
/// filtered afterwards. It exists because the caller that discards
/// `Piece::Unresolved` is the common one by a wide margin — every ByteLevel and
/// every tiktoken-style vocabulary, i.e. every vocabulary with no byte fallback
/// configured — and building a `Vec<Piece>` (24 bytes per entry, sized by input
/// length) only to filter it into a `Vec<u32>` costs that caller an allocation
/// and a second pass per chunk, on the hottest path in the crate.
///
/// Takes no `seeds`: the presegmented seeding exists solely to reproduce
/// HuggingFace's byte-fallback ordering, so it is only ever reached with a
/// fallback configured, which is exactly when [`merge_and_collect`] is used
/// instead.
/// Appends to `out` rather than returning a fresh `Vec`, so a caller encoding
/// many chunks into one buffer pays no allocation per chunk.
pub(super) fn merge_and_collect_ids_into(
    piece: &[u8],
    nodes: &mut [Node],
    merge_ranks: RankLookup<'_>,
    id_encoder: &Encoder,
    out: &mut Vec<u32>,
) {
    let live = link_and_merge(piece, nodes, merge_ranks);

    // One id per surviving node, except where a node's surface resolves to no
    // token and is spelled out byte by byte instead — rare enough (it needs a
    // vocabulary that does not cover its own alphabet) that sizing for it would
    // over-allocate every ordinary chunk. Reserving rather than sizing exactly:
    // `out` is shared across chunks and usually already has the room.
    out.reserve(live);
    let mut curr = 0;

    while curr != usize::MAX {
        let node = &nodes[curr];
        let slice = &piece[node.start..node.start + node.len];

        match id_encoder.get(slice) {
            Some(id) => out.push(id),
            // Same contract as `merge_and_collect`'s fallback branch, minus the
            // span bookkeeping: a byte that resolves is emitted, one that does
            // not is dropped.
            None => out.extend(slice.iter().filter_map(|b| id_encoder.get(&[*b][..]))),
        }
        curr = nodes[curr].next;
    }
}
