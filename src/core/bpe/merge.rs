use crate::core::encoder::Encoder;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::encode::{Piece, Seed};
use super::nodes::Node;
use super::ranks::RankLookup;

/// Symbol count at or below which merges are selected by scanning a
/// stack-resident rank table instead of a queue.
///
/// The two strategies are asymptotically opposite and there is no single right
/// answer, so the threshold is measured rather than guessed. Selection cost per
/// merge is O(symbols) for the scan and O(log symbols) for the heap. Measured on
/// llama-3 over novel words, the scan wins up to roughly 100 symbols and the
/// heap wins above it, diverging fast in both directions (at 8000 symbols the
/// heap is ~46x ahead; at 16 symbols the scan is ~25% ahead).
///
/// That measurement predates [`super::scratch`], which amortizes the heap's
/// allocation per thread rather than per piece, so the crossover is lower than
/// it was. Re-measured since: for ASCII it is still above 64, which is why
/// [`GATE_ASCII`] does not lower it; for multi-byte scripts it is far below,
/// which is what [`GATE_MULTI`] exists to express.
///
/// 64 sits deliberately on the safe side of that crossover. It also covers the
/// entire realistic range of the dominant workload: a pre-tokenizer emits
/// chunks of a handful of symbols, so ordinary text never reaches the heap at
/// all, while a `tokenizer.json` with no splitting stage (`pre_tokenizer:
/// null`, which is how Mistral's AWQ and GPTQ files are shaped) hands over the
/// whole document and needs the heap's asymptote to stay usable.
///
/// The table is `[u32; 64]` — 256 bytes, four cache lines, no allocation.
///
/// This is the scan's *capacity*, not the whole of the choice: which strategy a
/// piece actually gets also depends on its script, via [`prefers_scan`].
pub(super) const SCAN_SYMBOL_LIMIT: usize = 64;

/// Longest piece, in bytes, still handed to the scan when its content starts
/// with an ASCII byte.
///
/// Equal to [`SCAN_SYMBOL_LIMIT`], i.e. no gate at all: an ASCII piece has one
/// symbol per byte, so the symbol limit already binds first. Lowering it was
/// measured and does not pay — English is a hair worse at 24, because the scan
/// is genuinely the better strategy at these lengths.
const GATE_ASCII: u8 = 64;

/// The same for a piece whose content starts outside ASCII, and for a piece that
/// is only whitespace.
///
/// Lower than [`GATE_ASCII`] because the two are not counting the same thing. A
/// multi-byte character is several symbols, so a non-Latin piece reaches any
/// symbol count in a fraction of the characters an English word needs, and the
/// scan is quadratic in symbols.
///
/// Swept against llama-3 over the per-script corpora. The answer depends on how
/// expensive the queue is, and moved once [`merge_by_queue`] got cheaper: with
/// the old heap, seeded with every adjacent pair, the curve did not flatten
/// until 32 and 8 cost Arabic ~12%; with the cold/hot split it is flat from 8 to
/// 16 and rises from 24. 16 is the middle of that flat region rather than its
/// edge, so the constant is not sitting on a cliff.
const GATE_MULTI: u8 = 16;

/// Longest scannable piece per leading content byte.
const fn build_byte_gate() -> [u8; 256] {
    let mut gate = [GATE_MULTI; 256];
    let mut b = 0;
    while b < 0x80 {
        gate[b] = GATE_ASCII;
        b += 1;
    }
    // Reached only by a piece that is *nothing but* whitespace, since a piece
    // with content is classified past its delimiter. Such a piece merges no
    // further than its own run, so it belongs on the cheap gate.
    gate[b' ' as usize] = GATE_MULTI;
    gate[b'\t' as usize] = GATE_MULTI;
    gate[b'\n' as usize] = GATE_MULTI;
    gate[b'\r' as usize] = GATE_MULTI;
    gate
}

static BYTE_GATE: [u8; 256] = build_byte_gate();

/// Where a piece's content begins, past whatever word delimiter the
/// pre-tokenizer put in front of it.
///
/// Classifying byte 0 would classify the delimiter — every ByteLevel piece
/// starts `Ġ` and every Metaspace piece starts `▁`, so every piece in a corpus
/// would look alike and the gate would say nothing about the script.
#[inline]
fn content_start(piece: &[u8]) -> usize {
    match piece {
        // Metaspace `▁` (U+2581).
        [0xE2, 0x96, 0x81, rest @ ..] if !rest.is_empty() => 3,
        // ByteLevel `Ġ` (U+0120), the byte-level spelling of a leading space.
        [0xC4, 0xA0, rest @ ..] if !rest.is_empty() => 2,
        // A literal leading space, which ByteLevel also hands over unmapped.
        [ws, rest @ ..] if ws.is_ascii_whitespace() && !rest.is_empty() => 1,
        _ => 0,
    }
}

/// Whether `piece`, holding `symbols` symbols, should be merged by scan rather
/// than by heap.
///
/// The single source of the decision: the caller picks the node buffer from it
/// and [`link_and_merge`] picks the strategy from it, so the two cannot
/// disagree about which one a piece is getting.
#[inline]
pub(super) fn prefers_scan(piece: &[u8], symbols: usize) -> bool {
    if symbols > SCAN_SYMBOL_LIMIT {
        return false;
    }
    // No gate is lower than the smallest one, so a piece that short is scanned
    // whatever its script — and classifying it would be pure overhead on the
    // shortest, most common pieces there are.
    if piece.len() <= GATE_MULTI as usize {
        return true;
    }
    piece.len() <= BYTE_GATE[piece[content_start(piece)] as usize] as usize
}

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

/// A queued merge candidate: its rank in the high half, the index of its left
/// node in the low half.
///
/// Packed so that comparing two keys as integers *is* the ordering BPE
/// prescribes — lowest rank first, and on a tie the lowest left index, which is
/// the leftmost pair (node indices are assigned in list order and never
/// reassigned). The tie-break is load-bearing for bit-exactness with tiktoken;
/// `test_tiebreak_leftmost_wins` locks it in.
#[inline]
fn key(rank: u32, left: usize) -> u64 {
    ((rank as u64) << 32) | (left as u64 & 0xFFFF_FFFF)
}

#[inline]
fn key_rank(key: u64) -> u32 {
    (key >> 32) as u32
}

#[inline]
fn key_left(key: u64) -> usize {
    (key & 0xFFFF_FFFF) as usize
}

/// The buffers [`merge_by_queue`] works in, reused across pieces.
#[derive(Default)]
pub(super) struct QueueScratch {
    /// Current merge rank of the pair starting at each node, `u32::MAX` when
    /// that pair cannot merge or the node has been absorbed.
    ///
    /// This is what makes a stale key recognizable without re-deriving it: a
    /// key is live exactly when `ranks[left]` still equals the rank it was
    /// pushed with.
    pub(super) ranks: Vec<u32>,
    /// The pairs the piece starts with. All of them are known before the first
    /// merge, so they are sorted once and read front to back — no heap.
    pub(super) cold: Vec<u64>,
    /// The pairs merges create. Only these need a live priority queue, and
    /// there are at most two per merge.
    pub(super) hot: BinaryHeap<Reverse<u64>>,
}

impl QueueScratch {
    /// Empty every buffer, keeping the capacity that makes reuse worthwhile.
    pub(super) fn clear(&mut self) {
        self.ranks.clear();
        self.cold.clear();
        self.hot.clear();
    }

    /// Whether every buffer is empty. Only the scratch's own tests ask.
    #[cfg(test)]
    pub(super) fn is_empty(&self) -> bool {
        self.ranks.is_empty() && self.cold.is_empty() && self.hot.is_empty()
    }
}

/// Merge selection for long pieces, split across two queues.
///
/// The insight is that the initial pairs and the created pairs have completely
/// different access patterns. Every initial pair is known before the first
/// merge, so sorting them once and walking the result is strictly cheaper than
/// heapifying them and paying a sift-down per pop. Only the pairs a merge
/// *creates* arrive unpredictably, and there are at most two per merge, so the
/// live heap stays small. Seeding one heap with everything, as this used to,
/// pays heap prices for the majority of candidates that never needed them.
///
/// Superseded entries are left where they are and skipped when they surface,
/// so no entry is ever found and removed.
fn merge_by_queue(
    piece: &[u8],
    nodes: &mut [Node],
    merge_ranks: RankLookup<'_>,
    q: &mut QueueScratch,
) -> usize {
    let count = nodes.len();
    q.ranks.clear();
    q.ranks.resize(count, u32::MAX);
    q.cold.clear();
    q.hot.clear();

    // `saturating_sub` because the callers guarantee a non-empty list through
    // their own guards, and this may not become an underflow if a third ever
    // appears.
    for i in 0..count.saturating_sub(1) {
        let rank = rank_of(piece, nodes, i, i + 1, merge_ranks);
        q.ranks[i] = rank;
        if rank != u32::MAX {
            q.cold.push(key(rank, i));
        }
    }
    q.cold.sort_unstable();

    let mut cursor = 0usize;
    let mut live = count;
    loop {
        // Retire stale entries from the front of each queue, so the two
        // candidates compared below are both live.
        while let Some(&k) = q.cold.get(cursor) {
            if q.ranks[key_left(k)] == key_rank(k) {
                break;
            }
            cursor += 1;
        }
        while let Some(&Reverse(k)) = q.hot.peek() {
            if q.ranks[key_left(k)] == key_rank(k) {
                break;
            }
            q.hot.pop();
        }

        let next = match (q.cold.get(cursor).copied(), q.hot.peek().map(|r| r.0)) {
            (None, None) => return live,
            (Some(cold), None) => {
                cursor += 1;
                cold
            }
            (None, Some(hot)) => {
                q.hot.pop();
                hot
            }
            (Some(cold), Some(hot)) => {
                if cold < hot {
                    cursor += 1;
                    cold
                } else {
                    q.hot.pop();
                    hot
                }
            }
        };

        // A live key's rank is the current rank of `(left, left.next)`, and a
        // rankable pair has a right node, so this cannot be the tail.
        let left = key_left(next);
        let right = nodes[left].next;
        let new_next = absorb(nodes, left, right);
        live -= 1;
        q.ranks[right] = u32::MAX;

        // The merge changed at most two adjacencies, around the merged node.
        // Recording their new ranks is what invalidates their old keys.
        let prev = nodes[left].prev;
        if prev != usize::MAX {
            let rank = rank_of(piece, nodes, prev, left, merge_ranks);
            q.ranks[prev] = rank;
            if rank != u32::MAX {
                q.hot.push(Reverse(key(rank, prev)));
            }
        }
        let rank = rank_of(piece, nodes, left, new_next, merge_ranks);
        q.ranks[left] = rank;
        if rank != u32::MAX {
            q.hot.push(Reverse(key(rank, left)));
        }
    }
}

/// Link `nodes` into a list and run the merge loop over them, returning how
/// many nodes are still live.
///
/// `nodes` arrives with `start`/`len` set and `prev`/`next` ignored. `queue` is
/// touched only on the heap branch; the scan branch never looks at it, so a
/// caller that knows it is below the threshold may pass an empty one.
fn link_and_merge(
    piece: &[u8],
    nodes: &mut [Node],
    merge_ranks: RankLookup<'_>,
    queue: &mut QueueScratch,
) -> usize {
    let count = nodes.len();
    for (i, node) in nodes.iter_mut().enumerate() {
        node.prev = if i == 0 { usize::MAX } else { i - 1 };
        node.next = if i + 1 == count { usize::MAX } else { i + 1 };
    }

    if prefers_scan(piece, count) {
        merge_by_scan(piece, nodes, merge_ranks)
    } else {
        merge_by_queue(piece, nodes, merge_ranks, queue)
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
    nodes: &mut [Node],
    merge_ranks: RankLookup<'_>,
    id_encoder: &Encoder,
    seeds: Option<&[Seed]>,
    queue: &mut QueueScratch,
) -> Vec<Piece> {
    link_and_merge(piece, nodes, merge_ranks, queue);

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
    queue: &mut QueueScratch,
) {
    let live = link_and_merge(piece, nodes, merge_ranks, queue);

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
