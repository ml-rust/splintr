use super::*;
use proptest::prelude::*;
use rustc_hash::FxHashMap;

use super::ranks::{BytePairRanks, RankLookup};

/// `byte_pair_encode_ids_seeded_into` collected, so the property below can
/// compare it against the piece-reporting form as a value.
fn ids_seeded(
    piece: &[u8],
    merge_ranks: &FxHashMap<Vec<u8>, u32>,
    id_encoder: &FxHashMap<Vec<u8>, u32>,
    char_granular: bool,
) -> Vec<u32> {
    let mut out = Vec::new();
    byte_pair_encode_ids_seeded_into(
        piece,
        RankLookup::new(merge_ranks),
        id_encoder,
        char_granular,
        &mut out,
    );
    out
}

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
    byte_pair_encode_reference_seeded(piece, merge_ranks, id_encoder, false)
}

/// The oracle with the same seeding switch as
/// [`byte_pair_encode_pieces_seeded`], so character granularity is checked
/// against an independently-written implementation too. Byte granularity
/// stays exactly as pinned.
fn byte_pair_encode_reference_seeded(
    piece: &[u8],
    merge_ranks: &FxHashMap<Vec<u8>, u32>,
    id_encoder: &FxHashMap<Vec<u8>, u32>,
    char_granular: bool,
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

    // Initialize linked list - one node per byte, or per whole UTF-8
    // character when `char_granular` (byte seeding on invalid UTF-8).
    let spans: Vec<(usize, usize)> = match char_granular
        .then(|| std::str::from_utf8(piece).ok())
        .flatten()
    {
        Some(text) => text
            .char_indices()
            .map(|(start, c)| (start, c.len_utf8()))
            .collect(),
        None => (0..piece.len()).map(|start| (start, 1)).collect(),
    };

    let mut nodes: Vec<RefNode> = Vec::with_capacity(spans.len());
    for (i, &(start, len)) in spans.iter().enumerate() {
        nodes.push(RefNode {
            prev: if i == 0 { usize::MAX } else { i - 1 },
            next: if i + 1 == spans.len() {
                usize::MAX
            } else {
                i + 1
            },
            rank: u32::MAX,
            start,
            len,
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

/// The `Token` ids of a piece list, dropping the unresolved spans — the
/// same projection [`byte_pair_encode_with_ranks`] applies, so results can
/// be compared against the oracle's token vector.
fn tokens_only(pieces: Vec<Piece>) -> Vec<u32> {
    pieces
        .into_iter()
        .filter_map(|p| match p {
            Piece::Token(id) => Some(id),
            Piece::Unresolved { .. } => None,
        })
        .collect()
}

/// The same leftmost tiebreak as [`test_tiebreak_leftmost_wins`], over a
/// multi-byte alphabet under character seeding.
///
/// There a node index is a *character* index rather than a byte index, so
/// this pins that ordering too: `"▁▁▁"` must resolve `[▁▁][▁]`, not
/// `[▁][▁▁]`, exactly as `"aaa"` does.
#[test]
fn test_tiebreak_leftmost_wins_multibyte_chars() {
    let mut encoder = FxHashMap::default();
    encoder.insert("▁".as_bytes().to_vec(), 0);
    encoder.insert("▁▁".as_bytes().to_vec(), 1);

    let piece = "▁▁▁".as_bytes();
    assert_eq!(
        tokens_only(byte_pair_encode_pieces_seeded(
            piece,
            RankLookup::new(&encoder),
            &encoder,
            true
        )),
        vec![1, 0]
    );
    assert_eq!(
        byte_pair_encode_reference_seeded(piece, &encoder, &encoder, true),
        vec![1, 0]
    );
}

/// tiktoken-style vocabulary: the id doubles as the merge rank.
fn prop_encoder() -> FxHashMap<Vec<u8>, u32> {
    let mut encoder = FxHashMap::default();
    let tokens: [&[u8]; 18] = [
        b"a", b"b", b"c", b"d", b"aa", b"ab", b"ba", b"bb", b"cd", b"dc", b"cc", b"aaa", b"aab",
        b"abab", b"aaaa", b"bcd", b"abcd", b"abc",
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

/// HuggingFace-style vocabulary over a deliberately mixed-width alphabet:
/// ASCII (1 byte), `▁` (3 bytes), a CJK character (3 bytes) and an emoji
/// (4 bytes) — the widths that byte seeding cannot reassemble. Ranks tie so
/// the leftmost tiebreak is exercised, and `中😀` is deliberately given a
/// merge rank but NO id so the unresolved fallback path is covered too.
fn char_prop_maps() -> (FxHashMap<Vec<u8>, u32>, FxHashMap<Vec<u8>, u32>) {
    let ranked: [(&str, u32); 10] = [
        ("aa", 1),
        ("ab", 1),
        ("▁a", 1),
        ("b▁", 2),
        ("中😀", 2),
        ("😀😀", 2),
        ("aab", 3),
        ("ab▁", 3),
        ("中😀中", 4),
        ("aaaa", 5),
    ];
    let mut merge_ranks = FxHashMap::default();
    for (token, rank) in ranked {
        merge_ranks.insert(token.as_bytes().to_vec(), rank);
    }

    // Ids in an order unrelated to the merge ranks, and missing `中😀`.
    let ids: [&str; 14] = [
        "a",
        "b",
        "▁",
        "中",
        "😀",
        "aaaa",
        "中😀中",
        "ab▁",
        "aab",
        "😀😀",
        "b▁",
        "▁a",
        "ab",
        "aa",
    ];
    let mut id_encoder = FxHashMap::default();
    for (i, token) in ids.iter().enumerate() {
        id_encoder.insert(token.as_bytes().to_vec(), i as u32);
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
/// to exercise `byte_pair_encode_pieces_seeded`'s `Unresolved` reporting.
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
        byte_pair_encode_pieces_seeded(b"abc", RankLookup::new(&encoder), &encoder, false),
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
        byte_pair_encode_pieces_seeded(b"abcd", RankLookup::new(&encoder), &encoder, false),
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
        byte_pair_encode_pieces_seeded(b"bab", RankLookup::new(&encoder), &encoder, false),
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
        let pieces =
            byte_pair_encode_pieces_seeded(piece, RankLookup::new(&encoder), &encoder, false);
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
    assert_eq!(
        byte_pair_encode_pieces_seeded(b"", RankLookup::new(&encoder), &encoder, false),
        empty
    );
    assert_eq!(
        byte_pair_encode_pieces_seeded(b"a", RankLookup::new(&encoder), &encoder, false),
        vec![Piece::Token(0)]
    );
    assert_eq!(
        byte_pair_encode_pieces_seeded(b"b", RankLookup::new(&encoder), &encoder, false),
        vec![Piece::Unresolved { start: 0, len: 1 }]
    );
}

proptest! {
    /// Single-map (tiktoken-style) path over a dense small alphabet, where
    /// merges actually chain.
    #[test]
    fn prop_matches_reference_single_map(
        piece in prop::collection::vec(prop::sample::select(vec![b'a', b'b', b'c', b'd']), 0..200)
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
        piece in prop::collection::vec(any::<u8>(), 0..200)
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
        piece in prop::collection::vec(prop::sample::select(vec![b'a', b'b', b'c', b'd', b'z']), 0..200)
    ) {
        let (merge_ranks, id_encoder) = prop_two_maps();
        prop_assert_eq!(
            byte_pair_encode_with_ranks(&piece, &merge_ranks, &id_encoder),
            byte_pair_encode_reference(&piece, &merge_ranks, &id_encoder)
        );
    }

    /// `byte_pair_encode_pieces_seeded` filtered down to its `Token` ids must
    /// agree with `byte_pair_encode_with_ranks` exactly — the latter is
    /// now defined in terms of the former, but this pins the equivalence
    /// as an independent property over arbitrary bytes (so the
    /// `Unresolved`-reporting fallback path is exercised too).
    #[test]
    fn prop_pieces_tokens_match_with_ranks(
        piece in prop::collection::vec(any::<u8>(), 0..200)
    ) {
        let encoder = prop_encoder();
        let tokens_only: Vec<u32> = byte_pair_encode_pieces_seeded(&piece, RankLookup::new(&encoder), &encoder, false)
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

    /// Character-granular counterpart of the three properties above, over
    /// valid UTF-8 drawn from a mixed-width alphabet (1, 3 and 4 byte
    /// characters), against the character-seeded oracle.
    #[test]
    fn prop_char_seeded_matches_reference(
        chars in prop::collection::vec(
            prop::sample::select(vec!['a', 'b', '▁', '中', '😀']), 0..200)
    ) {
        let (merge_ranks, id_encoder) = char_prop_maps();
        let text: String = chars.into_iter().collect();
        let piece = text.as_bytes();
        prop_assert_eq!(
            tokens_only(byte_pair_encode_pieces_seeded(piece, RankLookup::new(&merge_ranks), &id_encoder, true)),
            byte_pair_encode_reference_seeded(piece, &merge_ranks, &id_encoder, true)
        );
    }

    /// The two-byte index must answer exactly what the map answers.
    ///
    /// It fronts the map for a subset of keys, so it is only safe if it is
    /// indistinguishable from it — including the cases where the two could
    /// plausibly disagree: a two-byte key absent from the map, and a two-byte
    /// key the vocabulary maps to the `u32::MAX` sentinel, which the map path
    /// reports as unmergeable and the table stores as its own "absent" marker.
    #[test]
    fn prop_byte_pair_table_agrees_with_the_map(
        entries in prop::collection::vec(
            (prop::collection::vec(any::<u8>(), 1..5), any::<u32>()), 0..40),
        probes in prop::collection::vec(prop::collection::vec(any::<u8>(), 0..5), 1..40)
    ) {
        let map: FxHashMap<Vec<u8>, u32> = entries.into_iter().collect();
        let pairs = BytePairRanks::build(&map);
        let plain = RankLookup::new(&map);
        let fronted = RankLookup::with_pairs(&map, &pairs);
        for probe in &probes {
            prop_assert_eq!(fronted.get(probe), plain.get(probe), "diverged on {:?}", probe);
        }
    }

    /// The id-only entry point must equal the piece-reporting one filtered
    /// down to its tokens, on every input.
    ///
    /// `byte_pair_encode_ids_seeded` is not a wrapper — it has its own fast
    /// paths and its own collection loop, and it carries essentially all
    /// production traffic (every vocabulary without a byte fallback). The
    /// two implementations have to be checked against each other rather
    /// than assumed equal. Arbitrary bytes, so the unresolved-byte branch
    /// where they differ most is covered.
    #[test]
    fn prop_ids_seeded_matches_pieces_seeded(
        piece in prop::collection::vec(any::<u8>(), 0..200),
        char_granular in any::<bool>()
    ) {
        let (merge_ranks, id_encoder) = prop_two_maps();
        prop_assert_eq!(
            ids_seeded(&piece, &merge_ranks, &id_encoder, char_granular),
            tokens_only(byte_pair_encode_pieces_seeded(&piece, RankLookup::new(&merge_ranks), &id_encoder, char_granular
            ))
        );
    }

    /// Both merge-selection strategies must agree, on inputs that straddle
    /// the threshold that picks between them.
    ///
    /// The other properties compare against the oracle at whatever size
    /// proptest generates; this one forces the comparison to span
    /// `SCAN_SYMBOL_LIMIT` by construction — a piece is generated near the
    /// boundary and checked at lengths on both sides of it, so a divergence
    /// that only appears once the heap takes over cannot hide behind a
    /// generator that happens to favour short inputs.
    #[test]
    fn prop_selection_strategies_agree_across_the_threshold(
        piece in prop::collection::vec(
            prop::sample::select(vec![b'a', b'b', b'c', b'd']), 130..260)
    ) {
        let encoder = prop_encoder();
        for len in [32usize, 63, 64, 65, 129, piece.len()] {
            let slice = &piece[..len.min(piece.len())];
            prop_assert_eq!(
                byte_pair_encode(slice, &encoder),
                byte_pair_encode_reference(slice, &encoder, &encoder),
                "diverged at {} symbols", slice.len()
            );
        }
    }
}
