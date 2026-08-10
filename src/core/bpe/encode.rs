use crate::core::encoder::Encoder;

use super::merge::{
    merge_and_collect, merge_and_collect_ids_into, prefers_scan, QueueScratch, SCAN_SYMBOL_LIMIT,
};
use super::nodes::Node;
use super::ranks::RankLookup;
use super::scratch::with_merge_scratch;

/// Perform byte-pair encoding on a piece of text using a linked-list approach.
///
/// This is the core BPE algorithm that:
/// 1. Initializes a linked list with one node per byte
/// 2. Ranks every adjacent pair the vocabulary can merge
/// 3. Takes the pair with the lowest rank (leftmost on a tie)
/// 4. Merges it by updating linked list pointers
/// 5. Re-ranks the pairs the merge created
/// 6. Continues until no more merges are possible
///
/// The linked list makes each splice O(1); how the next merge is *selected*
/// depends on the piece length — see the module docs for the two strategies and
/// the measurement behind the threshold between them.
pub fn byte_pair_encode(piece: &[u8], encoder: &Encoder) -> Vec<u32> {
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
    merge_ranks: &Encoder,
    id_encoder: &Encoder,
) -> Vec<u32> {
    // Byte granularity: this entry point is the tiktoken-shaped one, whose
    // merges operate on bytes. Going through `byte_pair_encode_pieces_seeded`
    // and filtering to its tokens gives the same answer — `prop_ids_seeded_
    // matches_pieces_seeded` pins that — but reaches it through a `Vec<Piece>`
    // and a `filter_map` whose size hint is not exact, so the output vector
    // grows by doubling.
    let mut out = Vec::new();
    byte_pair_encode_ids_seeded_into(
        piece,
        RankLookup::new(merge_ranks),
        id_encoder,
        false,
        &mut out,
    );
    out
}
/// One output of the merge phase: either a resolved token, or a run of bytes
/// the vocabulary could not represent at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Piece {
    Token(u32),
    /// Byte range `[start, start + len)` within the input piece that resolved
    /// to no token — neither as a whole nor byte-by-byte.
    Unresolved {
        start: usize,
        len: usize,
    },
}

/// Byte-pair encoding that reports byte spans the vocabulary could not
/// resolve, instead of silently dropping them, with the merge granularity made
/// explicit.
///
/// `char_granular` selects what a merge starts from:
///
/// - `false` — one node per **byte**, which is what tiktoken-style
///   vocabularies merge over. Their merges genuinely operate on bytes, and
///   they contain tokens that are not valid UTF-8 at all (cl100k_base alone
///   has 773), so nothing else can reproduce them.
/// - `true` — one node per whole **UTF-8 character**, which is what
///   HuggingFace-style `merges` operate over. Seeding those by byte strands
///   every character of 3 bytes or more: reassembling `▁` (U+2581 = E2 96 81)
///   would need a rank for the partial prefix `E2 96`, which is never a
///   vocabulary entry, so the chain stalls and the character shatters into
///   `<0xNN>` byte fallbacks. (2-byte characters survive byte seeding because
///   the concatenation of their two bytes *is* the whole character, which does
///   have a base rank — which is also why ByteLevel vocabularies, whose
///   alphabet is entirely ≤2 UTF-8 bytes, are safe either way.)
///
/// Only the seeding differs; everything downstream is granularity-agnostic.
/// In particular the unresolved-run traversal still walks a node's slice one
/// byte at a time, which is exactly HuggingFace's ByteFallback behavior.
pub(crate) fn byte_pair_encode_pieces_seeded(
    piece: &[u8],
    merge_ranks: RankLookup<'_>,
    id_encoder: &Encoder,
    char_granular: bool,
) -> Vec<Piece> {
    if piece.is_empty() {
        return vec![];
    }

    // Fast path: single byte
    if piece.len() == 1 {
        return match id_encoder.get(piece) {
            Some(r) => vec![Piece::Token(r)],
            None => vec![Piece::Unresolved { start: 0, len: 1 }],
        };
    }

    // Fast path: entire piece is a single token
    if let Some(id) = id_encoder.get(piece) {
        return vec![Piece::Token(id)];
    }

    with_merge_scratch(|s| {
        seed_nodes_reusing(
            piece,
            char_granular,
            symbol_count(piece, char_granular),
            &mut s.nodes,
        );
        merge_and_collect(
            piece,
            &mut s.nodes,
            // `Piece`-reporting callers keep the byte-keyed path: they exist for
            // byte fallback, whose symbols the id seeding does not cover.
            merge_ranks.without_ids(),
            id_encoder,
            None,
            &mut s.queue,
        )
    })
}

/// [`byte_pair_encode_pieces_seeded`] with the [`Piece`] layer removed: token
/// ids directly, dropping what the vocabulary cannot represent.
///
/// Same algorithm, same fast paths, same output as filtering
/// [`byte_pair_encode_pieces_seeded`] down to its `Piece::Token`s — which is
/// exactly what every caller without a byte fallback does, and that is every
/// ByteLevel and every tiktoken-style vocabulary. Going through `Vec<Piece>`
/// for them costs an allocation sized by input length plus a second pass, per
/// chunk, on the hottest path in the crate.
/// Byte-pair encoding straight to token ids, appended to a caller-owned buffer.
///
/// The form the encode path actually uses: a text is many chunks, and returning
/// a `Vec` per chunk means an allocation and a copy per chunk for a result that
/// is immediately concatenated with its neighbours.
///
/// The node list is stack-resident for pieces at or below
/// [`SCAN_SYMBOL_LIMIT`](super::merge::SCAN_SYMBOL_LIMIT) symbols, which is the
/// same bound the merge strategy switches on. Above it the buffers come from
/// [`with_merge_scratch`], so the first long piece on a thread pays the
/// allocation and the rest reuse it.
pub(crate) fn byte_pair_encode_ids_seeded_into(
    piece: &[u8],
    merge_ranks: RankLookup<'_>,
    id_encoder: &Encoder,
    char_granular: bool,
    out: &mut Vec<u32>,
) {
    if piece.is_empty() {
        return;
    }

    // Fast path: single byte. An unresolvable one is dropped, which is what the
    // `Piece::Unresolved` this would otherwise produce amounts to here.
    if piece.len() == 1 {
        out.extend(id_encoder.get(piece));
        return;
    }

    // Fast path: entire piece is a single token
    if let Some(id) = id_encoder.get(piece) {
        out.push(id);
        return;
    }

    // Merging by id needs an id for every symbol the merge starts from, which
    // may mean seeding at a different granularity: a ByteLevel vocabulary's
    // alphabet is its characters, so the individual bytes of a two-byte one are
    // not tokens and have no id. The two seedings agree there — the alphabet
    // takes the lowest merge ranks, so byte seeding reassembles exactly those
    // characters before any other merge runs — which is what makes the choice
    // free to make.
    let granular = char_granular || merge_ranks.by_id().is_some_and(|t| t.seeds_by_char());
    let symbols = symbol_count(piece, granular);
    if prefers_scan(piece, symbols) {
        let mut buf = [Node::PLACEHOLDER; SCAN_SYMBOL_LIMIT];
        let nodes = &mut buf[..symbols];
        seed_nodes_into(piece, granular, nodes);
        if !seed_ids(nodes, piece, merge_ranks) {
            return byte_pair_encode_ids_seeded_into(
                piece,
                merge_ranks.without_ids(),
                id_encoder,
                char_granular,
                out,
            );
        }
        // The scan never looks at the queue, and an empty `QueueScratch` holds
        // empty buffers, so this allocates nothing — and this branch is chosen
        // by the same predicate the strategy is, so the scan is what runs.
        merge_and_collect_ids_into(
            piece,
            nodes,
            merge_ranks,
            id_encoder,
            out,
            &mut QueueScratch::default(),
        );
    } else {
        let seeded = with_merge_scratch(|s| {
            seed_nodes_reusing(piece, granular, symbols, &mut s.nodes);
            if !seed_ids(&mut s.nodes, piece, merge_ranks) {
                return false;
            }
            merge_and_collect_ids_into(
                piece,
                &mut s.nodes,
                merge_ranks,
                id_encoder,
                out,
                &mut s.queue,
            );
            true
        });
        if !seeded {
            byte_pair_encode_ids_seeded_into(
                piece,
                merge_ranks.without_ids(),
                id_encoder,
                char_granular,
                out,
            );
        }
    }
}

/// Resolve every seed symbol to a token id, reporting whether all of them did.
///
/// A `false` sends the piece back through the byte-keyed path at the
/// granularity that path has always used — nothing has been emitted yet, so the
/// retry is a clean start. Giving up per *piece* rather than per vocabulary is
/// what makes the id path safe to offer at all: the table is a property of the
/// vocabulary, but whether a piece's symbols are in it is a property of the
/// piece.
#[inline]
fn seed_ids(nodes: &mut [Node], piece: &[u8], merge_ranks: RankLookup<'_>) -> bool {
    let Some(table) = merge_ranks.by_id() else {
        return true;
    };
    for node in nodes.iter_mut() {
        node.id = table.seed_id(&piece[node.start..node.start + node.len]);
        if node.id == u32::MAX {
            return false;
        }
    }
    true
}

/// How many symbols [`seed_nodes_into`] will produce for `piece`.
///
/// Byte seeding is one per byte; character seeding is one per UTF-8 character,
/// falling back to bytes on input that is not valid UTF-8 — the same fallback
/// [`seed_nodes_into`] applies, so the two always agree.
#[inline]
fn symbol_count(piece: &[u8], char_granular: bool) -> usize {
    match char_granular
        .then(|| std::str::from_utf8(piece).ok())
        .flatten()
    {
        Some(text) => text.chars().count(),
        None => piece.len(),
    }
}

/// Seed the linked list: one node per byte, or one node per whole UTF-8
/// character when `char_granular`.
///
/// Invalid UTF-8 has no character boundaries to walk, so it falls back to byte
/// seeding — `piece` comes from a `&str` in practice, but this function takes
/// bytes and must not panic on ones that are not. `start`/`len` are collected
/// directly into `nodes` (capacity `piece.len()`, an exact bound for byte
/// seeding and an upper bound for char seeding, since a UTF-8 char is never
/// longer than its own byte count) so no separate spans buffer is allocated;
/// `prev`/`next` only depend on final list length, so they are filled in by the
/// merge phase once that length is known.
/// Seed `nodes` into a buffer the caller is reusing, so a long piece costs no
/// allocation once that buffer has grown to fit one.
fn seed_nodes_reusing(piece: &[u8], char_granular: bool, symbols: usize, nodes: &mut Vec<Node>) {
    nodes.clear();
    nodes.resize(symbols, Node::PLACEHOLDER);
    seed_nodes_into(piece, char_granular, nodes);
}

/// Seed nodes into storage the caller already has, which may be a stack
/// array. `nodes` must be exactly [`symbol_count`] long.
fn seed_nodes_into(piece: &[u8], char_granular: bool, nodes: &mut [Node]) {
    match char_granular
        .then(|| std::str::from_utf8(piece).ok())
        .flatten()
    {
        Some(text) => {
            for (node, (start, c)) in nodes.iter_mut().zip(text.char_indices()) {
                node.start = start;
                node.len = c.len_utf8();
            }
        }
        None => {
            for (start, node) in nodes.iter_mut().enumerate() {
                node.start = start;
                node.len = 1;
            }
        }
    }
}

/// One symbol the merge starts from, when the caller has already decided the
/// segmentation — see [`byte_pair_encode_pieces_presegmented`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Seed {
    /// Start of this symbol's surface within the buffer the seeds describe.
    pub(crate) start: usize,
    /// Length of that surface, in bytes.
    pub(crate) len: usize,
    /// The id this symbol already stands for, used verbatim when no merge
    /// absorbed it. `None` defers to the ordinary `id_encoder` lookup.
    pub(crate) id: Option<u32>,
}

/// [`byte_pair_encode_pieces_seeded`] over a segmentation the caller supplies,
/// rather than one derived from `piece` by byte or by character.
///
/// This exists for HuggingFace's byte-fallback order: `tokenizers`' BPE model
/// resolves an unrepresentable character to its `<0xNN>`/`<unk>` tokens
/// *before* merging, so those tokens are ordinary word symbols that the merge
/// list may combine with their neighbours. Reproducing that means merging over
/// a buffer whose unrepresentable characters have already been replaced by
/// those tokens' vocabulary spellings, split at the spellings' own boundaries —
/// which is exactly a caller-supplied segmentation (see
/// `Tokenizer::bpe_fallback_first`).
///
/// Deliberately without [`byte_pair_encode_pieces_seeded`]'s whole-piece fast
/// path: that shortcut answers "is this entire chunk one token?", a question
/// about the *input*, and the buffer here is a rewritten one whose
/// concatenation is not input text.
pub(crate) fn byte_pair_encode_pieces_presegmented(
    piece: &[u8],
    seeds: &[Seed],
    merge_ranks: RankLookup<'_>,
    id_encoder: &Encoder,
) -> Vec<Piece> {
    if seeds.is_empty() {
        return vec![];
    }
    with_merge_scratch(|s| {
        s.nodes.extend(seeds.iter().map(|seed| Node {
            prev: 0,
            next: 0,
            start: seed.start,
            len: seed.len,
            // The presegmented path is byte-keyed: this seeding exists to
            // reproduce HuggingFace's byte-fallback order, whose symbols are
            // vocabulary *spellings* the id path has no seed table for.
            id: u32::MAX,
        }));
        merge_and_collect(
            piece,
            &mut s.nodes,
            merge_ranks.without_ids(),
            id_encoder,
            Some(seeds),
            &mut s.queue,
        )
    })
}
