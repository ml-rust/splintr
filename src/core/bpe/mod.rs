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
//! node into the left one and rewires two pointers. Nothing moves.
//!
//! # Selecting the next merge
//!
//! The splice is only half the problem — *selecting* which pair to merge next
//! has to be cheap too. There is no single best way to do it, because the two
//! obvious ways are good in opposite regimes, and this crate is subject to
//! both:
//!
//! - **Scan** ([`merge::merge_by_scan`]) keeps a rank per node in a
//!   stack-resident table and rescans it for the minimum on every merge. That
//!   is O(N) per merge and quadratic overall, but it allocates nothing and the
//!   table is contiguous, so at small N it beats anything with a heap in it.
//! - **Queue** ([`merge::merge_by_queue`]) splits candidates by where they came
//!   from. The pairs the piece *starts with* are all known before the first
//!   merge, so they are sorted once and read front to back. Only the pairs a
//!   merge *creates* — at most two per merge — go in a live binary heap. Both
//!   tiers use lazy deletion: a superseded entry is left where it is and skipped
//!   when it surfaces, so no entry is ever found and removed.
//!
//! Which regime applies is decided by the *pre-tokenizer* and by the *script*,
//! not by the input size. A tokenizer that splits (nearly all of them) hands the
//! merge loop one short word at a time; one that does not split at all —
//! `pre_tokenizer: null`, which is how Mistral's AWQ and GPTQ `tokenizer.json`
//! files are shaped — hands over the entire document and needs the queue's
//! asymptote. Script matters because symbols are counted in ByteLevel space,
//! where one CJK character is several, so a non-Latin piece leaves the scan's
//! regime in a handful of characters. The choice is made per piece by
//! [`merge::prefers_scan`], from [`merge::SCAN_SYMBOL_LIMIT`] and a per-script
//! byte gate; those constants carry the measurements behind the crossover.
//!
//! Both strategies resolve equal ranks LEFTMOST and treat `u32::MAX` as
//! unmergeable, so they are bit-exact with each other and with tiktoken. The
//! property tests check them against a slow, obviously-correct reference on
//! generated inputs that span the threshold in both directions.
//!
//! # Complexity Analysis
//!
//! - **Time**: O(N log N) for a piece above the threshold; O(N²) below it, on
//!   an N bounded by a small constant, which is why it is the faster half
//! - **Space**: O(N) for the node list, plus O(N) for the queue above the
//!   threshold and nothing below it. Both come from a per-thread scratch
//!   ([`scratch`]), so they are allocated once per thread rather than per piece
//!
//! # Algorithm Steps
//!
//! 1. Initialize linked list with one node per byte (or per character)
//! 2. Rank every adjacent pair the vocabulary can merge
//! 3. Take the best candidate: lowest rank, leftmost position on a tie
//! 4. Merge it by updating pointers (O(1)), tombstoning the absorbed node
//! 5. Re-rank the two pairs the merge created, around the merged node
//! 6. Repeat until no rankable pair is left

mod encode;
mod merge;
mod nodes;
mod ranks;
mod scratch;
#[cfg(test)]
mod tests;

pub use encode::byte_pair_encode;
pub(crate) use encode::{
    byte_pair_encode_pieces_presegmented, byte_pair_encode_pieces_seeded, byte_pair_merge_ids_into,
    Piece, Seed, Seeding,
};
pub(crate) use ranks::{merge_ranks, BytePairRanks, PairRanks, RankLookup};

// Reachable from `tests` (via its `use super::*`) and from the intra-doc links
// in the algorithm modules, without widening anything past this module.
// `byte_pair_encode_with_ranks` is the ranked entry point: public surface of the
// module, but the crate's own encode path calls the seeded form directly, so only
// the tests bind it here.
#[allow(unused_imports)]
use encode::byte_pair_encode_with_ranks;
#[allow(unused_imports)]
use nodes::Node;
