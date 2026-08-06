use rustc_hash::FxHashMap;

/// Build a bytes → merge-rank map (lower rank = merged first) from a model's
/// ordered merge list and the vocabulary it was built over.
///
/// Merge priority is independent of token id (RoBERTa orders its merges
/// differently from GPT-2, and GGUF vocabularies disagree with their own id
/// order), so the ranks come from the list, not from the ids. The map covers two
/// groups, ranked so the first always wins:
///
/// 1. **Base alphabet** — vocabulary entries that are never a merge *result*
///    (the byte-level single chars). They take the lowest ranks `0..b` so that
///    wherever a base entry is reachable as a merge of two adjacent pieces it
///    forms before any real merge runs. Under byte seeding that only rescues
///    2-byte characters, whose two bytes concatenate to the whole character;
///    a ≥3-byte character has no rank for its partial prefix and can never
///    coalesce from bytes at all, which is why HuggingFace-style vocabularies
///    seed by character instead (`char_granular` in
///    [`byte_pair_encode_pieces_seeded`]).
/// 2. **Merges** — each merged token (`a ++ b`) at rank `b + merge_index`.
///
/// `merged` holds the already-concatenated result of each merge, in list order.
/// `vocab_in_id_order` yields every vocabulary token, lowest id first, so the
/// base ranks are deterministic.
pub(crate) fn merge_ranks<'a>(
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
