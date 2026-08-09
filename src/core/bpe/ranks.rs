use crate::core::token_bytes::{Encoder, TokenBytes};
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
    merged: Vec<String>,
    vocab_in_id_order: impl Iterator<Item = &'a str>,
) -> Encoder {
    // Both sized up front and both on FxHash: this runs over a whole
    // vocabulary, where the default hasher and a table that doubles from empty
    // are each worth more than the work they serve.
    let mut merge_set: rustc_hash::FxHashSet<&str> =
        rustc_hash::FxHashSet::with_capacity_and_hasher(merged.len(), rustc_hash::FxBuildHasher);
    merge_set.extend(merged.iter().map(String::as_str));

    // The base alphabet adds a few hundred entries on top of the merges.
    let mut ranks: Encoder =
        Encoder::with_capacity_and_hasher(merged.len() + 512, rustc_hash::FxBuildHasher);

    // Base alphabet first, in id order for determinism.
    for token in vocab_in_id_order.filter(|t| !merge_set.contains(t)) {
        let next = ranks.len() as u32;
        ranks
            .entry(TokenBytes::from(token.as_bytes().to_vec()))
            .or_insert(next);
    }

    // Then the merges, preserving list priority. The set borrowed from
    // `merged`; dropping it first lets each token be *moved* into its key
    // rather than copied, which is one allocation saved per merge over a list
    // as long as the vocabulary.
    drop(merge_set);
    let base_count = ranks.len() as u32;
    for (i, token) in merged.into_iter().enumerate() {
        ranks
            .entry(TokenBytes::Owned(token.into_bytes().into_boxed_slice()))
            .or_insert(base_count + i as u32);
    }
    ranks
}

/// Merge ranks of every two-byte sequence, indexed directly by the two bytes.
///
/// Every merge the loop performs starts from a pair of adjacent seed symbols,
/// and when seeding is byte-granular — every tiktoken vocabulary and every
/// ByteLevel one, i.e. almost all traffic — that pair is exactly two bytes. So
/// the seeding pass probes `N-1` two-byte keys per piece, and further pairs of
/// single-byte symbols keep appearing as merging proceeds; measured on real
/// text they are roughly a third of all rank lookups.
///
/// A hash lookup for those is pure overhead: two bytes index a table directly.
/// The table is `256 * 256` ranks — 256 KB — but only the entries text actually
/// produces are ever touched, so the working set follows the text's own
/// statistics rather than the table's size.
///
/// Measured against the map it fronts: ~30% faster over a probe mix with the
/// two-byte share real text produces.
pub(crate) struct BytePairRanks {
    /// `u32::MAX` marks a pair the vocabulary cannot merge — the same sentinel
    /// [`RankLookup::get`] returns for an absent key, and the same one the merge
    /// loop already treats as unmergeable.
    ranks: Box<[u32]>,
}

impl BytePairRanks {
    /// Index every two-byte entry of `map`.
    pub(crate) fn build(map: &Encoder) -> Self {
        let mut ranks = vec![u32::MAX; 256 * 256];
        for (key, &rank) in map {
            if let [hi, lo] = key[..] {
                ranks[(hi as usize) << 8 | lo as usize] = rank;
            }
        }
        Self {
            ranks: ranks.into_boxed_slice(),
        }
    }

    #[inline]
    fn get(&self, hi: u8, lo: u8) -> u32 {
        self.ranks[(hi as usize) << 8 | lo as usize]
    }
}

/// Where the merge loop reads ranks from: the authoritative map, optionally
/// fronted by [`BytePairRanks`] for the two-byte keys.
///
/// Borrowed rather than owned so building one costs nothing — the merge loop
/// constructs it per piece, and the public entry points that have no table
/// construct it with `pairs: None` and behave exactly as before.
#[derive(Clone, Copy)]
pub(crate) struct RankLookup<'a> {
    map: &'a Encoder,
    pairs: Option<&'a BytePairRanks>,
}

impl<'a> RankLookup<'a> {
    /// A lookup that only consults the map.
    pub(crate) fn new(map: &'a Encoder) -> Self {
        Self { map, pairs: None }
    }

    /// A lookup fronted by a two-byte table.
    pub(crate) fn with_pairs(map: &'a Encoder, pairs: &'a BytePairRanks) -> Self {
        Self {
            map,
            pairs: Some(pairs),
        }
    }

    /// The merge rank of `key`, or `u32::MAX` when the vocabulary cannot merge
    /// it.
    ///
    /// `u32::MAX` doubles as the "unmergeable" sentinel, so a vocabulary that
    /// maps a token to it is treated as unmergeable — which is what the map-only
    /// path has always done, and what the table reproduces by storing the same
    /// value for an absent pair.
    #[inline]
    pub(crate) fn get(&self, key: &[u8]) -> u32 {
        if let (Some(pairs), [hi, lo]) = (self.pairs, key) {
            return pairs.get(*hi, *lo);
        }
        self.map.get(key).copied().unwrap_or(u32::MAX)
    }
}
