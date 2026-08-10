use crate::core::encoder::Encoder;

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
    let mut ranks: Encoder = Encoder::with_capacity(merged.len() + 512);

    // Base alphabet first, in id order for determinism.
    for token in vocab_in_id_order.filter(|t| !merge_set.contains(t)) {
        let next = ranks.len() as u32;
        ranks.insert_if_absent(token.as_bytes(), next);
    }

    // Then the merges, preserving list priority. The set borrows from
    // `merged`, so drop it first — `merged.into_iter()` below consumes the
    // vec it points into.
    drop(merge_set);
    let base_count = ranks.len() as u32;
    for (i, token) in merged.into_iter().enumerate() {
        ranks.insert_if_absent(token.as_bytes(), base_count + i as u32);
    }
    ranks
}

/// Shortest key the [`ShortRanks`] table indexes; below it the direct two-byte
/// table already answers without probing.
const SHORT_MIN: usize = 3;
/// Longest key it indexes — the point where a key stops fitting in a `u32`.
const SHORT_MAX: usize = 4;

/// Highest rank a packed slot can carry. Excluding the top value keeps
/// [`ShortRanks::EMPTY`] distinct from every real entry.
const SHORT_MAX_RANK: u32 = u32::MAX >> 1;

/// Merge ranks of every short key: two bytes indexed directly, three and four
/// through [`ShortRanks`].
///
/// Every merge the loop performs starts from a pair of adjacent seed symbols,
/// and when seeding is byte-granular — every tiktoken vocabulary and every
/// ByteLevel one, i.e. almost all traffic — that pair is exactly two bytes. So
/// the seeding pass probes `N-1` two-byte keys per piece, and further pairs of
/// single-byte symbols keep appearing as merging proceeds; measured on real
/// text they are roughly half of all rank lookups.
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
    /// Covers the three- and four-byte keys, or `None` when the vocabulary
    /// could not be indexed completely (see [`ShortRanks::build`]).
    short: Option<ShortRanks>,
}

impl BytePairRanks {
    /// Index every two-byte entry of `map`, and every three- and four-byte one.
    pub(crate) fn build(map: &Encoder) -> Self {
        let mut ranks = vec![u32::MAX; 256 * 256];
        for (key, rank) in map {
            if let [hi, lo] = key[..] {
                ranks[(hi as usize) << 8 | lo as usize] = rank;
            }
        }
        Self {
            ranks: ranks.into_boxed_slice(),
            short: ShortRanks::build(map),
        }
    }

    #[inline]
    fn get(&self, hi: u8, lo: u8) -> u32 {
        self.ranks[(hi as usize) << 8 | lo as usize]
    }
}

/// Merge ranks of every three- and four-byte key, in one open-addressed table.
///
/// The two-byte table above one step further out: measured over real text, keys
/// of four bytes and under are over 80% of all rank probes, yet they are a
/// quarter to a third of the map serving them. Three and four bytes will not
/// index directly, so this is a hash table and the win is density, not the
/// absence of a probe — a slot is one `u64` holding key, length and rank, so a
/// probe is an aligned load and an integer compare against the map's pointer
/// chase and `memcmp`.
///
/// The table is complete for the lengths it covers, so a miss here is a miss in
/// the map and needs no second lookup — which is why [`Self::build`] gives up
/// entirely rather than skipping an entry it cannot pack.
struct ShortRanks {
    /// Packed slots, always a power of two so the index is a mask.
    slots: Box<[u64]>,
    mask: usize,
}

impl ShortRanks {
    /// A slot holding no entry. Distinct from every packed entry because
    /// [`SHORT_MAX_RANK`] excludes the rank that would produce it.
    const EMPTY: u64 = u64::MAX;

    /// The key's bytes and length in the low bits, leaving the rest for the
    /// rank. Lengths are [`SHORT_MIN`]`..=`[`SHORT_MAX`], so one bit records
    /// which — without it a three-byte key would collide with the four-byte key
    /// that has a zero as its last byte.
    #[inline]
    fn pack_key(key: &[u8]) -> u64 {
        let mut bytes = [0u8; 4];
        bytes[..key.len()].copy_from_slice(key);
        u32::from_le_bytes(bytes) as u64 | ((key.len() - SHORT_MIN) as u64) << 32
    }

    /// Fibonacci hashing: the multiply spreads the key's high entropy — the
    /// bytes themselves — down into the bits the mask selects.
    #[inline]
    fn slot_of(&self, packed_key: u64) -> usize {
        (packed_key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32) as usize & self.mask
    }

    /// Index every three- and four-byte entry of `map`, or `None` if any of
    /// them cannot be packed.
    ///
    /// All-or-nothing on purpose: callers rely on a miss being authoritative,
    /// which only holds if every such key is present. A rank above
    /// [`SHORT_MAX_RANK`] is the only thing that can prevent it, and no real
    /// vocabulary comes close.
    fn build(map: &Encoder) -> Option<Self> {
        let count = map
            .keys()
            .filter(|k| (SHORT_MIN..=SHORT_MAX).contains(&k.len()))
            .count();

        // Half full at most: linear probing degrades sharply past that, and the
        // table is small enough that the headroom costs little.
        let capacity = (count * 2).next_power_of_two().max(16);
        let mut table = Self {
            slots: vec![Self::EMPTY; capacity].into_boxed_slice(),
            mask: capacity - 1,
        };

        for (key, rank) in map {
            if !(SHORT_MIN..=SHORT_MAX).contains(&key.len()) {
                continue;
            }
            if rank > SHORT_MAX_RANK {
                return None;
            }
            let packed_key = Self::pack_key(key);
            let mut slot = table.slot_of(packed_key);
            // Terminates: the table is at most half full, so an empty slot is
            // always reachable. A duplicate key cannot occur — `map` is a map.
            while table.slots[slot] != Self::EMPTY {
                slot = (slot + 1) & table.mask;
            }
            table.slots[slot] = packed_key | (rank as u64) << 33;
        }
        Some(table)
    }

    /// The rank of `key`, or `u32::MAX` when the vocabulary cannot merge it.
    #[inline]
    fn get(&self, key: &[u8]) -> u32 {
        let packed_key = Self::pack_key(key);
        let mut slot = self.slot_of(packed_key);
        loop {
            let entry = self.slots[slot];
            if entry == Self::EMPTY {
                return u32::MAX;
            }
            if entry & 0x1_FFFF_FFFF == packed_key {
                return (entry >> 33) as u32;
            }
            slot = (slot + 1) & self.mask;
        }
    }
}

/// Where the merge loop reads ranks from: the authoritative map, optionally
/// fronted by [`BytePairRanks`] for the short keys.
///
/// Borrowed rather than owned so building one costs nothing — the merge loop
/// constructs it per piece, and the public entry points that have no table
/// construct it with `pairs: None` and behave exactly as before.
#[derive(Clone, Copy)]
pub(crate) struct RankLookup<'a> {
    map: &'a Encoder,
    pairs: Option<&'a BytePairRanks>,
    /// Held here rather than reached through `pairs` so a vocabulary without
    /// one tests a register instead of loading through it on every probe.
    short: Option<&'a ShortRanks>,
}

impl<'a> RankLookup<'a> {
    /// A lookup that only consults the map.
    pub(crate) fn new(map: &'a Encoder) -> Self {
        Self {
            map,
            pairs: None,
            short: None,
        }
    }

    /// A lookup fronted by a two-byte table.
    pub(crate) fn with_pairs(map: &'a Encoder, pairs: &'a BytePairRanks) -> Self {
        Self {
            map,
            pairs: Some(pairs),
            short: pairs.short.as_ref(),
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
        if let Some(pairs) = self.pairs {
            if let [hi, lo] = key {
                return pairs.get(*hi, *lo);
            }
        }
        if let Some(short) = self.short {
            if (SHORT_MIN..=SHORT_MAX).contains(&key.len()) {
                return short.get(key);
            }
        }
        self.map.get(key).unwrap_or(u32::MAX)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encoder(entries: &[(&[u8], u32)]) -> Encoder {
        entries.iter().copied().collect()
    }

    /// The length bit in a packed key. Without it a three-byte key and the
    /// four-byte key that appends a zero pack identically, and one would answer
    /// with the other's rank.
    #[test]
    fn a_trailing_zero_byte_is_not_the_shorter_key() {
        let map = encoder(&[(b"abc", 7), (b"abc\0", 9)]);
        let short = ShortRanks::build(&map).unwrap();
        assert_eq!(short.get(b"abc"), 7);
        assert_eq!(short.get(b"abc\0"), 9);
    }

    /// A miss in the table is authoritative — the merge loop does not consult
    /// the map afterwards — so every short key of the map must be present and
    /// everything else must report unmergeable.
    #[test]
    fn the_table_answers_for_every_short_key_and_only_those() {
        let map = encoder(&[(b"ab", 1), (b"xyz", 2), (b"wxyz", 3), (b"abcde", 4)]);
        let short = ShortRanks::build(&map).unwrap();
        assert_eq!(short.get(b"xyz"), 2);
        assert_eq!(short.get(b"wxyz"), 3);
        assert_eq!(short.get(b"qqq"), u32::MAX);
    }

    /// A rank the packing cannot carry disables the table rather than leaving a
    /// hole in it, since a hole would read as unmergeable.
    #[test]
    fn an_unpackable_rank_gives_up_on_the_whole_table() {
        assert!(ShortRanks::build(&encoder(&[(b"abc", u32::MAX)])).is_none());
    }

    /// The lookup agrees with the map it fronts across every key length.
    #[test]
    fn the_fronted_lookup_agrees_with_the_map() {
        let entries: &[(&[u8], u32)] = &[
            (b"ab", 1),
            (b"abc", 2),
            (b"abcd", 3),
            (b"abcde", 4),
            (b"abcdef", 5),
        ];
        let map = encoder(entries);
        let pairs = BytePairRanks::build(&map);
        let fronted = RankLookup::with_pairs(&map, &pairs);
        let plain = RankLookup::new(&map);

        for (key, rank) in entries {
            assert_eq!(fronted.get(key), *rank);
            assert_eq!(plain.get(key), *rank);
        }
        for miss in [&b"zz"[..], b"zzz", b"zzzz", b"zzzzz"] {
            assert_eq!(fronted.get(miss), u32::MAX);
            assert_eq!(plain.get(miss), u32::MAX);
        }
    }
}
