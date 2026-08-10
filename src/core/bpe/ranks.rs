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

/// Widest id [`PairRanks`] can address, and the widest merge rank it can carry.
///
/// A slot is one `u64`: two ids for the key and one for the merged token, so
/// three fields of 20 bits leave four spare and keep every real entry below
/// [`PairRanks::EMPTY`]. No published vocabulary is within two orders of
/// magnitude of the limit; one that were simply keeps the byte-keyed path.
const PAIR_ID_BITS: u32 = 20;
const PAIR_ID_LIMIT: u32 = 1 << PAIR_ID_BITS;

/// Merge ranks keyed by the **ids of the pair being merged**, rather than by the
/// bytes the pair concatenates to.
///
/// The byte-keyed tables above are complete only for short keys — two bytes
/// directly, three and four through [`ShortRanks`] — and a merge's key grows as
/// the merge proceeds. That is invisible on Latin text, whose merges stay
/// short, and dominant on every other script: a CJK character is three bytes, so
/// the *first* merge already produces a six-byte key and every one after it is
/// nine, twelve, fifteen. All of those fall through to a hash of a byte string
/// and a `memcmp`.
///
/// Keyed by id there is no such gradient. Every symbol the merge loop holds is a
/// vocabulary token — a seed symbol is one by construction, and a merged one is
/// one because a pair only merges when the vocabulary contains what it
/// concatenates to — so a pair is two `u32`s whatever its surface is, and a probe
/// is a multiply and an aligned load.
///
/// The table answers for exactly the pairs the byte-keyed map answers for: every
/// split of every mergeable surface into two halves the vocabulary also has.
/// That is what keeps the two paths bit-exact rather than merely similar — it is
/// the same relation, addressed differently.
pub(crate) struct PairRanks {
    /// Open-addressed, always a power of two. A slot packs the pair in the low
    /// `2 * PAIR_ID_BITS` bits and the merged token's id above them.
    slots: Box<[u64]>,
    mask: usize,
    /// Merged id of every pair whose operands are both below [`DENSE_IDS`],
    /// indexed directly rather than probed.
    ///
    /// A vocabulary's alphabet takes its lowest ids, and every merge starts from
    /// the alphabet, so this answers the pairs a piece is seeded with — the
    /// largest single group of lookups, and the one whose operands repeat most
    /// across pieces and so stay resident.
    dense: Box<[u32]>,
    /// Merge priority per token id, when the model orders its merges
    /// independently of its ids. `None` for tiktoken-style vocabularies, where a
    /// token's id *is* its rank and the indirection would be an identity.
    rank_by_id: Option<Box<[u32]>>,
    /// Id of each single byte, `u32::MAX` where the vocabulary has no such
    /// token. Byte-seeded pieces start here.
    byte_ids: Box<[u32; 256]>,
    /// Id of each two-byte token, indexed directly. This is what a ByteLevel
    /// vocabulary seeds its non-ASCII characters from, and there is one lookup
    /// per character of every chunk that reaches the merge.
    pair_byte_ids: Box<[u32]>,
    /// Id of every token of one to four bytes: what character-seeded pieces
    /// start from, since a UTF-8 character is never longer.
    symbol_ids: SymbolIds,
    /// Whether a merge must start from whole characters rather than bytes,
    /// because the vocabulary does not contain every single byte as a token.
    seeds_by_char: bool,
    /// Id of the token each **raw** input byte stands for, for a ByteLevel
    /// vocabulary — that is, of the alphabet character the byte maps to.
    ///
    /// `None` unless every one of the 256 resolves. With it a piece can be
    /// merged without ever entering ByteLevel space: the merge only ever needs
    /// its symbols' ids, and this supplies them from the input bytes directly.
    raw_byte_ids: Option<Box<[u32; 256]>>,
    /// Id per character a merge may start from already assembled, indexed by
    /// codepoint — see [`CharSeeds`] and [`PairRanks::char_seeds`].
    char_seeds: Option<CharSeeds>,
}

impl PairRanks {
    /// The id of the token a single byte is, or `u32::MAX`.
    #[inline]
    pub(crate) fn byte_id(&self, byte: u8) -> u32 {
        self.byte_ids[byte as usize]
    }

    /// The id a seed symbol of `bytes` stands for, or `u32::MAX` when the
    /// vocabulary has no such token and the piece must take the byte path.
    ///
    /// One and two bytes are indexed directly. Between them they are every
    /// symbol a ByteLevel vocabulary can seed from — its alphabet is 256
    /// characters, none of them wider — so the hash below serves only the
    /// character-seeded vocabularies whose alphabet is genuine text.
    #[inline]
    pub(crate) fn seed_id(&self, bytes: &[u8]) -> u32 {
        match bytes {
            [byte] => self.byte_ids[*byte as usize],
            [hi, lo] => self.pair_byte_ids[(*hi as usize) << 8 | *lo as usize],
            _ => self.symbol_ids.get(bytes),
        }
    }

    /// Whether pieces must be seeded by character for this vocabulary.
    #[inline]
    pub(crate) fn seeds_by_char(&self) -> bool {
        self.seeds_by_char
    }

    /// Whether this vocabulary can merge a piece straight from its raw input
    /// bytes, with no mapping into ByteLevel space.
    #[inline]
    pub(crate) fn seeds_raw(&self) -> bool {
        self.raw_byte_ids.is_some()
    }

    /// The id of the whole character `bytes` spells, or `u32::MAX` when it must
    /// be merged up from its bytes instead.
    ///
    /// A direct index rather than a hash, because this is asked once per
    /// character of every chunk that reaches the merge and the answer is often
    /// *no* — a script whose characters mostly fail the test would otherwise pay
    /// a probe per character to learn nothing.
    #[inline]
    pub(crate) fn char_seed(&self, bytes: &[u8]) -> u32 {
        match &self.char_seeds {
            Some(seeds) => seeds.get(bytes),
            None => u32::MAX,
        }
    }

    /// Whether any character can be seeded whole.
    #[inline]
    pub(crate) fn seeds_chars(&self) -> bool {
        self.char_seeds.is_some()
    }

    /// The id of the token a raw input byte stands for. Only called when
    /// [`Self::seeds_raw`] holds, where every byte resolves.
    #[inline]
    pub(crate) fn raw_byte_id(&self, byte: u8) -> u32 {
        match &self.raw_byte_ids {
            Some(ids) => ids[byte as usize],
            None => u32::MAX,
        }
    }
}

/// Ids below this are indexed directly by [`PairRanks::dense`]. The square of
/// it is the table's size in entries, so it buys the alphabet and the earliest
/// merges, not the vocabulary.
const DENSE_IDS: u32 = 512;

impl PairRanks {
    /// A slot holding no entry. Distinct from every real one because all three
    /// packed fields are bounded by [`PAIR_ID_LIMIT`].
    const EMPTY: u64 = u64::MAX;

    /// Index every pair `rank_map` can merge, or `None` when the vocabulary
    /// cannot be addressed by id.
    ///
    /// All-or-nothing, like [`ShortRanks::build`] and for the same reason: the
    /// merge loop takes a miss as authoritative and never consults the map
    /// afterwards, which is only sound if every mergeable pair is present.
    pub(crate) fn build(
        rank_map: &Encoder,
        id_encoder: &Encoder,
        raw_encoder: Option<&Encoder>,
    ) -> Option<Self> {
        let mut max_id = 0u32;
        for (_, id) in id_encoder {
            if id >= PAIR_ID_LIMIT {
                return None;
            }
            max_id = max_id.max(id);
        }

        // Every split of every mergeable surface whose two halves are also
        // tokens. Collected first so the table can be sized exactly rather than
        // walking the vocabulary twice.
        let mut pairs: Vec<(u64, u32)> = Vec::with_capacity(rank_map.len() * 2);
        for (key, _) in rank_map {
            let Some(merged) = id_encoder.get(key) else {
                continue;
            };
            for split in 1..key.len() {
                // Left first and only then right: most splits of a token are not
                // two tokens, and building this walks the whole vocabulary, so
                // the probe that is skipped is the majority of the work.
                let Some(left) = id_encoder.get(&key[..split]) else {
                    continue;
                };
                let Some(right) = id_encoder.get(&key[split..]) else {
                    continue;
                };
                pairs.push((Self::pack(left, right), merged));
            }
        }

        // Seeding by byte needs every byte to be a token. Where one is not — a
        // ByteLevel vocabulary, whose alphabet is characters — the merge must
        // start from characters instead, which is only equivalent because the
        // alphabet holds the lowest merge ranks and byte seeding would
        // reassemble exactly those characters first. A vocabulary whose ranks
        // *are* its ids has no such alphabet ordering to rely on, so it does not
        // get the id path at all unless byte seeding already works for it.
        let seeds_by_char = (0..=u8::MAX).any(|byte| id_encoder.get(&[byte][..]).is_none());
        if seeds_by_char && std::ptr::eq(rank_map, id_encoder) {
            return None;
        }

        // Half full at most: linear probing degrades sharply past that.
        let capacity = (pairs.len() * 2).next_power_of_two().max(16);
        let mut table = Self {
            slots: vec![Self::EMPTY; capacity].into_boxed_slice(),
            mask: capacity - 1,
            dense: vec![u32::MAX; (DENSE_IDS * DENSE_IDS) as usize].into_boxed_slice(),
            rank_by_id: None,
            byte_ids: Box::new([u32::MAX; 256]),
            pair_byte_ids: vec![u32::MAX; 256 * 256].into_boxed_slice(),
            symbol_ids: SymbolIds::build(id_encoder)?,
            seeds_by_char,
            // All or nothing: the raw path has no way to render a byte the
            // alphabet does not cover, and falling back mid-piece would mean
            // merging raw bytes against a map keyed in ByteLevel space.
            char_seeds: None,
            raw_byte_ids: raw_encoder.and_then(|raw| {
                let mut ids = Box::new([u32::MAX; 256]);
                for (byte, slot) in ids.iter_mut().enumerate() {
                    *slot = raw.get(&[byte as u8][..])?;
                }
                Some(ids)
            }),
        };
        for (key, merged) in pairs {
            let (left, right) = (key >> PAIR_ID_BITS, key & ((1 << PAIR_ID_BITS) - 1));
            if left < DENSE_IDS as u64 && right < DENSE_IDS as u64 {
                table.dense[(left * DENSE_IDS as u64 + right) as usize] = merged;
                continue;
            }
            let mut slot = table.slot_of(key);
            // A duplicate key cannot occur: a pair of ids determines the bytes
            // it concatenates to, which the map holds at most once.
            while table.slots[slot] != Self::EMPTY {
                slot = (slot + 1) & table.mask;
            }
            table.slots[slot] = key | (merged as u64) << (2 * PAIR_ID_BITS);
        }

        for byte in 0..=u8::MAX {
            if let Some(id) = id_encoder.get(&[byte][..]) {
                table.byte_ids[byte as usize] = id;
            }
        }
        for (key, id) in id_encoder {
            if let [hi, lo] = key {
                table.pair_byte_ids[(*hi as usize) << 8 | *lo as usize] = id;
            }
        }

        // Identical maps mean the id is the rank, and the array would be
        // `rank_by_id[i] == i`.
        let ranks = (!std::ptr::eq(rank_map, id_encoder)).then(|| {
            let mut ranks = vec![u32::MAX; max_id as usize + 1].into_boxed_slice();
            for (key, rank) in rank_map {
                if let Some(id) = id_encoder.get(key) {
                    ranks[id as usize] = rank;
                }
            }
            ranks
        });
        let rank_of = |id: u32| match &ranks {
            Some(ranks) => ranks[id as usize],
            None => id,
        };

        table.char_seeds = raw_encoder.map(|raw| Self::char_seeds(raw, &rank_of));

        if let Some(ranks) = ranks {
            // The merge loop only ever *compares* ranks, so a vocabulary that
            // numbers its tokens in merge order — which is what a model whose
            // ids were assigned as its merges were learned does — needs no
            // ranks at all: comparing ids is the same comparison. Dropping the
            // array takes a dependent load out of the loop's inner step, and
            // that load is a random access into a table the size of the
            // vocabulary.
            let ordered = ranks
                .iter()
                .filter(|&&rank| rank != u32::MAX)
                .is_sorted_by(|a, b| a < b);
            if !ordered {
                table.rank_by_id = Some(ranks);
            }
        }

        Some(table)
    }

    /// The characters a merge may start from already assembled, keyed by their
    /// raw bytes.
    ///
    /// A merge that seeds a character whole skips the merges that would have
    /// built it out of its bytes. That is only the same computation if those
    /// merges were going to happen anyway, before anything else could touch the
    /// character's bytes — which is what this decides, per character, from the
    /// vocabulary alone.
    ///
    /// A character `c` qualifies when **every token that could take a piece of
    /// it from a neighbour ranks above `c` itself**: no token ending in a proper
    /// prefix of `c` merges earlier than `c` does, and none beginning with a
    /// proper suffix of `c` does either. Then nothing can reach into `c` before
    /// it is whole, so byte seeding always assembles it first and pre-assembling
    /// it changes nothing.
    ///
    /// That single condition is enough on its own. It might look as though a
    /// merge *consuming* the finished character also needs checking — under byte
    /// seeding such a merge has to wait for `c`, and pre-assembly would let it
    /// fire earlier — but any token containing `c` is built either from `c`
    /// itself or from a token straddling `c`'s edge, and the condition already
    /// puts both above `c`. So no merge involving the whole character can
    /// outrank the character.
    ///
    /// Characters failing the test are simply left out, and seed as their bytes
    /// exactly as before; the decision is per character and needs nothing from
    /// its neighbours.
    fn char_seeds(raw: &Encoder, rank_of: &impl Fn(u32) -> u32) -> CharSeeds {
        // Lowest rank among tokens having each short byte string as a *proper*
        // suffix, and as a proper prefix. Only lengths a character's proper
        // prefixes and suffixes can take are worth recording.
        let mut min_suffix: rustc_hash::FxHashMap<u32, u32> = rustc_hash::FxHashMap::default();
        let mut min_prefix: rustc_hash::FxHashMap<u32, u32> = rustc_hash::FxHashMap::default();
        for (key, id) in raw {
            let rank = rank_of(id);
            for len in 1..key.len().min(SYMBOL_MAX) {
                let head = SymbolIds::pack_key(&key[..len]) as u32;
                let tail = SymbolIds::pack_key(&key[key.len() - len..]) as u32;
                let slot = min_prefix.entry(head).or_insert(u32::MAX);
                *slot = (*slot).min(rank);
                let slot = min_suffix.entry(tail).or_insert(u32::MAX);
                *slot = (*slot).min(rank);
            }
        }

        let mut safe: Vec<(&[u8], u32)> = Vec::new();
        for (key, id) in raw {
            if key.len() > SYMBOL_MAX || !Self::is_one_character(key) {
                continue;
            }
            let rank = rank_of(id);
            let reachable = (1..key.len()).any(|at| {
                let head = SymbolIds::pack_key(&key[..at]) as u32;
                let tail = SymbolIds::pack_key(&key[at..]) as u32;
                min_suffix.get(&head).is_some_and(|&r| r < rank)
                    || min_prefix.get(&tail).is_some_and(|&r| r < rank)
            });
            if !reachable {
                safe.push((key, id));
            }
        }
        CharSeeds::new(&safe)
    }

    /// Whether `key` spells exactly one character.
    fn is_one_character(key: &[u8]) -> bool {
        std::str::from_utf8(key).is_ok_and(|text| text.chars().nth(1).is_none())
    }

    #[inline]
    fn pack(left: u32, right: u32) -> u64 {
        (left as u64) << PAIR_ID_BITS | right as u64
    }

    /// Fibonacci hashing, as [`ShortRanks::slot_of`].
    #[inline]
    fn slot_of(&self, key: u64) -> usize {
        (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32) as usize & self.mask
    }

    /// The id the pair merges to, or `u32::MAX` when it does not merge.
    #[inline]
    fn merged(&self, left: u32, right: u32) -> u32 {
        if left < DENSE_IDS && right < DENSE_IDS {
            return self.dense[(left * DENSE_IDS + right) as usize];
        }
        let key = Self::pack(left, right);
        let mut slot = self.slot_of(key);
        loop {
            let entry = self.slots[slot];
            if entry == Self::EMPTY {
                return u32::MAX;
            }
            if entry & ((1 << (2 * PAIR_ID_BITS)) - 1) == key {
                return (entry >> (2 * PAIR_ID_BITS)) as u32;
            }
            slot = (slot + 1) & self.mask;
        }
    }

    /// The merge priority of the token `id` stands for.
    #[inline]
    fn rank(&self, id: u32) -> u32 {
        match &self.rank_by_id {
            Some(ranks) => ranks[id as usize],
            None => id,
        }
    }
}

/// Lowest codepoint spelled with two UTF-8 bytes, and the lowest spelled with
/// three — the bases the two tables are indexed from.
const TWO_BYTE_BASE: u32 = 0x80;
const THREE_BYTE_BASE: u32 = 0x800;
/// One past the highest codepoint spelled with three bytes.
const FOUR_BYTE_BASE: u32 = 0x1_0000;

/// Id per character that may be seeded whole, indexed by codepoint.
///
/// `u32::MAX` where the character is absent from the vocabulary or failed the
/// test in [`PairRanks::char_seeds`]. Both tables together are a quarter of a
/// megabyte and answer in one load, which is the point: the question is asked
/// per character and the answer is often no.
#[derive(Clone)]
struct CharSeeds {
    two: Box<[u32]>,
    three: Box<[u32]>,
}

impl CharSeeds {
    fn new(entries: &[(&[u8], u32)]) -> Self {
        let mut seeds = Self {
            two: vec![u32::MAX; (THREE_BYTE_BASE - TWO_BYTE_BASE) as usize].into_boxed_slice(),
            three: vec![u32::MAX; (FOUR_BYTE_BASE - THREE_BYTE_BASE) as usize].into_boxed_slice(),
        };
        for (key, id) in entries {
            let Some(codepoint) = std::str::from_utf8(key)
                .ok()
                .and_then(|text| text.chars().next())
                .map(u32::from)
            else {
                continue;
            };
            match key.len() {
                2 => seeds.two[(codepoint - TWO_BYTE_BASE) as usize] = *id,
                3 => seeds.three[(codepoint - THREE_BYTE_BASE) as usize] = *id,
                // One byte is already one symbol, and four-byte characters are
                // too rare to spend a third table on.
                _ => {}
            }
        }
        seeds
    }

    /// The id of the character `bytes` spells, or `u32::MAX`.
    ///
    /// `bytes` is one whole character as the seeding walk cut it, so its length
    /// is what selects the table and its continuation bytes need no validation.
    #[inline]
    fn get(&self, bytes: &[u8]) -> u32 {
        match *bytes {
            [b0, b1] => {
                let codepoint = (b0 as u32 & 0x1F) << 6 | (b1 as u32 & 0x3F);
                match codepoint.checked_sub(TWO_BYTE_BASE) {
                    Some(at) => self.two[at as usize],
                    None => u32::MAX,
                }
            }
            [b0, b1, b2] => {
                let codepoint =
                    (b0 as u32 & 0x0F) << 12 | (b1 as u32 & 0x3F) << 6 | (b2 as u32 & 0x3F);
                match codepoint.checked_sub(THREE_BYTE_BASE) {
                    Some(at) => self.three[at as usize],
                    None => u32::MAX,
                }
            }
            _ => u32::MAX,
        }
    }
}

/// Longest symbol [`SymbolIds`] indexes: a UTF-8 character is never longer, and
/// nothing else seeds a merge.
const SYMBOL_MAX: usize = 4;

/// Id of every token short enough to seed a merge, in one open-addressed table.
///
/// [`ShortRanks`] one map over: the same packing, keyed the same way, but
/// answering with an id rather than a rank and covering one byte as well, since
/// a seed symbol may be a single character.
struct SymbolIds {
    slots: Box<[u64]>,
    mask: usize,
}

impl SymbolIds {
    const EMPTY: u64 = u64::MAX;

    /// The key's bytes in the low 32 bits and its length above them, leaving the
    /// rest for the id. Two length bits, so a three-byte key cannot collide with
    /// the four-byte key that ends in a zero.
    #[inline]
    fn pack_key(key: &[u8]) -> u64 {
        let mut bytes = [0u8; 4];
        bytes[..key.len()].copy_from_slice(key);
        u32::from_le_bytes(bytes) as u64 | ((key.len() - 1) as u64) << 32
    }

    #[inline]
    fn slot_of(&self, packed_key: u64) -> usize {
        (packed_key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32) as usize & self.mask
    }

    /// Index every token of [`SYMBOL_MAX`] bytes or fewer, or `None` when one
    /// carries an id too wide to pack.
    fn build(map: &Encoder) -> Option<Self> {
        let count = map.keys().filter(|k| k.len() <= SYMBOL_MAX).count();
        let capacity = (count * 2).next_power_of_two().max(16);
        let mut table = Self {
            slots: vec![Self::EMPTY; capacity].into_boxed_slice(),
            mask: capacity - 1,
        };
        for (key, id) in map {
            if key.is_empty() || key.len() > SYMBOL_MAX {
                continue;
            }
            if id >= PAIR_ID_LIMIT {
                return None;
            }
            let packed_key = Self::pack_key(key);
            let mut slot = table.slot_of(packed_key);
            while table.slots[slot] != Self::EMPTY {
                slot = (slot + 1) & table.mask;
            }
            table.slots[slot] = packed_key | (id as u64) << 34;
        }
        Some(table)
    }

    /// The id of `key`, or `u32::MAX` when the vocabulary does not have it.
    #[inline]
    fn get(&self, key: &[u8]) -> u32 {
        if key.is_empty() || key.len() > SYMBOL_MAX {
            return u32::MAX;
        }
        let packed_key = Self::pack_key(key);
        let mut slot = self.slot_of(packed_key);
        loop {
            let entry = self.slots[slot];
            if entry == Self::EMPTY {
                return u32::MAX;
            }
            if entry & 0x3_FFFF_FFFF == packed_key {
                return (entry >> 34) as u32;
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
    /// The id-keyed table, when the vocabulary could be indexed by id.
    by_id: Option<&'a PairRanks>,
}

impl<'a> RankLookup<'a> {
    /// A lookup that only consults the map.
    pub(crate) fn new(map: &'a Encoder) -> Self {
        Self {
            map,
            pairs: None,
            short: None,
            by_id: None,
        }
    }

    /// A lookup fronted by a two-byte table.
    pub(crate) fn with_pairs(map: &'a Encoder, pairs: &'a BytePairRanks) -> Self {
        Self {
            map,
            pairs: Some(pairs),
            short: pairs.short.as_ref(),
            by_id: None,
        }
    }

    /// The same lookup, also carrying the id-keyed table.
    pub(crate) fn with_ids(mut self, by_id: Option<&'a PairRanks>) -> Self {
        self.by_id = by_id;
        self
    }

    /// The same lookup with the id-keyed table dropped.
    ///
    /// The table is a property of the vocabulary, but seeding by id is a
    /// property of the *piece*: a symbol the vocabulary does not have has no id
    /// to merge by. A piece that cannot be seeded takes this and runs exactly
    /// the byte-keyed path it always did.
    pub(crate) fn without_ids(mut self) -> Self {
        self.by_id = None;
        self
    }

    /// The id-keyed table, when this lookup has one.
    #[inline]
    pub(crate) fn by_id(&self) -> Option<&'a PairRanks> {
        self.by_id
    }

    /// The merge rank of the pair `(left, right)` and the id it merges to, or
    /// `(u32::MAX, u32::MAX)` when the vocabulary cannot merge it.
    #[inline]
    pub(crate) fn pair(&self, table: &PairRanks, left: u32, right: u32) -> (u32, u32) {
        let merged = table.merged(left, right);
        match merged {
            u32::MAX => (u32::MAX, u32::MAX),
            id => (table.rank(id), id),
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
