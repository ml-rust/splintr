//! The pre-token chunk cache.
//!
//! Pre-tokenizers cut text at word boundaries and prose reuses words heavily,
//! so memoizing chunk → ids pays on ordinary input.
//!
//! This cache sits on the one path `encode_batch` runs concurrently. A single
//! lock around it does not merely reduce the speedup, it inverts it — every
//! worker queues twice per chunk, and adding threads makes it worse. Sharding
//! the lock is what makes the parallel path parallel.
//!
//! # Why the entries are inline
//!
//! Only the chunks that miss the whole-piece vocabulary lookup get here, and
//! nearly all of them are short. A map keyed by `Vec<u8>` with a `Vec<u32>`
//! value serves that population with two heap allocations and two pointer
//! chases per entry, and it dominated the allocation count of every encode.
//!
//! A slot instead holds its key and ids in place: nothing to allocate, one
//! memory access to read.
//!
//! # Why there is a second tier
//!
//! An inline slot has to fix a maximum length, and "too long" must not mean
//! "not cached". Which chunks reach this cache is script-dependent, and the
//! dependence is severe: measured per corpus, English and code barely use it at
//! all — the whole-piece lookup answers them — while for Chinese *no* chunk
//! fits in 16 bytes, and under ByteLevel, where every byte expands, none fits
//! in 32 either. Sizing a single inline slot therefore cannot be done from one
//! script's statistics; a cliff sized on English silently disables the cache
//! for CJK, which is the text that needs it most.
//!
//! So a chunk that does not fit inline goes to a second, growable tier instead
//! of being declined. Short chunks keep the allocation-free path; long ones
//! overwrite the buffers already in their slot, so a slot allocates while it is
//! growing to the longest chunk that has landed in it and not afterwards. That
//! matters because "long" is the whole population for CJK, where replacing the
//! buffers per insert made every miss cost two allocations and two frees.
//!
//! The two tiers hold the same number of slots. Giving the long tier a fraction
//! of the inline tier's looks like it saves memory on the English workloads that
//! never use it, but "long" and "short" are not two slices of one population —
//! they are different scripts, and each is the whole of its own. Starving the
//! long tier therefore does not right-size it, it evicts CJK's working set and
//! makes those chunks re-merge: measured, a quarter share costs up to a fifth of
//! the instructions on Chinese and Japanese and saves nothing on English. Empty
//! slots are two `Vec` headers, which is not a price worth optimising against
//! model weights.

use rustc_hash::FxHasher;
use std::hash::{Hash, Hasher};
use std::sync::RwLock;

/// Number of independently-locked shards.
///
/// Fixed rather than derived from the core count, so a tokenizer behaves the
/// same on every machine. It sits comfortably above the thread counts this
/// crate is used at, which is what keeps contention low.
const SHARDS: usize = 64;

/// Longest chunk an inline slot holds. Anything longer goes to the boxed tier.
const MAX_KEY: usize = 16;

/// Most ids an inline slot holds.
///
/// Equal to [`MAX_KEY`] on purpose: every id consumes at least one byte of the
/// chunk, so a key that fits inline can never produce a result that does not,
/// and the inline tier needs no length test on its ids.
const MAX_IDS: usize = MAX_KEY;

/// One slot: the chunk and its ids, stored in place.
///
/// Whether a slot holds anything is recorded by its tag, not here; `klen` is
/// only the length of the key it holds.
#[derive(Clone, Copy)]
struct Slot {
    key: [u8; MAX_KEY],
    ids: [u32; MAX_IDS],
    klen: u8,
    nids: u8,
}

const EMPTY: u8 = 0;

const VACANT: Slot = Slot {
    key: [0; MAX_KEY],
    ids: [0; MAX_IDS],
    klen: EMPTY,
    nids: 0,
};

/// A chunk too long for an inline slot, with its ids.
///
/// Vacant slots hold an empty key, which allocates nothing. `Vec` rather than
/// `Box<[_]>` so an overwrite reuses the buffer: the slot keeps the capacity of
/// the longest chunk it has held, which is what makes a steady-state insert
/// allocation-free.
#[derive(Clone, Default)]
struct LongSlot {
    key: Vec<u8>,
    ids: Vec<u32>,
}

/// Slots a key may occupy, counted from its home slot.
///
/// Direct-mapping (one slot per key) is a slot cheaper to search and much worse
/// to live in: two hot chunks whose hashes land together evict each other on
/// every alternation, forever, and no amount of capacity fixes it because the
/// rest of the table is unreachable to them. Sixteen ways turns that permanent
/// conflict into an ordinary capacity question, and a window of sixteen tags is
/// two `u64` loads to search.
const WAYS: usize = 16;

/// A one-byte digest of a key's hash, kept beside its slot so a probe rejects
/// non-matching slots without touching them.
///
/// `0` means vacant, so a real tag is never `0` — the bias below costs one bit
/// of tag entropy and buys a vacancy test that is the same comparison as the
/// match test.
#[inline]
fn tag_of(hash: u64) -> u8 {
    ((hash >> 24) as u8) | 1
}

const VACANT_TAG: u8 = 0;

/// Byte lanes of `word` equal to `tag`, marked by their high bit.
///
/// The classic zero-byte search: XOR makes a matching lane zero, and only a
/// zero lane both borrows from its neighbour and keeps a clear high bit.
#[inline]
fn matching_lanes(word: u64, tag: u8) -> u64 {
    const LOW: u64 = 0x0101_0101_0101_0101;
    const HIGH: u64 = 0x8080_8080_8080_8080;
    let x = word ^ LOW.wrapping_mul(tag as u64);
    x.wrapping_sub(LOW) & !x & HIGH
}

/// Call `visit` with each slot in `home`'s window whose tag is `tag`, until it
/// returns true.
///
/// Eight lanes are tested per `u64`, so a sixteen-way window is two loads and
/// two bit-scans rather than sixteen dependent byte compares.
#[inline]
fn probe(tags: &[u8], home: usize, tag: u8, mut visit: impl FnMut(usize) -> bool) -> bool {
    let mut base = home;
    while base < home + WAYS {
        // The table is padded by `WAYS`, so a full window always fits; `get`
        // rather than a slice so a future sizing bug cannot become a panic.
        let Some(lanes) = tags.get(base..base + 8).and_then(|s| s.try_into().ok()) else {
            return false;
        };
        let mut hits = matching_lanes(u64::from_le_bytes(lanes), tag);
        while hits != 0 {
            let lane = hits.trailing_zeros() as usize / 8;
            if visit(base + lane) {
                return true;
            }
            hits &= hits - 1;
        }
        base += 8;
    }
    false
}

/// One shard's two tiers, behind a single lock so a chunk takes one lock
/// whichever tier it belongs to.
///
/// Each tier carries a tag per slot, in its own array so a probe reads only
/// tags — sixteen of them in two loads — instead of striding over whole slots.
struct Tiers {
    inline_tags: Box<[u8]>,
    inline: Box<[Slot]>,
    long_tags: Box<[u8]>,
    long: Box<[LongSlot]>,
}

type Shard = RwLock<Tiers>;

/// A chunk → token-ids cache striped across [`SHARDS`] independently-locked
/// slabs, each [`WAYS`]-way set-associative.
///
/// Capacity is divided evenly across the shards, so eviction is per-shard
/// rather than global: a skewed key distribution could evict from a hot shard
/// while a cold one has room. Keys are distributed by a hash of the chunk
/// bytes, so that skew does not arise in practice.
///
/// A slot stores the chunk bytes themselves, not a bare hash, and a hit
/// compares them — so a collision cannot return another chunk's ids. It can
/// only evict, which costs a merge and never correctness.
pub(crate) struct ChunkCache {
    shards: Box<[Shard]>,
    /// Slots per shard in each tier, always a power of two so the index is a
    /// mask.
    mask: usize,
}

impl ChunkCache {
    /// Build a cache holding up to `capacity` chunks in total.
    pub(crate) fn new(capacity: usize) -> Self {
        let per_shard = (capacity / SHARDS).next_power_of_two().max(1);
        // Padded by a window, so the last home slot's probe stays in bounds
        // without wrapping — wrapping would make the window of a high slot
        // overlap the window of a low one and bias eviction toward the table's
        // start.
        let slots = per_shard + WAYS;
        Self {
            shards: (0..SHARDS)
                .map(|_| {
                    RwLock::new(Tiers {
                        inline_tags: vec![VACANT_TAG; slots].into_boxed_slice(),
                        inline: vec![VACANT; slots].into_boxed_slice(),
                        long_tags: vec![VACANT_TAG; slots].into_boxed_slice(),
                        long: vec![LongSlot::default(); slots].into_boxed_slice(),
                    })
                })
                .collect(),
            mask: per_shard - 1,
        }
    }

    /// The shard hash of `key`, so a caller that both looks up and inserts can
    /// compute it once: a miss walks this cache twice, a failed
    /// [`Self::extend_into`] then a [`Self::put`].
    pub(crate) fn shard_hash(key: &[u8]) -> u64 {
        let mut hasher = FxHasher::default();
        key.hash(&mut hasher);
        hasher.finish()
    }

    /// The shard a hash belongs to, and the slot within it. Which *tier* that
    /// slot is in is the caller's business, and both tiers are the same size.
    ///
    /// The high bits pick the shard and the low bits the slot, so the two
    /// selections cannot correlate — `FxHasher`'s low bits are its weakest, and
    /// reusing them for both would cluster every shard's traffic into the same
    /// few slots.
    #[inline]
    fn locate(&self, hash: u64) -> (&Shard, usize) {
        (
            &self.shards[(hash >> 32) as usize % SHARDS],
            hash as usize & self.mask,
        )
    }

    /// Append the ids cached for `key` to `out`, reporting whether there were
    /// any.
    ///
    /// A read lock, so concurrent hits do not serialise, and the ids are copied
    /// out from the slot itself — no pointer to follow.
    pub(crate) fn extend_into(&self, hash: u64, key: &[u8], out: &mut Vec<u32>) -> bool {
        let (shard, home) = self.locate(hash);
        let Ok(tiers) = shard.read() else {
            return false;
        };
        let tag = tag_of(hash);
        // A tag match only narrows the search — the slot's own bytes still have
        // to equal the key, so a tag collision costs a comparison and never an
        // answer from the wrong chunk.
        if key.len() > MAX_KEY {
            return probe(&tiers.long_tags, home, tag, |i| {
                let slot = &tiers.long[i];
                if slot.key != key {
                    return false;
                }
                out.extend_from_slice(&slot.ids);
                true
            });
        }
        probe(&tiers.inline_tags, home, tag, |i| {
            let slot = &tiers.inline[i];
            if slot.klen as usize != key.len() || &slot.key[..key.len()] != key {
                return false;
            }
            // Almost every chunk resolves to one id, and `extend_from_slice`
            // spends a length-driven `memcpy` and a capacity check on it.
            match slot.nids {
                1 => out.push(slot.ids[0]),
                n => out.extend_from_slice(&slot.ids[..n as usize]),
            }
            true
        })
    }

    /// The slot in `home`'s window that an insert should take: the one already
    /// holding this key, else a vacant one, else a victim.
    ///
    /// Preferring the key's own slot is what keeps a repeated chunk in one place
    /// instead of filling the window with copies of itself.
    fn insert_slot(
        tags: &[u8],
        home: usize,
        hash: u64,
        tag: u8,
        matches: impl Fn(usize) -> bool,
    ) -> usize {
        let mut found = None;
        probe(tags, home, tag, |i| {
            if matches(i) {
                found = Some(i);
                return true;
            }
            false
        });
        if let Some(i) = found {
            return i;
        }
        if let Some(i) = (home..home + WAYS).find(|&i| tags.get(i) == Some(&VACANT_TAG)) {
            return i;
        }
        // Full window. Spread victims across it with hash bits the home slot did
        // not use, so a hot home slot is not the one always overwritten.
        home + (hash >> 56) as usize % WAYS
    }

    /// Cache `ids` for `key`, whose shard hash the caller already has.
    ///
    /// A chunk longer than an inline slot goes to the growable tier, whose
    /// buffers are overwritten in place rather than replaced — allocating only
    /// while the slot grows to the longest chunk it has held. The inline tier's
    /// id count needs no check: see [`MAX_IDS`].
    pub(crate) fn put(&self, hash: u64, key: &[u8], ids: &[u32]) {
        if key.is_empty() {
            return;
        }
        let (shard, home) = self.locate(hash);
        let Ok(mut tiers) = shard.write() else {
            return;
        };
        let tag = tag_of(hash);
        if key.len() > MAX_KEY {
            let index = Self::insert_slot(&tiers.long_tags, home, hash, tag, |i| {
                tiers.long[i].key == key
            });
            let slot = &mut tiers.long[index];
            slot.key.clear();
            slot.key.extend_from_slice(key);
            slot.ids.clear();
            slot.ids.extend_from_slice(ids);
            tiers.long_tags[index] = tag;
            return;
        }
        debug_assert!(ids.len() <= MAX_IDS, "ids outnumber the chunk's bytes");
        let index = Self::insert_slot(&tiers.inline_tags, home, hash, tag, |i| {
            let slot = &tiers.inline[i];
            slot.klen as usize == key.len() && slot.key[..key.len()] == *key
        });
        let slot = &mut tiers.inline[index];
        slot.key[..key.len()].copy_from_slice(key);
        slot.ids[..ids.len()].copy_from_slice(ids);
        slot.klen = key.len() as u8;
        slot.nids = ids.len() as u8;
        tiers.inline_tags[index] = tag;
    }

    /// Drop every entry.
    pub(crate) fn clear(&self) {
        for shard in &self.shards {
            if let Ok(mut tiers) = shard.write() {
                tiers.inline_tags.fill(VACANT_TAG);
                tiers.inline.fill(VACANT);
                tiers.long_tags.fill(VACANT_TAG);
                tiers.long.fill_with(LongSlot::default);
            }
        }
    }

    /// Entries currently held, across all shards.
    pub(crate) fn len(&self) -> usize {
        self.shards
            .iter()
            .map(|shard| {
                shard
                    .read()
                    .map(|tiers| {
                        // The tags are the occupancy record for both tiers, and
                        // reading them touches two bytes per slot instead of two
                        // whole slots.
                        tiers
                            .inline_tags
                            .iter()
                            .filter(|&&t| t != VACANT_TAG)
                            .count()
                            + tiers.long_tags.iter().filter(|&&t| t != VACANT_TAG).count()
                    })
                    .unwrap_or(0)
            })
            .sum()
    }
}

impl std::fmt::Debug for ChunkCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChunkCache")
            .field("shards", &SHARDS)
            .field("len", &self.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(cache: &ChunkCache, key: &[u8], ids: &[u32]) -> Option<Vec<u32>> {
        let hash = ChunkCache::shard_hash(key);
        cache.put(hash, key, ids);
        let mut out = Vec::new();
        cache.extend_into(hash, key, &mut out).then_some(out)
    }

    /// The regression this tier exists to prevent: a chunk longer than an
    /// inline slot must be cached, not declined. Which chunks are long is
    /// script-dependent — no Chinese chunk fits inline, and under ByteLevel not
    /// even a short one does — so declining them disables the cache exactly
    /// where it carries the most work.
    #[test]
    fn a_chunk_too_long_to_inline_is_still_cached() {
        let cache = ChunkCache::new(1024);
        let key = "那么，线性代数又是如何来解决这些问题的呢".as_bytes();
        assert!(key.len() > MAX_KEY, "test key must exercise the boxed tier");
        assert_eq!(
            roundtrip(&cache, key, &[1, 2, 3, 4, 5]).as_deref(),
            Some(&[1, 2, 3, 4, 5][..])
        );
    }

    /// Overwriting a long slot must replace its contents, not append to them —
    /// the buffers are reused rather than rebuilt, so a missing `clear` would
    /// leave the previous entry's tail behind and answer with too many ids.
    #[test]
    fn overwriting_a_long_slot_replaces_its_ids() {
        let cache = ChunkCache::new(1024);
        let key = "那么，线性代数又是如何来解决这些问题的呢".as_bytes();
        assert_eq!(
            roundtrip(&cache, key, &[1, 2, 3, 4, 5]).as_deref(),
            Some(&[1, 2, 3, 4, 5][..])
        );
        assert_eq!(roundtrip(&cache, key, &[9]).as_deref(), Some(&[9][..]));
    }

    /// The reason the table is set-associative. Two keys sharing a home slot
    /// must both be holdable — direct-mapped, the second overwrote the first and
    /// an alternating pair missed every single time no matter how much spare
    /// capacity the table had.
    #[test]
    fn keys_sharing_a_home_slot_do_not_evict_each_other() {
        let cache = ChunkCache::new(1024);
        // The caller supplies the hash, so the collision is exact rather than
        // hunted for.
        let hash = 0x1234_5678_9abc_def0;
        cache.put(hash, b"alpha", &[1]);
        cache.put(hash, b"bravo", &[2]);

        let mut out = Vec::new();
        assert!(cache.extend_into(hash, b"alpha", &mut out));
        assert_eq!(out, [1], "the first key was evicted by the second");
        out.clear();
        assert!(cache.extend_into(hash, b"bravo", &mut out));
        assert_eq!(out, [2]);
    }

    /// A full window's worth of colliding keys all survive, and re-inserting a
    /// key already present updates it in place rather than consuming a way.
    #[test]
    fn a_window_holds_a_full_set_of_colliding_keys() {
        let cache = ChunkCache::new(1024);
        let hash = 0xdead_beef_0000_0000;
        let keys: Vec<Vec<u8>> = (0..WAYS)
            .map(|i| format!("key{i:02}").into_bytes())
            .collect();
        for (i, key) in keys.iter().enumerate() {
            cache.put(hash, key, &[i as u32]);
        }
        // Re-inserting the first must not displace any of the others.
        cache.put(hash, &keys[0], &[99]);

        let mut out = Vec::new();
        for (i, key) in keys.iter().enumerate().skip(1) {
            out.clear();
            assert!(cache.extend_into(hash, key, &mut out), "lost key {i}");
            assert_eq!(out, [i as u32], "key {i} came back with the wrong ids");
        }
        out.clear();
        assert!(cache.extend_into(hash, &keys[0], &mut out));
        assert_eq!(out, [99], "re-inserting a present key did not update it");
    }

    #[test]
    fn a_short_chunk_round_trips_through_the_inline_tier() {
        let cache = ChunkCache::new(1024);
        assert_eq!(
            roundtrip(&cache, b"hello", &[7, 8]).as_deref(),
            Some(&[7, 8][..])
        );
    }

    /// Long and short keys are counted and cleared alike.
    #[test]
    fn clear_empties_both_tiers() {
        let cache = ChunkCache::new(1024);
        roundtrip(&cache, b"short", &[1]);
        roundtrip(&cache, "那么，线性代数又是如何来解决".as_bytes(), &[2]);
        assert_eq!(cache.len(), 2);
        cache.clear();
        assert_eq!(cache.len(), 0);
    }

    /// A miss must not answer with a neighbour's ids, in either tier.
    #[test]
    fn a_different_key_is_a_miss() {
        let cache = ChunkCache::new(1024);
        roundtrip(&cache, b"alpha", &[1]);
        let mut out = Vec::new();
        assert!(!cache.extend_into(ChunkCache::shard_hash(b"beta"), b"beta", &mut out));
        let long = "这是一个很长的中文词组用来测试".as_bytes();
        roundtrip(&cache, long, &[9]);
        let other = "另一个完全不同的中文词组测试内容".as_bytes();
        assert!(!cache.extend_into(ChunkCache::shard_hash(other), other, &mut out));
        assert!(out.is_empty());
    }
}
