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
//! So a chunk that does not fit inline goes to a second, boxed tier instead of
//! being declined. Short chunks keep the allocation-free path; long ones pay
//! one allocation on insert, which is what they cost before any of this.

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

/// Slots in the boxed tier, as a fraction of the inline tier's.
///
/// Long chunks are the minority in mixed text but the whole of it in CJK, so
/// this is a compromise rather than a measurement — sized to hold a realistic
/// working set of long chunks without doubling the cache's memory.
const LONG_SHARE: usize = 4;

/// One slot: the chunk and its ids, stored in place.
///
/// `klen == EMPTY` marks a slot never written. No pre-tokenizer produces an
/// empty chunk, so the sentinel cannot collide with a real entry.
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
/// Vacant slots hold an empty key, which allocates nothing.
#[derive(Clone, Default)]
struct LongSlot {
    key: Box<[u8]>,
    ids: Box<[u32]>,
}

/// One shard's two tiers, behind a single lock so a chunk takes one lock
/// whichever tier it belongs to.
struct Tiers {
    inline: Box<[Slot]>,
    long: Box<[LongSlot]>,
}

type Shard = RwLock<Tiers>;

/// A chunk → token-ids cache striped across [`SHARDS`] independently-locked
/// slabs, each direct-mapped.
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
    /// Inline slots per shard, always a power of two so the index is a mask.
    mask: usize,
    /// The same for the boxed tier.
    long_mask: usize,
}

impl ChunkCache {
    /// Build a cache holding up to `capacity` chunks in total.
    pub(crate) fn new(capacity: usize) -> Self {
        let per_shard = (capacity / SHARDS).next_power_of_two().max(1);
        let long_per_shard = (per_shard / LONG_SHARE).next_power_of_two().max(1);
        Self {
            shards: (0..SHARDS)
                .map(|_| {
                    RwLock::new(Tiers {
                        inline: vec![VACANT; per_shard].into_boxed_slice(),
                        long: vec![LongSlot::default(); long_per_shard].into_boxed_slice(),
                    })
                })
                .collect(),
            mask: per_shard - 1,
            long_mask: long_per_shard - 1,
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

    /// The shard a hash belongs to, and the slot within it.
    ///
    /// The high bits pick the shard and the low bits the slot, so the two
    /// selections cannot correlate — `FxHasher`'s low bits are its weakest, and
    /// reusing them for both would cluster every shard's traffic into the same
    /// few slots.
    #[inline]
    fn locate(&self, hash: u64, key_len: usize) -> (&Shard, usize) {
        let mask = if key_len > MAX_KEY {
            self.long_mask
        } else {
            self.mask
        };
        (
            &self.shards[(hash >> 32) as usize % SHARDS],
            hash as usize & mask,
        )
    }

    /// Append the ids cached for `key` to `out`, reporting whether there were
    /// any.
    ///
    /// A read lock, so concurrent hits do not serialise, and the ids are copied
    /// out from the slot itself — no pointer to follow.
    pub(crate) fn extend_into(&self, hash: u64, key: &[u8], out: &mut Vec<u32>) -> bool {
        let (shard, index) = self.locate(hash, key.len());
        let Ok(tiers) = shard.read() else {
            return false;
        };
        if key.len() > MAX_KEY {
            let slot = &tiers.long[index];
            if &*slot.key != key {
                return false;
            }
            out.extend_from_slice(&slot.ids);
            return true;
        }
        let slot = &tiers.inline[index];
        if slot.klen as usize != key.len() || &slot.key[..key.len()] != key {
            return false;
        }
        out.extend_from_slice(&slot.ids[..slot.nids as usize]);
        true
    }

    /// Cache `ids` for `key`, whose shard hash the caller already has.
    ///
    /// A chunk longer than an inline slot goes to the boxed tier, which costs
    /// one allocation for the key and one for the ids. The inline tier's id
    /// count needs no check: see [`MAX_IDS`].
    pub(crate) fn put(&self, hash: u64, key: &[u8], ids: &[u32]) {
        if key.is_empty() {
            return;
        }
        let (shard, index) = self.locate(hash, key.len());
        let Ok(mut tiers) = shard.write() else {
            return;
        };
        if key.len() > MAX_KEY {
            tiers.long[index] = LongSlot {
                key: key.into(),
                ids: ids.into(),
            };
            return;
        }
        debug_assert!(ids.len() <= MAX_IDS, "ids outnumber the chunk's bytes");
        let slot = &mut tiers.inline[index];
        slot.key[..key.len()].copy_from_slice(key);
        slot.ids[..ids.len()].copy_from_slice(ids);
        slot.klen = key.len() as u8;
        slot.nids = ids.len() as u8;
    }

    /// Drop every entry.
    pub(crate) fn clear(&self) {
        for shard in &self.shards {
            if let Ok(mut tiers) = shard.write() {
                tiers.inline.fill(VACANT);
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
                        tiers.inline.iter().filter(|s| s.klen != EMPTY).count()
                            + tiers.long.iter().filter(|s| !s.key.is_empty()).count()
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
