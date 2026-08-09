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
//! memory access to read. A chunk too long for a slot is not cached and simply
//! recomputes.

use rustc_hash::FxHasher;
use std::hash::{Hash, Hasher};
use std::sync::RwLock;

/// Number of independently-locked shards.
///
/// Fixed rather than derived from the core count, so a tokenizer behaves the
/// same on every machine. It sits comfortably above the thread counts this
/// crate is used at, which is what keeps contention low.
const SHARDS: usize = 64;

/// Longest chunk a slot can hold. Anything longer is not cached.
const MAX_KEY: usize = 16;

/// Most ids a slot can hold.
///
/// Equal to [`MAX_KEY`] on purpose: every id consumes at least one byte of the
/// chunk, so a key that fits can never produce a result that does not. Sizing
/// this to the *common* id count instead would decline the long results —
/// precisely the chunks that took the most merges to produce.
const MAX_IDS: usize = MAX_KEY;

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

type Shard = RwLock<Box<[Slot]>>;

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
    /// Slots per shard, always a power of two so the index is a mask.
    mask: usize,
}

impl ChunkCache {
    /// Build a cache holding up to `capacity` chunks in total.
    pub(crate) fn new(capacity: usize) -> Self {
        let per_shard = (capacity / SHARDS).next_power_of_two().max(1);
        Self {
            shards: (0..SHARDS)
                .map(|_| RwLock::new(vec![VACANT; per_shard].into_boxed_slice()))
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

    /// The shard a hash belongs to, and the slot within it.
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
        if key.len() > MAX_KEY {
            return false;
        }
        let (shard, index) = self.locate(hash);
        let Ok(slab) = shard.read() else {
            return false;
        };
        let slot = &slab[index];
        if slot.klen as usize != key.len() || &slot.key[..key.len()] != key {
            return false;
        }
        out.extend_from_slice(&slot.ids[..slot.nids as usize]);
        true
    }

    /// Cache `ids` for `key`, whose shard hash the caller already has.
    ///
    /// A chunk longer than a slot is declined rather than stored elsewhere,
    /// which costs only a recomputation. The id count needs no such check: see
    /// [`MAX_IDS`].
    pub(crate) fn put(&self, hash: u64, key: &[u8], ids: &[u32]) {
        if key.len() > MAX_KEY || key.is_empty() {
            return;
        }
        debug_assert!(ids.len() <= MAX_IDS, "ids outnumber the chunk's bytes");
        let (shard, index) = self.locate(hash);
        if let Ok(mut slab) = shard.write() {
            let slot = &mut slab[index];
            slot.key[..key.len()].copy_from_slice(key);
            slot.ids[..ids.len()].copy_from_slice(ids);
            slot.klen = key.len() as u8;
            slot.nids = ids.len() as u8;
        }
    }

    /// Drop every entry.
    pub(crate) fn clear(&self) {
        for shard in &self.shards {
            if let Ok(mut slab) = shard.write() {
                slab.fill(VACANT);
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
                    .map(|slab| slab.iter().filter(|s| s.klen != EMPTY).count())
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
