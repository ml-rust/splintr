//! The pre-token chunk cache.
//!
//! Encoding the same chunk twice is common — pre-tokenizers cut text at word
//! boundaries and ordinary prose reuses words heavily — so memoizing chunk →
//! ids is a large win on repetitive input.
//!
//! The subtlety is that this cache sits on the one path `encode_batch` runs
//! concurrently. A single mutex around it makes every rayon worker queue on the
//! same lock twice per chunk, which does not merely reduce the speedup, it
//! inverts it: measured on a 24-core machine over cache-missing text, a
//! single-mutex cache made `encode_batch` *slower* than encoding the same texts
//! one at a time, and adding threads past four made it worse. Sharding the
//! lock, so concurrent chunks almost never contend, took the same workload from
//! 17.7ms to 1.94ms at 16 threads.
//!
//! So the shards are not a micro-optimization; they are what makes the parallel
//! path parallel.

use lru::LruCache;
use rustc_hash::FxHasher;
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::Mutex;

/// Number of independently-locked shards.
///
/// Fixed rather than derived from `available_parallelism()` so a tokenizer
/// built on one machine behaves identically on another, and so the eviction
/// behaviour a caller measures does not change under them when the process
/// moves to a different core count. 64 is comfortably above the thread counts
/// this crate is used at, which is what keeps the collision probability — and
/// therefore the contention — low.
const SHARDS: usize = 64;

type Shard = Mutex<LruCache<Vec<u8>, Vec<u32>, BuildHasherDefault<FxHasher>>>;

/// A chunk → token-ids cache striped across [`SHARDS`] independently-locked
/// LRUs.
///
/// Capacity is divided evenly across the shards, so eviction is per-shard
/// rather than global: the cache holds up to the requested number of entries,
/// but a pathologically skewed key distribution could evict from a hot shard
/// while a cold one has room. That is the standard trade for striping, and the
/// distribution here is a hash of the chunk bytes, so skew of that kind does
/// not arise in practice.
///
/// Keyed by the chunk bytes themselves (not a bare hash) so a hash collision
/// cannot return another chunk's token ids — the `lru` crate hashes AND
/// compares the key, making a wrong-chunk hit structurally impossible. FxHash
/// stays as the hasher for throughput on this hot path.
pub(crate) struct ChunkCache {
    shards: Box<[Shard]>,
}

impl ChunkCache {
    /// Build a cache holding up to `capacity` entries in total.
    ///
    /// A capacity below one entry per shard still yields one per shard: `lru`
    /// requires a non-zero capacity, and rounding down to zero would silently
    /// turn the cache off rather than make it small.
    pub(crate) fn new(capacity: usize) -> Self {
        let per_shard = NonZeroUsize::new(capacity / SHARDS).unwrap_or(NonZeroUsize::MIN);
        Self {
            shards: (0..SHARDS)
                .map(|_| {
                    Mutex::new(LruCache::with_hasher(
                        per_shard,
                        BuildHasherDefault::default(),
                    ))
                })
                .collect(),
        }
    }

    /// The shard a key belongs to.
    ///
    /// The high bits are used because `FxHasher`'s low bits are the weakest
    /// part of its output, and `SHARDS` is a power of two so only a few bits
    /// select the shard.
    fn shard(&self, key: &[u8]) -> &Shard {
        let mut hasher = FxHasher::default();
        key.hash(&mut hasher);
        &self.shards[(hasher.finish() >> 32) as usize % SHARDS]
    }

    /// Append the ids cached for `key` to `out`, reporting whether there were
    /// any.
    ///
    /// Appending rather than returning a `Vec` is what makes a cache hit
    /// allocation-free: the ids are copied straight into the caller's buffer,
    /// which already has the capacity, instead of into a fresh vector that the
    /// caller then copies again and drops. The copy happens under the shard
    /// lock, but it is a `memcpy` of a few token ids — shorter than the lookup
    /// that preceded it.
    ///
    /// The lookup itself allocates nothing either, borrowing via
    /// `Vec<u8>: Borrow<[u8]>`.
    pub(crate) fn extend_into(&self, key: &[u8], out: &mut Vec<u32>) -> bool {
        let Ok(mut shard) = self.shard(key).lock() else {
            return false;
        };
        match shard.get(key) {
            Some(ids) => {
                out.extend_from_slice(ids);
                true
            }
            None => false,
        }
    }

    /// Cache `ids` for `key`.
    pub(crate) fn put(&self, key: &[u8], ids: &[u32]) {
        if let Ok(mut shard) = self.shard(key).lock() {
            shard.put(key.to_vec(), ids.to_vec());
        }
    }

    /// Drop every entry.
    pub(crate) fn clear(&self) {
        for shard in &self.shards {
            if let Ok(mut shard) = shard.lock() {
                shard.clear();
            }
        }
    }

    /// Total entries currently held, across all shards.
    pub(crate) fn len(&self) -> usize {
        self.shards
            .iter()
            .map(|shard| shard.lock().map(|s| s.len()).unwrap_or(0))
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
