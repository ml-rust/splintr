//! The token bytes → id table used to encode.
//!
//! # Why this is not a `HashMap<Vec<u8>, u32>`
//!
//! A vocabulary is 100k-200k tokens averaging ~7 bytes, and a map keyed by an
//! owned byte string pays one allocation per token to hold them — twice over,
//! once here and once wherever else the same bytes are needed. Loading a
//! `tokenizer.json` spent roughly three allocations per token on nothing but
//! key ownership.
//!
//! The bytes are all known at load and none of them ever change, so they go
//! into one buffer and a slot addresses a span of it. A whole vocabulary is a
//! handful of allocations rather than a per-token stream of them.
//!
//! # Why the slot carries its span
//!
//! A slot holds `(offset, len, id)` rather than an index into a side table of
//! spans. Both were measured against the map, replaying a recorded probe trace
//! of real pre-tokens:
//!
//! | | instructions | cache misses |
//! | --- | ---: | ---: |
//! | `HashMap<TokenBytes, u32>` | 341.8 M | 2.0-2.3 M |
//! | slot carries the span | 324.9 M | 2.0-3.3 M |
//! | slot carries an index into spans | 349.8 M | 4.1-4.4 M |
//!
//! The side-table variant loses on both counts: it costs a second dependent
//! load where this one costs the same single chase the map's key pointer did.
//!
//! The table is deliberately kept at half load. Tightening it to 0.76 — which
//! looks like free density — measured **+15.7% instructions**, because linear
//! probing degrades faster than the smaller footprint repays.

use std::hash::Hasher;

use rustc_hash::{FxHashMap, FxHasher};

/// One entry: where its key lives in the arena, and the id it maps to.
#[derive(Clone, Copy)]
struct Entry {
    offset: u32,
    len: u32,
    id: u32,
}

/// A slot holding no entry. `len` is the marker because a real key may sit at
/// offset 0 and may be empty — the empty token is a legitimate vocabulary
/// entry.
const VACANT: Entry = Entry {
    offset: 0,
    len: u32::MAX,
    id: 0,
};

/// Vocabulary as token bytes → id.
#[derive(Clone)]
pub struct Encoder {
    /// Every key's bytes, concatenated in insertion order.
    arena: Vec<u8>,
    /// Open-addressed, always a power of two so `len - 1` is the index mask.
    /// Boxed rather than a `Vec`: it is exactly sized and never grows in place.
    slots: Box<[Entry]>,
    len: usize,
}

impl Default for Encoder {
    fn default() -> Self {
        Self::with_capacity(0)
    }
}

/// Smallest slot count. Keeps `mask` meaningful for an empty table.
const MIN_SLOTS: usize = 16;

impl Encoder {
    /// A table sized to hold `tokens` entries without growing.
    pub fn with_capacity(tokens: usize) -> Self {
        let slots = (tokens * 2).next_power_of_two().max(MIN_SLOTS);
        Self {
            arena: Vec::new(),
            slots: vec![VACANT; slots].into_boxed_slice(),
            len: 0,
        }
    }

    /// A table over an arena that is already filled.
    ///
    /// For a vocabulary that arrives as one contiguous buffer — the packed
    /// `.splv` payload — every token's bytes are already laid out inside it, so
    /// the arena is that buffer and [`Self::insert_span`] records where each
    /// token sits rather than copying it out. One bulk copy replaces 100k-200k
    /// small ones. The buffer's framing bytes ride along unused, which is a few
    /// hundred KB against not copying the vocabulary token by token.
    pub fn with_arena(arena: Vec<u8>, tokens: usize) -> Self {
        let slots = (tokens * 2).next_power_of_two().max(MIN_SLOTS);
        Self {
            arena,
            slots: vec![VACANT; slots].into_boxed_slice(),
            len: 0,
        }
    }

    /// Map the arena bytes at `offset..offset + len` to `id`.
    ///
    /// For a key already inside the arena — see [`Self::with_arena`]. Keeps the
    /// FIRST id when a byte sequence repeats, as vocabulary files require.
    ///
    /// # Panics
    ///
    /// If the span runs past the arena.
    pub fn insert_span(&mut self, offset: u32, len: u32, id: u32) {
        assert!(
            offset as usize + len as usize <= self.arena.len(),
            "span runs past the arena"
        );
        if self.len * 2 >= self.slots.len() {
            self.resize(self.slots.len() * 2);
        }
        let mask = self.mask();
        let key_start = offset as usize;
        let key_end = key_start + len as usize;
        let mut slot = Self::slot_of(Self::hash(&self.arena[key_start..key_end]), mask);
        loop {
            let entry = self.slots[slot];
            if entry.len == u32::MAX {
                break;
            }
            if entry.len == len
                && Self::key_at(&self.arena, &entry) == &self.arena[key_start..key_end]
            {
                return;
            }
            slot = (slot + 1) & mask;
        }
        self.slots[slot] = Entry { offset, len, id };
        self.len += 1;
    }

    /// Make room for `additional` more entries.
    pub fn reserve(&mut self, additional: usize) {
        let wanted = (self.len + additional) * 2;
        if wanted > self.slots.len() {
            self.resize(wanted.next_power_of_two().max(MIN_SLOTS));
        }
        // ~7 bytes per token, so this is the right order without being exact;
        // the arena grows itself if a vocabulary runs long.
        self.arena.reserve(additional * 8);
    }

    /// `Hasher::write` directly, not `key.hash(..)`.
    ///
    /// `Hash for [u8]` writes a length prefix before the bytes, which costs a
    /// whole extra round of the hasher on every lookup and buys nothing here:
    /// an entry's length is compared separately, so two keys of different
    /// lengths are already told apart without it.
    #[inline]
    pub fn hash_of(key: &[u8]) -> u64 {
        Self::hash(key)
    }

    #[inline]
    fn hash(key: &[u8]) -> u64 {
        let mut hasher = FxHasher::default();
        hasher.write(key);
        hasher.finish()
    }

    /// Slots are a power of two, so the mask is the index of the last one.
    #[inline]
    fn mask(&self) -> usize {
        self.slots.len() - 1
    }

    /// The multiply spreads the key's entropy into the bits the mask selects.
    #[inline]
    fn slot_of(hash: u64, mask: usize) -> usize {
        (hash.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32) as usize & mask
    }

    /// Whether two keys of equal length hold the same bytes.
    ///
    /// `==` on slices of unknown length is a `memcmp` call, and a vocabulary
    /// key is a word: measured on `deepseek-v4`, that call is 8% of an encode's
    /// instructions on its own. Dispatching on the length first gives the
    /// comparison a *constant* size, which compiles to one word compare and no
    /// call at all; only a longer key is worth the call it used to make.
    ///
    /// A scalar byte loop was tried instead and is worse than the call — the
    /// win here is the constant length, not the avoidance of vectors.
    #[inline]
    fn same_bytes(a: &[u8], b: &[u8]) -> bool {
        match a.len() {
            0 => true,
            1 => a[..1] == b[..1],
            2 => a[..2] == b[..2],
            3 => a[..3] == b[..3],
            4 => a[..4] == b[..4],
            5 => a[..5] == b[..5],
            6 => a[..6] == b[..6],
            7 => a[..7] == b[..7],
            8 => a[..8] == b[..8],
            9 => a[..9] == b[..9],
            10 => a[..10] == b[..10],
            11 => a[..11] == b[..11],
            12 => a[..12] == b[..12],
            13 => a[..13] == b[..13],
            14 => a[..14] == b[..14],
            15 => a[..15] == b[..15],
            16 => a[..16] == b[..16],
            _ => a == b,
        }
    }

    #[inline]
    fn key_at<'a>(arena: &'a [u8], entry: &Entry) -> &'a [u8] {
        let start = entry.offset as usize;
        &arena[start..start + entry.len as usize]
    }

    /// The id `key` maps to, if the vocabulary holds it.
    #[inline]
    pub fn get(&self, key: &[u8]) -> Option<u32> {
        self.get_with_hash(key, Self::hash_of(key))
    }

    /// [`Encoder::get`] for a caller that has already hashed `key`.
    ///
    /// The encode path asks the vocabulary and then the chunk cache about the
    /// very same bytes, and both are keyed on the same hash of them — so it is
    /// computed once and handed to each. `hash` must be [`Encoder::hash_of`] of
    /// `key`; anything else answers about a different key.
    #[inline]
    pub fn get_with_hash(&self, key: &[u8], hash: u64) -> Option<u32> {
        let mask = self.mask();
        let mut slot = Self::slot_of(hash, mask);
        loop {
            let entry = self.slots[slot];
            if entry.len == u32::MAX {
                return None;
            }
            if entry.len as usize == key.len()
                && Self::same_bytes(Self::key_at(&self.arena, &entry), key)
            {
                return Some(entry.id);
            }
            slot = (slot + 1) & mask;
        }
    }

    /// Whether the vocabulary holds `key`.
    #[inline]
    pub fn contains_key(&self, key: &[u8]) -> bool {
        self.get(key).is_some()
    }

    /// Map `key` to `id`, replacing any id it already had and reporting it.
    ///
    /// A replaced key keeps its original arena bytes, so re-inserting does not
    /// grow the arena.
    pub fn insert(&mut self, key: &[u8], id: u32) -> Option<u32> {
        if self.len * 2 >= self.slots.len() {
            self.resize(self.slots.len() * 2);
        }
        let mask = self.mask();
        let mut slot = Self::slot_of(Self::hash(key), mask);
        loop {
            let entry = self.slots[slot];
            if entry.len == u32::MAX {
                break;
            }
            if entry.len as usize == key.len() && Self::key_at(&self.arena, &entry) == key {
                let previous = entry.id;
                self.slots[slot].id = id;
                return Some(previous);
            }
            slot = (slot + 1) & mask;
        }

        let offset = self.arena.len() as u32;
        self.arena.extend_from_slice(key);
        self.slots[slot] = Entry {
            offset,
            len: key.len() as u32,
            id,
        };
        self.len += 1;
        None
    }

    /// Map `key` to `id` only if it is not already present, and report the id
    /// it ends up with.
    ///
    /// Loading a vocabulary that lists a byte sequence twice must keep the
    /// FIRST id, so this is not `insert`.
    pub fn insert_if_absent(&mut self, key: &[u8], id: u32) -> u32 {
        match self.get(key) {
            Some(existing) => existing,
            None => {
                self.insert(key, id);
                id
            }
        }
    }

    fn resize(&mut self, slots: usize) {
        let mask = slots - 1;
        let mut fresh = vec![VACANT; slots];
        for entry in self.slots.iter().filter(|e| e.len != u32::MAX) {
            let key = Self::key_at(&self.arena, entry);
            let mut slot = Self::slot_of(Self::hash(key), mask);
            while fresh[slot].len != u32::MAX {
                slot = (slot + 1) & mask;
            }
            fresh[slot] = *entry;
        }
        self.slots = fresh.into_boxed_slice();
    }

    /// How many tokens the vocabulary holds.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Every token and its id, in no particular order.
    pub fn iter(&self) -> impl Iterator<Item = (&[u8], u32)> + '_ {
        self.slots
            .iter()
            .filter(|entry| entry.len != u32::MAX)
            .map(move |entry| (Self::key_at(&self.arena, entry), entry.id))
    }

    /// Every token's bytes, in no particular order.
    pub fn keys(&self) -> impl Iterator<Item = &[u8]> + '_ {
        self.iter().map(|(key, _)| key)
    }

    /// Every id, in no particular order.
    pub fn values(&self) -> impl Iterator<Item = u32> + '_ {
        self.iter().map(|(_, id)| id)
    }
}

impl<'a> IntoIterator for &'a Encoder {
    type Item = (&'a [u8], u32);
    type IntoIter = Box<dyn Iterator<Item = (&'a [u8], u32)> + 'a>;

    fn into_iter(self) -> Self::IntoIter {
        Box::new(self.iter())
    }
}

impl FromIterator<(Vec<u8>, u32)> for Encoder {
    fn from_iter<I: IntoIterator<Item = (Vec<u8>, u32)>>(entries: I) -> Self {
        let entries = entries.into_iter();
        let mut encoder = Self::with_capacity(entries.size_hint().0);
        for (key, id) in entries {
            encoder.insert(&key, id);
        }
        encoder
    }
}

impl<'a> FromIterator<(&'a [u8], u32)> for Encoder {
    fn from_iter<I: IntoIterator<Item = (&'a [u8], u32)>>(entries: I) -> Self {
        let entries = entries.into_iter();
        let mut encoder = Self::with_capacity(entries.size_hint().0);
        for (key, id) in entries {
            encoder.insert(key, id);
        }
        encoder
    }
}

impl std::fmt::Debug for Encoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Encoder")
            .field("len", &self.len)
            .field("bytes", &self.arena.len())
            .finish()
    }
}

/// Convert an owned vocabulary map into the internal representation.
///
/// What the public constructors — which speak `FxHashMap<Vec<u8>, u32>` — hand
/// to the internal ones.
pub fn encoder_from_owned(map: FxHashMap<Vec<u8>, u32>) -> Encoder {
    let mut encoder = Encoder::with_capacity(map.len());
    for (key, id) in map {
        encoder.insert(&key, id);
    }
    encoder
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_what_it_was_given() {
        let mut encoder = Encoder::default();
        encoder.insert(b"hello", 1);
        encoder.insert(b"world", 2);
        assert_eq!(encoder.get(b"hello"), Some(1));
        assert_eq!(encoder.get(b"world"), Some(2));
        assert_eq!(encoder.get(b"absent"), None);
        assert_eq!(encoder.len(), 2);
    }

    /// The empty token is a real vocabulary entry, and the vacancy marker must
    /// not swallow it.
    #[test]
    fn the_empty_token_is_a_usable_key() {
        let mut encoder = Encoder::default();
        encoder.insert(b"", 50256);
        assert_eq!(encoder.get(b""), Some(50256));
    }

    /// A key stored at arena offset 0 must not read as vacant either.
    #[test]
    fn the_first_key_inserted_is_found() {
        let mut encoder = Encoder::default();
        encoder.insert(b"first", 7);
        for i in 0..500u32 {
            encoder.insert(format!("filler{i}").as_bytes(), i + 100);
        }
        assert_eq!(encoder.get(b"first"), Some(7));
    }

    #[test]
    fn growing_preserves_every_entry() {
        let mut encoder = Encoder::with_capacity(4);
        for i in 0..2000u32 {
            encoder.insert(format!("token{i}").as_bytes(), i);
        }
        assert_eq!(encoder.len(), 2000);
        for i in 0..2000u32 {
            assert_eq!(encoder.get(format!("token{i}").as_bytes()), Some(i));
        }
    }

    #[test]
    fn reinserting_replaces_the_id_without_growing() {
        let mut encoder = Encoder::default();
        encoder.insert(b"key", 1);
        let arena = encoder.arena.len();
        assert_eq!(encoder.insert(b"key", 2), Some(1));
        assert_eq!(encoder.get(b"key"), Some(2));
        assert_eq!(encoder.len(), 1);
        assert_eq!(encoder.arena.len(), arena, "arena grew on replacement");
    }

    /// Vocabulary files list a byte sequence twice; the lowest id wins.
    #[test]
    fn insert_if_absent_keeps_the_first_id() {
        let mut encoder = Encoder::default();
        assert_eq!(encoder.insert_if_absent(b"key", 1), 1);
        assert_eq!(encoder.insert_if_absent(b"key", 9), 1);
        assert_eq!(encoder.get(b"key"), Some(1));
    }

    #[test]
    fn iteration_yields_every_entry_once() {
        let mut encoder = Encoder::default();
        for i in 0..100u32 {
            encoder.insert(format!("t{i}").as_bytes(), i);
        }
        let mut seen: Vec<(Vec<u8>, u32)> = encoder.iter().map(|(k, v)| (k.to_vec(), v)).collect();
        seen.sort();
        assert_eq!(seen.len(), 100);
        assert_eq!(encoder.keys().count(), 100);
        assert_eq!(encoder.values().sum::<u32>(), (0..100).sum::<u32>());
    }

    /// Keys that collide in the table must stay distinguishable.
    #[test]
    fn distinct_keys_of_equal_length_do_not_alias() {
        let mut encoder = Encoder::default();
        for i in 0..1000u32 {
            encoder.insert(&i.to_le_bytes(), i);
        }
        for i in 0..1000u32 {
            assert_eq!(encoder.get(&i.to_le_bytes()), Some(i));
        }
    }
}
