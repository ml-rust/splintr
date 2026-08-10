//! The id → bytes table used to decode.
//!
//! # Why this is not a map
//!
//! Decoding asks one question — what bytes does this id stand for — of a key
//! space that is already dense: a vocabulary numbers its tokens `0..n`. A
//! `HashMap<u32, _>` hashes an integer to find a slot it could have indexed,
//! and pays a per-token allocation to own the bytes it stores.
//!
//! Both costs are avoidable together. The bytes go into one contiguous buffer
//! and each id indexes a span of it, so the table costs two allocations for a
//! whole vocabulary rather than one per token, and a lookup is an index.
//!
//! Building it from an encoder used to be where a runtime-loaded vocabulary was
//! copied a second time, once per token; it is still copied once, but into the
//! buffer rather than into 200k separate boxes.
//!
//! # Sparse ids
//!
//! Special tokens are often numbered far above the base vocabulary, and nothing
//! stops a `tokenizer.json` from naming an id near `u32::MAX`. Indexing by id
//! alone would then size the span table by the largest id rather than by the
//! vocabulary, so ids past a limit set by the token count go to an overflow map
//! instead. Real vocabularies never reach it.

use rustc_hash::FxHashMap;

use crate::core::token_bytes::Encoder;

/// Where one token's bytes sit in the buffer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Span {
    offset: u32,
    len: u32,
}

impl Span {
    /// An id the vocabulary does not use. A real span may have `len == 0` — the
    /// empty token is a legitimate vocabulary entry — so absence is marked on
    /// the offset instead.
    const ABSENT: Span = Span {
        offset: u32::MAX,
        len: 0,
    };

    #[inline]
    fn is_absent(self) -> bool {
        self.offset == u32::MAX
    }
}

/// Token bytes addressed by token id.
#[derive(Default, Clone)]
pub struct DecodeTable {
    bytes: Vec<u8>,
    /// Indexed by id, for ids below [`Self::dense_limit`].
    dense: Vec<Span>,
    /// Ids at or above it, which are not worth a dense slot each.
    sparse: FxHashMap<u32, Span>,
    count: usize,
}

impl DecodeTable {
    /// Largest id given a dense slot, as a function of how many tokens the
    /// table holds. Generous enough that a vocabulary numbering its specials
    /// above its base tokens still lands entirely in the dense half.
    fn dense_limit(token_count: usize) -> u32 {
        (token_count.saturating_mul(4).saturating_add(4096)).min(u32::MAX as usize) as u32
    }

    /// Build the table from an encoder, copying each token's bytes once.
    pub fn from_encoder(encoder: &Encoder) -> Self {
        let total: usize = encoder.keys().map(|key| key.len()).sum();
        let limit = Self::dense_limit(encoder.len());
        let dense_len = encoder
            .values()
            .filter(|&&id| id < limit)
            .max()
            .map_or(0, |&id| id as usize + 1);

        let mut table = Self {
            bytes: Vec::with_capacity(total),
            dense: vec![Span::ABSENT; dense_len],
            sparse: FxHashMap::default(),
            count: 0,
        };
        for (key, &id) in encoder {
            table.insert(id, key);
        }
        table
    }

    /// Record `bytes` as the spelling of `id`, replacing any it already had.
    ///
    /// A replaced entry leaves its bytes in the buffer. Replacement only
    /// happens where a special token is layered over a vocabulary that already
    /// claimed its id, which is a handful of entries.
    pub fn insert(&mut self, id: u32, bytes: &[u8]) {
        let span = Span {
            offset: self.bytes.len() as u32,
            len: bytes.len() as u32,
        };
        self.bytes.extend_from_slice(bytes);

        let previous = if (id as usize) < self.dense.len() {
            std::mem::replace(&mut self.dense[id as usize], span)
        } else {
            self.sparse.insert(id, span).unwrap_or(Span::ABSENT)
        };
        if previous.is_absent() {
            self.count += 1;
        }
    }

    /// The bytes `id` stands for, if the vocabulary uses it.
    #[inline]
    pub fn get(&self, id: u32) -> Option<&[u8]> {
        let span = match self.dense.get(id as usize) {
            Some(&span) => span,
            None => *self.sparse.get(&id)?,
        };
        if span.is_absent() {
            return None;
        }
        let start = span.offset as usize;
        Some(&self.bytes[start..start + span.len as usize])
    }

    /// How many ids the table holds.
    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// The largest id the table holds, or `None` when it holds none.
    pub fn max_id(&self) -> Option<u32> {
        let dense = self
            .dense
            .iter()
            .rposition(|span| !span.is_absent())
            .map(|index| index as u32);
        self.sparse.keys().copied().max().max(dense)
    }

    /// Every id the table holds, with its bytes, in no particular order.
    pub fn iter(&self) -> impl Iterator<Item = (u32, &[u8])> + '_ {
        let dense = self
            .dense
            .iter()
            .enumerate()
            .filter(|(_, span)| !span.is_absent())
            .map(|(index, _)| index as u32);
        dense
            .chain(self.sparse.keys().copied())
            .filter_map(move |id| self.get(id).map(|bytes| (id, bytes)))
    }
}

impl std::fmt::Debug for DecodeTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecodeTable")
            .field("len", &self.count)
            .field("bytes", &self.bytes.len())
            .finish()
    }
}

impl FromIterator<(u32, Vec<u8>)> for DecodeTable {
    fn from_iter<I: IntoIterator<Item = (u32, Vec<u8>)>>(entries: I) -> Self {
        let mut table = Self::default();
        for (id, bytes) in entries {
            // Grow the dense half as ids arrive: a caller collecting into the
            // table has no count to size it from up front.
            if id as usize >= table.dense.len() && id < Self::dense_limit(table.count + 1) {
                table.dense.resize(id as usize + 1, Span::ABSENT);
            }
            table.insert(id, &bytes);
        }
        table
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::token_bytes::TokenBytes;

    fn encoder(entries: &[(&[u8], u32)]) -> Encoder {
        entries
            .iter()
            .map(|(bytes, id)| (TokenBytes::from(bytes.to_vec()), *id))
            .collect()
    }

    #[test]
    fn resolves_every_id_it_was_built_from() {
        let table = DecodeTable::from_encoder(&encoder(&[(b"hello", 0), (b"world", 1)]));
        assert_eq!(table.get(0), Some(&b"hello"[..]));
        assert_eq!(table.get(1), Some(&b"world"[..]));
        assert_eq!(table.get(2), None);
        assert_eq!(table.len(), 2);
    }

    /// The empty token is a real vocabulary entry, so `len == 0` must not read
    /// as an unused id.
    #[test]
    fn the_empty_token_is_present_not_absent() {
        let table = DecodeTable::from_encoder(&encoder(&[(b"", 7)]));
        assert_eq!(table.get(7), Some(&b""[..]));
        assert_eq!(table.get(6), None);
        assert_eq!(table.len(), 1);
    }

    /// An id far above the vocabulary must not size the dense table.
    #[test]
    fn a_distant_id_goes_to_the_overflow_map() {
        let table = DecodeTable::from_encoder(&encoder(&[(b"a", 0), (b"b", 4_000_000_000)]));
        assert_eq!(table.get(0), Some(&b"a"[..]));
        assert_eq!(table.get(4_000_000_000), Some(&b"b"[..]));
        assert!(table.dense.len() < 8192, "dense table sized by the outlier");
        assert_eq!(table.max_id(), Some(4_000_000_000));
    }

    #[test]
    fn a_later_insert_replaces_an_id_without_double_counting() {
        let mut table = DecodeTable::from_encoder(&encoder(&[(b"old", 3)]));
        table.insert(3, b"new");
        assert_eq!(table.get(3), Some(&b"new"[..]));
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn iteration_yields_dense_and_sparse_alike() {
        let table = DecodeTable::from_encoder(&encoder(&[(b"a", 1), (b"b", 3_000_000_000)]));
        let mut seen: Vec<(u32, Vec<u8>)> = table
            .iter()
            .map(|(id, bytes)| (id, bytes.to_vec()))
            .collect();
        seen.sort();
        assert_eq!(
            seen,
            vec![(1, b"a".to_vec()), (3_000_000_000, b"b".to_vec())]
        );
    }
}
