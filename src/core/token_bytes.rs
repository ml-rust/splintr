//! Token bytes that are either borrowed from an embedded vocabulary or owned.
//!
//! # Why this is not `Vec<u8>`
//!
//! A vocabulary is 100k-200k tokens averaging ~7 bytes. Keying its maps with
//! `Vec<u8>` means that many heap allocations and that many `memcpy`s, twice
//! over — once building the encoder, once building the decoder — and on
//! `cl100k_base` those two steps are over 90% of `from_vocab`.
//!
//! The bundled vocabularies do not need any of it. `pretrained.rs` embeds them
//! with `include_bytes!`, so their token bytes are already laid out
//! contiguously in `&'static [u8]` that outlives every tokenizer. A key can
//! point at them instead of copying them, which is what [`TokenBytes::Static`]
//! is for. Vocabularies read at runtime — a `tokenizer.json`, a GGUF file, a
//! `.tiktoken` path — have no such guarantee and take [`TokenBytes::Owned`].
//!
//! # Why not `Cow<'static, [u8]>`
//!
//! `Cow` is the same shape but does not implement `Borrow<[u8]>`, so
//! `map.get(bytes)` would not compile against it and every lookup would have to
//! construct a `Cow` first. That matters because those lookups are the encode
//! hot path, not the load path. The whole point of a custom type here is the
//! `Borrow` impl below: **it keeps `.get(&[u8])` exactly as it was**, so
//! nothing on the hot path changes shape.
//!
//! # The invariant that makes `Borrow` sound
//!
//! `Borrow<[u8]>` requires that `TokenBytes` hash and compare *identically* to
//! the `[u8]` it borrows to — otherwise a key inserted as `TokenBytes` could
//! not be found by its slice. Both impls below delegate to the slice for
//! exactly this reason, and neither may be replaced with a derive: a derived
//! `Hash` would mix in the enum discriminant, and the same bytes stored
//! `Static` in one vocabulary and `Owned` in another would hash differently.

use std::borrow::Borrow;
use std::hash::{Hash, Hasher};
use std::ops::Deref;

use rustc_hash::FxHashMap;

/// The bytes of one vocabulary token.
#[derive(Debug, Clone, Eq)]
pub enum TokenBytes {
    /// Borrowed from a vocabulary embedded in the binary. No allocation.
    Static(&'static [u8]),
    /// Owned, for a vocabulary read at runtime.
    Owned(Box<[u8]>),
}

impl TokenBytes {
    /// The bytes, whichever way they are held.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        match self {
            TokenBytes::Static(b) => b,
            TokenBytes::Owned(b) => b,
        }
    }
}

impl Deref for TokenBytes {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

/// Delegates to the slice. See the module docs: this is what lets a map keyed
/// by `TokenBytes` be queried with a plain `&[u8]`.
impl Borrow<[u8]> for TokenBytes {
    #[inline]
    fn borrow(&self) -> &[u8] {
        self.as_slice()
    }
}

/// Hashes the bytes and nothing else — **not** derived, so that `Static` and
/// `Owned` holding the same bytes hash alike, and alike to the bare slice.
impl Hash for TokenBytes {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state);
    }
}

/// Compares the bytes and nothing else, for the same reason as [`Hash`].
impl PartialEq for TokenBytes {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl PartialEq<[u8]> for TokenBytes {
    #[inline]
    fn eq(&self, other: &[u8]) -> bool {
        self.as_slice() == other
    }
}

impl From<Vec<u8>> for TokenBytes {
    /// Takes ownership without copying: `into_boxed_slice` only reallocates
    /// when the vector has spare capacity, and the vocabulary loaders build
    /// each token at its exact length.
    #[inline]
    fn from(bytes: Vec<u8>) -> Self {
        TokenBytes::Owned(bytes.into_boxed_slice())
    }
}

impl From<&'static [u8]> for TokenBytes {
    #[inline]
    fn from(bytes: &'static [u8]) -> Self {
        TokenBytes::Static(bytes)
    }
}

impl AsRef<[u8]> for TokenBytes {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

/// Vocabulary as token bytes → id.
pub type Encoder = FxHashMap<TokenBytes, u32>;

/// Vocabulary as id → token bytes.
///
/// Not a map: see [`DecodeTable`](crate::core::DecodeTable).
pub type Decoder = crate::core::decode_table::DecodeTable;

/// Convert an owned vocabulary map into the internal representation.
///
/// Every key moves rather than copies, so this costs one pass over the map and
/// no allocation per token. It is what the public constructors — which still
/// speak `FxHashMap<Vec<u8>, u32>` — hand to the internal ones.
pub fn encoder_from_owned(map: FxHashMap<Vec<u8>, u32>) -> Encoder {
    map.into_iter()
        .map(|(bytes, id)| (TokenBytes::from(bytes), id))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;

    fn hash_of<T: Hash + ?Sized>(value: &T) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    /// The `Borrow` contract: a key must hash the same as what it borrows to,
    /// or a map keyed by `TokenBytes` cannot be queried with `&[u8]`.
    #[test]
    fn hashes_identically_to_the_bare_slice() {
        let bytes: &'static [u8] = b"hello";
        assert_eq!(hash_of(&TokenBytes::Static(bytes)), hash_of(bytes));
        assert_eq!(
            hash_of(&TokenBytes::Owned(bytes.to_vec().into_boxed_slice())),
            hash_of(bytes)
        );
    }

    /// The two variants are interchangeable as keys. If this ever failed, a
    /// bundled vocabulary and a file-loaded one would disagree about the same
    /// token.
    #[test]
    fn static_and_owned_are_the_same_key() {
        let static_key = TokenBytes::Static(b"token");
        let owned_key = TokenBytes::from(b"token".to_vec());
        assert_eq!(static_key, owned_key);
        assert_eq!(hash_of(&static_key), hash_of(&owned_key));

        let mut map: Encoder = Encoder::default();
        map.insert(static_key, 7);
        assert_eq!(map.get(&owned_key), Some(&7));
        assert_eq!(
            map.get(b"token".as_slice()),
            Some(&7),
            "slice lookup failed"
        );
    }

    #[test]
    fn lookup_by_slice_finds_both_variants() {
        let mut map: Encoder = Encoder::default();
        map.insert(TokenBytes::Static(b"a"), 1);
        map.insert(TokenBytes::from(b"b".to_vec()), 2);
        assert_eq!(map.get(b"a".as_slice()), Some(&1));
        assert_eq!(map.get(b"b".as_slice()), Some(&2));
        assert_eq!(map.get(b"c".as_slice()), None);
    }

    #[test]
    fn the_empty_token_is_a_usable_key() {
        let mut map: Encoder = Encoder::default();
        map.insert(TokenBytes::Static(b""), 50256);
        assert_eq!(map.get(b"".as_slice()), Some(&50256));
    }

    #[test]
    fn owned_conversion_preserves_bytes() {
        let key = TokenBytes::from(vec![0xE2, 0x96, 0x81]);
        assert_eq!(key.as_slice(), &[0xE2, 0x96, 0x81]);
        assert_eq!(&*key, &[0xE2, 0x96, 0x81]);
    }
}
