//! The artifact a trainer produces.

use rustc_hash::FxHashMap;

/// How a word's bytes are cut into the symbols BPE starts from.
///
/// The two settings are not stylistic — they follow from what the pre-tokenizer
/// handed over, and picking the wrong one produces a vocabulary that cannot
/// spell its own corpus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Seeding {
    /// One symbol per byte. The tiktoken-style shape: no `ByteLevel`
    /// pre-tokenizer stage, words are ordinary text, and the 256 byte values are
    /// the alphabet — so every possible input is spellable and no unknown token
    /// is needed.
    #[default]
    Bytes,
    /// One symbol per character. The HuggingFace byte-level shape: a `ByteLevel`
    /// stage has already mapped each raw byte to a printable code point, so a
    /// *character* is what a byte became. Cutting those by byte instead would
    /// split the multi-byte UTF-8 of `Ġ` down the middle and merge halves of a
    /// code point.
    Chars,
}

impl Seeding {
    /// Cut `word` into the byte spans that become its initial symbols.
    pub(crate) fn units(self, word: &[u8]) -> Vec<&[u8]> {
        match self {
            Seeding::Bytes => word.chunks(1).collect(),
            Seeding::Chars => match std::str::from_utf8(word) {
                Ok(text) => text
                    .char_indices()
                    .map(|(offset, ch)| &word[offset..offset + ch.len_utf8()])
                    .collect(),
                // A word that is not valid UTF-8 cannot be cut into characters;
                // falling back to bytes keeps it spellable rather than dropping
                // it. A `ByteLevel` pre-tokenizer never produces one.
                Err(_) => word.chunks(1).collect(),
            },
        }
    }
}

/// A trained vocabulary: the pieces, the merges that built them, and the special
/// tokens to sit above them.
///
/// Ids are assigned so that **an id is its own merge rank** — the seed alphabet
/// takes `0..alphabet_len`, then each merge takes the next id in merge order.
/// That is not a convenience: splintr's BPE engine ranks merges by the token a
/// merge *produces* rather than by the pair it joins (see `core::bpe::ranks`),
/// so a vocabulary in this order is one its encoder can consume directly, and a
/// `.tiktoken` rank file states it with nothing lost.
#[derive(Debug, Clone)]
pub struct TrainedVocab {
    pieces: Vec<Vec<u8>>,
    alphabet_len: usize,
    merges: Vec<(u32, u32)>,
    specials: Vec<String>,
    seeding: Seeding,
}

impl TrainedVocab {
    pub(crate) fn new(
        pieces: Vec<Vec<u8>>,
        alphabet_len: usize,
        merges: Vec<(u32, u32)>,
        specials: Vec<String>,
        seeding: Seeding,
    ) -> Self {
        Self {
            pieces,
            alphabet_len,
            merges,
            specials,
            seeding,
        }
    }

    /// Every piece, lowest id first. Index is the id.
    pub fn pieces(&self) -> &[Vec<u8>] {
        &self.pieces
    }

    /// The bytes of one id, if the vocabulary has it.
    pub fn piece(&self, id: u32) -> Option<&[u8]> {
        self.pieces.get(id as usize).map(Vec::as_slice)
    }

    /// How many pieces are seeds rather than merge results.
    pub fn alphabet_len(&self) -> usize {
        self.alphabet_len
    }

    /// The `(left, right)` piece ids each merge joined, in merge order. Merge
    /// `i` produced piece `alphabet_len + i`.
    pub fn merges(&self) -> &[(u32, u32)] {
        &self.merges
    }

    /// The special tokens, which are numbered above every piece.
    pub fn specials(&self) -> &[String] {
        &self.specials
    }

    pub fn seeding(&self) -> Seeding {
        self.seeding
    }

    /// Pieces plus specials.
    pub fn len(&self) -> usize {
        self.pieces.len() + self.specials.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty() && self.specials.is_empty()
    }

    /// The vocabulary as the `FxHashMap<Vec<u8>, u32>` splintr's `Tokenizer::new`
    /// takes, so a freshly trained vocabulary can be encoded with without going
    /// through a file at all.
    pub fn encoder(&self) -> FxHashMap<Vec<u8>, u32> {
        self.pieces
            .iter()
            .enumerate()
            .map(|(id, bytes)| (bytes.clone(), id as u32))
            .collect()
    }

    /// The special tokens as the `FxHashMap<String, u32>` splintr takes,
    /// numbered from the end of the piece list.
    pub fn special_encoder(&self) -> FxHashMap<String, u32> {
        let base = self.pieces.len() as u32;
        self.specials
            .iter()
            .enumerate()
            .map(|(i, token)| (token.clone(), base + i as u32))
            .collect()
    }
}
