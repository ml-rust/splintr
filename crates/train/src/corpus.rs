//! Turning text into the word counts a trainer consumes.
//!
//! A trainer never sees raw documents. It sees *words* — whatever the
//! normalizer and pre-tokenizer cut the text into — with how often each occurs,
//! because BPE only ever merges within a pre-token and identical words merge
//! identically.
//!
//! The normalizer and pre-tokenizer here are splintr's own, the same types
//! `Tokenizer::with_normalizer` and `Tokenizer::with_pre_tokenizer` take. That
//! is the point of training inside splintr rather than importing someone else's
//! output: the boundaries a vocabulary is trained on and the boundaries it is
//! later encoded against come from one implementation, so they cannot drift.

use std::hash::Hasher;

use rustc_hash::FxHasher;
use splintr::{Normalizer, PreTokStage, PreTokenizer, SplitBehavior, SplitPattern};

use crate::error::TrainError;

/// The pre-tokenizer shapes worth naming, so a caller does not have to assemble
/// a stage list to get an ordinary one.
///
/// [`PreTokenizer`] itself is still accepted by
/// [`Corpus::with_pre_tokenizer`] for anything these do not cover.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreTok {
    /// Split on whitespace and isolate punctuation. The word-level default.
    Whitespace,
    /// GPT-2 byte level: pieces come out already mapped into the byte-level
    /// alphabet, so train them with [`Seeding::Chars`](crate::Seeding::Chars).
    ByteLevel,
    /// Split with a regex, tiktoken-style — the expression matches the pieces
    /// themselves rather than the separators.
    Pattern(String),
    /// No splitting at all: a document is one word. Rarely wanted, since BPE
    /// will then merge straight across word boundaries.
    None,
}

impl PreTok {
    /// Compile to a [`PreTokenizer`], or `None` for [`PreTok::None`].
    ///
    /// # Errors
    /// [`TrainError::PreTokenizer`] if a [`PreTok::Pattern`] does not compile.
    pub fn build(&self) -> Result<Option<PreTokenizer>, TrainError> {
        let stages = match self {
            PreTok::None => return Ok(None),
            PreTok::Whitespace => vec![
                PreTokStage::WhitespaceSplit,
                PreTokStage::Punctuation {
                    behavior: SplitBehavior::Isolated,
                },
            ],
            PreTok::ByteLevel => vec![PreTokStage::ByteLevel {
                use_regex: true,
                add_prefix_space: false,
            }],
            PreTok::Pattern(pattern) => vec![PreTokStage::Split {
                pattern: SplitPattern::Regex(pattern.clone()),
                behavior: SplitBehavior::Isolated,
                // The expression names the pieces, so the spans *between*
                // matches are the separators.
                invert: true,
            }],
        };
        Ok(Some(PreTokenizer::new(stages)?))
    }
}

/// One distinct word: where its bytes end in the arena, and how often it
/// occurred.
///
/// Only the *end* is stored — words are appended in insertion order, so a
/// word's start is the previous word's end. That halves the per-word bookkeeping
/// and leaves no length ceiling.
#[derive(Debug, Clone, Copy)]
struct Entry {
    end: u64,
    count: u64,
}

/// An index slot: the top half of the hash, and the entry index plus one so
/// that an all-zero slot means empty.
const EMPTY: u64 = 0;

fn hash(word: &[u8]) -> u64 {
    let mut hasher = FxHasher::default();
    hasher.write(word);
    hasher.finish()
}

/// Words and their frequencies, the input every trainer takes.
///
/// Keyed by raw bytes rather than `String`: a pre-tokenizer with a `ByteLevel`
/// stage emits pieces already mapped into the byte-level alphabet, and one
/// without it emits ordinary text, but both are handed on as the bytes the
/// vocabulary will actually be keyed by. See [`Seeding`](crate::Seeding) for how
/// those bytes are then cut into initial symbols.
///
/// The words live end to end in one arena rather than in a `Vec<u8>` each. A
/// hash map keyed by owned bytes needs an allocation per *occurrence* to look a
/// word up at all, and keeps one per distinct word forever; on a gigabyte of
/// text that is tens of millions of allocations for a few hundred megabytes of
/// actual bytes, and the fragmentation costs more than the data. Here a repeat
/// occurrence allocates nothing and a new word appends.
#[derive(Debug, Clone)]
pub struct WordCounts {
    arena: Vec<u8>,
    entries: Vec<Entry>,
    /// Open-addressed, linear-probed, always a power of two.
    slots: Vec<u64>,
    total: u64,
}

impl WordCounts {
    pub fn new() -> Self {
        Self {
            arena: Vec::new(),
            entries: Vec::new(),
            slots: Vec::new(),
            total: 0,
        }
    }

    /// Preallocate for `words` distinct words, avoiding the rehashes that
    /// growing from empty would otherwise cost.
    pub fn with_capacity(words: usize) -> Self {
        let mut counts = Self::new();
        if words > 0 {
            counts.entries.reserve(words);
            counts.slots = vec![EMPTY; (words * 2).next_power_of_two().max(16)];
        }
        counts
    }

    /// Record one occurrence of `word`.
    pub fn add(&mut self, word: &[u8]) {
        self.add_n(word, 1);
    }

    /// Record `n` occurrences of `word`.
    pub fn add_n(&mut self, word: &[u8], n: u64) {
        if word.is_empty() {
            return;
        }
        // Keep the table under three-quarters full; past that, linear probing
        // starts clustering badly.
        if (self.entries.len() + 1) * 4 > self.slots.len() * 3 {
            self.grow();
        }

        let hash = hash(word);
        let tag = hash >> 32;
        let mask = self.slots.len() - 1;
        let mut probe = hash as usize & mask;
        loop {
            let slot = self.slots[probe];
            if slot == EMPTY {
                self.arena.extend_from_slice(word);
                self.entries.push(Entry {
                    end: self.arena.len() as u64,
                    count: n,
                });
                self.slots[probe] = tag << 32 | self.entries.len() as u64;
                self.total += n;
                return;
            }
            // Compare the stored hash half first: a mismatch here is settled
            // without touching the arena, which is the expensive read.
            if slot >> 32 == tag {
                let index = (slot as u32 - 1) as usize;
                if self.word_at(index) == word {
                    self.entries[index].count += n;
                    self.total += n;
                    return;
                }
            }
            probe = (probe + 1) & mask;
        }
    }

    /// The bytes of the `index`th distinct word.
    fn word_at(&self, index: usize) -> &[u8] {
        let start = if index == 0 {
            0
        } else {
            self.entries[index - 1].end as usize
        };
        &self.arena[start..self.entries[index].end as usize]
    }

    /// Double the index and reinsert. Hashes are recomputed from the arena
    /// rather than stored: a full hash per word would cost more memory than the
    /// occasional rehash costs time.
    fn grow(&mut self) {
        let capacity = (self.slots.len() * 2).max(16);
        let mask = capacity - 1;
        let mut slots = vec![EMPTY; capacity];
        let mut start = 0usize;
        for (index, entry) in self.entries.iter().enumerate() {
            let end = entry.end as usize;
            let hash = hash(&self.arena[start..end]);
            start = end;
            let mut probe = hash as usize & mask;
            while slots[probe] != EMPTY {
                probe = (probe + 1) & mask;
            }
            slots[probe] = (hash >> 32) << 32 | (index as u64 + 1);
        }
        self.slots = slots;
    }

    /// Fold another set of counts into this one.
    pub fn merge(&mut self, other: WordCounts) {
        for (word, count) in other.iter() {
            self.add_n(word, count);
        }
    }

    /// How many distinct words were seen. The trainer's cost is driven by this
    /// rather than by corpus size, so it is worth knowing before starting.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Total occurrences across all words.
    pub fn total(&self) -> u64 {
        self.total
    }

    /// How many bytes the counts occupy, arena and index together. Worth
    /// checking before a long run: this is the floor under a trainer's memory.
    pub fn memory_bytes(&self) -> usize {
        self.arena.capacity()
            + self.entries.capacity() * std::mem::size_of::<Entry>()
            + self.slots.capacity() * std::mem::size_of::<u64>()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&[u8], u64)> + '_ {
        let mut start = 0usize;
        self.entries.iter().map(move |entry| {
            let end = entry.end as usize;
            let word = &self.arena[start..end];
            start = end;
            (word, entry.count)
        })
    }
}

impl Default for WordCounts {
    fn default() -> Self {
        Self::new()
    }
}

impl FromIterator<(Vec<u8>, u64)> for WordCounts {
    fn from_iter<I: IntoIterator<Item = (Vec<u8>, u64)>>(entries: I) -> Self {
        let mut counts = Self::new();
        for (word, count) in entries {
            counts.add_n(&word, count);
        }
        counts
    }
}

/// Reads text into [`WordCounts`] through a normalizer and a pre-tokenizer.
///
/// Both are optional and independently so: a `.tiktoken`-style vocabulary is
/// defined by a pre-tokenizer pattern and no normalizer, while a BERT-style one
/// uses both.
pub struct Corpus {
    normalizer: Option<Normalizer>,
    pre_tokenizer: Option<PreTokenizer>,
    word_marker: Option<char>,
    counts: WordCounts,
    /// Reused by [`Corpus::add_word`] so that marking a word does not allocate
    /// once per occurrence.
    marked: Vec<u8>,
}

/// SentencePiece's word-boundary marker, U+2581 LOWER ONE EIGHTH BLOCK.
pub const METASPACE: char = '\u{2581}';

impl Corpus {
    /// A reader that splits on nothing: the whole text of each document is one
    /// word. Useful for testing and for vocabularies that declare no
    /// pre-tokenizer, but not what a real training run wants — without a
    /// pre-tokenizer, BPE will happily merge across word boundaries.
    pub fn new() -> Self {
        Self {
            normalizer: None,
            pre_tokenizer: None,
            word_marker: None,
            counts: WordCounts::new(),
            marked: Vec::new(),
        }
    }

    pub fn with_normalizer(mut self, normalizer: Normalizer) -> Self {
        self.normalizer = Some(normalizer);
        self
    }

    pub fn with_pre_tokenizer(mut self, pre_tokenizer: PreTokenizer) -> Self {
        self.pre_tokenizer = Some(pre_tokenizer);
        self
    }

    /// A reader using one of the named [`PreTok`] shapes.
    ///
    /// # Errors
    /// [`TrainError::PreTokenizer`] if a [`PreTok::Pattern`] does not compile.
    pub fn with_pre_tok(pre_tok: PreTok) -> Result<Self, TrainError> {
        let mut corpus = Self::new();
        corpus.pre_tokenizer = pre_tok.build()?;
        Ok(corpus)
    }

    /// Mark the start of every word with [`METASPACE`], the SentencePiece
    /// convention.
    ///
    /// **Required for any vocabulary a SentencePiece-style segmenter will
    /// load.** Those segmenters prepend the marker themselves before matching,
    /// so a vocabulary trained without it cannot spell the very first character
    /// of any word: measured, every word then picked up a spurious unknown
    /// token and the corpus needed ~1.9x the tokens it should have.
    ///
    /// It is also what makes word boundaries recoverable at decode time — the
    /// marker is where the space went — so this is a property of the vocabulary,
    /// not a preprocessing detail.
    #[must_use]
    pub fn with_word_marker(mut self, marker: char) -> Self {
        self.word_marker = Some(marker);
        self
    }

    /// [`with_word_marker`](Self::with_word_marker) with SentencePiece's own
    /// marker.
    #[must_use]
    pub fn with_metaspace(self) -> Self {
        self.with_word_marker(METASPACE)
    }

    /// Feed one document.
    pub fn feed(&mut self, text: &str) {
        let normalized = match &self.normalizer {
            Some(normalizer) => normalizer.normalize(text),
            None => std::borrow::Cow::Borrowed(text),
        };
        match &self.pre_tokenizer {
            Some(pre) => {
                for piece in pre.split(&normalized) {
                    self.add_word(&piece);
                }
            }
            None => self.add_word(&normalized),
        }
    }

    /// Record one word, marked if a marker was configured.
    fn add_word(&mut self, word: &str) {
        match self.word_marker {
            Some(marker) => {
                let marked = &mut self.marked;
                marked.clear();
                let mut encoded = [0u8; 4];
                marked.extend_from_slice(marker.encode_utf8(&mut encoded).as_bytes());
                marked.extend_from_slice(word.as_bytes());
                self.counts.add(marked);
            }
            None => self.counts.add(word.as_bytes()),
        }
    }

    /// Feed every document in an iterator.
    pub fn feed_all<I, S>(&mut self, documents: I)
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        for document in documents {
            self.feed(document.as_ref());
        }
    }

    /// Feed a file, one document per line.
    ///
    /// Line endings are kept: whether a newline belongs to the preceding word is
    /// the pre-tokenizer's decision, and stripping it here would answer that
    /// question differently from how encoding will.
    pub fn feed_file(&mut self, path: impl AsRef<std::path::Path>) -> Result<(), TrainError> {
        self.feed_reader(std::fs::File::open(path)?)
    }

    /// Feed anything readable, one document per line, a line at a time.
    ///
    /// Nothing larger than the current line is ever held, so corpus size does
    /// not enter the memory cost — only the number of *distinct* words does.
    /// Prefer this over reading a file into a `String` and calling
    /// [`feed`](Self::feed): the two produce identical counts, but the string
    /// costs the whole corpus in resident memory on top of the counts.
    ///
    /// # Errors
    /// [`TrainError::Io`] if the reader fails or yields text that is not UTF-8.
    pub fn feed_reader(&mut self, reader: impl std::io::Read) -> Result<(), TrainError> {
        use std::io::BufRead;

        let mut reader = std::io::BufReader::with_capacity(1 << 20, reader);
        let mut line = String::new();
        loop {
            line.clear();
            if reader.read_line(&mut line)? == 0 {
                break;
            }
            self.feed(&line);
        }
        Ok(())
    }

    /// The counts collected so far.
    pub fn counts(&self) -> &WordCounts {
        &self.counts
    }

    /// Take the counts, consuming the reader.
    pub fn into_counts(self) -> WordCounts {
        self.counts
    }
}

impl Default for Corpus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustc_hash::FxHashMap;
    use splintr::{PreTokStage, SplitBehavior, SplitPattern};

    fn whitespace() -> PreTokenizer {
        PreTokenizer::new(vec![PreTokStage::Split {
            pattern: SplitPattern::Regex(r"\s+".into()),
            behavior: SplitBehavior::Removed,
            invert: false,
        }])
        .expect("a whitespace split compiles")
    }

    #[test]
    fn counts_repeated_words_once_each() {
        let mut corpus = Corpus::new().with_pre_tokenizer(whitespace());
        corpus.feed("the cat the cat the");
        let counts = corpus.into_counts();
        assert_eq!(counts.len(), 2);
        assert_eq!(counts.total(), 5);
        let map: FxHashMap<Vec<u8>, u64> = counts.iter().map(|(w, c)| (w.to_vec(), c)).collect();
        assert_eq!(map[b"the".as_slice()], 3);
        assert_eq!(map[b"cat".as_slice()], 2);
    }

    /// With no pre-tokenizer the whole document is one word, which is what
    /// makes a pre-tokenizer effectively mandatory for real training.
    #[test]
    fn without_a_pre_tokenizer_a_document_is_one_word() {
        let mut corpus = Corpus::new();
        corpus.feed("the cat");
        assert_eq!(corpus.counts().len(), 1);
    }

    /// A `ByteLevel` stage emits pieces already in the byte-level alphabet, and
    /// they are recorded as exactly those bytes — the space becomes `Ġ`.
    #[test]
    fn byte_level_pieces_are_recorded_as_encoded() {
        let pre = PreTokenizer::new(vec![PreTokStage::ByteLevel {
            use_regex: true,
            add_prefix_space: false,
        }])
        .expect("a byte-level stage compiles");
        let mut corpus = Corpus::new().with_pre_tokenizer(pre);
        corpus.feed("a b");
        let words: Vec<String> = corpus
            .counts()
            .iter()
            .map(|(w, _)| String::from_utf8_lossy(w).into_owned())
            .collect();
        assert!(words.iter().any(|w| w == "Ġb"), "got {words:?}");
    }

    /// The marker goes on the front of every word, which is what a
    /// SentencePiece-style segmenter expects to match against.
    #[test]
    fn the_word_marker_prefixes_every_word() {
        let mut corpus = Corpus::new()
            .with_pre_tokenizer(whitespace())
            .with_metaspace();
        corpus.feed("the cat");
        let words: Vec<String> = corpus
            .counts()
            .iter()
            .map(|(w, _)| String::from_utf8_lossy(w).into_owned())
            .collect();
        assert!(words.contains(&"\u{2581}the".to_string()), "got {words:?}");
        assert!(words.contains(&"\u{2581}cat".to_string()), "got {words:?}");
    }

    /// Every named shape compiles, so the CLI cannot offer one that fails only
    /// when a user picks it.
    #[test]
    fn every_named_pre_tokenizer_builds() {
        assert!(PreTok::Whitespace.build().unwrap().is_some());
        assert!(PreTok::ByteLevel.build().unwrap().is_some());
        assert!(PreTok::Pattern(r"\s*\S+".into()).build().unwrap().is_some());
        assert!(PreTok::None.build().unwrap().is_none());
    }

    /// A pattern that does not compile is reported rather than silently
    /// dropping the stage, which would change the ids with nothing to point at.
    #[test]
    fn a_broken_pattern_is_an_error() {
        assert!(PreTok::Pattern("(unclosed".into()).build().is_err());
    }

    #[test]
    fn empty_words_are_not_counted() {
        let mut counts = WordCounts::new();
        counts.add(b"");
        assert!(counts.is_empty());
    }

    /// Enough distinct words to force the index through several growths, with
    /// every count checked afterwards: a probe or rehash that loses an entry
    /// would show up as a wrong count rather than a crash.
    #[test]
    fn many_words_survive_repeated_growth() {
        let mut counts = WordCounts::new();
        let words: Vec<Vec<u8>> = (0..10_000u32)
            .map(|i| format!("word{i}").into_bytes())
            .collect();
        for (i, word) in words.iter().enumerate() {
            counts.add_n(word, i as u64 + 1);
        }
        // Again, so every add is a hit rather than an insert.
        for word in &words {
            counts.add(word);
        }

        assert_eq!(counts.len(), words.len());
        let map: FxHashMap<Vec<u8>, u64> = counts.iter().map(|(w, c)| (w.to_vec(), c)).collect();
        assert_eq!(map.len(), words.len());
        for (i, word) in words.iter().enumerate() {
            assert_eq!(map[word], i as u64 + 2, "count for {i}");
        }
        assert_eq!(counts.total(), map.values().sum::<u64>());
    }

    /// Words are stored end to end, so a word that is a prefix of another must
    /// still compare unequal — the length has to come from the entry, not from
    /// scanning the arena.
    #[test]
    fn prefixes_are_distinct_words() {
        let mut counts = WordCounts::new();
        counts.add_n(b"can", 1);
        counts.add_n(b"candle", 2);
        counts.add_n(b"c", 3);
        counts.add_n(b"can", 4);

        let map: FxHashMap<Vec<u8>, u64> = counts.iter().map(|(w, c)| (w.to_vec(), c)).collect();
        assert_eq!(map[b"can".as_slice()], 5);
        assert_eq!(map[b"candle".as_slice()], 2);
        assert_eq!(map[b"c".as_slice()], 3);
        assert_eq!(counts.len(), 3);
        assert_eq!(counts.total(), 10);
    }

    #[test]
    fn merging_sums_shared_words() {
        let mut left = WordCounts::new();
        left.add_n(b"the", 2);
        left.add_n(b"cat", 1);
        let mut right = WordCounts::new();
        right.add_n(b"the", 3);
        right.add_n(b"dog", 5);
        left.merge(right);

        let map: FxHashMap<Vec<u8>, u64> = left.iter().map(|(w, c)| (w.to_vec(), c)).collect();
        assert_eq!(map[b"the".as_slice()], 5);
        assert_eq!(map[b"cat".as_slice()], 1);
        assert_eq!(map[b"dog".as_slice()], 5);
        assert_eq!(left.total(), 11);
    }

    /// Streaming a reader and feeding the same text as one string must give the
    /// same counts, or the memory saving would be a change of behaviour.
    #[test]
    fn streaming_a_reader_matches_feeding_the_text() {
        let text = "the cat sat\non the mat\nthe cat\n";

        let mut streamed = Corpus::new().with_pre_tokenizer(whitespace());
        streamed
            .feed_reader(text.as_bytes())
            .expect("reading from memory cannot fail");

        let mut fed = Corpus::new().with_pre_tokenizer(whitespace());
        for line in text.split_inclusive('\n') {
            fed.feed(line);
        }

        let streamed: FxHashMap<Vec<u8>, u64> = streamed
            .counts()
            .iter()
            .map(|(w, c)| (w.to_vec(), c))
            .collect();
        let fed: FxHashMap<Vec<u8>, u64> =
            fed.counts().iter().map(|(w, c)| (w.to_vec(), c)).collect();
        assert_eq!(streamed, fed);
    }

    /// The arena is the point: distinct words cost their bytes plus fixed
    /// bookkeeping, and repeats cost nothing at all.
    #[test]
    fn repeats_do_not_grow_the_arena() {
        let mut counts = WordCounts::with_capacity(4);
        counts.add(b"repeated");
        let after_first = counts.memory_bytes();
        for _ in 0..1000 {
            counts.add(b"repeated");
        }
        assert_eq!(counts.memory_bytes(), after_first);
        assert_eq!(counts.len(), 1);
        assert_eq!(counts.total(), 1001);
    }
}
