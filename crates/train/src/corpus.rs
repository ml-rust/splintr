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

use rustc_hash::FxHashMap;
use splintr::{Normalizer, PreTokenizer};

use crate::error::TrainError;

/// Words and their frequencies, the input every trainer takes.
///
/// Keyed by raw bytes rather than `String`: a pre-tokenizer with a `ByteLevel`
/// stage emits pieces already mapped into the byte-level alphabet, and one
/// without it emits ordinary text, but both are handed on as the bytes the
/// vocabulary will actually be keyed by. See [`Seeding`](crate::Seeding) for how
/// those bytes are then cut into initial symbols.
#[derive(Debug, Default, Clone)]
pub struct WordCounts {
    counts: FxHashMap<Vec<u8>, u64>,
}

impl WordCounts {
    pub fn new() -> Self {
        Self::default()
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
        *self.counts.entry(word.to_vec()).or_insert(0) += n;
    }

    /// Fold another set of counts into this one.
    pub fn merge(&mut self, other: WordCounts) {
        for (word, count) in other.counts {
            *self.counts.entry(word).or_insert(0) += count;
        }
    }

    /// How many distinct words were seen. The trainer's cost is driven by this
    /// rather than by corpus size, so it is worth knowing before starting.
    pub fn len(&self) -> usize {
        self.counts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }

    /// Total occurrences across all words.
    pub fn total(&self) -> u64 {
        self.counts.values().sum()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&[u8], u64)> + '_ {
        self.counts
            .iter()
            .map(|(word, &count)| (word.as_slice(), count))
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
                let mut marked = String::with_capacity(word.len() + marker.len_utf8());
                marked.push(marker);
                marked.push_str(word);
                self.counts.add(marked.as_bytes());
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
        use std::io::BufRead;

        let file = std::fs::File::open(path)?;
        let mut reader = std::io::BufReader::with_capacity(1 << 20, file);
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

    #[test]
    fn empty_words_are_not_counted() {
        let mut counts = WordCounts::new();
        counts.add(b"");
        assert!(counts.is_empty());
    }
}
