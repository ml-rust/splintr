//! The recipe a vocabulary was trained under, for the formats that cannot state
//! it themselves.
//!
//! `.tiktoken`, `vocab.txt` and a SentencePiece model are all lists of pieces
//! and nothing else. Load any of them against different word boundaries than
//! they were trained on and you get different ids, silently — so the recipe is
//! written beside them.

use std::io::Write;
use std::path::Path;

use serde_json::{json, Value};

use crate::corpus::Recipe;
use crate::error::TrainError;
use crate::unigram::UnigramVocab;
use crate::vocab::TrainedVocab;
use crate::wordpiece::WordPieceVocab;

/// A trained vocabulary that may know how its corpus was cut.
pub trait Trained {
    /// See [`Recipe`]. `None` when the pre-tokenizer could not be written down —
    /// a hand-assembled one, or counts assembled without a [`Corpus`](crate::Corpus).
    fn recipe(&self) -> Option<&Recipe>;
    /// The special tokens, which these formats also leave unstated.
    fn special_tokens(&self) -> &[String];
}

impl Trained for TrainedVocab {
    fn recipe(&self) -> Option<&Recipe> {
        TrainedVocab::recipe(self)
    }
    fn special_tokens(&self) -> &[String] {
        self.specials()
    }
}

impl Trained for WordPieceVocab {
    fn recipe(&self) -> Option<&Recipe> {
        WordPieceVocab::recipe(self)
    }
    fn special_tokens(&self) -> &[String] {
        &self.tokens()[..self.special_count()]
    }
}

impl Trained for UnigramVocab {
    fn recipe(&self) -> Option<&Recipe> {
        UnigramVocab::recipe(self)
    }
    fn special_tokens(&self) -> &[String] {
        &self.tokens()[..self.special_count()]
    }
}

/// What the piece list cannot say about itself: the boundaries it was trained
/// against, and the special tokens that sit above it.
///
/// `None` when the vocabulary carries no [`Recipe`], since inventing one would
/// be worse than admitting there is none.
pub fn recipe_json<V: Trained>(vocab: &V) -> Option<Value> {
    let recipe = vocab.recipe()?;
    Some(json!({
        "pre_tokenizer": recipe.pre_tokenizer_json(),
        "pattern": recipe.pattern(),
        "word_marker": recipe.word_marker.map(String::from),
        "special_tokens": vocab.special_tokens(),
    }))
}

/// [`recipe_json`], written to a file.
///
/// Returns whether anything was written — a vocabulary with no recorded recipe
/// has nothing to say here.
///
/// # Errors
/// [`TrainError::Io`] if the file cannot be written.
pub fn recipe_json_file<V: Trained>(vocab: &V, path: impl AsRef<Path>) -> Result<bool, TrainError> {
    let Some(value) = recipe_json(vocab) else {
        return Ok(false);
    };
    let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
    file.write_all(serde_json::to_string_pretty(&value)?.as_bytes())?;
    file.write_all(b"\n")?;
    file.flush()?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Corpus, PreTok, UnigramTrainer, WordPieceTrainer, METASPACE};

    fn corpus(pre: PreTok, metaspace: bool) -> crate::WordCounts {
        let mut corpus = Corpus::with_pre_tok(pre).unwrap();
        if metaspace {
            corpus = corpus.with_metaspace();
        }
        corpus.feed("the cat sat on the mat the cat sat down");
        corpus.into_counts()
    }

    /// `vocab.txt` states no boundaries, so the WordPiece vocabulary has to
    /// carry them the same way the BPE one does.
    #[test]
    fn a_wordpiece_vocabulary_carries_its_recipe() {
        let counts = corpus(PreTok::Whitespace, false);
        let vocab = WordPieceTrainer::builder()
            .vocab_size(300)
            .min_frequency(1)
            .special_tokens(["[UNK]"])
            .build()
            .train(&counts)
            .unwrap();

        assert_eq!(
            vocab.recipe().map(|r| &r.pre_tok),
            Some(&PreTok::Whitespace)
        );
        let sidecar = recipe_json(&vocab).expect("a PreTok-built corpus records its recipe");
        assert_eq!(sidecar["pre_tokenizer"]["type"], "Sequence");
        assert_eq!(sidecar["special_tokens"][0], "[UNK]");
    }

    /// The Unigram recipe has to record the word marker, because a
    /// SentencePiece segmenter prepends one before matching and a vocabulary
    /// trained without it cannot spell any word's first character.
    #[test]
    fn a_unigram_vocabulary_records_its_word_marker() {
        let counts = corpus(PreTok::Whitespace, true);
        let vocab = UnigramTrainer::builder()
            .vocab_size(200)
            .min_frequency(1)
            .special_tokens(["<unk>"])
            .build()
            .train(&counts)
            .unwrap();

        assert_eq!(vocab.recipe().and_then(|r| r.word_marker), Some(METASPACE));
        let sidecar = recipe_json(&vocab).expect("a PreTok-built corpus records its recipe");
        assert_eq!(sidecar["word_marker"], METASPACE.to_string());
    }

    /// Trained without a marker, the vocabulary must not claim one — that claim
    /// is what makes a SentencePiece segmenter emit an unknown per word.
    #[test]
    fn an_unmarked_unigram_vocabulary_claims_no_marker() {
        let counts = corpus(PreTok::Whitespace, false);
        let vocab = UnigramTrainer::builder()
            .vocab_size(200)
            .min_frequency(1)
            .build()
            .train(&counts)
            .unwrap();

        assert_eq!(vocab.recipe().and_then(|r| r.word_marker), None);
        let json = crate::write::unigram_json(&vocab, &Default::default());
        assert_ne!(
            json["pre_tokenizer"]["type"], "Metaspace",
            "an unmarked vocabulary must not be written as a Metaspace model"
        );
    }

    /// And trained *with* one, the JSON declares Metaspace using that very
    /// character rather than whatever the options happened to say.
    #[test]
    fn a_marked_unigram_vocabulary_is_written_as_metaspace() {
        let counts = corpus(PreTok::Whitespace, true);
        let vocab = UnigramTrainer::builder()
            .vocab_size(200)
            .min_frequency(1)
            .build()
            .train(&counts)
            .unwrap();

        let json = crate::write::unigram_json(&vocab, &Default::default());
        assert_eq!(json["pre_tokenizer"]["type"], "Metaspace");
        assert_eq!(json["pre_tokenizer"]["replacement"], METASPACE.to_string());
    }
}
