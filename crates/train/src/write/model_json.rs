//! `tokenizer.json` output for the WordPiece and Unigram models.
//!
//! Both are read back by splintr's `from_json_path`, which dispatches on
//! `model.type`. The shapes differ in more than a label: WordPiece's `vocab` is
//! an object mapping token to id, Unigram's is an array of `[token, score]`
//! pairs, and the loader uses exactly that difference to tell them apart when a
//! file omits the type.

use std::io::Write;
use std::path::Path;

use serde_json::{json, Map, Value};

use crate::corpus::METASPACE;
use crate::error::TrainError;
use crate::unigram::UnigramVocab;
use crate::wordpiece::WordPieceVocab;

/// What to declare alongside a WordPiece vocabulary.
pub struct WordPieceJsonOptions {
    /// The token stood in for anything unspellable. Must be in the vocabulary.
    pub unk_token: String,
    /// The continuation marker the vocabulary was trained with.
    pub continuing_subword_prefix: String,
    /// Words longer than this become the unknown token outright.
    pub max_input_chars_per_word: usize,
    /// Emit the BERT normalizer (lowercasing off, accents kept).
    pub bert_normalizer: bool,
}

impl Default for WordPieceJsonOptions {
    fn default() -> Self {
        Self {
            unk_token: "[UNK]".to_string(),
            continuing_subword_prefix: crate::wordpiece::DEFAULT_CONTINUING_PREFIX.to_string(),
            max_input_chars_per_word: 100,
            bert_normalizer: true,
        }
    }
}

/// A WordPiece vocabulary as a `tokenizer.json` value.
///
/// # Errors
/// [`TrainError::MissingToken`] if the declared unknown token is not in the
/// vocabulary — a WordPiece model cannot work without one, and a file naming a
/// token it does not contain fails at load rather than at write.
pub fn wordpiece_json(
    vocab: &WordPieceVocab,
    options: &WordPieceJsonOptions,
) -> Result<Value, TrainError> {
    if vocab.id(&options.unk_token).is_none() {
        return Err(TrainError::MissingToken {
            token: options.unk_token.clone(),
        });
    }

    let mut model_vocab = Map::with_capacity(vocab.len());
    for (id, token) in vocab.tokens().iter().enumerate() {
        model_vocab.insert(token.clone(), json!(id));
    }

    let normalizer = if options.bert_normalizer {
        json!({
            "type": "BertNormalizer",
            "clean_text": true,
            "handle_chinese_chars": true,
            "strip_accents": null,
            "lowercase": false,
        })
    } else {
        Value::Null
    };

    Ok(json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added_tokens(vocab.tokens(), vocab.special_count(), 0),
        "normalizer": normalizer,
        // What actually cut the words, when the vocabulary knows. A file that
        // declares boundaries it was not trained on is wrong about itself, and
        // nothing downstream can tell.
        "pre_tokenizer": match vocab.recipe() {
            Some(recipe) => recipe.pre_tokenizer_json(),
            None => json!({ "type": "BertPreTokenizer" }),
        },
        "post_processor": null,
        "decoder": {
            "type": "WordPiece",
            "prefix": options.continuing_subword_prefix,
            "cleanup": true,
        },
        "model": {
            "type": "WordPiece",
            "unk_token": options.unk_token,
            "continuing_subword_prefix": options.continuing_subword_prefix,
            "max_input_chars_per_word": options.max_input_chars_per_word,
            "vocab": model_vocab,
        },
    }))
}

/// What to declare alongside a Unigram vocabulary.
pub struct UnigramJsonOptions {
    /// Id of the unknown piece, if the vocabulary has one.
    pub unk_id: Option<u32>,
    /// The word-boundary marker the vocabulary was trained with.
    pub replacement: char,
    /// Whether the segmenter prepends the marker to the first word too.
    pub prepend_scheme_always: bool,
}

impl Default for UnigramJsonOptions {
    fn default() -> Self {
        Self {
            unk_id: Some(0),
            replacement: METASPACE,
            prepend_scheme_always: true,
        }
    }
}

/// A Unigram vocabulary as a `tokenizer.json` value.
///
/// The `vocab` is an array of `[token, score]` pairs — not an object — which is
/// the shape the loader identifies a Unigram model by when no type is declared.
pub fn unigram_json(vocab: &UnigramVocab, options: &UnigramJsonOptions) -> Value {
    let entries: Vec<Value> = vocab
        .tokens()
        .iter()
        .zip(vocab.scores())
        .map(|(token, score)| json!([token, score]))
        .collect();

    // A Metaspace pre-tokenizer is only right if the corpus was actually marked
    // with one, and by the same character. Declaring it otherwise is the failure
    // measured in `.claude/prep/train-checklist.md`: the segmenter prepends a
    // marker the vocabulary cannot spell, and every word picks up an unknown.
    let marker = match vocab.recipe() {
        Some(recipe) => recipe.word_marker,
        None => Some(options.replacement),
    };
    let metaspace = match marker {
        Some(replacement) => json!({
            "type": "Metaspace",
            "replacement": replacement.to_string(),
            "prepend_scheme": if options.prepend_scheme_always { "always" } else { "never" },
            "split": true,
        }),
        None => match vocab.recipe() {
            Some(recipe) => recipe.pre_tokenizer_json(),
            None => Value::Null,
        },
    };

    json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added_tokens(vocab.tokens(), vocab.special_count(), 0),
        "normalizer": { "type": "Precompiled", "precompiled_charsmap": null },
        "pre_tokenizer": metaspace,
        "post_processor": null,
        "decoder": metaspace,
        "model": {
            "type": "Unigram",
            "unk_id": options.unk_id,
            "byte_fallback": false,
            "vocab": entries,
        },
    })
}

/// The leading `special_count` tokens, declared as added tokens.
fn added_tokens(tokens: &[String], special_count: usize, base: u32) -> Vec<Value> {
    tokens
        .iter()
        .take(special_count)
        .enumerate()
        .map(|(i, content)| {
            json!({
                "id": base + i as u32,
                "content": content,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true,
            })
        })
        .collect()
}

/// [`wordpiece_json`], written to a file.
pub fn wordpiece_json_file(
    vocab: &WordPieceVocab,
    options: &WordPieceJsonOptions,
    path: impl AsRef<Path>,
) -> Result<(), TrainError> {
    write_json(&wordpiece_json(vocab, options)?, path)
}

/// [`unigram_json`], written to a file.
pub fn unigram_json_file(
    vocab: &UnigramVocab,
    options: &UnigramJsonOptions,
    path: impl AsRef<Path>,
) -> Result<(), TrainError> {
    write_json(&unigram_json(vocab, options), path)
}

fn write_json(value: &Value, path: impl AsRef<Path>) -> Result<(), TrainError> {
    let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
    serde_json::to_writer_pretty(&mut file, value)?;
    file.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Corpus, UnigramTrainer, WordPieceTrainer};

    fn corpus(marked: bool) -> Corpus {
        let mut corpus = if marked {
            Corpus::new().with_metaspace()
        } else {
            Corpus::new()
        };
        for word in [
            "playing", "played", "player", "running", "runner", "reads", "reader",
        ] {
            for _ in 0..8 {
                corpus.feed(word);
            }
        }
        corpus
    }

    fn wordpiece() -> WordPieceVocab {
        WordPieceTrainer::builder()
            .vocab_size(120)
            .special_tokens(["[UNK]", "[CLS]", "[SEP]"])
            .build()
            .train(corpus(false).counts())
            .expect("training succeeds")
    }

    fn unigram() -> UnigramVocab {
        UnigramTrainer::builder()
            .vocab_size(120)
            .special_tokens(["<unk>"])
            .min_frequency(1)
            .build()
            .train(corpus(true).counts())
            .expect("training succeeds")
    }

    /// The end-to-end claim for WordPiece: written, then loaded by splintr's own
    /// `tokenizer.json` reader.
    #[test]
    fn splintr_loads_the_wordpiece_model() {
        let vocab = wordpiece();
        let text =
            serde_json::to_vec(&wordpiece_json(&vocab, &WordPieceJsonOptions::default()).unwrap())
                .unwrap();
        let tokenizer = splintr::from_json_bytes(&text).expect("splintr loads it");
        let ids = tokenizer.encode("playing");
        assert!(!ids.is_empty());
    }

    /// And for Unigram.
    #[test]
    fn splintr_loads_the_unigram_model() {
        let vocab = unigram();
        let text =
            serde_json::to_vec(&unigram_json(&vocab, &UnigramJsonOptions::default())).unwrap();
        let tokenizer = splintr::from_json_bytes(&text).expect("splintr loads it");
        let ids = tokenizer.encode("playing");
        assert!(!ids.is_empty());
    }

    /// A Unigram `vocab` is an array of pairs, which is what the loader tells
    /// the model apart by when the type is absent.
    #[test]
    fn the_unigram_vocabulary_is_pairs_not_an_object() {
        let vocab = unigram();
        let value = unigram_json(&vocab, &UnigramJsonOptions::default());
        let entries = value["model"]["vocab"].as_array().expect("an array");
        assert_eq!(entries.len(), vocab.len());
        assert!(entries[0][0].is_string(), "first column is the token");
        assert!(entries[0][1].is_number(), "second column is the score");
    }

    /// A WordPiece `vocab` is an object, the other half of that distinction.
    #[test]
    fn the_wordpiece_vocabulary_is_an_object() {
        let vocab = wordpiece();
        let value = wordpiece_json(&vocab, &WordPieceJsonOptions::default()).unwrap();
        let map = value["model"]["vocab"].as_object().expect("an object");
        assert_eq!(map.len(), vocab.len());
        assert_eq!(map["[UNK]"], 0);
    }

    /// Naming an unknown token the vocabulary lacks fails at write, where it can
    /// be pointed at, rather than at load.
    #[test]
    fn refuses_an_unk_token_the_vocabulary_lacks() {
        let vocab = wordpiece();
        let options = WordPieceJsonOptions {
            unk_token: "[NOPE]".to_string(),
            ..WordPieceJsonOptions::default()
        };
        assert!(matches!(
            wordpiece_json(&vocab, &options).unwrap_err(),
            TrainError::MissingToken { .. }
        ));
    }

    #[test]
    fn both_write_to_a_file() {
        let dir = tempfile::tempdir().unwrap();

        let wp = dir.path().join("wordpiece.json");
        wordpiece_json_file(&wordpiece(), &WordPieceJsonOptions::default(), &wp).unwrap();
        let parsed: Value = serde_json::from_str(&std::fs::read_to_string(&wp).unwrap()).unwrap();
        assert_eq!(parsed["model"]["type"], "WordPiece");

        let uni = dir.path().join("unigram.json");
        unigram_json_file(&unigram(), &UnigramJsonOptions::default(), &uni).unwrap();
        let parsed: Value = serde_json::from_str(&std::fs::read_to_string(&uni).unwrap()).unwrap();
        assert_eq!(parsed["model"]["type"], "Unigram");
    }
}
