//! `tokenizer.json` output.
//!
//! Unlike a `.tiktoken`, this format states the whole tokenizer: the vocabulary,
//! the ordered merge list, the pre-tokenizer and decoder chain, and the added
//! tokens. splintr reads it through `from_json_path`, and so does every other
//! tool that speaks HuggingFace.
//!
//! The merges are written as **pairs**, which is the format's own model: HF
//! ranks a merge by the pair it joins, while splintr ranks it by the token it
//! produces (`core::bpe::ranks`). Both readings are derivable from a trained
//! vocabulary, since the trainer records the operands of every merge — so this
//! writer states the pairs and splintr's loader converts on the way back in.

use std::io::Write;
use std::path::Path;

use serde_json::{json, Map, Value};

use crate::error::TrainError;
use crate::vocab::{Seeding, TrainedVocab};

/// What to declare alongside the vocabulary.
///
/// The defaults describe the byte-level shape, because that is the one a
/// `tokenizer.json` is normally written for — a vocabulary trained under
/// [`Seeding::Bytes`] has pieces that are raw bytes and may not be valid UTF-8,
/// which this format cannot key by.
pub struct BpeJsonOptions {
    /// Emit the `ByteLevel` pre-tokenizer and decoder pair.
    pub byte_level: bool,
    /// `ByteLevel { add_prefix_space }`.
    pub add_prefix_space: bool,
    /// Split with the GPT-2 word regex before byte-encoding.
    pub use_regex: bool,
}

impl Default for BpeJsonOptions {
    fn default() -> Self {
        Self {
            byte_level: true,
            add_prefix_space: false,
            use_regex: true,
        }
    }
}

/// The vocabulary as a `tokenizer.json` value.
///
/// # Errors
/// [`TrainError::NotUtf8`] if a piece is not valid UTF-8. This format keys its
/// vocabulary by string, so such a piece has no key — which is why a vocabulary
/// meant for it is trained under [`Seeding::Chars`] over byte-level
/// pre-tokenized text, where every piece is printable by construction.
pub fn bpe_json(vocab: &TrainedVocab, options: &BpeJsonOptions) -> Result<Value, TrainError> {
    let mut model_vocab = Map::with_capacity(vocab.pieces().len());
    for (id, piece) in vocab.pieces().iter().enumerate() {
        let key = std::str::from_utf8(piece).map_err(|_| TrainError::NotUtf8 { id: id as u32 })?;
        model_vocab.insert(key.to_string(), json!(id));
    }

    // Merges as `["left", "right"]`, in merge order. Every operand is a piece
    // with a lower id than the result, so both lookups are always present.
    let mut merges = Vec::with_capacity(vocab.merges().len());
    for &(left, right) in vocab.merges() {
        let left_piece = piece_str(vocab, left)?;
        let right_piece = piece_str(vocab, right)?;
        merges.push(json!([left_piece, right_piece]));
    }

    let base = vocab.pieces().len() as u32;
    let added: Vec<Value> = vocab
        .specials()
        .iter()
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
        .collect();

    // The recipe the vocabulary was trained under wins over the option. The
    // option describes what the caller wants the file to claim; the recipe is
    // what actually cut the words, and a file claiming boundaries it was not
    // trained on is the failure this exists to prevent.
    if let Some(recipe) = vocab.recipe() {
        let pre_tokenizer = recipe.pre_tokenizer_json();
        let decoder = if matches!(recipe.pre_tok, crate::PreTok::ByteLevel) {
            pre_tokenizer.clone()
        } else {
            Value::Null
        };
        return Ok(assemble(
            vocab,
            model_vocab,
            merges,
            added,
            pre_tokenizer,
            decoder,
        ));
    }

    let (pre_tokenizer, decoder) = if options.byte_level {
        (
            json!({
                "type": "ByteLevel",
                "add_prefix_space": options.add_prefix_space,
                "trim_offsets": true,
                "use_regex": options.use_regex,
            }),
            json!({ "type": "ByteLevel", "add_prefix_space": options.add_prefix_space, "trim_offsets": true, "use_regex": options.use_regex }),
        )
    } else {
        (Value::Null, Value::Null)
    };

    Ok(assemble(
        vocab,
        model_vocab,
        merges,
        added,
        pre_tokenizer,
        decoder,
    ))
}

fn assemble(
    vocab: &TrainedVocab,
    model_vocab: Map<String, Value>,
    merges: Vec<Value>,
    added: Vec<Value>,
    pre_tokenizer: Value,
    decoder: Value,
) -> Value {
    json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added,
        "normalizer": null,
        "pre_tokenizer": pre_tokenizer,
        "post_processor": null,
        "decoder": decoder,
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": false,
            "byte_fallback": matches!(vocab.seeding(), Seeding::Bytes),
            "ignore_merges": false,
            "vocab": model_vocab,
            "merges": merges,
        },
    })
}

fn piece_str(vocab: &TrainedVocab, id: u32) -> Result<&str, TrainError> {
    let piece = vocab.piece(id).ok_or(TrainError::NotUtf8 { id })?;
    std::str::from_utf8(piece).map_err(|_| TrainError::NotUtf8 { id })
}

/// [`bpe_json`], written to a file.
pub fn bpe_json_file(
    vocab: &TrainedVocab,
    options: &BpeJsonOptions,
    path: impl AsRef<Path>,
) -> Result<(), TrainError> {
    let value = bpe_json(vocab, options)?;
    let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
    serde_json::to_writer_pretty(&mut file, &value)?;
    file.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BpeTrainer, Seeding, WordCounts};

    fn byte_level_vocab() -> TrainedVocab {
        // Character seeding over printable text, which is what a byte-level
        // pipeline hands over: every piece is a valid UTF-8 string.
        let counts: WordCounts = [
            ("lower".as_bytes().to_vec(), 5u64),
            ("lowest".as_bytes().to_vec(), 4),
            ("newer".as_bytes().to_vec(), 6),
        ]
        .into_iter()
        .collect();
        BpeTrainer::builder()
            .vocab_size(40)
            .seeding(Seeding::Chars)
            .special_tokens(["<|endoftext|>"])
            .build()
            .train(&counts)
            .expect("training succeeds")
    }

    #[test]
    fn states_the_vocabulary_and_merges() {
        let vocab = byte_level_vocab();
        let value = bpe_json(&vocab, &BpeJsonOptions::default()).unwrap();
        let model = &value["model"];
        assert_eq!(model["type"], "BPE");
        assert_eq!(
            model["vocab"].as_object().unwrap().len(),
            vocab.pieces().len()
        );
        assert_eq!(
            model["merges"].as_array().unwrap().len(),
            vocab.merges().len()
        );
    }

    /// Each written merge names the two pieces whose concatenation is the piece
    /// that merge produced — the property a reader relies on to rebuild ranks.
    #[test]
    fn every_merge_names_its_operands() {
        let vocab = byte_level_vocab();
        let value = bpe_json(&vocab, &BpeJsonOptions::default()).unwrap();
        let merges = value["model"]["merges"].as_array().unwrap();
        for (i, merge) in merges.iter().enumerate() {
            let left = merge[0].as_str().unwrap();
            let right = merge[1].as_str().unwrap();
            let produced = vocab.piece((vocab.alphabet_len() + i) as u32).unwrap();
            assert_eq!(format!("{left}{right}").as_bytes(), produced);
        }
    }

    #[test]
    fn specials_are_added_tokens_above_the_vocabulary() {
        let vocab = byte_level_vocab();
        let value = bpe_json(&vocab, &BpeJsonOptions::default()).unwrap();
        let added = value["added_tokens"].as_array().unwrap();
        assert_eq!(added.len(), 1);
        assert_eq!(added[0]["content"], "<|endoftext|>");
        assert_eq!(added[0]["id"], vocab.pieces().len());
        assert_eq!(added[0]["special"], true);
    }

    /// A byte-seeded vocabulary holds pieces that are not valid UTF-8, and this
    /// format cannot key them — refused rather than written wrong.
    #[test]
    fn refuses_a_piece_that_is_not_utf8() {
        let counts: WordCounts = [(vec![0xFF, 0xFE], 3u64)].into_iter().collect();
        let vocab = BpeTrainer::builder()
            .vocab_size(260)
            .build()
            .train(&counts)
            .unwrap();
        let error = bpe_json(&vocab, &BpeJsonOptions::default()).unwrap_err();
        assert!(matches!(error, TrainError::NotUtf8 { .. }));
    }

    #[test]
    fn byte_level_can_be_left_out() {
        let vocab = byte_level_vocab();
        let options = BpeJsonOptions {
            byte_level: false,
            ..BpeJsonOptions::default()
        };
        let value = bpe_json(&vocab, &options).unwrap();
        assert!(value["pre_tokenizer"].is_null());
    }

    /// The whole point of the recipe: a vocabulary trained on one set of
    /// boundaries cannot be written out claiming another, however the options
    /// are set. Before this, `byte_level: true` here would have produced a file
    /// declaring a pre-tokenizer the vocabulary was never trained under.
    #[test]
    fn the_training_recipe_overrides_the_requested_pre_tokenizer() {
        use crate::{Corpus, PreTok};

        let mut corpus = Corpus::with_pre_tok(PreTok::Whitespace).unwrap();
        corpus.feed("the cat sat on the mat the cat");
        let counts = corpus.into_counts();
        let vocab = BpeTrainer::builder()
            .vocab_size(300)
            .min_frequency(1)
            .seeding(Seeding::Chars)
            .build()
            .train(&counts)
            .unwrap();

        assert_eq!(
            vocab.recipe().map(|r| &r.pre_tok),
            Some(&PreTok::Whitespace)
        );

        let options = BpeJsonOptions {
            byte_level: true,
            ..Default::default()
        };
        let value = bpe_json(&vocab, &options).unwrap();
        assert_eq!(value["pre_tokenizer"]["type"], "Sequence");
        assert_eq!(
            value["pre_tokenizer"]["pretokenizers"][0]["type"],
            "WhitespaceSplit"
        );
        // Only a byte-level vocabulary gets a byte-level decoder.
        assert!(value["decoder"].is_null());
    }

    /// A pattern-trained vocabulary states its pattern, which is what a
    /// `.tiktoken` needs and cannot hold.
    #[test]
    fn a_pattern_recipe_reaches_the_sidecar() {
        use crate::{write::recipe_json, Corpus, PreTok};

        let pattern = r"\s*\S+";
        let mut corpus = Corpus::with_pre_tok(PreTok::Pattern(pattern.into())).unwrap();
        corpus.feed("the cat sat on the mat the cat");
        let counts = corpus.into_counts();
        let vocab = BpeTrainer::builder()
            .vocab_size(300)
            .min_frequency(1)
            .special_tokens(["<eos>"])
            .build()
            .train(&counts)
            .unwrap();

        let sidecar = recipe_json(&vocab).expect("a PreTok-built corpus records its recipe");
        assert_eq!(sidecar["pattern"], pattern);
        assert_eq!(sidecar["special_tokens"][0], "<eos>");
        assert_eq!(sidecar["pre_tokenizer"]["type"], "Split");
    }

    /// Counts assembled by hand describe no recipe, and must not invent one.
    #[test]
    fn hand_built_counts_carry_no_recipe() {
        let counts: WordCounts = [(b"low".to_vec(), 5u64)].into_iter().collect();
        let vocab = BpeTrainer::builder()
            .vocab_size(300)
            .min_frequency(1)
            .build()
            .train(&counts)
            .unwrap();
        assert!(vocab.recipe().is_none());
        assert!(crate::write::recipe_json(&vocab).is_none());
    }

    #[test]
    fn writes_to_a_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tokenizer.json");
        let vocab = byte_level_vocab();
        bpe_json_file(&vocab, &BpeJsonOptions::default(), &path).unwrap();
        let parsed: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(parsed["model"]["type"], "BPE");
    }
}
