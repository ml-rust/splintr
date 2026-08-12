//! Every trainer, through every writer it has, to a real file, loaded back by
//! splintr.
//!
//! The unit tests check each writer against splintr's reader in memory. These
//! go through the filesystem and the `PreTok` shapes the CLI actually uses, so a
//! change that breaks the path a user takes — rather than the one the tests
//! took — fails here.

use splintr::Tokenize;
use splintr_train::{
    write, BpeTrainer, Corpus, PreTok, Seeding, UnigramTrainer, WordCounts, WordPieceTrainer,
};

/// Enough repeated English that the trainers have merges worth taking.
fn corpus_text() -> String {
    let lines = [
        "the tokenizer encodes text into tokens and decodes them again",
        "training produces a vocabulary the encoder can load and reuse",
        "a trained vocabulary is written to a file and read back later",
        "the corpus repeats words so that the merges are worth taking",
        "vocabulary training and vocabulary loading share one pre tokenizer",
    ];
    let mut text = String::new();
    for _ in 0..60 {
        for line in lines {
            text.push_str(line);
            text.push('\n');
        }
    }
    text
}

fn counts(pre: PreTok, metaspace: bool) -> WordCounts {
    let mut corpus = Corpus::with_pre_tok(pre).expect("the pre-tokenizer compiles");
    if metaspace {
        corpus = corpus.with_metaspace();
    }
    corpus.feed(&corpus_text());
    corpus.into_counts()
}

const SAMPLE: &str = "the tokenizer encodes text";

/// BPE to a `.tiktoken` rank file, loaded with the pattern it was trained under.
#[test]
fn bpe_tiktoken_file_loads_and_round_trips() {
    const PATTERN: &str = r"\s*\S+";
    let vocab = BpeTrainer::builder()
        .vocab_size(800)
        .seeding(Seeding::Bytes)
        .special_tokens(["<|endoftext|>"])
        .build()
        .train(&counts(PreTok::Pattern(PATTERN.into()), false))
        .expect("training succeeds");

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v.tiktoken");
    write::tiktoken_file(&vocab, &path).expect("writes");

    let tokenizer =
        splintr::Tokenizer::from_file(path.to_str().unwrap(), PATTERN, vocab.special_encoder())
            .expect("splintr loads it");

    let ids = tokenizer.encode_ordinary(SAMPLE);
    assert!(!ids.is_empty());
    assert_eq!(tokenizer.decode(&ids).expect("decodes"), SAMPLE);
}

/// BPE to a `tokenizer.json`, which carries its own pre-tokenizer.
#[test]
fn bpe_json_file_loads_and_round_trips() {
    let vocab = BpeTrainer::builder()
        .vocab_size(400)
        .seeding(Seeding::Chars)
        .build()
        .train(&counts(PreTok::ByteLevel, false))
        .expect("training succeeds");

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tokenizer.json");
    write::bpe_json_file(&vocab, &Default::default(), &path).expect("writes");

    let tokenizer = splintr::from_json_path(path.to_str().unwrap()).expect("splintr loads it");
    let ids = tokenizer.encode(SAMPLE);
    assert!(!ids.is_empty());
    assert_eq!(tokenizer.decode(&ids).expect("decodes"), SAMPLE);
}

/// WordPiece to a `vocab.txt`, the BERT-family list.
#[test]
fn wordpiece_vocab_txt_loads_and_segments() {
    let vocab = WordPieceTrainer::builder()
        .vocab_size(800)
        .special_tokens(["[UNK]"])
        .build()
        .train(&counts(PreTok::Whitespace, false))
        .expect("training succeeds");

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("vocab.txt");
    write::vocab_txt_file(&vocab, &path).expect("writes");

    let tokens: Vec<String> = std::fs::read_to_string(&path)
        .unwrap()
        .lines()
        .map(str::to_string)
        .collect();
    assert_eq!(tokens, vocab.tokens(), "the file is the vocabulary");

    let segmenter = splintr::WordPieceTokenizer::new(tokens, 0, 512, false);
    for word in SAMPLE.split_whitespace() {
        let ids = segmenter.encode(word);
        assert!(!ids.is_empty(), "{word} produced nothing");
        assert!(!ids.contains(&0), "{word} fell back to [UNK]: {ids:?}");
    }
}

/// WordPiece to a `tokenizer.json`.
#[test]
fn wordpiece_json_file_loads() {
    let vocab = WordPieceTrainer::builder()
        .vocab_size(800)
        .special_tokens(["[UNK]", "[CLS]", "[SEP]"])
        .build()
        .train(&counts(PreTok::Whitespace, false))
        .expect("training succeeds");

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tokenizer.json");
    write::wordpiece_json_file(&vocab, &Default::default(), &path).expect("writes");

    let tokenizer = splintr::from_json_path(path.to_str().unwrap()).expect("splintr loads it");
    assert!(!tokenizer.encode(SAMPLE).is_empty());
}

/// Unigram to a `.spm`, the format splintr's SentencePiece loader reads.
#[test]
fn unigram_spm_file_loads_and_round_trips() {
    let vocab = UnigramTrainer::builder()
        .vocab_size(800)
        .special_tokens(["<unk>"])
        .build()
        .train(&counts(PreTok::Whitespace, true))
        .expect("training succeeds");

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v.spm");
    write::spm_file(&vocab, &path).expect("writes");

    let loaded = splintr::core::load_spm_vocab(&std::fs::read(&path).unwrap()).expect("loads");
    let tokenizer = splintr::SentencePieceTokenizer::new(
        loaded.pieces.clone(),
        loaded.scores.iter().map(|s| *s as f64).collect(),
        None,
        0,
    )
    .expect("builds a tokenizer");

    let ids = tokenizer.encode(SAMPLE);
    assert!(!ids.is_empty());
    assert_eq!(tokenizer.decode(&ids).expect("decodes"), SAMPLE);
}

/// Unigram to a `tokenizer.json`.
#[test]
fn unigram_json_file_loads_and_round_trips() {
    let vocab = UnigramTrainer::builder()
        .vocab_size(800)
        .special_tokens(["<unk>"])
        .build()
        .train(&counts(PreTok::Whitespace, true))
        .expect("training succeeds");

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tokenizer.json");
    write::unigram_json_file(&vocab, &Default::default(), &path).expect("writes");

    let tokenizer = splintr::from_json_path(path.to_str().unwrap()).expect("splintr loads it");
    let ids = tokenizer.encode(SAMPLE);
    assert!(!ids.is_empty());
    assert_eq!(tokenizer.decode(&ids).expect("decodes"), SAMPLE);
}

/// A Unigram vocabulary trained *without* the marker cannot spell the first
/// character of any word, because the segmenter prepends one before matching.
/// This is the failure that read as "Unigram is 1.9x worse than BPE" until it
/// was diagnosed, so it is pinned rather than left to be rediscovered.
#[test]
fn an_unmarked_unigram_vocabulary_segments_worse_than_a_marked_one() {
    let train = |metaspace: bool| {
        let vocab = UnigramTrainer::builder()
            .vocab_size(800)
            .special_tokens(["<unk>"])
            .build()
            .train(&counts(PreTok::Whitespace, metaspace))
            .expect("training succeeds");
        let (tokens, scores) = vocab.into_parts();
        let tokenizer =
            splintr::SentencePieceTokenizer::new(tokens, scores, None, 0).expect("builds");
        SAMPLE
            .split_whitespace()
            .map(|word| tokenizer.encode(word).len())
            .sum::<usize>()
    };

    let marked = train(true);
    let unmarked = train(false);
    assert!(
        marked < unmarked,
        "marking must help: marked={marked} unmarked={unmarked}"
    );
}
