//! The proof obligation for the whole crate: a vocabulary trained here loads
//! into splintr and encodes the corpus it was trained on, losslessly.
//!
//! Every unit test in the crate checks a structure in isolation — that the merge
//! deltas balance, that ids follow merge order, that base64 round-trips. None of
//! them would catch a vocabulary that is internally consistent and still refused
//! by the loader, or one that loads and then cannot spell its own training text.
//! That is what these check, and they go through the *public* path a user would:
//! train, write a file, load the file, encode, decode.

use splintr::{PreTokStage, PreTokenizer, SplitBehavior, SplitPattern};
use splintr_train::{write, BpeTrainer, Corpus, Seeding, TrainedVocab};

/// The pre-tokenizer used for both training and encoding. Being the same object
/// in both places is the point — a vocabulary is only meaningful against the
/// boundaries it was trained on.
const PATTERN: &str = r"\s*\S+";

fn pre_tokenizer() -> PreTokenizer {
    PreTokenizer::new(vec![PreTokStage::Split {
        pattern: SplitPattern::Regex(PATTERN.into()),
        behavior: SplitBehavior::Isolated,
        invert: true,
    }])
    .expect("the training pattern compiles")
}

fn corpus_text() -> String {
    // Repeated so pairs actually recur; varied so there is something to learn.
    let lines = [
        "the quick brown fox jumps over the lazy dog",
        "the quick brown cat sleeps under the lazy sun",
        "a tokenizer learns the merges that compress the corpus",
        "training produces a vocabulary the encoder can load",
        "the corpus repeats words so the merges are worth taking",
    ];
    let mut text = String::new();
    for _ in 0..40 {
        for line in lines {
            text.push_str(line);
            text.push('\n');
        }
    }
    text
}

fn train(vocab_size: usize) -> (TrainedVocab, String) {
    let text = corpus_text();
    let mut corpus = Corpus::new().with_pre_tokenizer(pre_tokenizer());
    corpus.feed(&text);
    let vocab = BpeTrainer::builder()
        .vocab_size(vocab_size)
        .seeding(Seeding::Bytes)
        .special_tokens(["<|endoftext|>"])
        .build()
        .train(corpus.counts())
        .expect("training succeeds");
    (vocab, text)
}

/// The end-to-end claim: written to disk, loaded by splintr's own `.tiktoken`
/// loader, and able to reproduce the training text exactly.
#[test]
fn a_trained_vocabulary_loads_and_round_trips() {
    let (vocab, text) = train(1_000);

    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("trained.tiktoken");
    write::tiktoken_file(&vocab, &path).expect("the vocabulary writes");

    let tokenizer = splintr::Tokenizer::from_file(
        path.to_str().expect("a utf-8 path"),
        PATTERN,
        vocab.special_encoder(),
    )
    .expect("splintr loads the vocabulary it was given");

    for line in text.lines() {
        let ids = tokenizer.encode_ordinary(line);
        let decoded = tokenizer.decode(&ids).expect("every id decodes");
        assert_eq!(decoded, line, "round trip failed for {line:?}");
    }
}

/// Text the vocabulary was never trained on still encodes, because byte seeding
/// puts all 256 bytes in the alphabet. A trained vocabulary with holes would
/// pass the test above and fail here.
#[test]
fn unseen_text_still_round_trips() {
    let (vocab, _) = train(600);
    let tokenizer = splintr::Tokenizer::new(vocab.encoder(), vocab.special_encoder(), PATTERN)
        .expect("the trained vocabulary builds a tokenizer");

    for text in [
        "wholly unrelated sentences with punctuation!",
        "digits 0123456789 and symbols #$%^&*",
        "unicode: héllo wörld — naïve café 日本語 🎉",
        "\ttabs\nand\r\nnewlines",
    ] {
        let ids = tokenizer.encode_ordinary(text);
        let decoded = tokenizer.decode(&ids).expect("every id decodes");
        assert_eq!(decoded, text, "round trip failed for {text:?}");
    }
}

/// Training is supposed to *compress*: a larger vocabulary must not need more
/// tokens for the same text than a smaller one. This is the check that would
/// catch merges being learned but never actually applied by the encoder — the
/// failure mode the rank-versus-pair difference could have caused.
#[test]
fn a_larger_vocabulary_compresses_at_least_as_well() {
    let text = corpus_text();
    let sample: String = text.lines().take(20).collect::<Vec<_>>().join("\n");

    let mut lengths = Vec::new();
    for size in [300usize, 500, 800, 1_200] {
        let (vocab, _) = train(size);
        let tokenizer = splintr::Tokenizer::new(vocab.encoder(), vocab.special_encoder(), PATTERN)
            .expect("the trained vocabulary builds a tokenizer");
        lengths.push((size, tokenizer.encode_ordinary(&sample).len()));
    }

    for window in lengths.windows(2) {
        let (small, long) = window[0];
        let (large, short) = window[1];
        assert!(
            short <= long,
            "vocab {large} needed {short} tokens where vocab {small} needed {long}"
        );
    }

    // And it must actually be learning something, not just seeding bytes.
    let (_, bytes_only) = lengths[0];
    let (_, trained) = lengths[lengths.len() - 1];
    assert!(
        trained < bytes_only,
        "training bought no compression at all ({trained} vs {bytes_only})"
    );
}

/// Every merge the trainer recorded is reachable by the encoder: encoding the
/// piece's own text yields that single id. A merge whose operands could not
/// recombine would be dead weight in the file and a silently worse tokenizer.
#[test]
fn every_merged_piece_encodes_to_its_own_id() {
    let (vocab, _) = train(800);
    let tokenizer = splintr::Tokenizer::new(vocab.encoder(), vocab.special_encoder(), PATTERN)
        .expect("the trained vocabulary builds a tokenizer");

    let mut unreachable = Vec::new();
    for id in vocab.alphabet_len()..vocab.pieces().len() {
        let piece = &vocab.pieces()[id];
        // Only pieces that are whole pre-tokens can be asked for directly; a
        // piece the pre-tokenizer would itself split is not a fair question.
        let Ok(text) = std::str::from_utf8(piece) else {
            continue;
        };
        if pre_tokenizer().split(text).len() != 1 {
            continue;
        }
        let ids = tokenizer.encode_ordinary(text);
        if ids != [id as u32] {
            unreachable.push((id, text.to_string(), ids));
        }
    }
    assert!(
        unreachable.is_empty(),
        "pieces the encoder cannot reach: {unreachable:?}"
    );
}

/// The `.tiktoken` text and the in-memory vocabulary describe the same thing,
/// so loading from the file cannot differ from building the encoder directly.
#[test]
fn the_written_file_agrees_with_the_in_memory_vocabulary() {
    let (vocab, text) = train(700);
    let dir = tempfile::tempdir().expect("a temp dir");
    let path = dir.path().join("trained.tiktoken");
    write::tiktoken_file(&vocab, &path).expect("the vocabulary writes");

    let from_file = splintr::Tokenizer::from_file(
        path.to_str().expect("a utf-8 path"),
        PATTERN,
        vocab.special_encoder(),
    )
    .expect("the file loads");
    let from_memory = splintr::Tokenizer::new(vocab.encoder(), vocab.special_encoder(), PATTERN)
        .expect("the encoder builds");

    for line in text.lines().take(50) {
        assert_eq!(
            from_file.encode_ordinary(line),
            from_memory.encode_ordinary(line),
            "file and memory disagree on {line:?}"
        );
    }
}

/// Special tokens are recognised as tokens rather than encoded as their letters.
#[test]
fn special_tokens_are_reachable() {
    let (vocab, _) = train(500);
    let tokenizer = splintr::Tokenizer::new(vocab.encoder(), vocab.special_encoder(), PATTERN)
        .expect("the trained vocabulary builds a tokenizer");

    let eot = vocab.special_encoder()["<|endoftext|>"];
    let ids = tokenizer.encode_with_special("a<|endoftext|>b");
    assert!(ids.contains(&eot), "the special token was not recognised");
    assert_eq!(tokenizer.decode(&ids).expect("decodes"), "a<|endoftext|>b");
}
