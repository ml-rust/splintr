//! Our BPE merge loop against HuggingFace's, on identical input.
//!
//! The rest of the suite checks that this crate is self-consistent: that the
//! merge deltas balance, that a written file loads back, that a bigger
//! vocabulary compresses better. None of that would catch a merge loop that is
//! internally coherent and simply *selects differently* from every other BPE
//! implementation — which would produce a vocabulary that is defensible in
//! isolation and wrong against the literature.
//!
//! So this pins the selection itself. The reference is a committed fixture
//! produced by `tokenizers` 0.22.1 (regenerate with
//! `scripts/generate_hf_reference.py`), so the test needs neither Python nor a
//! network.
//!
//! The configuration is chosen so any difference is the merge loop rather than
//! the setup: whitespace pre-tokenization, which both sides reproduce exactly,
//! over a corpus with no punctuation for the two punctuation policies to
//! disagree about; `min_frequency` 1; no special tokens; character seeding,
//! since HuggingFace seeds from the characters present where byte seeding would
//! add all 256.
//!
//! The corpus is 250 KB of Latin and Cyrillic text — Cyrillic because it is two
//! bytes per character, so the merge loop is exercised over characters wider
//! than a byte. Scripts with very large character inventories are left out on
//! purpose: several thousand Han characters would exceed the target vocabulary
//! as an alphabet, leaving nothing to merge and reducing this to a check that
//! both sides can list the corpus's characters.
//!
//! Scale matters here. An earlier version of this fixture ran dry at 101 pieces
//! and 78 merges, which pins the first few decisions and says nothing about
//! whether the two implementations still agree once the corpus is thoroughly
//! merged. It now covers 2000 pieces and 1861 merges, and a separate run
//! against a live `tokenizers` at a 32000-piece target on 9 MB of text agreed
//! on all 32000 pieces and all 31673 merges in order.
//!
//! A failure here is a finding, not a flake. It means our selection moved, or
//! theirs did — and either is worth knowing before shipping a vocabulary.

use rustc_hash::FxHashSet;
use splintr_train::{BpeTrainer, Corpus, PreTok, Seeding};

const VOCAB_SIZE: usize = 2000;

fn reference() -> serde_json::Value {
    let text = include_str!("fixtures/hf_bpe_reference.json");
    serde_json::from_str(text).expect("the reference fixture parses")
}

fn ours() -> (Vec<String>, usize, Vec<(u32, u32)>) {
    let mut corpus = Corpus::with_pre_tok(PreTok::Whitespace).expect("the pre-tokenizer compiles");
    corpus.feed(include_str!("fixtures/parity_corpus.txt"));

    let vocab = BpeTrainer::builder()
        .vocab_size(VOCAB_SIZE)
        .min_frequency(1)
        .seeding(Seeding::Chars)
        .build()
        .train(corpus.counts())
        .expect("training succeeds");

    let pieces = vocab
        .pieces()
        .iter()
        .map(|piece| String::from_utf8(piece.clone()).expect("character seeding keeps text"))
        .collect();
    (pieces, vocab.alphabet_len(), vocab.merges().to_vec())
}

/// The same set of pieces, neither more nor fewer.
#[test]
fn the_vocabularies_hold_the_same_pieces() {
    let (pieces, _, _) = ours();
    let reference = reference();
    let theirs: FxHashSet<&str> = reference["vocab"]
        .as_object()
        .expect("an object")
        .keys()
        .map(String::as_str)
        .collect();
    let mine: FxHashSet<&str> = pieces.iter().map(String::as_str).collect();

    let missing: Vec<&&str> = theirs.difference(&mine).collect();
    let extra: Vec<&&str> = mine.difference(&theirs).collect();
    assert!(
        missing.is_empty() && extra.is_empty(),
        "missing {missing:?}, extra {extra:?}"
    );
}

/// The stronger claim: the merges are chosen in the *same order*.
///
/// Order is what a BPE vocabulary is — the ids follow it, and two vocabularies
/// holding the same pieces in different orders segment text differently.
#[test]
fn the_merges_are_chosen_in_the_same_order() {
    let (pieces, alphabet_len, merges) = ours();
    let reference = reference();

    // HuggingFace states a merge as its two operands; ours is the piece the
    // merge produced. Comparing the concatenation compares the same thing.
    let theirs: Vec<String> = reference["merges"]
        .as_array()
        .expect("an array")
        .iter()
        .map(|merge| match merge {
            // Newer files store `["a", "b"]`, older ones `"a b"`.
            serde_json::Value::Array(pair) => format!(
                "{}{}",
                pair[0].as_str().expect("a string"),
                pair[1].as_str().expect("a string")
            ),
            serde_json::Value::String(joined) => joined.replace(' ', ""),
            other => panic!("unexpected merge shape: {other}"),
        })
        .collect();

    let mine: Vec<&String> = (0..merges.len())
        .map(|i| &pieces[alphabet_len + i])
        .collect();

    assert_eq!(
        mine.len(),
        theirs.len(),
        "different merge counts: ours {} theirs {}",
        mine.len(),
        theirs.len()
    );
    for (i, (mine, theirs)) in mine.iter().zip(&theirs).enumerate() {
        assert_eq!(
            *mine, theirs,
            "merge {i} differs: ours {mine:?}, theirs {theirs:?}"
        );
    }
}

/// Every operand of every merge is itself a piece, and the pair concatenates to
/// what the merge produced. Checked against our own output rather than theirs,
/// because it is the invariant splintr's rank-keyed encoder depends on.
#[test]
fn every_merge_is_its_operands_joined() {
    let (pieces, alphabet_len, merges) = ours();
    for (i, &(left, right)) in merges.iter().enumerate() {
        let produced = &pieces[alphabet_len + i];
        let expected = format!("{}{}", pieces[left as usize], pieces[right as usize]);
        assert_eq!(*produced, expected, "merge {i}");
    }
}
