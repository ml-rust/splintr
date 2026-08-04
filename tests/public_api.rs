//! Reachability guards for splintr's public API surface.
//!
//! Integration tests compile as a **separate crate**, so everything here is
//! resolved exactly as a downstream user would resolve it. That is the whole
//! point of this file: an in-crate test can name an item through `crate::`
//! whether it is `pub` or merely `pub(crate)`, so only a test that lives
//! outside the crate can tell "public" apart from "public in name only".
//!
//! A builder method whose parameter type cannot be named by a caller is
//! uncallable no matter what its own visibility says. Each test below
//! therefore *constructs* the argument through crate-root paths and then
//! actually calls the method — compiling is most of the assertion.
use splintr::{NormOp, Normalizer, Tokenizer};

/// The normalizer pipeline is nameable and constructible from outside the
/// crate, and [`Tokenizer::with_normalizer`] accepts what it produces.
#[test]
fn normalizer_can_be_built_and_attached_from_outside_the_crate() {
    let normalizer = Normalizer::new(vec![NormOp::Lowercase]);
    assert!(!normalizer.is_empty());
    assert_eq!(normalizer.normalize("MiXeD"), "mixed");

    let encoder = [(b"hello".to_vec(), 0u32), (b"HELLO".to_vec(), 1u32)]
        .into_iter()
        .collect();
    let tokenizer = Tokenizer::new(encoder, splintr::FxHashMap::default(), r"\S+|\s+")
        .expect("tokenizer construction")
        .with_normalizer(normalizer);

    // Lowercasing happens before the vocabulary is consulted, so the uppercase
    // spelling must resolve to the lowercase entry's id.
    assert_eq!(tokenizer.encode("HELLO"), vec![0]);
}

/// [`Normalizer::new`] takes the ops by value in order, and an empty pipeline
/// is a valid no-op rather than an error — both are part of the contract a
/// downstream caller depends on.
#[test]
fn empty_normalizer_is_a_valid_no_op() {
    let normalizer = Normalizer::new(vec![]);
    assert!(normalizer.is_empty());
    assert_eq!(normalizer.normalize("Unchanged"), "Unchanged");
}
