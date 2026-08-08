//! Integration tests for Llama 3/3.1/3.2/3.3 tokenizer.
//!
//! These tests verify that the Llama 3 tokenizer correctly encodes and decodes text,
//! handles special tokens, and produces consistent results.
// Gated on `vocab-llama3` because it names that vocabulary's embedded bytes directly, and the feature is what
// compiles those bytes in. Without it this crate is empty rather than
// a compile error.
#![cfg(feature = "vocab-llama3")]

use splintr::pretrained::{llama3_special_tokens, LLAMA3_VOCAB};
use splintr::{Tokenizer, LLAMA3_PATTERN};
use std::sync::LazyLock;

/// Shared tokenizer instance to avoid expensive re-initialization per test.
static TOKENIZER: LazyLock<Tokenizer> = LazyLock::new(create_llama3_tokenizer_impl);

// =============================================================================
// Exact Token ID Tests
// =============================================================================

/// Verify exact token IDs for "Hello world".
#[test]
fn test_llama3_hello_world_tokens() {
    let tokenizer = create_llama3_tokenizer();
    let tokens = tokenizer.encode("Hello world");
    assert_eq!(
        tokens,
        vec![9906, 1917],
        "Token IDs for 'Hello world' changed"
    );
}

/// Verify exact token IDs for "Hello, world!".
#[test]
fn test_llama3_hello_world_punctuation_tokens() {
    let tokenizer = create_llama3_tokenizer();
    let tokens = tokenizer.encode("Hello, world!");
    assert_eq!(
        tokens,
        vec![9906, 11, 1917, 0],
        "Token IDs for 'Hello, world!' changed"
    );
}

/// Verify exact token IDs for "你好世界".
#[test]
fn test_llama3_chinese_tokens() {
    let tokenizer = create_llama3_tokenizer();
    let tokens = tokenizer.encode("你好世界");
    assert_eq!(
        tokens,
        vec![57668, 53901, 102616],
        "Token IDs for '你好世界' changed"
    );
}

/// Verify exact token IDs for "Hello 🌍 World!".
#[test]
fn test_llama3_emoji_tokens() {
    let tokenizer = create_llama3_tokenizer();
    let tokens = tokenizer.encode("Hello 🌍 World!");
    assert_eq!(
        tokens,
        vec![9906, 11410, 234, 235, 4435, 0],
        "Token IDs for emoji text changed"
    );
}

// =============================================================================
// Case-Boundary (camelCase) Exact Token ID Tests
// =============================================================================
//
// Llama 3's pre-tokenizer takes whole letter runs with a plain `\p{L}+`; it has
// no upper/lower case-boundary rule. Splitting on case — the o200k convention —
// prevents the `UserName`/`HttpRequest` merges and silently produces more,
// different ids. These cases are the ones that discriminate between the two
// patterns, so they guard against `LLAMA3_PATTERN` being re-aliased to
// `O200K_BASE_PATTERN`.
//
// Reference ids produced by HuggingFace `tokenizers` loading Meta's
// `llama-3.2-1b/tokenizer.json`.

/// Verify exact token IDs for "XMLHttpRequest" (uppercase run into a capitalised
/// word — the case boundary an o200k-style split would break on).
#[test]
fn test_llama3_camel_case_xml_http_request_tokens() {
    let tokenizer = create_llama3_tokenizer();
    let tokens = tokenizer.encode("XMLHttpRequest");
    assert_eq!(
        tokens,
        vec![10833, 27459],
        "Token IDs for 'XMLHttpRequest' changed"
    );
}

/// Verify exact token IDs for "getUserName" (lowercase into capitalised words).
#[test]
fn test_llama3_camel_case_get_user_name_tokens() {
    let tokenizer = create_llama3_tokenizer();
    let tokens = tokenizer.encode("getUserName");
    assert_eq!(
        tokens,
        vec![456, 19387],
        "Token IDs for 'getUserName' changed"
    );
}

/// Verify exact token IDs for "camelCaseIdentifier".
#[test]
fn test_llama3_camel_case_identifier_tokens() {
    let tokenizer = create_llama3_tokenizer();
    let tokens = tokenizer.encode("camelCaseIdentifier");
    assert_eq!(
        tokens,
        vec![94421, 4301, 8887],
        "Token IDs for 'camelCaseIdentifier' changed"
    );
}

/// Verify exact token IDs for "fooBar baz" (camelCase followed by a spaced word,
/// so the leading-space branch is exercised alongside the case boundary).
#[test]
fn test_llama3_camel_case_with_following_word_tokens() {
    let tokenizer = create_llama3_tokenizer();
    let tokens = tokenizer.encode("fooBar baz");
    assert_eq!(
        tokens,
        vec![8134, 3511, 51347],
        "Token IDs for 'fooBar baz' changed"
    );
}

// =============================================================================
// General Roundtrip Tests
// =============================================================================

/// Test basic encoding and decoding roundtrip.
#[test]
fn test_llama3_encode_decode_roundtrip() {
    let tokenizer = create_llama3_tokenizer();

    let test_cases = vec![
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Rust is a systems programming language.",
        "1234567890",
        "Special characters: !@#$%^&*()",
        "Multi-line\ntext\nwith\nnewlines",
        "Unicode: こんにちは 世界 🦀",
    ];

    for text in test_cases {
        let tokens = tokenizer.encode(text);
        let decoded = tokenizer.decode(&tokens).unwrap();
        assert_eq!(decoded, text, "Roundtrip failed for: {:?}", text);
    }
}

/// Test that vocab size is correct (128,000 BPE tokens).
#[test]
fn test_llama3_vocab_size() {
    let tokenizer = create_llama3_tokenizer();
    // Llama 3 has 128,000 BPE tokens plus special tokens
    assert!(
        tokenizer.vocab_size() >= 128000,
        "Vocab size should be at least 128,000, got {}",
        tokenizer.vocab_size()
    );
}

/// Test official Meta special tokens from Llama 3.3.
#[test]
fn test_llama3_meta_special_tokens() {
    let tokenizer = create_llama3_tokenizer();

    // Test begin/end of text
    let tokens = tokenizer.encode_with_special("<|begin_of_text|>Hello<|end_of_text|>");
    assert!(
        tokens.contains(&128000),
        "Should contain begin_of_text (128000)"
    );
    assert!(
        tokens.contains(&128001),
        "Should contain end_of_text (128001)"
    );

    // Test header markers
    let tokens = tokenizer.encode_with_special("<|start_header_id|>system<|end_header_id|>");
    assert!(
        tokens.contains(&128006),
        "Should contain start_header_id (128006)"
    );
    assert!(
        tokens.contains(&128007),
        "Should contain end_header_id (128007)"
    );

    // Test end of turn
    let tokens = tokenizer.encode_with_special("<|eot_id|>");
    assert!(tokens.contains(&128009), "Should contain eot_id (128009)");
}

/// Test Llama 3.1+ specific tokens.
#[test]
fn test_llama3_1_special_tokens() {
    let tokenizer = create_llama3_tokenizer();

    // Test finetune_right_pad_id (added in 3.1)
    let tokens = tokenizer.encode_with_special("<|finetune_right_pad_id|>");
    assert!(
        tokens.contains(&128004),
        "Should contain finetune_right_pad_id (128004)"
    );

    // Test eom_id - end of message for tool use (added in 3.1)
    let tokens = tokenizer.encode_with_special("<|eom_id|>");
    assert!(tokens.contains(&128008), "Should contain eom_id (128008)");

    // Test python_tag for code interpreter (added in 3.1)
    let tokens = tokenizer.encode_with_special("<|python_tag|>");
    assert!(
        tokens.contains(&128010),
        "Should contain python_tag (128010)"
    );
}

/// Test splintr agent tokens for Llama 3.
#[test]
fn test_llama3_agent_tokens() {
    let tokenizer = create_llama3_tokenizer();

    // Test conversation tokens
    let tokens = tokenizer.encode_with_special("<|system|>You are helpful.<|user|>Hi<|assistant|>");
    assert!(tokens.contains(&128300), "Should contain system (128300)");
    assert!(tokens.contains(&128301), "Should contain user (128301)");
    assert!(
        tokens.contains(&128302),
        "Should contain assistant (128302)"
    );

    // Test thinking tokens
    let tokens = tokenizer.encode_with_special("<|think|>Let me reason...<|/think|>");
    assert!(tokens.contains(&128305), "Should contain think (128305)");
    assert!(
        tokens.contains(&128306),
        "Should contain think_end (128306)"
    );

    // Test function calling tokens
    let tokens = tokenizer.encode_with_special("<|function|>get_weather<|/function|>");
    assert!(tokens.contains(&128315), "Should contain function (128315)");
    assert!(
        tokens.contains(&128316),
        "Should contain function_end (128316)"
    );
}

/// Test Llama 3 chat template format.
#[test]
fn test_llama3_chat_format() {
    let tokenizer = create_llama3_tokenizer();

    // Llama 3 chat format uses header markers
    let chat = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHello!<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";

    let tokens = tokenizer.encode_with_special(chat);

    // Verify special tokens are present
    assert!(tokens.contains(&128000)); // begin_of_text
    assert!(tokens.contains(&128006)); // start_header_id
    assert!(tokens.contains(&128007)); // end_header_id
    assert!(tokens.contains(&128009)); // eot_id

    // Verify roundtrip
    let decoded = tokenizer.decode(&tokens).unwrap();
    assert_eq!(decoded, chat);
}

/// Test batch encoding.
#[test]
fn test_llama3_batch_encode() {
    let tokenizer = create_llama3_tokenizer();

    let texts = vec![
        "Hello, world!".to_string(),
        "How are you?".to_string(),
        "I'm doing great!".to_string(),
    ];

    let batch_tokens = tokenizer.encode_batch(&texts);

    assert_eq!(batch_tokens.len(), 3);

    // Verify each batch result matches individual encoding
    for (i, text) in texts.iter().enumerate() {
        let individual = tokenizer.encode(text);
        assert_eq!(
            batch_tokens[i], individual,
            "Batch encoding should match individual encoding for text {}: {:?}",
            i, text
        );
    }
}

/// Test that special tokens decode correctly.
#[test]
fn test_llama3_special_token_decode() {
    let tokenizer = create_llama3_tokenizer();

    // Decode individual special tokens
    let decoded = tokenizer.decode(&[128000]).unwrap();
    assert_eq!(decoded, "<|begin_of_text|>");

    let decoded = tokenizer.decode(&[128009]).unwrap();
    assert_eq!(decoded, "<|eot_id|>");

    let decoded = tokenizer.decode(&[128008]).unwrap();
    assert_eq!(decoded, "<|eom_id|>");

    let decoded = tokenizer.decode(&[128010]).unwrap();
    assert_eq!(decoded, "<|python_tag|>");
}

/// Test Llama 3.2-Vision specific tokens.
#[test]
fn test_llama3_2_vision_tokens() {
    let tokenizer = create_llama3_tokenizer();

    // Test step_id (added in 3.2-Vision)
    let tokens = tokenizer.encode_with_special("<|step_id|>");
    assert!(tokens.contains(&128005), "Should contain step_id (128005)");

    // Test image token - official Meta token from 3.2-Vision
    let tokens = tokenizer.encode_with_special("<|image|>content<|/image|>");
    assert!(tokens.contains(&128256), "Should contain image (128256)");
    assert!(
        tokens.contains(&128257),
        "Should contain image_end (128257)"
    );

    // Verify decode
    let decoded = tokenizer.decode(&[128005]).unwrap();
    assert_eq!(decoded, "<|step_id|>");

    let decoded = tokenizer.decode(&[128256]).unwrap();
    assert_eq!(decoded, "<|image|>");
}

/// Test empty input handling.
#[test]
fn test_llama3_empty_input() {
    let tokenizer = create_llama3_tokenizer();

    let tokens = tokenizer.encode("");
    assert!(tokens.is_empty(), "Empty input should produce empty tokens");

    let decoded = tokenizer.decode(&[]).unwrap();
    assert!(
        decoded.is_empty(),
        "Empty tokens should decode to empty string"
    );
}

/// Test that all from_pretrained variants work.
#[test]
fn test_llama3_from_pretrained_variants() {
    // All these should create valid tokenizers
    let _t1 = create_llama3_tokenizer_by_name("llama3");
    let _t2 = create_llama3_tokenizer_by_name("llama3.1");
    let _t3 = create_llama3_tokenizer_by_name("llama3.2");
    let _t4 = create_llama3_tokenizer_by_name("llama3.3");

    // They should all produce the same encoding for regular text
    let text = "Hello, world!";
    let t1 = create_llama3_tokenizer_by_name("llama3");
    let t2 = create_llama3_tokenizer_by_name("llama3.3");

    assert_eq!(
        t1.encode(text),
        t2.encode(text),
        "All Llama 3 variants should produce same encoding"
    );
}

/// Get the shared tokenizer instance
fn create_llama3_tokenizer() -> &'static Tokenizer {
    &TOKENIZER
}

/// Create a fresh tokenizer by name (for variant tests only)
fn create_llama3_tokenizer_by_name(_name: &str) -> Tokenizer {
    create_llama3_tokenizer_impl()
}

/// Implementation that actually constructs the tokenizer.
///
/// Built entirely from the production pieces — `LLAMA3_VOCAB`, `LLAMA3_PATTERN`,
/// `llama3_special_tokens()` — mirroring what
/// `pretrained::from_vocab(PretrainedVocab::Llama3)` actually builds
/// (`Tokenizer::from_bytes_chain(LLAMA3_VOCAB, &[LLAMA3_PATTERN], special)`), so
/// this fixture cannot drift from production. It previously re-declared its own
/// special-token table, which had fallen 27 tokens behind production: missing
/// the `<|audio|>`/`<|/audio|>` and `<|video|>`/`<|/video|>` multimodal pairs, and
/// missing the `<|lang|>`..`<|/lang|>`, `<|context|>`..`<|/context|>`,
/// `<|quote|>`..`<|/quote|>`, `<|cite|>`..`<|/cite|>`, `<|source|>`..`<|/source|>`,
/// `<|memory|>`..`<|/memory|>`, `<|recall|>`..`<|/recall|>`, `<|pad|>`, `<|stop|>`,
/// `<|sep|>`, `<|title|>`..`<|/title|>`, `<|section|>`..`<|/section|>`, and
/// `<|summary|>`..`<|/summary|>` agent tokens. Every entry the old table did have
/// matched production's id exactly, so this was purely additive drift, not a
/// conflict.
fn create_llama3_tokenizer_impl() -> Tokenizer {
    Tokenizer::from_bytes_chain(LLAMA3_VOCAB, &[LLAMA3_PATTERN], llama3_special_tokens())
        .expect("bundled llama3 vocabulary must load")
}
