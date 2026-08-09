//! Integration tests for o200k_base tokenizer (GPT-4o).
//!
//! These tests verify that the o200k_base tokenizer correctly encodes and decodes text,
//! handles special tokens, and produces consistent results.
// Gated on `vocab-o200k` because it names that vocabulary's embedded bytes directly, and the feature is what
// compiles those bytes in. Without it this crate is empty rather than
// a compile error.
#![cfg(feature = "vocab-o200k")]

use splintr::pretrained::{o200k_base_special_tokens, O200K_BASE_VOCAB_PACKED};
use splintr::{Tokenizer, O200K_BASE_PATTERN};
use std::sync::LazyLock;

/// Shared tokenizer instance to avoid expensive re-initialization per test.
static TOKENIZER: LazyLock<Tokenizer> = LazyLock::new(create_o200k_tokenizer_impl);

// =============================================================================
// Exact Token ID Tests
// =============================================================================

/// Verify exact token IDs for "Hello world".
#[test]
fn test_o200k_hello_world_tokens() {
    let tokenizer = create_o200k_tokenizer();
    let tokens = tokenizer.encode("Hello world");
    assert_eq!(
        tokens,
        vec![13225, 2375],
        "Token IDs for 'Hello world' changed"
    );
}

/// Verify exact token IDs for "Hello, world!".
#[test]
fn test_o200k_hello_world_punctuation_tokens() {
    let tokenizer = create_o200k_tokenizer();
    let tokens = tokenizer.encode("Hello, world!");
    assert_eq!(
        tokens,
        vec![13225, 11, 2375, 0],
        "Token IDs for 'Hello, world!' changed"
    );
}

/// Verify exact token IDs for "你好世界".
#[test]
fn test_o200k_chinese_tokens() {
    let tokenizer = create_o200k_tokenizer();
    let tokens = tokenizer.encode("你好世界");
    assert_eq!(
        tokens,
        vec![177519, 28428],
        "Token IDs for '你好世界' changed"
    );
}

/// Verify exact token IDs for "Hello 🌍 World!".
#[test]
fn test_o200k_emoji_tokens() {
    let tokenizer = create_o200k_tokenizer();
    let tokens = tokenizer.encode("Hello 🌍 World!");
    assert_eq!(
        tokens,
        vec![13225, 130321, 235, 5922, 0],
        "Token IDs for emoji text changed"
    );
}

/// Verify that `O200K_BASE_PATTERN` is actually in effect, not
/// `CL100K_BASE_PATTERN` (or `LLAMA3_PATTERN`).
///
/// o200k_base's letter run splits on an uppercase/lowercase case boundary and
/// folds a trailing contraction suffix onto the preceding word; cl100k_base's
/// letter run is a plain `\p{L}+` with no case rule, and treats the contraction
/// suffix as its own leading token. Every input below produces different ids
/// under the two patterns, so a mixup here cannot pass silently the way it did
/// for DeepSeek V3 (see `tests/deepseek_v3.rs`).
///
/// Every expected id sequence was produced by `tiktoken.get_encoding("o200k_base").encode(text)`,
/// never by recording splintr's own output.
#[test]
fn test_o200k_letter_runs_split_on_case() {
    let tokenizer = create_o200k_tokenizer();

    // camelCase identifier: cl100k_base wrongly gives [456, 19387] (no split).
    assert_eq!(
        tokenizer.encode("getUserName"),
        vec![522, 1844, 864],
        "camelCase identifier not split on the case boundary"
    );

    // PascalCase with a leading acronym run: cl100k_base wrongly gives
    // [10833, 27459] (no split).
    assert_eq!(
        tokenizer.encode("XMLHttpRequest"),
        vec![13836, 4682, 2303],
        "PascalCase identifier with acronym not split on the case boundary"
    );

    // PascalCase, all-caps acronym: cl100k_base wrongly gives [64865, 3126].
    assert_eq!(
        tokenizer.encode("HTTPRequestHandler"),
        vec![159684, 4139],
        "PascalCase identifier with acronym not split on the case boundary"
    );

    // Leading punctuation immediately followed by a letter run, no space:
    // cl100k_base wrongly gives [33261, 4886].
    assert_eq!(
        tokenizer.encode(".isValidEmail"),
        vec![3109, 5258, 6622],
        "punctuation-then-letters branch mis-split"
    );

    // A method-call chain exercising the punctuation branch again:
    // cl100k_base wrongly gives [1710, 12165, 368].
    assert_eq!(
        tokenizer.encode("config.getValue()"),
        vec![3154, 775, 1638, 416],
        "punctuation-then-letters branch mis-split"
    );

    // Contraction: o200k_base folds the apostrophe-suffix onto the preceding
    // letter run; cl100k_base wrongly gives [285, 77, 956] (suffix split off).
    assert_eq!(
        tokenizer.encode("isn't"),
        vec![276, 3023],
        "contraction handling mis-split"
    );

    // Long digit run: both patterns chunk `\p{N}{1,3}`, but the two vocabularies
    // assign different ids to the same chunks. cl100k_base wrongly gives
    // [4513, 10961, 16474, 11531, 12901, 17458, 1954].
    assert_eq!(
        tokenizer.encode("12345678901234567890"),
        vec![7633, 19354, 29338, 19267, 22901, 30833, 2744],
        "long digit run mis-split or mis-encoded"
    );

    // Punctuation + digit-run mix. cl100k_base wrongly gives
    // [726, 8033, 2120, 11, 220, 2983, 8].
    assert_eq!(
        tokenizer.encode("self.assertEqual(x, 42)"),
        vec![1156, 6429, 5676, 4061, 11, 220, 4689, 8],
        "punctuation+identifier / digit-run mix mis-split"
    );

    // CJK ideograph run. cl100k_base wrongly gives
    // [70090, 23530, 56235, 85315, 222, 24775].
    assert_eq!(
        tokenizer.encode("北京市海淀区"),
        vec![141026, 12426, 11787, 222, 5243],
        "CJK ideograph run mis-split"
    );

    // Mixed-script text, no whitespace between scripts. cl100k_base wrongly
    // gives [87533, 85315, 115, 40862, 1199, 88435].
    assert_eq!(
        tokenizer.encode("Mixed混合Text文字"),
        vec![97258, 85591, 4377, 1279, 79831],
        "mixed-script run mis-split"
    );
}

// =============================================================================
// General Roundtrip Tests
// =============================================================================

/// Test basic encoding and decoding roundtrip.
#[test]
fn test_o200k_encode_decode_roundtrip() {
    let tokenizer = create_o200k_tokenizer();

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

/// Test that vocab size is correct (199,998 BPE tokens for o200k).
#[test]
fn test_o200k_vocab_size() {
    let tokenizer = create_o200k_tokenizer();
    // o200k_base has 199,998 BPE tokens plus special tokens
    assert!(
        tokenizer.vocab_size() >= 199998,
        "Vocab size should be at least 199,998, got {}",
        tokenizer.vocab_size()
    );
}

/// Test OpenAI standard special tokens.
#[test]
fn test_o200k_openai_special_tokens() {
    let tokenizer = create_o200k_tokenizer();

    // Test endoftext
    let tokens = tokenizer.encode_with_special("Hello<|endoftext|>World");
    assert!(
        tokens.contains(&199999),
        "Should contain endoftext (199999)"
    );

    // Test endofprompt
    let tokens = tokenizer.encode_with_special("<|endofprompt|>");
    assert!(
        tokens.contains(&200018),
        "Should contain endofprompt (200018)"
    );
}

/// Test splintr agent tokens for o200k.
#[test]
fn test_o200k_agent_tokens() {
    let tokenizer = create_o200k_tokenizer();

    // Test conversation tokens
    let tokens = tokenizer.encode_with_special("<|system|>You are helpful.<|user|>Hi<|assistant|>");
    assert!(tokens.contains(&200019), "Should contain system (200019)");
    assert!(tokens.contains(&200020), "Should contain user (200020)");
    assert!(
        tokens.contains(&200021),
        "Should contain assistant (200021)"
    );

    // Test thinking tokens
    let tokens = tokenizer.encode_with_special("<|think|>Let me reason...<|/think|>");
    assert!(tokens.contains(&200024), "Should contain think (200024)");
    assert!(
        tokens.contains(&200025),
        "Should contain think_end (200025)"
    );

    // Test function calling tokens
    let tokens = tokenizer.encode_with_special("<|function|>get_weather<|/function|>");
    assert!(tokens.contains(&200034), "Should contain function (200034)");
    assert!(
        tokens.contains(&200035),
        "Should contain function_end (200035)"
    );
}

/// Test ChatML format commonly used with GPT models.
#[test]
fn test_o200k_chatml_format() {
    let tokenizer = create_o200k_tokenizer();

    // ChatML format uses im_start/im_end
    let chat = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n";

    let tokens = tokenizer.encode_with_special(chat);

    // Verify special tokens are present
    assert!(tokens.contains(&200022)); // im_start
    assert!(tokens.contains(&200023)); // im_end

    // Verify roundtrip
    let decoded = tokenizer.decode(&tokens).unwrap();
    assert_eq!(decoded, chat);
}

/// Test batch encoding.
#[test]
fn test_o200k_batch_encode() {
    let tokenizer = create_o200k_tokenizer();

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
fn test_o200k_special_token_decode() {
    let tokenizer = create_o200k_tokenizer();

    // Decode individual special tokens
    let decoded = tokenizer.decode(&[199999]).unwrap();
    assert_eq!(decoded, "<|endoftext|>");

    let decoded = tokenizer.decode(&[200018]).unwrap();
    assert_eq!(decoded, "<|endofprompt|>");
}

/// Test empty input handling.
#[test]
fn test_o200k_empty_input() {
    let tokenizer = create_o200k_tokenizer();

    let tokens = tokenizer.encode("");
    assert!(tokens.is_empty(), "Empty input should produce empty tokens");

    let decoded = tokenizer.decode(&[]).unwrap();
    assert!(
        decoded.is_empty(),
        "Empty tokens should decode to empty string"
    );
}

/// Test code-related content (GPT-4o is commonly used for code).
#[test]
fn test_o200k_code_content() {
    let tokenizer = create_o200k_tokenizer();

    let code = r#"
def hello_world():
    print("Hello, World!")

if __name__ == "__main__":
    hello_world()
"#;

    let tokens = tokenizer.encode(code);
    let decoded = tokenizer.decode(&tokens).unwrap();
    assert_eq!(decoded, code);
}

/// Test multimodal placeholder tokens (GPT-4o supports vision).
#[test]
fn test_o200k_multimodal_tokens() {
    let tokenizer = create_o200k_tokenizer();

    // Test image tokens. The spelling is `<|image|>`/`<|/image|>`, matching
    // `insert_agent_tokens` (`pretrained.rs:632`) and every other vocabulary;
    // this test previously asserted an `<|image_start|>` spelling that no
    // production code has ever emitted.
    let tokens = tokenizer.encode_with_special("<|image|>image data<|/image|>");
    assert!(tokens.contains(&200061), "Should contain image (200061)");
    assert!(
        tokens.contains(&200062),
        "Should contain image_end (200062)"
    );

    // Test audio tokens
    let tokens = tokenizer.encode_with_special("<|audio|>audio data<|/audio|>");
    assert!(tokens.contains(&200063), "Should contain audio (200063)");
    assert!(
        tokens.contains(&200064),
        "Should contain audio_end (200064)"
    );
}

/// Test that o200k has larger vocab than cl100k.
#[test]
fn test_o200k_larger_than_cl100k() {
    let o200k = create_o200k_tokenizer();

    // o200k should have ~200k tokens vs cl100k's ~100k
    assert!(
        o200k.vocab_size() > 150000,
        "o200k should have more than 150k tokens"
    );
}

/// Get the shared tokenizer instance
fn create_o200k_tokenizer() -> &'static Tokenizer {
    &TOKENIZER
}

/// Implementation that actually constructs the tokenizer.
///
/// Built entirely from the production pieces — `O200K_BASE_VOCAB_PACKED`,
/// `O200K_BASE_PATTERN`, `o200k_base_special_tokens()` — so this fixture cannot
/// drift from what `pretrained::from_vocab(PretrainedVocab::O200kBase)` actually
/// builds. It previously re-declared its own special-token table, and such a
/// second source of truth is what let `tests/deepseek_v3.rs` sit green while the
/// production loader used the wrong pre-tokenizer. The stale copy here had also
/// fallen 29 agent tokens behind and had invented an `<|image_start|>` spelling
/// that no production code emits (the canonical pair is `<|image|>`/`<|/image|>`,
/// `pretrained.rs:632`).
fn create_o200k_tokenizer_impl() -> Tokenizer {
    Tokenizer::from_packed_chain(
        O200K_BASE_VOCAB_PACKED,
        &[O200K_BASE_PATTERN],
        o200k_base_special_tokens(),
    )
    .expect("bundled o200k_base vocabulary must load")
}
