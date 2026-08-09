//! Integration tests for cl100k_base tokenizer (GPT-4, GPT-3.5-turbo).
//!
//! These tests verify that the cl100k_base tokenizer correctly encodes and decodes text,
//! handles special tokens, and produces consistent results.
// Gated on `vocab-cl100k` because it names that vocabulary's embedded bytes directly, and the feature is what
// compiles those bytes in. Without it this crate is empty rather than
// a compile error.
#![cfg(feature = "vocab-cl100k")]

use splintr::pretrained::{cl100k_base_special_tokens, CL100K_BASE_VOCAB_PACKED};
use splintr::{Tokenizer, CL100K_BASE_PATTERN};
use std::sync::LazyLock;

/// Shared tokenizer instance to avoid expensive re-initialization per test.
static TOKENIZER: LazyLock<Tokenizer> = LazyLock::new(create_cl100k_tokenizer_impl);

// =============================================================================
// Exact Token ID Tests
// =============================================================================

/// Verify exact token IDs for "Hello world".
#[test]
fn test_cl100k_hello_world_tokens() {
    let tokenizer = create_cl100k_tokenizer();
    let tokens = tokenizer.encode("Hello world");
    assert_eq!(
        tokens,
        vec![9906, 1917],
        "Token IDs for 'Hello world' changed"
    );
}

/// Verify exact token IDs for "Hello, world!".
#[test]
fn test_cl100k_hello_world_punctuation_tokens() {
    let tokenizer = create_cl100k_tokenizer();
    let tokens = tokenizer.encode("Hello, world!");
    assert_eq!(
        tokens,
        vec![9906, 11, 1917, 0],
        "Token IDs for 'Hello, world!' changed"
    );
}

/// Verify exact token IDs for "你好世界".
#[test]
fn test_cl100k_chinese_tokens() {
    let tokenizer = create_cl100k_tokenizer();
    let tokens = tokenizer.encode("你好世界");
    assert_eq!(
        tokens,
        vec![57668, 53901, 3574, 244, 98220],
        "Token IDs for '你好世界' changed"
    );
}

/// Verify exact token IDs for "Hello 🌍 World!".
#[test]
fn test_cl100k_emoji_tokens() {
    let tokenizer = create_cl100k_tokenizer();
    let tokens = tokenizer.encode("Hello 🌍 World!");
    assert_eq!(
        tokens,
        vec![9906, 11410, 234, 235, 4435, 0],
        "Token IDs for emoji text changed"
    );
}

/// Verify that `CL100K_BASE_PATTERN` is actually in effect, not
/// `O200K_BASE_PATTERN`.
///
/// cl100k_base's letter run is a plain `\p{L}+` with no case-boundary rule and
/// its contractions are a separate leading `'(?i:[sdmt]|ll|ve|re)` alternative;
/// o200k_base splits letter runs on an uppercase/lowercase boundary and folds
/// the contraction suffix onto the preceding word instead. Every input below
/// produces different ids under the two patterns, so a mixup here cannot pass
/// silently the way it did for DeepSeek V3 (see `tests/deepseek_v3.rs`).
///
/// Every expected id sequence was produced by `tiktoken.get_encoding("cl100k_base").encode(text)`,
/// never by recording splintr's own output.
#[test]
fn test_cl100k_letter_runs_not_split_on_case() {
    let tokenizer = create_cl100k_tokenizer();

    // camelCase identifier: o200k_base wrongly gives [522, 1844, 864].
    assert_eq!(
        tokenizer.encode("getUserName"),
        vec![456, 19387],
        "camelCase identifier split on a case boundary"
    );

    // PascalCase with a leading acronym run: o200k_base wrongly gives [13836, 4682, 2303].
    assert_eq!(
        tokenizer.encode("XMLHttpRequest"),
        vec![10833, 27459],
        "PascalCase identifier with acronym split on a case boundary"
    );

    // PascalCase, all-caps acronym: o200k_base wrongly gives [159684, 4139].
    assert_eq!(
        tokenizer.encode("HTTPRequestHandler"),
        vec![64865, 3126],
        "PascalCase identifier with acronym split on a case boundary"
    );

    // Leading punctuation immediately followed by a letter run, no space:
    // o200k_base wrongly gives [3109, 5258, 6622].
    assert_eq!(
        tokenizer.encode(".isValidEmail"),
        vec![33261, 4886],
        "punctuation-then-letters branch mis-split"
    );

    // A method-call chain exercising the punctuation branch again:
    // o200k_base wrongly gives [3154, 775, 1638, 416].
    assert_eq!(
        tokenizer.encode("config.getValue()"),
        vec![1710, 12165, 368],
        "punctuation-then-letters branch mis-split"
    );

    // Contraction: cl100k_base tokenizes the apostrophe-suffix as its own leading
    // token; o200k_base folds it onto the preceding letter run and wrongly gives
    // [276, 3023].
    assert_eq!(
        tokenizer.encode("isn't"),
        vec![285, 77, 956],
        "contraction handling mis-split"
    );

    // Long digit run: both patterns chunk `\p{N}{1,3}`, but the two vocabularies
    // assign different ids to the same chunks. o200k_base wrongly gives
    // [7633, 19354, 29338, 19267, 22901, 30833, 2744].
    assert_eq!(
        tokenizer.encode("12345678901234567890"),
        vec![4513, 10961, 16474, 11531, 12901, 17458, 1954],
        "long digit run mis-split or mis-encoded"
    );

    // Punctuation + digit-run mix. o200k_base wrongly gives
    // [1156, 6429, 5676, 4061, 11, 220, 4689, 8].
    assert_eq!(
        tokenizer.encode("self.assertEqual(x, 42)"),
        vec![726, 8033, 2120, 11, 220, 2983, 8],
        "punctuation+identifier / digit-run mix mis-split"
    );

    // CJK ideograph run. o200k_base wrongly gives [141026, 12426, 11787, 222, 5243].
    assert_eq!(
        tokenizer.encode("北京市海淀区"),
        vec![70090, 23530, 56235, 85315, 222, 24775],
        "CJK ideograph run mis-split"
    );

    // Mixed-script text, no whitespace between scripts. o200k_base wrongly gives
    // [97258, 85591, 4377, 1279, 79831].
    assert_eq!(
        tokenizer.encode("Mixed混合Text文字"),
        vec![87533, 85315, 115, 40862, 1199, 88435],
        "mixed-script run mis-split"
    );
}

// =============================================================================
// General Roundtrip Tests
// =============================================================================

/// Test basic encoding and decoding roundtrip.
#[test]
fn test_cl100k_encode_decode_roundtrip() {
    let tokenizer = create_cl100k_tokenizer();

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

/// Test that vocab size is correct (100,256 BPE tokens for cl100k).
#[test]
fn test_cl100k_vocab_size() {
    let tokenizer = create_cl100k_tokenizer();
    // cl100k_base has 100,256 BPE tokens plus special tokens
    assert!(
        tokenizer.vocab_size() >= 100256,
        "Vocab size should be at least 100,256, got {}",
        tokenizer.vocab_size()
    );
}

/// Test OpenAI standard special tokens.
#[test]
fn test_cl100k_openai_special_tokens() {
    let tokenizer = create_cl100k_tokenizer();

    // Test endoftext
    let tokens = tokenizer.encode_with_special("Hello<|endoftext|>World");
    assert!(
        tokens.contains(&100257),
        "Should contain endoftext (100257)"
    );

    // Test fim tokens
    let tokens = tokenizer.encode_with_special("<|fim_prefix|>code<|fim_middle|>");
    assert!(
        tokens.contains(&100258),
        "Should contain fim_prefix (100258)"
    );
    assert!(
        tokens.contains(&100259),
        "Should contain fim_middle (100259)"
    );

    // Test fim_suffix
    let tokens = tokenizer.encode_with_special("<|fim_suffix|>");
    assert!(
        tokens.contains(&100260),
        "Should contain fim_suffix (100260)"
    );

    // Test endofprompt
    let tokens = tokenizer.encode_with_special("<|endofprompt|>");
    assert!(
        tokens.contains(&100276),
        "Should contain endofprompt (100276)"
    );
}

/// Test splintr agent tokens for cl100k.
#[test]
fn test_cl100k_agent_tokens() {
    let tokenizer = create_cl100k_tokenizer();

    // Test conversation tokens
    let tokens = tokenizer.encode_with_special("<|system|>You are helpful.<|user|>Hi<|assistant|>");
    assert!(tokens.contains(&100277), "Should contain system (100277)");
    assert!(tokens.contains(&100278), "Should contain user (100278)");
    assert!(
        tokens.contains(&100279),
        "Should contain assistant (100279)"
    );

    // Test thinking tokens
    let tokens = tokenizer.encode_with_special("<|think|>Let me reason...<|/think|>");
    assert!(tokens.contains(&100282), "Should contain think (100282)");
    assert!(
        tokens.contains(&100283),
        "Should contain think_end (100283)"
    );

    // Test function calling tokens
    let tokens = tokenizer.encode_with_special("<|function|>get_weather<|/function|>");
    assert!(tokens.contains(&100292), "Should contain function (100292)");
    assert!(
        tokens.contains(&100293),
        "Should contain function_end (100293)"
    );
}

/// Test ChatML format commonly used with GPT models.
#[test]
fn test_cl100k_chatml_format() {
    let tokenizer = create_cl100k_tokenizer();

    // ChatML format uses im_start/im_end
    let chat = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n";

    let tokens = tokenizer.encode_with_special(chat);

    // Verify special tokens are present
    assert!(tokens.contains(&100280)); // im_start
    assert!(tokens.contains(&100281)); // im_end

    // Verify roundtrip
    let decoded = tokenizer.decode(&tokens).unwrap();
    assert_eq!(decoded, chat);
}

/// Test batch encoding.
#[test]
fn test_cl100k_batch_encode() {
    let tokenizer = create_cl100k_tokenizer();

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
fn test_cl100k_special_token_decode() {
    let tokenizer = create_cl100k_tokenizer();

    // Decode individual special tokens
    let decoded = tokenizer.decode(&[100257]).unwrap();
    assert_eq!(decoded, "<|endoftext|>");

    let decoded = tokenizer.decode(&[100258]).unwrap();
    assert_eq!(decoded, "<|fim_prefix|>");

    let decoded = tokenizer.decode(&[100276]).unwrap();
    assert_eq!(decoded, "<|endofprompt|>");
}

/// Test empty input handling.
#[test]
fn test_cl100k_empty_input() {
    let tokenizer = create_cl100k_tokenizer();

    let tokens = tokenizer.encode("");
    assert!(tokens.is_empty(), "Empty input should produce empty tokens");

    let decoded = tokenizer.decode(&[]).unwrap();
    assert!(
        decoded.is_empty(),
        "Empty tokens should decode to empty string"
    );
}

/// Test code-related content (GPT-4 is commonly used for code).
#[test]
fn test_cl100k_code_content() {
    let tokenizer = create_cl100k_tokenizer();

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

/// Test FIM (Fill-in-the-Middle) format used for code completion.
#[test]
fn test_cl100k_fim_format() {
    let tokenizer = create_cl100k_tokenizer();

    let fim = "<|fim_prefix|>def hello():\n    <|fim_suffix|>\n    return result<|fim_middle|>";

    let tokens = tokenizer.encode_with_special(fim);

    // Verify FIM tokens are present
    assert!(tokens.contains(&100258)); // fim_prefix
    assert!(tokens.contains(&100259)); // fim_middle
    assert!(tokens.contains(&100260)); // fim_suffix

    // Verify roundtrip
    let decoded = tokenizer.decode(&tokens).unwrap();
    assert_eq!(decoded, fim);
}

/// Get the shared tokenizer instance
fn create_cl100k_tokenizer() -> &'static Tokenizer {
    &TOKENIZER
}

/// Implementation that actually constructs the tokenizer.
///
/// Built entirely from the production pieces — `CL100K_BASE_VOCAB_PACKED`,
/// `CL100K_BASE_PATTERN`, `cl100k_base_special_tokens()` — mirroring what
/// `pretrained::from_vocab(PretrainedVocab::Cl100kBase)` actually builds
/// (`Tokenizer::from_packed_chain(CL100K_BASE_VOCAB_PACKED, &[CL100K_BASE_PATTERN],
/// special)`), so this fixture cannot drift from production. It previously
/// re-declared its own special-token table, which stopped at `<|/output|>`
/// (100301) and was missing the 29 agent tokens production adds after it:
/// `<|lang|>`..`<|/lang|>`, `<|context|>`..`<|/context|>`, `<|quote|>`..`<|/quote|>`,
/// `<|cite|>`..`<|/cite|>`, `<|source|>`..`<|/source|>`, `<|memory|>`..`<|/memory|>`,
/// `<|recall|>`..`<|/recall|>`, `<|pad|>`, `<|stop|>`, `<|sep|>`,
/// `<|image|>`..`<|/image|>`, `<|audio|>`..`<|/audio|>`, `<|video|>`..`<|/video|>`,
/// `<|title|>`..`<|/title|>`, `<|section|>`..`<|/section|>`, and
/// `<|summary|>`..`<|/summary|>`. Every entry the old table did have matched
/// production's id exactly, so this was purely additive drift, not a conflict.
fn create_cl100k_tokenizer_impl() -> Tokenizer {
    Tokenizer::from_packed_chain(
        CL100K_BASE_VOCAB_PACKED,
        &[CL100K_BASE_PATTERN],
        cl100k_base_special_tokens(),
    )
    .expect("bundled cl100k_base vocabulary must load")
}
