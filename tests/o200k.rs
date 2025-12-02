//! Integration tests for o200k_base tokenizer (GPT-4o).
//!
//! These tests verify that the o200k_base tokenizer correctly encodes and decodes text,
//! handles special tokens, and produces consistent results.

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

    // Test image tokens
    let tokens = tokenizer.encode_with_special("<|image_start|>image data<|/image_start|>");
    assert!(
        tokens.contains(&200061),
        "Should contain image_start (200061)"
    );
    assert!(
        tokens.contains(&200062),
        "Should contain image_start_end (200062)"
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

/// Implementation that actually constructs the tokenizer
fn create_o200k_tokenizer_impl() -> Tokenizer {
    // Load the embedded vocab
    let vocab_bytes = include_bytes!("../python/splintr/vocabs/o200k_base.tiktoken");

    let mut special = rustc_hash::FxHashMap::default();

    // OpenAI standard special tokens
    special.insert("<|endoftext|>".to_string(), 199999);
    special.insert("<|endofprompt|>".to_string(), 200018);

    // Agent tokens (200019+)
    special.insert("<|system|>".to_string(), 200019);
    special.insert("<|user|>".to_string(), 200020);
    special.insert("<|assistant|>".to_string(), 200021);
    special.insert("<|im_start|>".to_string(), 200022);
    special.insert("<|im_end|>".to_string(), 200023);
    special.insert("<|think|>".to_string(), 200024);
    special.insert("<|/think|>".to_string(), 200025);
    special.insert("<|plan|>".to_string(), 200026);
    special.insert("<|/plan|>".to_string(), 200027);
    special.insert("<|step|>".to_string(), 200028);
    special.insert("<|/step|>".to_string(), 200029);
    special.insert("<|act|>".to_string(), 200030);
    special.insert("<|/act|>".to_string(), 200031);
    special.insert("<|observe|>".to_string(), 200032);
    special.insert("<|/observe|>".to_string(), 200033);
    special.insert("<|function|>".to_string(), 200034);
    special.insert("<|/function|>".to_string(), 200035);
    special.insert("<|result|>".to_string(), 200036);
    special.insert("<|/result|>".to_string(), 200037);
    special.insert("<|error|>".to_string(), 200038);
    special.insert("<|/error|>".to_string(), 200039);
    special.insert("<|code|>".to_string(), 200040);
    special.insert("<|/code|>".to_string(), 200041);
    special.insert("<|output|>".to_string(), 200042);
    special.insert("<|/output|>".to_string(), 200043);

    // Multimodal tokens
    special.insert("<|image_start|>".to_string(), 200061);
    special.insert("<|/image_start|>".to_string(), 200062);
    special.insert("<|audio|>".to_string(), 200063);
    special.insert("<|/audio|>".to_string(), 200064);
    special.insert("<|video|>".to_string(), 200065);
    special.insert("<|/video|>".to_string(), 200066);

    Tokenizer::from_bytes(vocab_bytes, O200K_BASE_PATTERN, special).unwrap()
}
