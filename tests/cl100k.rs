//! Integration tests for cl100k_base tokenizer (GPT-4, GPT-3.5-turbo).
//!
//! These tests verify that the cl100k_base tokenizer correctly encodes and decodes text,
//! handles special tokens, and produces consistent results.

use splintr::{Tokenizer, CL100K_BASE_PATTERN};

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

// Helper function to create a cl100k tokenizer for testing
fn create_cl100k_tokenizer() -> Tokenizer {
    // Load the embedded vocab
    let vocab_bytes = include_bytes!("../python/splintr/vocabs/cl100k_base.tiktoken");

    let mut special = rustc_hash::FxHashMap::default();

    // OpenAI standard special tokens
    special.insert("<|endoftext|>".to_string(), 100257);
    special.insert("<|fim_prefix|>".to_string(), 100258);
    special.insert("<|fim_middle|>".to_string(), 100259);
    special.insert("<|fim_suffix|>".to_string(), 100260);
    special.insert("<|endofprompt|>".to_string(), 100276);

    // Agent tokens (100277+)
    special.insert("<|system|>".to_string(), 100277);
    special.insert("<|user|>".to_string(), 100278);
    special.insert("<|assistant|>".to_string(), 100279);
    special.insert("<|im_start|>".to_string(), 100280);
    special.insert("<|im_end|>".to_string(), 100281);
    special.insert("<|think|>".to_string(), 100282);
    special.insert("<|/think|>".to_string(), 100283);
    special.insert("<|plan|>".to_string(), 100284);
    special.insert("<|/plan|>".to_string(), 100285);
    special.insert("<|step|>".to_string(), 100286);
    special.insert("<|/step|>".to_string(), 100287);
    special.insert("<|act|>".to_string(), 100288);
    special.insert("<|/act|>".to_string(), 100289);
    special.insert("<|observe|>".to_string(), 100290);
    special.insert("<|/observe|>".to_string(), 100291);
    special.insert("<|function|>".to_string(), 100292);
    special.insert("<|/function|>".to_string(), 100293);
    special.insert("<|result|>".to_string(), 100294);
    special.insert("<|/result|>".to_string(), 100295);
    special.insert("<|error|>".to_string(), 100296);
    special.insert("<|/error|>".to_string(), 100297);
    special.insert("<|code|>".to_string(), 100298);
    special.insert("<|/code|>".to_string(), 100299);
    special.insert("<|output|>".to_string(), 100300);
    special.insert("<|/output|>".to_string(), 100301);

    Tokenizer::from_bytes(vocab_bytes, CL100K_BASE_PATTERN, special).unwrap()
}
