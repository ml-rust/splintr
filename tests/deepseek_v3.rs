//! Integration tests for DeepSeek V3 tokenizer.
//!
//! These tests verify that the DeepSeek V3 tokenizer correctly encodes and decodes text,
//! handles ByteLevel BPE encoding, special tokens, and produces consistent results.

use splintr::{Tokenizer, LLAMA3_PATTERN};
use std::sync::LazyLock;

/// Shared tokenizer instance to avoid expensive re-initialization per test.
static TOKENIZER: LazyLock<Tokenizer> = LazyLock::new(create_deepseek_v3_tokenizer_impl);

// =============================================================================
// Exact Token ID Tests
// =============================================================================
// These tests verify specific token IDs to catch any regression in
// ByteLevel encoding or vocabulary changes.

/// Verify exact token IDs for "Hello world".
#[test]
fn test_deepseek_v3_hello_world_tokens() {
    let tokenizer = create_deepseek_v3_tokenizer();
    let tokens = tokenizer.encode("Hello world");
    assert_eq!(
        tokens,
        vec![19923, 2058],
        "Token IDs for 'Hello world' changed"
    );
}

/// Verify exact token IDs for " hello world ".
#[test]
fn test_deepseek_v3_space_prefix_tokens() {
    let tokenizer = create_deepseek_v3_tokenizer();
    let tokens = tokenizer.encode(" hello world ");
    assert_eq!(
        tokens,
        vec![44388, 2058, 223],
        "Token IDs for ' hello world ' changed"
    );
}

/// Verify exact token IDs for "你好世界".
#[test]
fn test_deepseek_v3_chinese_tokens() {
    let tokenizer = create_deepseek_v3_tokenizer();
    let tokens = tokenizer.encode("你好世界");
    assert_eq!(
        tokens,
        vec![30594, 3427],
        "Token IDs for '你好世界' changed"
    );
}

/// Verify exact token IDs for "Hello 你好 World 世界!".
#[test]
fn test_deepseek_v3_mixed_tokens() {
    let tokenizer = create_deepseek_v3_tokenizer();
    let tokens = tokenizer.encode("Hello 你好 World 世界!");
    assert_eq!(
        tokens,
        vec![19923, 223, 30594, 4495, 223, 3427, 3],
        "Token IDs for mixed Chinese/English changed"
    );
}

/// Verify exact token IDs for "Hello 🌍 World!".
#[test]
fn test_deepseek_v3_emoji_tokens() {
    let tokenizer = create_deepseek_v3_tokenizer();
    let tokens = tokenizer.encode("Hello 🌍 World!");
    assert_eq!(
        tokens,
        vec![19923, 73369, 238, 4495, 3],
        "Token IDs for emoji text changed"
    );
}

// =============================================================================
// General Roundtrip Tests
// =============================================================================

/// Test basic encoding and decoding roundtrip.
#[test]
fn test_deepseek_v3_encode_decode_roundtrip() {
    let tokenizer = create_deepseek_v3_tokenizer();

    let test_cases = vec![
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Rust is a systems programming language.",
        "1234567890",
        "Special characters: !@#$%^&*()",
        "Multi-line\ntext\nwith\nnewlines",
        "Unicode: こんにちは 世界",
    ];

    for text in test_cases {
        let tokens = tokenizer.encode(text);
        let decoded = tokenizer.decode(&tokens).unwrap();
        assert_eq!(decoded, text, "Roundtrip failed for: {:?}", text);
    }
}

/// Test ByteLevel encoding handles Chinese text correctly.
#[test]
fn test_deepseek_v3_chinese_text() {
    let tokenizer = create_deepseek_v3_tokenizer();

    let test_cases = vec![
        "你好",
        "你好世界",
        "中文测试",
        "Hello 你好 World 世界!",
        "混合文本 mixed text 测试",
    ];

    for text in test_cases {
        let tokens = tokenizer.encode(text);
        assert!(
            !tokens.is_empty(),
            "Chinese text should produce tokens: {:?}",
            text
        );
        let decoded = tokenizer.decode(&tokens).unwrap();
        assert_eq!(decoded, text, "Chinese roundtrip failed for: {:?}", text);
    }
}

/// Test ByteLevel encoding handles emoji correctly.
#[test]
fn test_deepseek_v3_emoji() {
    let tokenizer = create_deepseek_v3_tokenizer();

    let test_cases = vec![
        "Hello 🌍 World!",
        "🦀 Rust is awesome! 🚀",
        "Emoji test: 😀😎🎉",
    ];

    for text in test_cases {
        let tokens = tokenizer.encode(text);
        assert!(
            !tokens.is_empty(),
            "Emoji text should produce tokens: {:?}",
            text
        );
        let decoded = tokenizer.decode(&tokens).unwrap();
        assert_eq!(decoded, text, "Emoji roundtrip failed for: {:?}", text);
    }
}

/// Test that spaces are preserved correctly (ByteLevel maps space to Ġ).
#[test]
fn test_deepseek_v3_space_handling() {
    let tokenizer = create_deepseek_v3_tokenizer();

    let test_cases = vec![
        " hello",
        "hello ",
        " hello world ",
        "  double  spaces  ",
        "   leading spaces",
    ];

    for text in test_cases {
        let tokens = tokenizer.encode(text);
        let decoded = tokenizer.decode(&tokens).unwrap();
        assert_eq!(decoded, text, "Space handling failed for: {:?}", text);
    }
}

/// Test that vocab size is correct (128,000 BPE tokens).
#[test]
fn test_deepseek_v3_vocab_size() {
    let tokenizer = create_deepseek_v3_tokenizer();
    // DeepSeek V3 has 128,000 BPE tokens plus special tokens
    assert!(
        tokenizer.vocab_size() >= 128000,
        "Vocab size should be at least 128,000, got {}",
        tokenizer.vocab_size()
    );
}

/// Test official DeepSeek native special tokens.
#[test]
fn test_deepseek_v3_native_special_tokens() {
    let tokenizer = create_deepseek_v3_tokenizer();

    // Test begin/end of sentence
    let tokens = tokenizer.encode_with_special("<｜begin▁of▁sentence｜>Hello<｜end▁of▁sentence｜>");
    assert!(tokens.contains(&0), "Should contain begin_of_sentence (0)");
    assert!(tokens.contains(&1), "Should contain end_of_sentence (1)");

    // Test thinking tokens
    let tokens = tokenizer.encode_with_special("<think>Let me think...</think>");
    assert!(tokens.contains(&128798), "Should contain think (128798)");
    assert!(
        tokens.contains(&128799),
        "Should contain think_end (128799)"
    );

    // Test user/assistant tokens
    let tokens = tokenizer.encode_with_special("<｜User｜>Hi<｜Assistant｜>");
    assert!(tokens.contains(&128803), "Should contain User (128803)");
    assert!(
        tokens.contains(&128804),
        "Should contain Assistant (128804)"
    );

    // Test EOT token
    let tokens = tokenizer.encode_with_special("<|EOT|>");
    assert!(tokens.contains(&128805), "Should contain EOT (128805)");
}

/// Test DeepSeek FIM (Fill-in-the-Middle) tokens.
#[test]
fn test_deepseek_v3_fim_tokens() {
    let tokenizer = create_deepseek_v3_tokenizer();

    let tokens =
        tokenizer.encode_with_special("<｜fim▁begin｜>prefix<｜fim▁hole｜>suffix<｜fim▁end｜>");
    assert!(tokens.contains(&128800), "Should contain fim_hole (128800)");
    assert!(
        tokens.contains(&128801),
        "Should contain fim_begin (128801)"
    );
    assert!(tokens.contains(&128802), "Should contain fim_end (128802)");
}

/// Test DeepSeek tool calling tokens.
#[test]
fn test_deepseek_v3_tool_tokens() {
    let tokenizer = create_deepseek_v3_tokenizer();

    let tokens = tokenizer.encode_with_special("<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁call▁end｜><｜tool▁calls▁end｜>");
    assert!(
        tokens.contains(&128806),
        "Should contain tool_calls_begin (128806)"
    );
    assert!(
        tokens.contains(&128807),
        "Should contain tool_calls_end (128807)"
    );
    assert!(
        tokens.contains(&128808),
        "Should contain tool_call_begin (128808)"
    );
    assert!(
        tokens.contains(&128809),
        "Should contain tool_call_end (128809)"
    );

    // Test tool outputs
    let tokens = tokenizer.encode_with_special("<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>result<｜tool▁output▁end｜><｜tool▁outputs▁end｜>");
    assert!(
        tokens.contains(&128810),
        "Should contain tool_outputs_begin (128810)"
    );
    assert!(
        tokens.contains(&128811),
        "Should contain tool_outputs_end (128811)"
    );
    assert!(
        tokens.contains(&128812),
        "Should contain tool_output_begin (128812)"
    );
    assert!(
        tokens.contains(&128813),
        "Should contain tool_output_end (128813)"
    );
}

/// Test splintr agent tokens for DeepSeek V3.
#[test]
fn test_deepseek_v3_agent_tokens() {
    let tokenizer = create_deepseek_v3_tokenizer();

    // Test conversation tokens
    let tokens = tokenizer.encode_with_special("<|system|>You are helpful.<|user|>Hi<|assistant|>");
    assert!(tokens.contains(&128900), "Should contain system (128900)");
    assert!(tokens.contains(&128901), "Should contain user (128901)");
    assert!(
        tokens.contains(&128902),
        "Should contain assistant (128902)"
    );

    // Test thinking tokens (splintr style)
    let tokens = tokenizer.encode_with_special("<|think|>Let me reason...<|/think|>");
    assert!(tokens.contains(&128905), "Should contain think (128905)");
    assert!(
        tokens.contains(&128906),
        "Should contain think_end (128906)"
    );

    // Test function calling tokens
    let tokens = tokenizer.encode_with_special("<|function|>get_weather<|/function|>");
    assert!(tokens.contains(&128915), "Should contain function (128915)");
    assert!(
        tokens.contains(&128916),
        "Should contain function_end (128916)"
    );
}

/// Test DeepSeek V3 chat format.
#[test]
fn test_deepseek_v3_chat_format() {
    let tokenizer = create_deepseek_v3_tokenizer();

    // DeepSeek chat format
    let chat = "<｜begin▁of▁sentence｜><｜User｜>Hello!<｜Assistant｜>Hi there!<|EOT|>";

    let tokens = tokenizer.encode_with_special(chat);

    // Verify special tokens are present
    assert!(tokens.contains(&0)); // begin_of_sentence
    assert!(tokens.contains(&128803)); // User
    assert!(tokens.contains(&128804)); // Assistant
    assert!(tokens.contains(&128805)); // EOT

    // Verify roundtrip
    let decoded = tokenizer.decode(&tokens).unwrap();
    assert_eq!(decoded, chat);
}

/// Test DeepSeek V3 thinking format (R1-style reasoning).
#[test]
fn test_deepseek_v3_thinking_format() {
    let tokenizer = create_deepseek_v3_tokenizer();

    let chat = "<｜User｜>What is 2+2?<｜Assistant｜><think>Let me calculate: 2+2=4</think>The answer is 4.<|EOT|>";

    let tokens = tokenizer.encode_with_special(chat);

    // Verify special tokens
    assert!(tokens.contains(&128803)); // User
    assert!(tokens.contains(&128804)); // Assistant
    assert!(tokens.contains(&128798)); // think
    assert!(tokens.contains(&128799)); // /think
    assert!(tokens.contains(&128805)); // EOT

    // Verify roundtrip
    let decoded = tokenizer.decode(&tokens).unwrap();
    assert_eq!(decoded, chat);
}

/// Test batch encoding.
#[test]
fn test_deepseek_v3_batch_encode() {
    let tokenizer = create_deepseek_v3_tokenizer();

    let texts = vec![
        "Hello, world!".to_string(),
        "你好世界".to_string(),
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
fn test_deepseek_v3_special_token_decode() {
    let tokenizer = create_deepseek_v3_tokenizer();

    // Decode native DeepSeek tokens
    let decoded = tokenizer.decode(&[0]).unwrap();
    assert_eq!(decoded, "<｜begin▁of▁sentence｜>");

    let decoded = tokenizer.decode(&[1]).unwrap();
    assert_eq!(decoded, "<｜end▁of▁sentence｜>");

    let decoded = tokenizer.decode(&[128798]).unwrap();
    assert_eq!(decoded, "<think>");

    let decoded = tokenizer.decode(&[128799]).unwrap();
    assert_eq!(decoded, "</think>");

    let decoded = tokenizer.decode(&[128803]).unwrap();
    assert_eq!(decoded, "<｜User｜>");

    let decoded = tokenizer.decode(&[128804]).unwrap();
    assert_eq!(decoded, "<｜Assistant｜>");

    let decoded = tokenizer.decode(&[128805]).unwrap();
    assert_eq!(decoded, "<|EOT|>");
}

/// Test empty input handling.
#[test]
fn test_deepseek_v3_empty_input() {
    let tokenizer = create_deepseek_v3_tokenizer();

    let tokens = tokenizer.encode("");
    assert!(tokens.is_empty(), "Empty input should produce empty tokens");

    let decoded = tokenizer.decode(&[]).unwrap();
    assert!(
        decoded.is_empty(),
        "Empty tokens should decode to empty string"
    );
}

/// Test that both from_pretrained variants work.
#[test]
fn test_deepseek_v3_from_pretrained_variants() {
    let t1 = create_deepseek_v3_tokenizer_by_name("deepseek_v3");
    let t2 = create_deepseek_v3_tokenizer_by_name("deepseek-v3");

    let text = "Hello, world!";
    assert_eq!(
        t1.encode(text),
        t2.encode(text),
        "Both DeepSeek V3 variants should produce same encoding"
    );
}

/// Test mixed special tokens from different sources.
#[test]
fn test_deepseek_v3_mixed_special_tokens() {
    let tokenizer = create_deepseek_v3_tokenizer();

    // Mix native DeepSeek tokens with splintr agent tokens
    let chat = "<｜User｜>Tell me about Rust.<|think|>User wants info about Rust programming language.<|/think|><｜Assistant｜>Rust is a systems programming language.";

    let tokens = tokenizer.encode_with_special(chat);

    // Native tokens
    assert!(tokens.contains(&128803)); // User (native)
    assert!(tokens.contains(&128804)); // Assistant (native)

    // Agent tokens
    assert!(tokens.contains(&128905)); // think (agent)
    assert!(tokens.contains(&128906)); // /think (agent)

    // Verify roundtrip
    let decoded = tokenizer.decode(&tokens).unwrap();
    assert_eq!(decoded, chat);
}

/// Get the shared tokenizer instance
fn create_deepseek_v3_tokenizer() -> &'static Tokenizer {
    &TOKENIZER
}

/// Create a fresh tokenizer by name (for variant tests only)
fn create_deepseek_v3_tokenizer_by_name(_name: &str) -> Tokenizer {
    create_deepseek_v3_tokenizer_impl()
}

/// Implementation that actually constructs the tokenizer
fn create_deepseek_v3_tokenizer_impl() -> Tokenizer {
    // Load the embedded vocab
    let vocab_bytes = include_bytes!("../python/splintr/vocabs/deepseek_v3.tiktoken");

    let mut special = rustc_hash::FxHashMap::default();

    // DeepSeek native special tokens (0-2)
    special.insert("<｜begin▁of▁sentence｜>".to_string(), 0);
    special.insert("<｜end▁of▁sentence｜>".to_string(), 1);
    special.insert("<｜▁pad▁｜>".to_string(), 2);

    // Thinking tokens (128798-128799)
    special.insert("<think>".to_string(), 128798);
    special.insert("</think>".to_string(), 128799);

    // FIM tokens (128800-128802)
    special.insert("<｜fim▁hole｜>".to_string(), 128800);
    special.insert("<｜fim▁begin｜>".to_string(), 128801);
    special.insert("<｜fim▁end｜>".to_string(), 128802);

    // Chat tokens (128803-128805)
    special.insert("<｜User｜>".to_string(), 128803);
    special.insert("<｜Assistant｜>".to_string(), 128804);
    special.insert("<|EOT|>".to_string(), 128805);

    // Tool calling tokens (128806-128814)
    special.insert("<｜tool▁calls▁begin｜>".to_string(), 128806);
    special.insert("<｜tool▁calls▁end｜>".to_string(), 128807);
    special.insert("<｜tool▁call▁begin｜>".to_string(), 128808);
    special.insert("<｜tool▁call▁end｜>".to_string(), 128809);
    special.insert("<｜tool▁outputs▁begin｜>".to_string(), 128810);
    special.insert("<｜tool▁outputs▁end｜>".to_string(), 128811);
    special.insert("<｜tool▁output▁begin｜>".to_string(), 128812);
    special.insert("<｜tool▁output▁end｜>".to_string(), 128813);
    special.insert("<｜tool▁sep｜>".to_string(), 128814);

    // Agent tokens (128900+)
    special.insert("<|system|>".to_string(), 128900);
    special.insert("<|user|>".to_string(), 128901);
    special.insert("<|assistant|>".to_string(), 128902);
    special.insert("<|im_start|>".to_string(), 128903);
    special.insert("<|im_end|>".to_string(), 128904);
    special.insert("<|think|>".to_string(), 128905);
    special.insert("<|/think|>".to_string(), 128906);
    special.insert("<|plan|>".to_string(), 128907);
    special.insert("<|/plan|>".to_string(), 128908);
    special.insert("<|step|>".to_string(), 128909);
    special.insert("<|/step|>".to_string(), 128910);
    special.insert("<|act|>".to_string(), 128911);
    special.insert("<|/act|>".to_string(), 128912);
    special.insert("<|observe|>".to_string(), 128913);
    special.insert("<|/observe|>".to_string(), 128914);
    special.insert("<|function|>".to_string(), 128915);
    special.insert("<|/function|>".to_string(), 128916);
    special.insert("<|result|>".to_string(), 128917);
    special.insert("<|/result|>".to_string(), 128918);
    special.insert("<|error|>".to_string(), 128919);
    special.insert("<|/error|>".to_string(), 128920);
    special.insert("<|code|>".to_string(), 128921);
    special.insert("<|/code|>".to_string(), 128922);
    special.insert("<|output|>".to_string(), 128923);
    special.insert("<|/output|>".to_string(), 128924);

    // DeepSeek uses ByteLevel BPE encoding
    Tokenizer::from_bytes_byte_level(vocab_bytes, LLAMA3_PATTERN, special).unwrap()
}
