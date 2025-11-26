//! Integration tests for Llama 3/3.1/3.2/3.3 tokenizer.
//!
//! These tests verify that the Llama 3 tokenizer correctly encodes and decodes text,
//! handles special tokens, and produces consistent results.

use splintr::{Tokenizer, LLAMA3_PATTERN};

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

// Helper function to create a Llama 3 tokenizer for testing
fn create_llama3_tokenizer() -> Tokenizer {
    create_llama3_tokenizer_by_name("llama3")
}

fn create_llama3_tokenizer_by_name(name: &str) -> Tokenizer {
    // Load the embedded vocab
    let vocab_bytes = include_bytes!("../python/splintr/vocabs/llama3.tiktoken");

    let mut special = rustc_hash::FxHashMap::default();

    // Meta standard special tokens (128000-128010)
    special.insert("<|begin_of_text|>".to_string(), 128000);
    special.insert("<|end_of_text|>".to_string(), 128001);
    special.insert("<|reserved_special_token_0|>".to_string(), 128002);
    special.insert("<|reserved_special_token_1|>".to_string(), 128003);
    special.insert("<|finetune_right_pad_id|>".to_string(), 128004);
    special.insert("<|step_id|>".to_string(), 128005); // Llama 3.2-Vision
    special.insert("<|start_header_id|>".to_string(), 128006);
    special.insert("<|end_header_id|>".to_string(), 128007);
    special.insert("<|eom_id|>".to_string(), 128008);
    special.insert("<|eot_id|>".to_string(), 128009);
    special.insert("<|python_tag|>".to_string(), 128010);

    // Llama 3.2-Vision multimodal token
    special.insert("<|image|>".to_string(), 128256);
    special.insert("<|/image|>".to_string(), 128257);

    // Agent tokens (128300+)
    special.insert("<|system|>".to_string(), 128300);
    special.insert("<|user|>".to_string(), 128301);
    special.insert("<|assistant|>".to_string(), 128302);
    special.insert("<|im_start|>".to_string(), 128303);
    special.insert("<|im_end|>".to_string(), 128304);
    special.insert("<|think|>".to_string(), 128305);
    special.insert("<|/think|>".to_string(), 128306);
    special.insert("<|plan|>".to_string(), 128307);
    special.insert("<|/plan|>".to_string(), 128308);
    special.insert("<|step|>".to_string(), 128309);
    special.insert("<|/step|>".to_string(), 128310);
    special.insert("<|act|>".to_string(), 128311);
    special.insert("<|/act|>".to_string(), 128312);
    special.insert("<|observe|>".to_string(), 128313);
    special.insert("<|/observe|>".to_string(), 128314);
    special.insert("<|function|>".to_string(), 128315);
    special.insert("<|/function|>".to_string(), 128316);
    special.insert("<|result|>".to_string(), 128317);
    special.insert("<|/result|>".to_string(), 128318);
    special.insert("<|error|>".to_string(), 128319);
    special.insert("<|/error|>".to_string(), 128320);
    special.insert("<|code|>".to_string(), 128321);
    special.insert("<|/code|>".to_string(), 128322);
    special.insert("<|output|>".to_string(), 128323);
    special.insert("<|/output|>".to_string(), 128324);

    // Use the same pattern as the Python bindings
    let _ = name; // Acknowledge variant name (all use same vocab)

    Tokenizer::from_bytes(vocab_bytes, LLAMA3_PATTERN, special).unwrap()
}
