//! Integration tests for Mistral V3/Tekken tokenizer.
//!
//! Mistral V3 (Tekken) uses Tiktoken-style BPE encoding (NOT SentencePiece).
//! Key characteristics:
//! - Vocab size: ~131,126 (131,072 base + 54 agent tokens)
//! - Uses Tiktoken encoding (same pattern as O200K)
//! - Much larger vocabulary than V1/V2 (4x larger)
//! - Used by: Mistral NeMo, Mistral Large 2, Pixtral

use splintr::from_pretrained;

// =============================================================================
// Loading Tests
// =============================================================================

#[test]
fn test_v3_load_mistral_v3() {
    let tok = from_pretrained("mistral_v3").expect("Failed to load mistral_v3");
    assert!(tok.vocab_size() > 130000);
}

// =============================================================================
// Native Special Tokens (BOS/EOS/UNK)
// =============================================================================

#[test]
fn test_v3_bos_token() {
    let tok = from_pretrained("mistral_v3").expect("Failed to load mistral_v3");

    // <s> = BOS = token 1
    let tokens = tok.encode_with_special("<s>");
    assert_eq!(tokens, vec![1], "<s> should be token 1");
}

#[test]
fn test_v3_eos_token() {
    let tok = from_pretrained("mistral_v3").expect("Failed to load mistral_v3");

    // </s> = EOS = token 2
    let tokens = tok.encode_with_special("</s>");
    assert_eq!(tokens, vec![2], "</s> should be token 2");
}

#[test]
fn test_v3_unk_token() {
    let tok = from_pretrained("mistral_v3").expect("Failed to load mistral_v3");

    // <unk> = UNK = token 0
    let tokens = tok.encode_with_special("<unk>");
    assert_eq!(tokens, vec![0], "<unk> should be token 0");
}

#[test]
fn test_v3_decode_bos_eos_unk() {
    let tok = from_pretrained("mistral_v3").expect("Failed to load mistral_v3");

    let decoded = tok.decode(&[0]).expect("Failed to decode");
    assert_eq!(decoded, "<unk>");

    let decoded = tok.decode(&[1]).expect("Failed to decode");
    assert_eq!(decoded, "<s>");

    let decoded = tok.decode(&[2]).expect("Failed to decode");
    assert_eq!(decoded, "</s>");
}

// =============================================================================
// Vocab Size Tests
// =============================================================================

#[test]
fn test_v3_vocab_size() {
    let tok = from_pretrained("mistral_v3").expect("Failed to load mistral_v3");
    // V3 vocab: 131,072 base tokens + 54 agent tokens = 131,126
    assert_eq!(tok.vocab_size(), 131126);
}

#[test]
fn test_v3_much_larger_than_v2() {
    let v2 = from_pretrained("mistral_v2").expect("Failed to load mistral_v2");
    let v3 = from_pretrained("mistral_v3").expect("Failed to load mistral_v3");

    // V3 should be ~4x larger than V2
    assert!(v3.vocab_size() > v2.vocab_size() * 3);
    assert_eq!(v2.vocab_size(), 32822);
    assert_eq!(v3.vocab_size(), 131126);
}

// =============================================================================
// Agent Tokens Tests
// =============================================================================

#[test]
fn test_v3_agent_tokens_conversation() {
    let tok = from_pretrained("mistral_v3").expect("Failed to load mistral_v3");

    // Agent tokens start at 131072 for V3
    // <|system|> = 131072 + 0 = 131072
    let tokens = tok.encode_with_special("<|system|>");
    assert_eq!(tokens, vec![131072]);

    // <|user|> = 131072 + 1 = 131073
    let tokens = tok.encode_with_special("<|user|>");
    assert_eq!(tokens, vec![131073]);

    // <|assistant|> = 131072 + 2 = 131074
    let tokens = tok.encode_with_special("<|assistant|>");
    assert_eq!(tokens, vec![131074]);
}

#[test]
fn test_v3_agent_tokens_thinking() {
    let tok = from_pretrained("mistral_v3").expect("Failed to load mistral_v3");

    // <|think|> = 131072 + 5 = 131077
    let tokens = tok.encode_with_special("<|think|>");
    assert_eq!(tokens, vec![131077]);

    // <|/think|> = 131072 + 6 = 131078
    let tokens = tok.encode_with_special("<|/think|>");
    assert_eq!(tokens, vec![131078]);
}

#[test]
fn test_v3_agent_tokens_function() {
    let tok = from_pretrained("mistral_v3").expect("Failed to load mistral_v3");

    // <|function|> = 131072 + 15 = 131087
    let tokens = tok.encode_with_special("<|function|>");
    assert_eq!(tokens, vec![131087]);

    // <|/function|> = 131072 + 16 = 131088
    let tokens = tok.encode_with_special("<|/function|>");
    assert_eq!(tokens, vec![131088]);
}

#[test]
fn test_v3_decode_agent_tokens() {
    let tok = from_pretrained("mistral_v3").expect("Failed to load mistral_v3");

    let decoded = tok.decode(&[131072]).expect("Failed to decode");
    assert_eq!(decoded, "<|system|>");

    let decoded = tok.decode(&[131073]).expect("Failed to decode");
    assert_eq!(decoded, "<|user|>");

    let decoded = tok.decode(&[131074]).expect("Failed to decode");
    assert_eq!(decoded, "<|assistant|>");

    let decoded = tok.decode(&[131077]).expect("Failed to decode");
    assert_eq!(decoded, "<|think|>");

    let decoded = tok.decode(&[131078]).expect("Failed to decode");
    assert_eq!(decoded, "<|/think|>");
}

// =============================================================================
// Special Token Roundtrip Tests
// =============================================================================

#[test]
fn test_v3_special_tokens_in_mixed_text() {
    let tok = from_pretrained("mistral_v3").expect("Failed to load mistral_v3");

    // Test that special tokens are properly recognized in mixed content
    let tokens = tok.encode_with_special("<|system|>Hi<|user|>Hello<|assistant|>World");

    // Verify special tokens are present
    assert!(tokens.contains(&131072)); // system
    assert!(tokens.contains(&131073)); // user
    assert!(tokens.contains(&131074)); // assistant

    // Verify we can decode back
    let decoded = tok.decode(&tokens).expect("Failed to decode");
    // Special tokens should decode correctly
    assert!(decoded.contains("<|system|>"));
    assert!(decoded.contains("<|user|>"));
    assert!(decoded.contains("<|assistant|>"));
}

#[test]
fn test_v3_thinking_tokens_mixed() {
    let tok = from_pretrained("mistral_v3").expect("Failed to load mistral_v3");

    let tokens = tok.encode_with_special("<|think|>reasoning<|/think|>");

    // Verify thinking tokens are present
    assert!(tokens.contains(&131077)); // think
    assert!(tokens.contains(&131078)); // /think

    let decoded = tok.decode(&tokens).expect("Failed to decode");
    assert!(decoded.contains("<|think|>"));
    assert!(decoded.contains("<|/think|>"));
}

// =============================================================================
// V3 vs V1/V2 Comparison Tests
// =============================================================================

#[test]
fn test_v3_different_from_v1() {
    let v1 = from_pretrained("mistral_v1").expect("Failed to load mistral_v1");
    let v3 = from_pretrained("mistral_v3").expect("Failed to load mistral_v3");

    let text = "Hello";
    let v1_tokens = v1.encode(text);
    let v3_tokens = v3.encode(text);

    // V3 should encode differently than V1 (completely different vocab)
    assert_ne!(v1_tokens, v3_tokens);
}

#[test]
fn test_v3_different_from_v2() {
    let v2 = from_pretrained("mistral_v2").expect("Failed to load mistral_v2");
    let v3 = from_pretrained("mistral_v3").expect("Failed to load mistral_v3");

    let text = "Test";
    let v2_tokens = v2.encode(text);
    let v3_tokens = v3.encode(text);

    // V3 should encode differently than V2 (completely different vocab)
    assert_ne!(v2_tokens, v3_tokens);
}

// =============================================================================
// Basic Encoding Tests
// =============================================================================

#[test]
fn test_v3_encodes_text() {
    let tok = from_pretrained("mistral_v3").expect("Failed to load mistral_v3");

    // V3 should be able to encode basic text
    let tokens = tok.encode("Hello");
    assert!(!tokens.is_empty());
}

#[test]
fn test_v3_empty_input() {
    let tok = from_pretrained("mistral_v3").expect("Failed to load mistral_v3");

    let tokens = tok.encode("");
    assert!(tokens.is_empty());

    let decoded = tok.decode(&[]).expect("Failed to decode");
    assert!(decoded.is_empty());
}

// =============================================================================
// Roundtrip Tests (encode -> decode should preserve text)
// =============================================================================

#[test]
fn test_v3_roundtrip_hello_world() {
    let tok = from_pretrained("mistral_v3").expect("Failed to load mistral_v3");

    let text = "Hello world";
    let tokens = tok.encode(text);
    let decoded = tok.decode(&tokens).expect("Failed to decode");
    assert_eq!(
        decoded, text,
        "Roundtrip failed: spaces should be preserved"
    );
}

#[test]
fn test_v3_roundtrip_with_punctuation() {
    let tok = from_pretrained("mistral_v3").expect("Failed to load mistral_v3");

    let text = "Hello, world!";
    let tokens = tok.encode(text);
    let decoded = tok.decode(&tokens).expect("Failed to decode");
    assert_eq!(decoded, text);
}

#[test]
fn test_v3_roundtrip_leading_space() {
    let tok = from_pretrained("mistral_v3").expect("Failed to load mistral_v3");

    let text = " hello world ";
    let tokens = tok.encode(text);
    let decoded = tok.decode(&tokens).expect("Failed to decode");
    assert_eq!(decoded, text, "Leading/trailing spaces should be preserved");
}

#[test]
fn test_v3_roundtrip_multiple_spaces() {
    let tok = from_pretrained("mistral_v3").expect("Failed to load mistral_v3");

    let text = "hello  world";
    let tokens = tok.encode(text);
    let decoded = tok.decode(&tokens).expect("Failed to decode");
    assert_eq!(decoded, text, "Multiple spaces should be preserved");
}

#[test]
fn test_v3_roundtrip_chinese() {
    let tok = from_pretrained("mistral_v3").expect("Failed to load mistral_v3");

    let text = "你好世界";
    let tokens = tok.encode(text);
    let decoded = tok.decode(&tokens).expect("Failed to decode");
    assert_eq!(decoded, text);
}

#[test]
fn test_v3_roundtrip_emoji() {
    let tok = from_pretrained("mistral_v3").expect("Failed to load mistral_v3");

    let text = "Hello 🌍 World!";
    let tokens = tok.encode(text);
    let decoded = tok.decode(&tokens).expect("Failed to decode");
    assert_eq!(decoded, text);
}

#[test]
fn test_v3_roundtrip_multiline() {
    let tok = from_pretrained("mistral_v3").expect("Failed to load mistral_v3");

    let text = "Multi-line\ntext\nwith\nnewlines";
    let tokens = tok.encode(text);
    let decoded = tok.decode(&tokens).expect("Failed to decode");
    assert_eq!(decoded, text);
}

#[test]
fn test_v3_roundtrip_code() {
    let tok = from_pretrained("mistral_v3").expect("Failed to load mistral_v3");

    let text = "def hello():\n    print('Hello')";
    let tokens = tok.encode(text);
    let decoded = tok.decode(&tokens).expect("Failed to decode");
    assert_eq!(decoded, text);
}
