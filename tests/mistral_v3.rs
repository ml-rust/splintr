//! Integration tests for Mistral V3/Tekken tokenizer.
//!
//! Mistral V3 (Tekken) uses Tiktoken-style BPE encoding (NOT SentencePiece).
//! Key characteristics:
//! - Vocab size: ~131,126 (131,072 base + 54 agent tokens)
//! - Uses Tiktoken-style encoding with its own `MISTRAL_V3_PATTERN` (O200K-like,
//!   but with no contraction branches and single-digit `\p{N}`)
//! - Much larger vocabulary than V1/V2 (4x larger)
//! - Used by: Mistral NeMo, Mistral Large 2, Pixtral

use splintr::{from_pretrained, SpecialDecode, Tokenize};

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
    let tokens = tok.encode("<s>");
    assert_eq!(tokens, vec![1], "<s> should be token 1");
}

#[test]
fn test_v3_eos_token() {
    let tok = from_pretrained("mistral_v3").expect("Failed to load mistral_v3");

    // </s> = EOS = token 2
    let tokens = tok.encode("</s>");
    assert_eq!(tokens, vec![2], "</s> should be token 2");
}

#[test]
fn test_v3_unk_token() {
    let tok = from_pretrained("mistral_v3").expect("Failed to load mistral_v3");

    // <unk> = UNK = token 0
    let tokens = tok.encode("<unk>");
    assert_eq!(tokens, vec![0], "<unk> should be token 0");
}

/// The declared markers do not come back from `decode`: V3 drops them, as V1
/// and V2 do (`sentencepiece` 0.2.0 and `tokenizers` 0.22.1 both decode
/// `[3, …]` on `mistral-7b-v0.3` as plain text). That choice is stated at
/// `special_decode_ids` in `src/core/pretrained.rs`, including the fact that
/// no Tekken reference exists on this machine to measure V3 itself against.
///
/// This used to assert `decode([1]) == "<s>"`, which pinned splintr's own
/// output rather than a reference and made the same markers behave differently
/// from every sibling vocabulary. The spellings are still reachable — see
/// `test_v3_markers_are_reachable_with_specials_rendered` — and a caller that
/// wants one asks the name→id map, not `decode`.
#[test]
fn test_v3_decode_bos_eos_unk() {
    let tok = from_pretrained("mistral_v3").expect("Failed to load mistral_v3");

    for id in [0, 1, 2] {
        let decoded = tok.decode(&[id]).expect("Failed to decode");
        assert_eq!(decoded, "", "control token {id} must decode to nothing");
    }
}

/// The dropped markers stay reachable through the explicit
/// [`SpecialDecode::Render`] mode, so nothing about the vocabulary became
/// unreachable — the same contract `mistral_v2` holds.
#[test]
fn test_v3_markers_are_reachable_with_specials_rendered() {
    let tok = from_pretrained("mistral_v3").expect("Failed to load mistral_v3");

    for (id, spelling) in [
        (0u32, "<unk>"),
        (1, "<s>"),
        (2, "</s>"),
        (3, "[INST]"),
        (4, "[/INST]"),
    ] {
        let decoded = tok
            .decode_with(&[id], SpecialDecode::Render)
            .expect("Failed to decode");
        assert_eq!(decoded, spelling);
    }
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
    let tokens = tok.encode("<|system|>");
    assert_eq!(tokens, vec![131072]);

    // <|user|> = 131072 + 1 = 131073
    let tokens = tok.encode("<|user|>");
    assert_eq!(tokens, vec![131073]);

    // <|assistant|> = 131072 + 2 = 131074
    let tokens = tok.encode("<|assistant|>");
    assert_eq!(tokens, vec![131074]);
}

#[test]
fn test_v3_agent_tokens_thinking() {
    let tok = from_pretrained("mistral_v3").expect("Failed to load mistral_v3");

    // <|think|> = 131072 + 5 = 131077
    let tokens = tok.encode("<|think|>");
    assert_eq!(tokens, vec![131077]);

    // <|/think|> = 131072 + 6 = 131078
    let tokens = tok.encode("<|/think|>");
    assert_eq!(tokens, vec![131078]);
}

#[test]
fn test_v3_agent_tokens_function() {
    let tok = from_pretrained("mistral_v3").expect("Failed to load mistral_v3");

    // <|function|> = 131072 + 15 = 131087
    let tokens = tok.encode("<|function|>");
    assert_eq!(tokens, vec![131087]);

    // <|/function|> = 131072 + 16 = 131088
    let tokens = tok.encode("<|/function|>");
    assert_eq!(tokens, vec![131088]);
}

/// Agent tokens are markers like the control tokens above and are dropped by
/// the same rule — they sit one id block past the vocabulary file's last id and
/// have no reference of their own, so they follow the block below them.
/// `SpecialDecode::Render` still spells them.
#[test]
fn test_v3_decode_agent_tokens() {
    let tok = from_pretrained("mistral_v3").expect("Failed to load mistral_v3");

    for (id, spelling) in [
        (131072u32, "<|system|>"),
        (131073, "<|user|>"),
        (131074, "<|assistant|>"),
        (131077, "<|think|>"),
        (131078, "<|/think|>"),
    ] {
        let decoded = tok.decode(&[id]).expect("Failed to decode");
        assert_eq!(decoded, "", "agent token {id} must decode to nothing");

        let rendered = tok
            .decode_with(&[id], SpecialDecode::Render)
            .expect("Failed to decode");
        assert_eq!(rendered, spelling);
    }
}

// =============================================================================
// Special Token Roundtrip Tests
// =============================================================================

#[test]
fn test_v3_special_tokens_in_mixed_text() {
    let tok = from_pretrained("mistral_v3").expect("Failed to load mistral_v3");

    // Test that special tokens are properly recognized in mixed content
    let tokens = tok.encode("<|system|>Hi<|user|>Hello<|assistant|>World");

    // Verify special tokens are present
    assert!(tokens.contains(&131072)); // system
    assert!(tokens.contains(&131073)); // user
    assert!(tokens.contains(&131074)); // assistant

    // The markers do not come back from the default decode (they are dropped,
    // as V1/V2's are), but the content between them does, and asking for them
    // reproduces the whole string.
    let decoded = tok.decode(&tokens).expect("Failed to decode");
    assert!(!decoded.contains("<|system|>"));
    assert!(!decoded.contains("<|user|>"));
    assert!(!decoded.contains("<|assistant|>"));
    assert!(decoded.contains("Hi") && decoded.contains("Hello") && decoded.contains("World"));
    assert_eq!(
        tok.decode_with(&tokens, SpecialDecode::Render)
            .expect("Failed to decode"),
        "<|system|>Hi<|user|>Hello<|assistant|>World"
    );
}

#[test]
fn test_v3_thinking_tokens_mixed() {
    let tok = from_pretrained("mistral_v3").expect("Failed to load mistral_v3");

    let tokens = tok.encode("<|think|>reasoning<|/think|>");

    // Verify thinking tokens are present
    assert!(tokens.contains(&131077)); // think
    assert!(tokens.contains(&131078)); // /think

    let decoded = tok.decode(&tokens).expect("Failed to decode");
    assert!(!decoded.contains("<|think|>"));
    assert!(!decoded.contains("<|/think|>"));
    assert!(decoded.contains("reasoning"));
    assert_eq!(
        tok.decode_with(&tokens, SpecialDecode::Render)
            .expect("Failed to decode"),
        "<|think|>reasoning<|/think|>"
    );
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
