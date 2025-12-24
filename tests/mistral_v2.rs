use splintr::from_pretrained;

#[test]
fn test_v2_control_tokens_inst() {
    let tok = from_pretrained("mistral_v2").expect("Failed to load mistral_v2");

    // Test [INST] token (ID 3)
    let tokens = tok.encode_with_special("[INST]");
    assert_eq!(tokens, vec![3]);

    // Test [/INST] token (ID 4)
    let tokens = tok.encode_with_special("[/INST]");
    assert_eq!(tokens, vec![4]);

    // Test instruction format
    let tokens = tok.encode_with_special("[INST]Hello[/INST]");
    assert!(tokens.contains(&3)); // [INST]
    assert!(tokens.contains(&4)); // [/INST]
}

#[test]
fn test_v2_control_tokens_tool_calls() {
    let tok = from_pretrained("mistral_v2").expect("Failed to load mistral_v2");

    // Test [TOOL_CALLS] token (ID 5)
    let tokens = tok.encode_with_special("[TOOL_CALLS]");
    assert_eq!(tokens, vec![5]);

    // Test [AVAILABLE_TOOLS] token (ID 6)
    let tokens = tok.encode_with_special("[AVAILABLE_TOOLS]");
    assert_eq!(tokens, vec![6]);
}

#[test]
fn test_v2_native_sentencepiece_tokens() {
    let tok = from_pretrained("mistral_v2").expect("Failed to load mistral_v2");

    // Test <s> token (ID 1)
    let tokens = tok.encode_with_special("<s>");
    assert_eq!(tokens, vec![1]);

    // Test </s> token (ID 2)
    let tokens = tok.encode_with_special("</s>");
    assert_eq!(tokens, vec![2]);

    // Test <unk> token (ID 0)
    let tokens = tok.encode_with_special("<unk>");
    assert_eq!(tokens, vec![0]);
}

// Note: V1 and V2 do NOT share the same vocab structure, so this test is removed.
// V1: IDs 3-31999 are BPE merges
// V2: IDs 0-770 are special/control tokens, IDs 771-1026 are byte fallback, IDs 1027-32767 are BPE merges
// They encode the same text to different token IDs, which is expected.

#[test]
fn test_v2_agent_tokens() {
    let tok = from_pretrained("mistral_v2").expect("Failed to load mistral_v2");

    // Agent tokens start at 32768 for V2
    // <|think|> is at offset 5 (after system, user, assistant, im_start, im_end)
    let tokens = tok.encode_with_special("<|think|>");
    assert_eq!(tokens, vec![32773]); // THINK token = 32768 + 5

    // <|function|> is at offset 15
    let tokens = tok.encode_with_special("<|function|>");
    assert_eq!(tokens, vec![32783]); // FUNCTION token = 32768 + 15
}

#[test]
fn test_v2_decode_control_tokens() {
    let tok = from_pretrained("mistral_v2").expect("Failed to load mistral_v2");

    // Decode [INST] token
    let text = tok.decode(&[3]).expect("Failed to decode");
    assert_eq!(text, "[INST]");

    // Decode [/INST] token
    let text = tok.decode(&[4]).expect("Failed to decode");
    assert_eq!(text, "[/INST]");

    // Decode [TOOL_CALLS] token
    let text = tok.decode(&[5]).expect("Failed to decode");
    assert_eq!(text, "[TOOL_CALLS]");

    // Decode [AVAILABLE_TOOLS] token
    let text = tok.decode(&[6]).expect("Failed to decode");
    assert_eq!(text, "[AVAILABLE_TOOLS]");
}

#[test]
fn test_v2_full_instruction_roundtrip() {
    let tok = from_pretrained("mistral_v2").expect("Failed to load mistral_v2");

    let text = "[INST]What is the weather today?[/INST]";
    let tokens = tok.encode_with_special(text);
    let decoded = tok.decode(&tokens).expect("Failed to decode");
    assert_eq!(decoded, text);
}

#[test]
fn test_v2_model_name_underscore() {
    let tok = from_pretrained("mistral_v2").unwrap();

    // V2 should recognize control tokens
    let tokens = tok.encode_with_special("[INST]");
    assert_eq!(tokens, vec![3]);
}

#[test]
fn test_v2_vocab_size() {
    let tok = from_pretrained("mistral_v2").expect("Failed to load mistral_v2");

    // V2 vocab: 32,768 base tokens (IDs 0-32767) + 54 agent tokens (IDs 32768-32821)
    // vocab_size() returns max_id + 1 = 32822
    assert_eq!(tok.vocab_size(), 32822);
}

#[test]
fn test_v2_eos_bos_tokens() {
    let tok = from_pretrained("mistral_v2").expect("Failed to load mistral_v2");

    // EOS token should be 2 (</s>)
    let tokens = tok.encode_with_special("</s>");
    assert_eq!(tokens, vec![2]);

    // BOS token should be 1 (<s>)
    let tokens = tok.encode_with_special("<s>");
    assert_eq!(tokens, vec![1]);
}
