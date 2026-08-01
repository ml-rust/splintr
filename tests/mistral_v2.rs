use splintr::{from_pretrained, Tokenize};

// =============================================================================
// Exact Token ID Tests
// =============================================================================

/// Reference ids from the `sentencepiece` Python package, version 0.2.0,
/// reading Mistral 7B v0.3's own `tokenizer.model` — the 32,768-piece model
/// that splintr bundles as `mistral_v2`, 15 of whose pieces carry the `-1e9`
/// "never merge" score sentinel.
///
/// These are the reference, not a snapshot: a mismatch means splintr diverged
/// from SentencePiece. They pin what the tiktoken-format vocabulary could not
/// express — without the file's scores, `SpmTokenizer` merged in token-id
/// order, which inverts SentencePiece's order for the whitespace-run pieces and
/// turned `" Hello world"` into `[1027, 16998, 2294]`.
#[test]
fn v2_matches_sentencepiece_exactly() {
    let tok = from_pretrained("mistral_v2").expect("mistral_v2 loads");
    let cases: &[(&str, &[u32])] = &[
        ("the sourdough", &[1040, 18961, 29483, 1668]),
        (" Hello world", &[29473, 23325, 2294]),
        ("Hello, world!", &[23325, 29493, 2294, 29576]),
        ("hello world", &[7080, 29477, 2294]),
    ];

    for (text, expected) in cases {
        assert_eq!(
            tok.encode_raw(text),
            *expected,
            "sentencepiece reference mismatch for {text:?}"
        );
    }
}

/// The reference ids must also decode back to the text they came from.
#[test]
fn v2_reference_cases_round_trip() {
    let tok = from_pretrained("mistral_v2").expect("mistral_v2 loads");
    for text in [
        "the sourdough",
        " Hello world",
        "Hello, world!",
        "hello world",
    ] {
        let ids = tok.encode_raw(text);
        let decoded = tok.decode(&ids).expect("decodes");
        assert_eq!(decoded, text, "round trip failed for {text:?}");
    }
}

/// SentencePiece applies `add_dummy_prefix` to the whole input *before* it
/// splits on added tokens, so a control token at byte 0 leaves the prefix with
/// nothing to attach to and it is emitted as the lone `▁` piece (id 29473).
///
/// Reference (`AutoTokenizer.from_pretrained("mistral-7b-v0.3",
/// use_fast=False)`, `add_special_tokens=False`):
/// `"[INST]"` -> `[29473, 3]`, `"[/INST]"` -> `[29473, 4]`,
/// `"[INST]Hello[/INST]"` -> `[29473, 3, 16998, 4]` — note `16998` is bare
/// `Hello`, not `▁Hello`: a gap that follows a control token gets no prefix of
/// its own.
#[test]
fn test_v2_control_tokens_inst() {
    let tok = from_pretrained("mistral_v2").expect("Failed to load mistral_v2");

    // Test [INST] token (ID 3)
    let tokens = tok.encode("[INST]");
    assert_eq!(tokens, vec![29473, 3]);

    // Test [/INST] token (ID 4)
    let tokens = tok.encode("[/INST]");
    assert_eq!(tokens, vec![29473, 4]);

    // Test instruction format
    let tokens = tok.encode("[INST]Hello[/INST]");
    assert_eq!(tokens, vec![29473, 3, 16998, 4]);
}

#[test]
fn test_v2_control_tokens_tool_calls() {
    let tok = from_pretrained("mistral_v2").expect("Failed to load mistral_v2");

    // Test [TOOL_CALLS] token (ID 5); leading `29473` is the standalone dummy
    // prefix, exactly as the reference emits it.
    let tokens = tok.encode("[TOOL_CALLS]");
    assert_eq!(tokens, vec![29473, 5]);

    // Test [AVAILABLE_TOOLS] token (ID 6)
    let tokens = tok.encode("[AVAILABLE_TOOLS]");
    assert_eq!(tokens, vec![29473, 6]);
}

/// The vocabulary's own sentinels are the exception: a leading `<s>`, `</s>` or
/// `<unk>` *swallows* the standalone dummy prefix, so these stay single ids
/// while `"[INST]"` above does not. Reference: `"<s>"` -> `[1]`, `"</s>"` ->
/// `[2]`, `"<unk>"` -> `[0]`, versus `"[INST]"` -> `[29473, 3]`.
#[test]
fn test_v2_native_sentencepiece_tokens() {
    let tok = from_pretrained("mistral_v2").expect("Failed to load mistral_v2");

    // Test <s> token (ID 1)
    let tokens = tok.encode("<s>");
    assert_eq!(tokens, vec![1]);

    // Test </s> token (ID 2)
    let tokens = tok.encode("</s>");
    assert_eq!(tokens, vec![2]);

    // Test <unk> token (ID 0)
    let tokens = tok.encode("<unk>");
    assert_eq!(tokens, vec![0]);
}

// Note: V1 and V2 do NOT share the same vocab structure, so this test is removed.
// V1: IDs 3-31999 are BPE merges
// V2: IDs 0-770 are special/control tokens, IDs 771-1026 are byte fallback, IDs 1027-32767 are BPE merges
// They encode the same text to different token IDs, which is expected.

#[test]
fn test_v2_agent_tokens() {
    let tok = from_pretrained("mistral_v2").expect("Failed to load mistral_v2");

    // Agent tokens start at 32768 for V2. They are ordinary added tokens, not
    // the vocabulary's BOS/EOS/UNK sentinels, so a leading one carries the
    // standalone dummy prefix (29473) just as `[INST]` does.
    // <|think|> is at offset 5 (after system, user, assistant, im_start, im_end)
    let tokens = tok.encode("<|think|>");
    assert_eq!(tokens, vec![29473, 32773]); // THINK token = 32768 + 5

    // <|function|> is at offset 15
    let tokens = tok.encode("<|function|>");
    assert_eq!(tokens, vec![29473, 32783]); // FUNCTION token = 32768 + 15
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
    let tokens = tok.encode(text);

    // Reference ids (`AutoTokenizer.from_pretrained("mistral-7b-v0.3",
    // use_fast=False)`, `add_special_tokens=False`): the lone dummy prefix,
    // `[INST]`, then bare `What` — the gap after a control token is not
    // re-prefixed — and `[/INST]`.
    assert_eq!(
        tokens,
        vec![29473, 3, 3963, 1117, 1040, 8854, 3922, 29572, 4]
    );

    // Because the dummy prefix is now the only space-like piece added, and
    // `decode` strips exactly that one, this string does round-trip. (HF's own
    // detokenizer prints "[INST] What ... [/INST]" — it re-spaces around added
    // tokens on the way out. The ids are the contract; that spacing is not.)
    let decoded = tok.decode(&tokens).expect("Failed to decode");
    assert_eq!(decoded, text);
}

#[test]
fn test_v2_model_name_underscore() {
    let tok = from_pretrained("mistral_v2").unwrap();

    // V2 should recognize control tokens (preceded by the standalone dummy
    // prefix, since `[INST]` opens the input).
    let tokens = tok.encode("[INST]");
    assert_eq!(tokens, vec![29473, 3]);
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
    let tokens = tok.encode("</s>");
    assert_eq!(tokens, vec![2]);

    // BOS token should be 1 (<s>)
    let tokens = tok.encode("<s>");
    assert_eq!(tokens, vec![1]);
}
