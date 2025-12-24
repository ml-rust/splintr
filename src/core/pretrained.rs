//! Pretrained tokenizer support for common vocabularies.
//!
//! This module provides ready-to-use tokenizers for popular model families:
//! - `cl100k_base` - OpenAI GPT-4, GPT-3.5-turbo (~100k tokens)
//! - `o200k_base` - OpenAI GPT-4o (~200k tokens)
//! - `llama3` - Meta Llama 3 family (~128k tokens)
//! - `deepseek_v3` - DeepSeek V3/R1 (~128k tokens)
//! - `mistral` - Mistral 7B family (~32k tokens)
//!
//! # Example
//!
//! ```rust
//! use splintr::pretrained::from_pretrained;
//!
//! let tokenizer = from_pretrained("llama3").unwrap();
//! let tokens = tokenizer.encode("Hello, world!");
//! ```

use rustc_hash::FxHashMap;

use super::tokenizer::{
    Tokenizer, TokenizerError, CL100K_BASE_PATTERN, O200K_BASE_PATTERN, SENTENCEPIECE_PATTERN,
};

// Embed vocabulary files at compile time
pub const CL100K_BASE_VOCAB: &[u8] =
    include_bytes!("../../python/splintr/vocabs/cl100k_base.tiktoken");
pub const O200K_BASE_VOCAB: &[u8] =
    include_bytes!("../../python/splintr/vocabs/o200k_base.tiktoken");
pub const LLAMA3_VOCAB: &[u8] = include_bytes!("../../python/splintr/vocabs/llama3.tiktoken");
pub const DEEPSEEK_V3_VOCAB: &[u8] =
    include_bytes!("../../python/splintr/vocabs/deepseek_v3.tiktoken");
pub const MISTRAL_VOCAB: &[u8] = include_bytes!("../../python/splintr/vocabs/mistral.tiktoken");
pub const MISTRAL_V2_VOCAB: &[u8] =
    include_bytes!("../../python/splintr/vocabs/mistral_v2.tiktoken");

/// Mistral V3/Tekken vocabulary file (Tiktoken-based, ~131k tokens).
pub const MISTRAL_V3_VOCAB: &[u8] =
    include_bytes!("../../python/splintr/vocabs/mistral_v3_tekken.tiktoken");

/// Supported pretrained vocabulary types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PretrainedVocab {
    /// OpenAI cl100k_base (GPT-4, GPT-3.5-turbo)
    Cl100kBase,
    /// OpenAI o200k_base (GPT-4o)
    O200kBase,
    /// Meta Llama 3 family
    Llama3,
    /// DeepSeek V3/R1
    DeepseekV3,
    /// Mistral V1 (7B v0.1/v0.2, Mixtral 8x7B) - 32k SentencePiece
    MistralV1,
    /// Mistral V2 (7B v0.3, Mixtral 8x22B, Codestral) - 32k + 768 control tokens
    MistralV2,
    /// Mistral V3/Tekken (NeMo, Large 2, Pixtral) - 131k Tiktoken-based
    MistralV3,
}

impl PretrainedVocab {
    /// Parse vocabulary name from string.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "cl100k_base" => Some(Self::Cl100kBase),
            "o200k_base" => Some(Self::O200kBase),
            "llama3" | "llama3.1" | "llama3.2" | "llama3.3" => Some(Self::Llama3),
            "deepseek_v3" | "deepseek-v3" => Some(Self::DeepseekV3),

            // Mistral V1: Default mistral → V1
            "mistral" | "mistral_v1" => Some(Self::MistralV1),

            // Mistral V2: Extended vocabulary with control tokens
            "mistral_v2" => Some(Self::MistralV2),

            // Mistral V3: Tekken-based high-efficiency vocabulary
            "mistral_v3" => Some(Self::MistralV3),

            _ => None,
        }
    }

    /// Get all supported vocabulary names.
    pub fn supported_names() -> &'static [&'static str] {
        &[
            "cl100k_base",
            "o200k_base",
            "llama3",
            "llama3.1",
            "llama3.2",
            "llama3.3",
            "deepseek_v3",
            "deepseek-v3",
            // Mistral
            "mistral",
            "mistral_v1",
            "mistral_v2",
            "mistral_v3",
        ]
    }
}

/// Create a pretrained tokenizer by vocabulary name.
///
/// # Supported Names
/// - `cl100k_base` - OpenAI GPT-4, GPT-3.5-turbo
/// - `o200k_base` - OpenAI GPT-4o
/// - `llama3`, `llama3.1`, `llama3.2`, `llama3.3` - Meta Llama 3 family
/// - `deepseek_v3`, `deepseek-v3` - DeepSeek V3/R1
/// - `mistral`, `mistral-7b` - Mistral 7B family
///
/// # Example
/// ```rust
/// use splintr::pretrained::from_pretrained;
///
/// let tokenizer = from_pretrained("llama3").unwrap();
/// ```
pub fn from_pretrained(name: &str) -> Result<Tokenizer, TokenizerError> {
    let vocab = PretrainedVocab::from_name(name).ok_or_else(|| {
        TokenizerError::UnknownPretrained(format!(
            "{}. Supported: {}",
            name,
            PretrainedVocab::supported_names().join(", ")
        ))
    })?;

    from_vocab(vocab)
}

/// Create a pretrained tokenizer from vocabulary enum.
pub fn from_vocab(vocab: PretrainedVocab) -> Result<Tokenizer, TokenizerError> {
    let special = special_tokens(vocab);
    let pat = pattern(vocab);

    match vocab {
        PretrainedVocab::Cl100kBase => Tokenizer::from_bytes(CL100K_BASE_VOCAB, pat, special),
        PretrainedVocab::O200kBase => Tokenizer::from_bytes(O200K_BASE_VOCAB, pat, special),
        PretrainedVocab::Llama3 => Tokenizer::from_bytes(LLAMA3_VOCAB, pat, special),
        PretrainedVocab::DeepseekV3 => {
            // DeepSeek uses ByteLevel BPE encoding
            Tokenizer::from_bytes_byte_level(DEEPSEEK_V3_VOCAB, pat, special)
        }
        PretrainedVocab::MistralV1 => {
            // Mistral V1 uses SentencePiece - decoder converts ▁ (U+2581) to space
            Tokenizer::from_bytes_sentencepiece(MISTRAL_VOCAB, pat, special)
        }
        PretrainedVocab::MistralV2 => {
            // Mistral V2 vocab file has byte fallback tokens that duplicate BPE merges
            // Use with_decoder variant to preserve all 32,768 token IDs
            Tokenizer::from_bytes_sentencepiece_with_decoder(MISTRAL_V2_VOCAB, pat, special)
        }
        PretrainedVocab::MistralV3 => {
            // V3 uses Tiktoken (NOT SentencePiece) - standard tiktoken encoding
            Tokenizer::from_bytes(MISTRAL_V3_VOCAB, pat, special)
        }
    }
}

/// Get the regex pattern for a vocabulary.
pub fn pattern(vocab: PretrainedVocab) -> &'static str {
    match vocab {
        PretrainedVocab::Cl100kBase => CL100K_BASE_PATTERN,
        PretrainedVocab::O200kBase => O200K_BASE_PATTERN,
        PretrainedVocab::Llama3 => O200K_BASE_PATTERN, // Llama3 uses same pattern as O200K
        PretrainedVocab::DeepseekV3 => O200K_BASE_PATTERN, // DeepSeek uses same pattern
        PretrainedVocab::MistralV1 | PretrainedVocab::MistralV2 => SENTENCEPIECE_PATTERN, // SentencePiece-style
        PretrainedVocab::MistralV3 => O200K_BASE_PATTERN, // Tekken uses similar pattern to O200K
    }
}

/// Check if a vocabulary uses ByteLevel encoding.
pub fn uses_byte_level(vocab: PretrainedVocab) -> bool {
    matches!(vocab, PretrainedVocab::DeepseekV3)
}

/// Get the EOS (end of sequence) token ID for a vocabulary.
pub fn eos_token_id(vocab: PretrainedVocab) -> u32 {
    match vocab {
        PretrainedVocab::Cl100kBase => 100257, // <|endoftext|>
        PretrainedVocab::O200kBase => 199999,  // <|endoftext|>
        PretrainedVocab::Llama3 => 128001,     // <|end_of_text|>
        PretrainedVocab::DeepseekV3 => 1,      // <｜end▁of▁sentence｜>
        PretrainedVocab::MistralV1 | PretrainedVocab::MistralV2 | PretrainedVocab::MistralV3 => 2, // </s>
    }
}

/// Get the EOS token ID by vocabulary name string.
pub fn eos_token_id_by_name(name: &str) -> u32 {
    PretrainedVocab::from_name(name)
        .map(eos_token_id)
        .unwrap_or(0)
}

/// Get the BOS (beginning of sequence) token ID for a vocabulary.
pub fn bos_token_id(vocab: PretrainedVocab) -> Option<u32> {
    match vocab {
        PretrainedVocab::Cl100kBase => None,     // No BOS token
        PretrainedVocab::O200kBase => None,      // No BOS token
        PretrainedVocab::Llama3 => Some(128000), // <|begin_of_text|>
        PretrainedVocab::DeepseekV3 => Some(0),  // <｜begin▁of▁sentence｜>
        PretrainedVocab::MistralV1 | PretrainedVocab::MistralV2 | PretrainedVocab::MistralV3 => {
            Some(1)
        } // <s>
    }
}

/// Get the PAD token ID for a vocabulary.
pub fn pad_token_id(vocab: PretrainedVocab) -> Option<u32> {
    match vocab {
        PretrainedVocab::Cl100kBase => Some(100316), // <|pad|> (agent token)
        PretrainedVocab::O200kBase => Some(200058),  // <|pad|> (agent token)
        PretrainedVocab::Llama3 => Some(128339),     // <|pad|> (agent token)
        PretrainedVocab::DeepseekV3 => Some(2),      // <｜▁pad▁｜>
        PretrainedVocab::MistralV1 => Some(32039),   // <|pad|> (agent token)
        PretrainedVocab::MistralV2 => Some(32807),   // <|pad|> (agent token, after control tokens)
        PretrainedVocab::MistralV3 => Some(131111),  // <|pad|> (agent token)
    }
}

/// Get the special tokens map for a vocabulary.
pub fn special_tokens(vocab: PretrainedVocab) -> FxHashMap<String, u32> {
    match vocab {
        PretrainedVocab::Cl100kBase => cl100k_base_special_tokens(),
        PretrainedVocab::O200kBase => o200k_base_special_tokens(),
        PretrainedVocab::Llama3 => llama3_special_tokens(),
        PretrainedVocab::DeepseekV3 => deepseek_v3_special_tokens(),
        PretrainedVocab::MistralV1 => mistral_v1_special_tokens(),
        PretrainedVocab::MistralV2 => mistral_v2_special_tokens(),
        PretrainedVocab::MistralV3 => mistral_v3_special_tokens(),
    }
}

// =============================================================================
// Special token definitions for each vocabulary
// =============================================================================

/// Get the standard special tokens for cl100k_base encoding (GPT-4, GPT-3.5-turbo).
pub fn cl100k_base_special_tokens() -> FxHashMap<String, u32> {
    let mut special = FxHashMap::default();
    // OpenAI standard special tokens (100257-100276)
    special.insert("<|endoftext|>".to_string(), 100257);
    special.insert("<|fim_prefix|>".to_string(), 100258);
    special.insert("<|fim_middle|>".to_string(), 100259);
    special.insert("<|fim_suffix|>".to_string(), 100260);
    special.insert("<|endofprompt|>".to_string(), 100276);

    // Agent tokens (100277+)
    insert_agent_tokens(&mut special, 100277);

    special
}

/// Get the standard special tokens for o200k_base encoding (GPT-4o).
pub fn o200k_base_special_tokens() -> FxHashMap<String, u32> {
    let mut special = FxHashMap::default();
    // OpenAI standard special tokens (199999-200018)
    special.insert("<|endoftext|>".to_string(), 199999);
    special.insert("<|endofprompt|>".to_string(), 200018);

    // Agent tokens (200019+)
    insert_agent_tokens(&mut special, 200019);

    special
}

/// Get the standard special tokens for Llama 3 encoding.
pub fn llama3_special_tokens() -> FxHashMap<String, u32> {
    let mut special = FxHashMap::default();

    // Meta standard special tokens (128000-128010)
    special.insert("<|begin_of_text|>".to_string(), 128000);
    special.insert("<|end_of_text|>".to_string(), 128001);
    special.insert("<|reserved_special_token_0|>".to_string(), 128002);
    special.insert("<|reserved_special_token_1|>".to_string(), 128003);
    special.insert("<|finetune_right_pad_id|>".to_string(), 128004);
    special.insert("<|step_id|>".to_string(), 128005);
    special.insert("<|start_header_id|>".to_string(), 128006);
    special.insert("<|end_header_id|>".to_string(), 128007);
    special.insert("<|eom_id|>".to_string(), 128008);
    special.insert("<|eot_id|>".to_string(), 128009);
    special.insert("<|python_tag|>".to_string(), 128010);

    // Multimodal tokens (128256+) - aligned with official Meta tokens
    special.insert("<|image|>".to_string(), 128256);
    special.insert("<|/image|>".to_string(), 128257);
    special.insert("<|audio|>".to_string(), 128258);
    special.insert("<|/audio|>".to_string(), 128259);
    special.insert("<|video|>".to_string(), 128260);
    special.insert("<|/video|>".to_string(), 128261);

    // Agent tokens (128300+)
    insert_agent_tokens_llama3(&mut special, 128300);

    special
}

/// Get the standard special tokens for DeepSeek V3 encoding.
pub fn deepseek_v3_special_tokens() -> FxHashMap<String, u32> {
    let mut special = FxHashMap::default();

    // DeepSeek native special tokens (0-2)
    special.insert("<｜begin▁of▁sentence｜>".to_string(), 0);
    special.insert("<｜end▁of▁sentence｜>".to_string(), 1);
    special.insert("<｜▁pad▁｜>".to_string(), 2);

    // Thinking tokens (128798-128799)
    special.insert("<think>".to_string(), 128798);
    special.insert("</think>".to_string(), 128799);

    // FIM (Fill-in-the-Middle) tokens (128800-128802)
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
    insert_agent_tokens(&mut special, 128900);

    special
}

/// Get the standard special tokens for Mistral V1 encoding.
pub fn mistral_v1_special_tokens() -> FxHashMap<String, u32> {
    let mut special = FxHashMap::default();

    // Mistral SentencePiece special tokens (0-2)
    special.insert("<unk>".to_string(), 0);
    special.insert("<s>".to_string(), 1);
    special.insert("</s>".to_string(), 2);

    // Agent tokens (32000+) - Mistral V1 vocab has ~32000 base tokens
    insert_agent_tokens(&mut special, 32000);

    special
}

/// Get the standard special tokens for Mistral V2 encoding.
pub fn mistral_v2_special_tokens() -> FxHashMap<String, u32> {
    let mut special = FxHashMap::default();

    // V2 control tokens (for Aho-Corasick matching in encode_with_special)
    // These are also in the vocab file, but adding them here allows clean matching
    special.insert("[INST]".to_string(), 3);
    special.insert("[/INST]".to_string(), 4);
    special.insert("[TOOL_CALLS]".to_string(), 5);
    special.insert("[AVAILABLE_TOOLS]".to_string(), 6);
    special.insert("[/AVAILABLE_TOOLS]".to_string(), 7);
    special.insert("[TOOL_RESULTS]".to_string(), 8);
    special.insert("[/TOOL_RESULTS]".to_string(), 9);

    // Agent tokens start at 32768 (after V2 control token range)
    insert_agent_tokens(&mut special, 32768);

    special
}

/// Get the standard special tokens for Mistral V3/Tekken encoding.
pub fn mistral_v3_special_tokens() -> FxHashMap<String, u32> {
    let mut special = FxHashMap::default();

    // V3 control tokens (for Aho-Corasick matching)
    special.insert("[INST]".to_string(), 3);
    special.insert("[/INST]".to_string(), 4);
    special.insert("[AVAILABLE_TOOLS]".to_string(), 5);
    special.insert("[/AVAILABLE_TOOLS]".to_string(), 6);
    special.insert("[TOOL_RESULTS]".to_string(), 7);
    special.insert("[/TOOL_RESULTS]".to_string(), 8);
    special.insert("[TOOL_CALLS]".to_string(), 9);

    // Agent tokens start at 131072 (after base vocab)
    insert_agent_tokens(&mut special, 131072);

    special
}

// =============================================================================
// Helper functions for agent tokens
// =============================================================================

/// Insert the standard 54 agent tokens starting at the given base ID.
/// Used by cl100k_base, o200k_base, and deepseek_v3.
fn insert_agent_tokens(special: &mut FxHashMap<String, u32>, base: u32) {
    // Core conversation structure
    special.insert("<|system|>".to_string(), base);
    special.insert("<|user|>".to_string(), base + 1);
    special.insert("<|assistant|>".to_string(), base + 2);
    special.insert("<|im_start|>".to_string(), base + 3);
    special.insert("<|im_end|>".to_string(), base + 4);

    // Reasoning/thinking tokens
    special.insert("<|think|>".to_string(), base + 5);
    special.insert("<|/think|>".to_string(), base + 6);

    // ReAct agent loop tokens
    special.insert("<|plan|>".to_string(), base + 7);
    special.insert("<|/plan|>".to_string(), base + 8);
    special.insert("<|step|>".to_string(), base + 9);
    special.insert("<|/step|>".to_string(), base + 10);
    special.insert("<|act|>".to_string(), base + 11);
    special.insert("<|/act|>".to_string(), base + 12);
    special.insert("<|observe|>".to_string(), base + 13);
    special.insert("<|/observe|>".to_string(), base + 14);

    // Tool/function calling
    special.insert("<|function|>".to_string(), base + 15);
    special.insert("<|/function|>".to_string(), base + 16);
    special.insert("<|result|>".to_string(), base + 17);
    special.insert("<|/result|>".to_string(), base + 18);
    special.insert("<|error|>".to_string(), base + 19);
    special.insert("<|/error|>".to_string(), base + 20);

    // Code execution
    special.insert("<|code|>".to_string(), base + 21);
    special.insert("<|/code|>".to_string(), base + 22);
    special.insert("<|output|>".to_string(), base + 23);
    special.insert("<|/output|>".to_string(), base + 24);
    special.insert("<|lang|>".to_string(), base + 25);
    special.insert("<|/lang|>".to_string(), base + 26);

    // RAG/context injection
    special.insert("<|context|>".to_string(), base + 27);
    special.insert("<|/context|>".to_string(), base + 28);
    special.insert("<|quote|>".to_string(), base + 29);
    special.insert("<|/quote|>".to_string(), base + 30);
    special.insert("<|cite|>".to_string(), base + 31);
    special.insert("<|/cite|>".to_string(), base + 32);
    special.insert("<|source|>".to_string(), base + 33);
    special.insert("<|/source|>".to_string(), base + 34);

    // Memory/state management
    special.insert("<|memory|>".to_string(), base + 35);
    special.insert("<|/memory|>".to_string(), base + 36);
    special.insert("<|recall|>".to_string(), base + 37);
    special.insert("<|/recall|>".to_string(), base + 38);

    // Control tokens
    special.insert("<|pad|>".to_string(), base + 39);
    special.insert("<|stop|>".to_string(), base + 40);
    special.insert("<|sep|>".to_string(), base + 41);

    // Multimodal placeholders
    special.insert("<|image|>".to_string(), base + 42);
    special.insert("<|/image|>".to_string(), base + 43);
    special.insert("<|audio|>".to_string(), base + 44);
    special.insert("<|/audio|>".to_string(), base + 45);
    special.insert("<|video|>".to_string(), base + 46);
    special.insert("<|/video|>".to_string(), base + 47);

    // Document structure
    special.insert("<|title|>".to_string(), base + 48);
    special.insert("<|/title|>".to_string(), base + 49);
    special.insert("<|section|>".to_string(), base + 50);
    special.insert("<|/section|>".to_string(), base + 51);
    special.insert("<|summary|>".to_string(), base + 52);
    special.insert("<|/summary|>".to_string(), base + 53);
}

/// Insert agent tokens for Llama3 (excludes multimodal since they're at 128256+).
fn insert_agent_tokens_llama3(special: &mut FxHashMap<String, u32>, base: u32) {
    // Core conversation structure
    special.insert("<|system|>".to_string(), base);
    special.insert("<|user|>".to_string(), base + 1);
    special.insert("<|assistant|>".to_string(), base + 2);
    special.insert("<|im_start|>".to_string(), base + 3);
    special.insert("<|im_end|>".to_string(), base + 4);

    // Reasoning/thinking tokens
    special.insert("<|think|>".to_string(), base + 5);
    special.insert("<|/think|>".to_string(), base + 6);

    // ReAct agent loop tokens
    special.insert("<|plan|>".to_string(), base + 7);
    special.insert("<|/plan|>".to_string(), base + 8);
    special.insert("<|step|>".to_string(), base + 9);
    special.insert("<|/step|>".to_string(), base + 10);
    special.insert("<|act|>".to_string(), base + 11);
    special.insert("<|/act|>".to_string(), base + 12);
    special.insert("<|observe|>".to_string(), base + 13);
    special.insert("<|/observe|>".to_string(), base + 14);

    // Tool/function calling
    special.insert("<|function|>".to_string(), base + 15);
    special.insert("<|/function|>".to_string(), base + 16);
    special.insert("<|result|>".to_string(), base + 17);
    special.insert("<|/result|>".to_string(), base + 18);
    special.insert("<|error|>".to_string(), base + 19);
    special.insert("<|/error|>".to_string(), base + 20);

    // Code execution
    special.insert("<|code|>".to_string(), base + 21);
    special.insert("<|/code|>".to_string(), base + 22);
    special.insert("<|output|>".to_string(), base + 23);
    special.insert("<|/output|>".to_string(), base + 24);
    special.insert("<|lang|>".to_string(), base + 25);
    special.insert("<|/lang|>".to_string(), base + 26);

    // RAG/context injection
    special.insert("<|context|>".to_string(), base + 27);
    special.insert("<|/context|>".to_string(), base + 28);
    special.insert("<|quote|>".to_string(), base + 29);
    special.insert("<|/quote|>".to_string(), base + 30);
    special.insert("<|cite|>".to_string(), base + 31);
    special.insert("<|/cite|>".to_string(), base + 32);
    special.insert("<|source|>".to_string(), base + 33);
    special.insert("<|/source|>".to_string(), base + 34);

    // Memory/state management
    special.insert("<|memory|>".to_string(), base + 35);
    special.insert("<|/memory|>".to_string(), base + 36);
    special.insert("<|recall|>".to_string(), base + 37);
    special.insert("<|/recall|>".to_string(), base + 38);

    // Control tokens
    special.insert("<|pad|>".to_string(), base + 39);
    special.insert("<|stop|>".to_string(), base + 40);
    special.insert("<|sep|>".to_string(), base + 41);

    // Note: Multimodal tokens are at 128256+ for Llama3, already inserted separately

    // Document structure
    special.insert("<|title|>".to_string(), base + 48);
    special.insert("<|/title|>".to_string(), base + 49);
    special.insert("<|section|>".to_string(), base + 50);
    special.insert("<|/section|>".to_string(), base + 51);
    special.insert("<|summary|>".to_string(), base + 52);
    special.insert("<|/summary|>".to_string(), base + 53);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_pretrained_llama3() {
        let tokenizer = from_pretrained("llama3").unwrap();
        assert!(tokenizer.vocab_size() > 100000);
    }

    #[test]
    fn test_from_pretrained_cl100k() {
        let tokenizer = from_pretrained("cl100k_base").unwrap();
        assert!(tokenizer.vocab_size() > 90000);
    }

    #[test]
    fn test_eos_token_ids() {
        assert_eq!(eos_token_id(PretrainedVocab::Cl100kBase), 100257);
        assert_eq!(eos_token_id(PretrainedVocab::O200kBase), 199999);
        assert_eq!(eos_token_id(PretrainedVocab::Llama3), 128001);
        assert_eq!(eos_token_id(PretrainedVocab::DeepseekV3), 1);
        assert_eq!(eos_token_id(PretrainedVocab::MistralV1), 2);
    }

    #[test]
    fn test_vocab_from_name() {
        assert_eq!(
            PretrainedVocab::from_name("llama3"),
            Some(PretrainedVocab::Llama3)
        );
        assert_eq!(
            PretrainedVocab::from_name("llama3.1"),
            Some(PretrainedVocab::Llama3)
        );
        assert_eq!(
            PretrainedVocab::from_name("deepseek_v3"),
            Some(PretrainedVocab::DeepseekV3)
        );
        assert_eq!(
            PretrainedVocab::from_name("mistral"),
            Some(PretrainedVocab::MistralV1)
        );
        assert_eq!(PretrainedVocab::from_name("unknown"), None);
    }

    #[test]
    fn test_from_pretrained_mistral() {
        let tokenizer = from_pretrained("mistral").unwrap();
        // Mistral has ~32k tokens (31997 regular + 3 special)
        assert!(tokenizer.vocab_size() >= 31000);
    }

    #[test]
    fn test_mistral_encode_decode() {
        let tokenizer = from_pretrained("mistral").unwrap();

        // Test basic encoding
        let text = "Hello, world!";
        let tokens = tokenizer.encode(text);
        assert!(!tokens.is_empty());

        // Test decoding
        let decoded = tokenizer.decode(&tokens).unwrap();
        // Test exact roundtrip
        assert_eq!(decoded, text, "Encoding should be reversible");
    }
}
