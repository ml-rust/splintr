//! Pretrained tokenizer support for common vocabularies.
//!
//! This module provides ready-to-use tokenizers for popular model families:
//! - `cl100k_base` - OpenAI GPT-4, GPT-3.5-turbo (~100k tokens)
//! - `o200k_base` - OpenAI GPT-4o (~200k tokens)
//! - `llama3` - Meta Llama 3 family (~128k tokens)
//! - `deepseek_v3` - DeepSeek V3/R1 (~128k tokens)
//! - `mistral` - Mistral 7B family (~32k tokens)
//! - `whisper` - OpenAI Whisper multilingual v1/v2/v3 (~51k tokens)
//!
//! Every loader here returns an [`AnyTokenizer`], the same universal handle the
//! HuggingFace-json and GGUF loaders return, so a consumer never has to branch
//! on where a vocabulary came from. `encode` means one thing across all of them:
//! special tokens present in the input text are recognized as single ids.
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

use super::any_tokenizer::{AnyTokenizer, Backend};
use super::policy::SpecialPolicy;
use super::spm::{SpmTokenizer, NEVER_MERGE};
use super::tokenizer::{
    Tokenizer, TokenizerError, CL100K_BASE_PATTERN, GPT2_PATTERN, LLAMA3_PATTERN,
    MISTRAL_V3_PATTERN, O200K_BASE_PATTERN, SENTENCEPIECE_PATTERN,
};
use super::vocab::{load_spm_vocab, place_special_pieces};
use super::whisper::{whisper_special_tokens, WhisperVariant};

// Embed vocabulary files at compile time
pub const CL100K_BASE_VOCAB: &[u8] =
    include_bytes!("../../python/splintr/vocabs/cl100k_base.tiktoken");
pub const O200K_BASE_VOCAB: &[u8] =
    include_bytes!("../../python/splintr/vocabs/o200k_base.tiktoken");
pub const LLAMA3_VOCAB: &[u8] = include_bytes!("../../python/splintr/vocabs/llama3.tiktoken");
pub const DEEPSEEK_V3_VOCAB: &[u8] =
    include_bytes!("../../python/splintr/vocabs/deepseek_v3.tiktoken");

/// Mistral V1 SentencePiece vocabulary (32,000 pieces with their scores).
///
/// Extracted straight from `tokenizer.model` by `scripts/extract_spm_vocab.py`,
/// so pieces keep their SentencePiece spelling (`<0x41>`, `▁▁`) and every score
/// survives — including the `-1e9` "never merge" sentinel on the 15 whitespace
/// runs, which the `.tiktoken` form of this vocabulary silently inverted into a
/// *preferred* merge.
pub const MISTRAL_SPM_VOCAB: &[u8] = include_bytes!("../../python/splintr/vocabs/mistral.spm");

/// Mistral V2 SentencePiece vocabulary (32,768 pieces with their scores).
pub const MISTRAL_V2_SPM_VOCAB: &[u8] =
    include_bytes!("../../python/splintr/vocabs/mistral_v2.spm");

/// Mistral V3/Tekken vocabulary file (Tiktoken-based, ~131k tokens).
pub const MISTRAL_V3_VOCAB: &[u8] =
    include_bytes!("../../python/splintr/vocabs/mistral_v3_tekken.tiktoken");

/// Whisper base BPE vocabulary (GPT-2 byte-level, 50,257 tokens).
///
/// Shared by every multilingual variant (v1/v2/v3) — they differ only in the
/// programmatically-generated special tokens. The English-only checkpoints use
/// a different base BPE and are not bundled; load those via
/// [`crate::from_json_path`].
pub const WHISPER_VOCAB: &[u8] = include_bytes!("../../python/splintr/vocabs/whisper.tiktoken");

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
    /// Whisper v1 multilingual (tiny..large, `<|nocaptions|>`, 99 languages)
    WhisperV1,
    /// Whisper v2 multilingual (large-v2, `<|nospeech|>`, 99 languages)
    WhisperV2,
    /// Whisper v3 multilingual (large-v3, adds Cantonese, 100 languages)
    WhisperV3,
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

            // Whisper multilingual. Default "whisper" → v2 (most common).
            "whisper_v1" | "whisper-v1" | "whisper-multilingual-v1" => Some(Self::WhisperV1),
            "whisper" | "whisper_v2" | "whisper-v2" | "whisper-multilingual" => {
                Some(Self::WhisperV2)
            }
            "whisper_v3" | "whisper-v3" | "whisper-large-v3" => Some(Self::WhisperV3),

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
            // Whisper (multilingual)
            "whisper",
            "whisper_v1",
            "whisper_v2",
            "whisper_v3",
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
/// - `whisper`, `whisper_v1`, `whisper_v2`, `whisper_v3` - OpenAI Whisper multilingual
///   (bare `whisper` → v2). English-only checkpoints use a different base BPE and are
///   not bundled — load those via [`crate::from_json_path`].
///
/// # Example
/// ```rust
/// use splintr::pretrained::from_pretrained;
///
/// let tokenizer = from_pretrained("llama3").unwrap();
/// ```
pub fn from_pretrained(name: &str) -> Result<AnyTokenizer, TokenizerError> {
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
///
/// The backend is always BPE; it is wrapped in an [`AnyTokenizer`] whose policy
/// carries the vocabulary's EOS id and its named special tokens, but **no**
/// boundary template — a bundled vocabulary states no such template, and a
/// chat server or trainer places BOS itself. `apply_single` is therefore a
/// passthrough and the encoded ids are exactly the backend's.
pub fn from_vocab(vocab: PretrainedVocab) -> Result<AnyTokenizer, TokenizerError> {
    let special = special_tokens(vocab);
    let named = special.clone();
    let pats = patterns(vocab);

    let tokenizer = match vocab {
        PretrainedVocab::Cl100kBase => {
            Tokenizer::from_bytes_chain(CL100K_BASE_VOCAB, pats, special)
        }
        PretrainedVocab::O200kBase => Tokenizer::from_bytes_chain(O200K_BASE_VOCAB, pats, special),
        PretrainedVocab::Llama3 => Tokenizer::from_bytes_chain(LLAMA3_VOCAB, pats, special),
        PretrainedVocab::DeepseekV3 => {
            // DeepSeek uses ByteLevel BPE encoding
            Tokenizer::from_bytes_byte_level_chain(DEEPSEEK_V3_VOCAB, pats, special)
        }
        // Mistral V1/V2 are SentencePiece, so they take the SPM-BPE backend and
        // return here rather than falling through to byte-level BPE.
        PretrainedVocab::MistralV1 => {
            return spm_from_vocab(MISTRAL_SPM_VOCAB, vocab, special, named)
        }
        PretrainedVocab::MistralV2 => {
            return spm_from_vocab(MISTRAL_V2_SPM_VOCAB, vocab, special, named)
        }
        PretrainedVocab::MistralV3 => {
            // V3 uses ByteLevel BPE (like DeepSeek/GPT-2) - Ġ represents space
            Tokenizer::from_bytes_byte_level_chain(MISTRAL_V3_VOCAB, pats, special)
        }
        PretrainedVocab::WhisperV1 | PretrainedVocab::WhisperV2 | PretrainedVocab::WhisperV3 => {
            // Whisper uses GPT-2 ByteLevel BPE; specials are generated per variant
            Tokenizer::from_bytes_byte_level_chain(WHISPER_VOCAB, pats, special)
        }
    }?;

    // In-text special-token matching is on for every bundled vocabulary, the
    // same as the json and GGUF loaders, so `AnyTokenizer::encode` means one
    // thing regardless of which loader produced the handle.
    Ok(AnyTokenizer::new(
        Backend::Bpe(tokenizer.with_added_token_matching(true)),
        SpecialPolicy::boundary(None, None, Some(eos_token_id(vocab)), named),
    ))
}

/// Build an SPM-BPE tokenizer from a bundled SentencePiece `.spm` file.
///
/// A SentencePiece vocabulary is a list of *pieces*, and its word-boundary
/// marker `▁` is U+2581 = `E2 96 81`. Byte-level BPE builds tokens by merging
/// adjacent **bytes**, and `E2 96` is not a piece any SentencePiece vocabulary
/// was trained on — that intermediate pair never existed — so `▁` survives only
/// when a whole chunk already happens to be one token, and otherwise shatters
/// into three byte-fallback ids. The damage is invisible from the outside: the
/// ids stay in range and decode back to the original text, while the model is
/// fed fragments it never saw during training.
///
/// So the vocabulary is loaded as pieces and merged by
/// [`SpmTokenizer`](super::spm::SpmTokenizer), which reproduces llama.cpp's
/// `llm_tokenizer_spm`.
///
/// **Scores come from the file, not from id order.** SentencePiece merges by
/// score, and the 15 whitespace-run pieces (`▁`, `▁▁`, …) carry a `-1e9`
/// "never merge" sentinel. Falling back to id order does not approximate that,
/// it inverts it: those sentinel pieces sit at low ids, so id order makes the
/// pieces SentencePiece refuses to merge the *first* ones merged. `" Hello
/// world"` came out as `▁▁` + `Hello` + `▁world` (`[259, 16230, 1526]`) instead
/// of `▁` + `▁Hello` + `▁world` (`[28705, 22557, 1526]`).
fn spm_from_vocab(
    data: &[u8],
    vocab: PretrainedVocab,
    special: FxHashMap<String, u32>,
    named: FxHashMap<String, u32>,
) -> Result<AnyTokenizer, TokenizerError> {
    let (mut pieces, mut scores) = load_spm_vocab(data)?;
    // Agent tokens live above the vocabulary file's last id; give them slots so
    // they decode and count towards `vocab_size`, not just match on encode.
    place_special_pieces(&mut pieces, &special)?;
    // Grow the scores in step: `SpmTokenizer` indexes them by id, so a shorter
    // vector is a length mismatch rather than a missing entry.
    //
    // `NEVER_MERGE` is SentencePiece's own "never merge" sentinel, and it is the
    // right value for these slots. Added tokens are matched verbatim before
    // merging, so their score should never decide anything; if a merge does
    // reach one — because the input literally spells it, or because a hole was
    // left by `place_special_pieces` — it must lose to every genuine merge
    // rather than swallow the surrounding text as a chat marker.
    scores.resize(pieces.len(), NEVER_MERGE);

    let eos = eos_token_id(vocab);
    let tokenizer = SpmTokenizer::new(pieces, scores, bos_token_id(vocab), Some(eos))?
        .with_added_tokens(&special);

    Ok(AnyTokenizer::new(
        Backend::Spm(tokenizer),
        SpecialPolicy::boundary(None, None, Some(eos), named),
    ))
}

/// Get the ordered pre-tokenizer pattern sequence for a vocabulary.
///
/// Every bundled vocabulary is currently a single-pass pre-tokenizer, so each
/// arm returns a one-element slice; the return type is a slice (rather than a
/// single pattern) so a vocabulary whose pre-tokenizer is a multi-pass
/// sequence — llama.cpp's `regex_exprs` — can be expressed without changing
/// the accessor's shape again.
pub fn patterns(vocab: PretrainedVocab) -> &'static [&'static str] {
    match vocab {
        PretrainedVocab::Cl100kBase => &[CL100K_BASE_PATTERN],
        PretrainedVocab::O200kBase => &[O200K_BASE_PATTERN],
        PretrainedVocab::Llama3 => &[LLAMA3_PATTERN], // Meta's own split; NOT the o200k pattern
        PretrainedVocab::DeepseekV3 => &[O200K_BASE_PATTERN], // pinned explicitly: DeepSeek is not Llama 3

        PretrainedVocab::MistralV1 | PretrainedVocab::MistralV2 => &[SENTENCEPIECE_PATTERN], // SentencePiece-style
        PretrainedVocab::MistralV3 => &[MISTRAL_V3_PATTERN], // Tekken has its own pattern (no contractions, single-digit numbers)
        PretrainedVocab::WhisperV1 | PretrainedVocab::WhisperV2 | PretrainedVocab::WhisperV3 => {
            &[GPT2_PATTERN]
        }
    }
}

/// Check if a vocabulary uses ByteLevel encoding.
pub fn uses_byte_level(vocab: PretrainedVocab) -> bool {
    matches!(
        vocab,
        PretrainedVocab::DeepseekV3
            | PretrainedVocab::WhisperV1
            | PretrainedVocab::WhisperV2
            | PretrainedVocab::WhisperV3
    )
}

/// Get the EOS (end of sequence) token ID for a vocabulary.
pub fn eos_token_id(vocab: PretrainedVocab) -> u32 {
    match vocab {
        PretrainedVocab::Cl100kBase => 100257, // <|endoftext|>
        PretrainedVocab::O200kBase => 199999,  // <|endoftext|>
        PretrainedVocab::Llama3 => 128001,     // <|end_of_text|>
        PretrainedVocab::DeepseekV3 => 1,      // <｜end▁of▁sentence｜>
        PretrainedVocab::MistralV1 | PretrainedVocab::MistralV2 | PretrainedVocab::MistralV3 => 2, // </s>
        // <|endoftext|>, derived per variant. Matched one variant at a time so the
        // compiler forces this to be revisited if a Whisper generation is added,
        // rather than resolving it through a fallible lookup at runtime.
        PretrainedVocab::WhisperV1 => WhisperVariant::V1Multilingual.eos_token_id(),
        PretrainedVocab::WhisperV2 => WhisperVariant::V2Multilingual.eos_token_id(),
        PretrainedVocab::WhisperV3 => WhisperVariant::V3Multilingual.eos_token_id(),
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
        // Whisper has no BOS; <|startoftranscript|> is a decoding prompt, not a BOS.
        PretrainedVocab::WhisperV1 | PretrainedVocab::WhisperV2 | PretrainedVocab::WhisperV3 => {
            None
        }
    }
}

/// Get the BOS token ID by vocabulary name string.
pub fn bos_token_id_by_name(name: &str) -> Option<u32> {
    PretrainedVocab::from_name(name).and_then(bos_token_id)
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
        // Whisper carries no agent/pad token.
        PretrainedVocab::WhisperV1 | PretrainedVocab::WhisperV2 | PretrainedVocab::WhisperV3 => {
            None
        }
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
        PretrainedVocab::WhisperV1 => whisper_special_tokens(WhisperVariant::V1Multilingual),
        PretrainedVocab::WhisperV2 => whisper_special_tokens(WhisperVariant::V2Multilingual),
        PretrainedVocab::WhisperV3 => whisper_special_tokens(WhisperVariant::V3Multilingual),
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

    // Mistral SentencePiece special tokens (0-2). They are control tokens in the
    // vocabulary file, so they must be matched verbatim in the input: under
    // SentencePiece a bare `<s>` is normalized to `▁<s>`, which is not a piece,
    // and would otherwise merge into `▁<` + `s` + `>` instead of id 1.
    special.insert("<unk>".to_string(), 0);
    special.insert("<s>".to_string(), 1);
    special.insert("</s>".to_string(), 2);

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

    // Native special tokens (same as V1/V2)
    special.insert("<unk>".to_string(), 0);
    special.insert("<s>".to_string(), 1);
    special.insert("</s>".to_string(), 2);

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
    use crate::core::Tokenize;

    /// Reach the concrete BPE tokenizer for assertions on `Tokenizer`-only APIs.
    fn bpe(tokenizer: AnyTokenizer) -> Tokenizer {
        match tokenizer.into_backend() {
            Backend::Bpe(t) => t,
            _ => panic!("this vocabulary does not load as byte-pair encoding"),
        }
    }

    /// The id of a piece in a bundled SentencePiece vocabulary, read from the
    /// vocabulary *file* rather than from tokenizer output — so these tests
    /// assert against the vocabulary, not against whatever the tokenizer
    /// currently happens to produce.
    fn spm_piece_id(vocab_data: &[u8], piece: &str) -> u32 {
        let (pieces, _) = load_spm_vocab(vocab_data).expect("vocabulary loads");
        let id = pieces
            .iter()
            .position(|p| p == piece)
            .unwrap_or_else(|| panic!("{piece:?} is not in the vocabulary"));
        id as u32
    }

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
    fn test_from_pretrained_whisper_variants() {
        // Base BPE is the bundled 50,257-entry GPT-2 vocab for every variant.
        for (name, variant) in [
            ("whisper_v1", WhisperVariant::V1Multilingual),
            ("whisper", WhisperVariant::V2Multilingual), // bare name → v2
            ("whisper_v2", WhisperVariant::V2Multilingual),
            ("whisper-v3", WhisperVariant::V3Multilingual),
        ] {
            let tok = from_pretrained(name).unwrap_or_else(|e| panic!("{name}: {e}"));
            // vocab_size == max special id + 1, which equals the variant's size.
            assert_eq!(tok.vocab_size(), variant.vocab_size(), "{name} vocab_size");
            assert_eq!(bpe(tok).encoder().len(), 50257, "{name} base vocab size");
        }
    }

    /// The policy is an empty boundary template plus the vocabulary's EOS and
    /// named specials: encoding must be untouched, but the ids must be askable.
    #[test]
    fn test_policy_is_passthrough_but_knows_its_specials() {
        let tok = from_pretrained("llama3").unwrap();
        assert_eq!(tok.eos_token_id(), Some(128001));
        assert!(tok.is_eos(128001));
        assert_eq!(tok.special_token_id("<|eot_id|>"), Some(128009));
        // No template: the wrapped ids equal the bare backend output.
        let text = "Hello, world!";
        assert_eq!(tok.encode(text), tok.encode_raw(text));
    }

    /// Batch encoding must agree with encoding each text on its own.
    #[test]
    fn test_encode_batch_matches_individual() {
        let tok = from_pretrained("llama3").unwrap();
        let texts = ["Hello, world!", "", "<|eot_id|>after", "你好世界"];
        let batch = tok.encode_batch(&texts);
        assert_eq!(batch.len(), texts.len());
        for (got, text) in batch.iter().zip(texts) {
            assert_eq!(got, &tok.encode(text), "batch mismatch for {text:?}");
        }
        // Matching is on, so an in-text special is one id, not BPE'd text.
        assert!(batch[2].starts_with(&[128009]));
    }

    #[test]
    fn test_whisper_special_tokens_wired() {
        // Specials must be emitted as single ids at the variant-correct offsets,
        // proving the bundled base aligns with the generated special block.
        let tok = from_pretrained("whisper_v3").unwrap();
        assert_eq!(tok.encode("<|en|>"), vec![50259]);
        assert_eq!(
            tok.encode("<|transcribe|>"),
            vec![WhisperVariant::V3Multilingual.transcribe_token_id()]
        );
        // <|yue|> only exists on v3.
        assert_eq!(tok.encode("<|yue|>"), vec![50259 + 99]);
    }

    #[test]
    fn test_whisper_roundtrip() {
        let tok = from_pretrained("whisper").unwrap();
        let text = "Hello, world! 123 héllo";
        assert_eq!(tok.decode(&tok.encode(text)).unwrap(), text);
    }

    #[test]
    fn test_whisper_name_mapping() {
        assert_eq!(
            PretrainedVocab::from_name("whisper"),
            Some(PretrainedVocab::WhisperV2)
        );
        assert_eq!(
            PretrainedVocab::from_name("whisper-large-v3"),
            Some(PretrainedVocab::WhisperV3)
        );
        // English-only is intentionally not bundled (different base BPE).
        assert_eq!(PretrainedVocab::from_name("whisper.en"), None);
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

    /// The defect the SPM-BPE backend exists to prevent, asserted against the
    /// vocabulary file rather than against reference ids.
    ///
    /// The SentencePiece word-boundary marker `▁` is U+2581 = `E2 96 81`.
    /// Merging adjacent *bytes* can never build it, because `E2 96` is not a
    /// piece any SentencePiece vocabulary was trained on, so under a byte-level
    /// merger every word boundary collapses into the three byte-fallback ids
    /// for `<0xE2> <0x96> <0x81>`. That failure is invisible from the outside —
    /// the ids stay in range and decode back to the original text — so it is
    /// pinned here directly.
    #[test]
    fn test_mistral_never_shatters_the_word_boundary_marker() {
        for (name, data) in [
            ("mistral", MISTRAL_SPM_VOCAB),
            ("mistral_v2", MISTRAL_V2_SPM_VOCAB),
        ] {
            let shattered = [
                spm_piece_id(data, "<0xE2>"),
                spm_piece_id(data, "<0x96>"),
                spm_piece_id(data, "<0x81>"),
            ];
            let tokenizer = from_pretrained(name).unwrap();
            let ids = tokenizer.encode("the sourdough starter rose overnight");
            assert!(
                !ids.windows(3).any(|w| w == shattered.as_slice()),
                "{name}: word boundary shattered into byte tokens {shattered:?} in {ids:?}"
            );
        }
    }

    /// Whole words in the vocabulary must be reachable. `▁the` and `▁sour` are
    /// both present in the Mistral V1 file, so encoding text that starts those
    /// words must produce them — a byte-level merger reaches neither.
    #[test]
    fn test_mistral_reaches_whole_word_pieces() {
        let the = spm_piece_id(MISTRAL_SPM_VOCAB, "▁the");
        let sour = spm_piece_id(MISTRAL_SPM_VOCAB, "▁sour");
        let ids = from_pretrained("mistral").unwrap().encode("the sourdough");
        assert!(ids.contains(&the), "▁the ({the}) missing from {ids:?}");
        assert!(ids.contains(&sour), "▁sour ({sour}) missing from {ids:?}");
    }

    /// A sentence must survive encode → decode *exactly*. `add_dummy_prefix`
    /// puts a word boundary before the first piece on encode, and decoding
    /// removes it again, so nothing differs at all: no leading space, no
    /// dropped character, no replacement char, no reordering. Reference:
    /// `sp.decode(sp.encode("Hello, world!")) == "Hello, world!"`.
    #[test]
    fn test_mistral_round_trips_a_sentence() {
        let tokenizer = from_pretrained("mistral").unwrap();
        let text = "The quick brown fox jumps over the lazy dog.";
        let decoded = tokenizer.decode(&tokenizer.encode(text)).unwrap();
        assert_eq!(decoded, text);
    }
}
