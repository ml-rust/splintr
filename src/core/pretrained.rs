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

use rustc_hash::{FxHashMap, FxHashSet};

use super::any_tokenizer::{AnyTokenizer, Backend};
use super::policy::SpecialPolicy;
use super::spm::{SpmPrefixScheme, SpmTokenizer, NEVER_MERGE};
use super::tokenizer::{
    Tokenizer, TokenizerError, CL100K_BASE_PATTERN, DEEPSEEK_V3_PATTERNS, GPT2_PATTERN,
    LLAMA3_PATTERN, MISTRAL_V3_PATTERN, O200K_BASE_PATTERN,
};
use super::vocab::{load_spm_vocab, place_special_pieces};
use super::whisper::{whisper_special_tokens, WhisperVariant};

// Embed vocabulary files at compile time
pub const CL100K_BASE_VOCAB: &[u8] = include_bytes!("../../vocabs/cl100k_base.tiktoken");
pub const O200K_BASE_VOCAB: &[u8] = include_bytes!("../../vocabs/o200k_base.tiktoken");
pub const LLAMA3_VOCAB: &[u8] = include_bytes!("../../vocabs/llama3.tiktoken");
pub const DEEPSEEK_V3_VOCAB: &[u8] = include_bytes!("../../vocabs/deepseek_v3.tiktoken");

/// Mistral V1 SentencePiece vocabulary (32,000 pieces with their scores).
///
/// Extracted straight from `tokenizer.model` by `scripts/extract_spm_vocab.py`,
/// so pieces keep their SentencePiece spelling (`<0x41>`, `▁▁`) and every score
/// survives — including the `-1e9` "never merge" sentinel on the 15 whitespace
/// runs, which the `.tiktoken` form of this vocabulary silently inverted into a
/// *preferred* merge.
pub const MISTRAL_SPM_VOCAB: &[u8] = include_bytes!("../../vocabs/mistral.spm");

/// Mistral V2 SentencePiece vocabulary (32,768 pieces with their scores).
pub const MISTRAL_V2_SPM_VOCAB: &[u8] = include_bytes!("../../vocabs/mistral_v2.spm");

/// Mistral V3/Tekken vocabulary file (Tiktoken-based, ~131k tokens).
pub const MISTRAL_V3_VOCAB: &[u8] = include_bytes!("../../vocabs/mistral_v3_tekken.tiktoken");

/// Whisper base BPE vocabulary (GPT-2 byte-level, 50,257 tokens).
///
/// Shared by every multilingual variant (v1/v2/v3) — they differ only in the
/// programmatically-generated special tokens. The English-only checkpoints use
/// a different base BPE and are not bundled; load those via
/// [`crate::from_json_path`].
pub const WHISPER_VOCAB: &[u8] = include_bytes!("../../vocabs/whisper.tiktoken");

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
    from_vocab(resolve_vocab(name)?)
}

/// Resolve a vocabulary name to its enum, or the same "unknown name" error
/// every name-taking entry point in this module reports.
///
/// Centralized so [`from_pretrained`] and [`base_vocab_size_by_name`] cannot
/// drift apart on what counts as a valid name or how the error reads.
fn resolve_vocab(name: &str) -> Result<PretrainedVocab, TokenizerError> {
    PretrainedVocab::from_name(name).ok_or_else(|| {
        TokenizerError::UnknownPretrained(format!(
            "{}. Supported: {}",
            name,
            PretrainedVocab::supported_names().join(", ")
        ))
    })
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
    // What this vocabulary's own reference tokenizer renders as nothing — see
    // `special_decode_ids`, which states the measurement per vocabulary.
    let skipped = special_decode_ids(vocab, &special);
    // Mistral V1/V2 are SentencePiece: they take the SPM-BPE backend, which has
    // no pre-tokenizer regex, and so must return before `patterns` is consulted.
    match vocab {
        PretrainedVocab::MistralV1 => {
            return spm_from_vocab(MISTRAL_SPM_VOCAB, vocab, special, named, skipped)
        }
        PretrainedVocab::MistralV2 => {
            return spm_from_vocab(MISTRAL_V2_SPM_VOCAB, vocab, special, named, skipped)
        }
        _ => {}
    }
    // Every remaining vocabulary is byte-level BPE and states a pattern. A
    // `None` here would reach `from_bytes_*_chain` as an empty list and surface
    // as `EmptyPatternList` rather than silently encoding without a split.
    let pats = patterns(vocab).unwrap_or(&[]);

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
        // Handled above, before `patterns` is consulted.
        PretrainedVocab::MistralV1 | PretrainedVocab::MistralV2 => {
            return Err(TokenizerError::UnknownPretrained(
                "Mistral V1/V2 take the SPM backend and are routed earlier".to_owned(),
            ))
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
    // thing regardless of which loader produced the handle. The decode-skipped
    // ids are stated for the same reason: the two loaders must not disagree
    // about what a marker id decodes to either.
    Ok(AnyTokenizer::new(
        Backend::Bpe(
            tokenizer
                .with_added_token_matching(true)
                .with_special_decode_ids(skipped),
        ),
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
    skipped: FxHashSet<u32>,
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
    // The two boundary ids reach the backend through the constructor, which is
    // what makes decode drop them (`sp.decode([1, …, 2])` renders neither) as
    // well as what the leading-sentinel prefix rule keys off. `<unk>` needs no
    // stating — the constructor resolves it by name from the vocabulary itself.
    // The control and agent tokens in `special` are declared decode-skipped too,
    // through `special_decode_ids`. They were once deliberately left rendered —
    // on the reading that a caller decoding `[INST]` back to `"[INST]"` is the
    // round trip these vocabularies are bundled for — but that made the *same*
    // vocabulary decode differently depending on which loader produced the
    // handle, and disagreed with both references: `sentencepiece` 0.2.0 and
    // `tokenizers` 0.22.1 on `mistral-7b-v0.3` each decode `[3, …]` as plain
    // `'hello'`. A caller that wants the marker spelled has the vocabulary's
    // own name→id map (`AnyTokenizer::special_token_id`); a caller decoding
    // model output wants what the reference gives.
    let tokenizer = SpmTokenizer::new(pieces, scores, bos_token_id(vocab), Some(eos))?
        .with_prefix_scheme(spm_prefix_scheme(vocab))
        .with_added_tokens(&special)?
        .with_special_decode_ids(skipped);

    Ok(AnyTokenizer::new(
        Backend::Spm(tokenizer),
        SpecialPolicy::boundary(None, None, Some(eos), named),
    ))
}

/// Where a bundled SentencePiece vocabulary places its dummy prefix.
///
/// This is **not** a property of the file format — it is HuggingFace's `legacy`
/// flag, declared per checkpoint in `tokenizer_config.json`, and the two Mistral
/// generations disagree. Measured with
/// `AutoTokenizer.from_pretrained(..., use_fast=False).tokenize("<s>x")`:
///
/// | vocabulary | `legacy` | result | scheme |
/// |---|---|---|---|
/// | Mistral V1 (`mistral-7b-awq-int4`) | `true`  | `['<s>', '▁x']` | [`AfterEachSpecial`](SpmPrefixScheme::AfterEachSpecial) |
/// | Mistral V2 (`mistral-7b-v0.3`)     | `false` | `['<s>', 'x']`  | [`Once`](SpmPrefixScheme::Once) |
///
/// `legacy = true` reproduces the pre-fix `LlamaTokenizer`, which prefixed every
/// stretch following a special token — the same rule llama.cpp still implements
/// (`llama-vocab.cpp`'s `is_prev_special`). `legacy = false` is the corrected
/// behaviour: one prefix for the whole input, applied before the split.
///
/// So a new bundled `.spm` vocabulary must have its checkpoint's `legacy` flag
/// read off and mapped here — never assumed from the fact that it came from a
/// `tokenizer.model`.
fn spm_prefix_scheme(vocab: PretrainedVocab) -> SpmPrefixScheme {
    match vocab {
        PretrainedVocab::MistralV1 => SpmPrefixScheme::AfterEachSpecial,
        _ => SpmPrefixScheme::Once,
    }
}

/// The ordered pre-tokenizer expression sequence for a vocabulary, or `None`
/// when it has no regex pre-tokenizer at all.
///
/// The return type is a slice rather than a single pattern so a multi-pass
/// pre-tokenizer — llama.cpp's `regex_exprs`, which DeepSeek V3 needs — can be
/// expressed without reshaping the accessor again.
///
/// `None` is not "unknown", it is "this vocabulary does not pre-tokenize with a
/// regex". Mistral V1/V2 are the case: they run on [`SpmTokenizer`], which
/// segments by merging pieces and never applies a split pattern. This arm used
/// to answer `SENTENCEPIECE_PATTERN`, which was a plausible-looking lie — no
/// caller could tell that the pattern it was handed is never applied to that
/// vocabulary, and `from_vocab` computed it only to discard it.
pub fn patterns(vocab: PretrainedVocab) -> Option<&'static [&'static str]> {
    match vocab {
        PretrainedVocab::Cl100kBase => Some(&[CL100K_BASE_PATTERN]),
        PretrainedVocab::O200kBase => Some(&[O200K_BASE_PATTERN]),
        // Meta's own split; NOT the o200k pattern.
        PretrainedVocab::Llama3 => Some(&[LLAMA3_PATTERN]),
        // DeepSeek's own three-pass split; see DEEPSEEK_V3_PATTERNS.
        PretrainedVocab::DeepseekV3 => Some(DEEPSEEK_V3_PATTERNS),
        // SPM-BPE: merges pieces, no pre-tokenizer regex.
        PretrainedVocab::MistralV1 | PretrainedVocab::MistralV2 => None,
        // Tekken has its own pattern (no contractions, single-digit numbers).
        PretrainedVocab::MistralV3 => Some(&[MISTRAL_V3_PATTERN]),
        PretrainedVocab::WhisperV1 | PretrainedVocab::WhisperV2 | PretrainedVocab::WhisperV3 => {
            Some(&[GPT2_PATTERN])
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

/// Get the base vocabulary size for a bundled vocabulary — the size the
/// upstream reference implementation (`tiktoken`, `tokenizers`, or
/// `sentencepiece`, depending on the vocabulary) reports, *without* the 54
/// agent tokens splintr adds on top.
///
/// This is what a consumer needs when sizing a model's embedding or logit
/// layer, or when identifying which vocabulary a checkpoint uses from the
/// shape of its token-embedding tensor — both must match the checkpoint's own
/// vocabulary, not splintr's extended one. [`Tokenize::vocab_size`](crate::Tokenize::vocab_size) /
/// [`Tokenizer::vocab_size`](super::tokenizer::Tokenizer::vocab_size) report
/// the *extended* size (base + agent tokens); this reports the base alone.
/// Agent tokens are always appended **above** every id the base vocabulary
/// uses (see the per-vocabulary special-token tables below), so this is also
/// exactly the id at which splintr's additions start — every id below it is
/// untouched, unshifted, and identical to the reference tokenizer's.
pub fn base_vocab_size(vocab: PretrainedVocab) -> u32 {
    match vocab {
        PretrainedVocab::Cl100kBase => CL100K_BASE_BASE_VOCAB_SIZE,
        PretrainedVocab::O200kBase => O200K_BASE_BASE_VOCAB_SIZE,
        PretrainedVocab::Llama3 => LLAMA3_BASE_VOCAB_SIZE,
        PretrainedVocab::DeepseekV3 => DEEPSEEK_V3_BASE_VOCAB_SIZE,
        PretrainedVocab::MistralV1 => MISTRAL_V1_BASE_VOCAB_SIZE,
        PretrainedVocab::MistralV2 => MISTRAL_V2_BASE_VOCAB_SIZE,
        PretrainedVocab::MistralV3 => MISTRAL_V3_BASE_VOCAB_SIZE,
        // Whisper carries no agent tokens at all (see `pad_token_id`), so its
        // base size *is* the full generated vocabulary size — there is no
        // separate "extended" size to subtract from.
        PretrainedVocab::WhisperV1 => WhisperVariant::V1Multilingual.vocab_size() as u32,
        PretrainedVocab::WhisperV2 => WhisperVariant::V2Multilingual.vocab_size() as u32,
        PretrainedVocab::WhisperV3 => WhisperVariant::V3Multilingual.vocab_size() as u32,
    }
}

/// [`base_vocab_size`] by vocabulary name string, for callers (e.g. the
/// Python bindings) that only have the name `from_pretrained` accepts.
pub fn base_vocab_size_by_name(name: &str) -> Result<u32, TokenizerError> {
    resolve_vocab(name).map(base_vocab_size)
}

/// The ids `from_pretrained` declares decode-skipped, i.e. those a bundled
/// vocabulary renders as nothing under HuggingFace's default
/// `skip_special_tokens=True` — the semantics [`AnyTokenizer::decode`] implements
/// for every loader.
///
/// Not simply "everything in [`special_tokens`]": what counts as skipped is a
/// per-vocabulary fact, so each arm below states what its own reference tool was
/// **measured** to do, and a marker the reference renders is left rendered here
/// too. Without this the bundled loader and the `from_json` loader answered
/// differently for the *same* vocabulary — `from_pretrained("mistral_v2")`
/// decoded id 3 as `"[INST]"` where `from_json` on `mistral-7b-v0.3`'s
/// `tokenizer.json` decoded it as nothing.
fn special_decode_ids(vocab: PretrainedVocab, special: &FxHashMap<String, u32>) -> FxHashSet<u32> {
    let all = || special.values().copied().collect::<FxHashSet<u32>>();
    match vocab {
        // Reference: `tiktoken` 0.8.0, which has no `skip_special_tokens` mode
        // at all and renders every one of these — `enc.decode([100257])` is
        // `'<|endoftext|>'`, `o200k`'s `decode([199999])` likewise. Nothing is
        // declared skipped rather than overriding a measured reference; see
        // `pretrained_special_decode_ids_follow_the_reference` for the pin.
        PretrainedVocab::Cl100kBase | PretrainedVocab::O200kBase => FxHashSet::default(),

        // Reference: `tokenizers` 0.22.1 on `llama-3.2-1b/tokenizer.json`, whose
        // 256 added tokens are *all* `special: true` — `decode([128000, …])`
        // drops `<|begin_of_text|>`, `<|eot_id|>`, `<|step_id|>` and the rest.
        // splintr's own multimodal (128256+) and agent (128300+) tokens are the
        // same kind of marker one id block higher, with no reference of their
        // own, so they follow the rule the reference sets for the block below.
        PretrainedVocab::Llama3 => all(),

        // Reference: `sentencepiece` 0.2.0 and `tokenizers` 0.22.1, which agree
        // on `mistral-7b-v0.3`'s vocabulary: `sp.decode([3] + ids)` and
        // `decode([3] + ids)` are both `'hello'`, and all 771 of the json's
        // added tokens are `special: true`. V1 names only `<unk>`/`<s>`/`</s>`,
        // which the SPM backend already skips; both add agent tokens above the
        // file's last id, which have no reference and follow the same rule.
        PretrainedVocab::MistralV1 | PretrainedVocab::MistralV2 => all(),

        // NOT MEASURED — inferred from the family. Every other arm here states
        // what its own reference tool does; this one cannot, because no Tekken
        // reference is installed or on the shelf (`mistral_common` is absent,
        // and there is no `tekken.json` / Tekken-converted `tokenizer.json`
        // among the model repos). What would settle it is either of those: run
        // `mistral_common`'s `MistralTokenizer` (or `tokenizers` over a Tekken
        // `tokenizer.json`) over `[INST]`-marked ids and see whether the
        // default decode renders the markers. `scripts/verify_external_models.py`
        // has the target wired up and reports MISSING until one appears.
        //
        // Dropping is the inference, not the measurement: V3 is the same
        // vendor's successor to V1/V2 with the same `[INST]`/`[/INST]` chat
        // markers, both of those were measured to drop (`sentencepiece` 0.2.0
        // and `tokenizers` 0.22.1 agree), and every reference in this project
        // that has an opinion about chat markers drops them. Rendering them
        // would also make `from_pretrained("mistral_v3")` disagree with
        // `from_json` on a Tekken file, the same split the V1/V2 change closed.
        // `decode_with(ids, SpecialDecode::Render)` recovers them either way.
        PretrainedVocab::MistralV3 => all(),

        // Reference: `tokenizers` 0.22.1 on `deepseek-v3-tokenizer/tokenizer.json`,
        // which declares 14 of its added tokens `special: false` and therefore
        // *renders* them: `decode([128803] + ids)` is `'<｜User｜>hello'`. Those
        // ids stay rendered here. Everything else it declares `special: true`
        // and drops, including 128798/128799 (which splintr names
        // `<think>`/`</think>` and the reference names as placeholders) and
        // `<|EOT|>` (128805).
        PretrainedVocab::DeepseekV3 => {
            let rendered: FxHashSet<u32> = (128800..=128804).chain(128806..=128814).collect();
            all().difference(&rendered).copied().collect()
        }

        // Reference: `tokenizers` 0.22.1 on `whisper-tiny/tokenizer.json`. Its
        // 107 control tokens (`<|endoftext|>`, `<|startoftranscript|>`, the
        // language table, `<|translate|>`…`<|notimestamps|>`) are `special:
        // true` and dropped; the 1,501 timestamp tokens `<|0.00|>`…`<|30.00|>`
        // are `special: false` and rendered — they carry the transcript's
        // timings, which is content, so they are left rendered here too.
        PretrainedVocab::WhisperV1 | PretrainedVocab::WhisperV2 | PretrainedVocab::WhisperV3 => {
            let variant = match vocab {
                PretrainedVocab::WhisperV2 => WhisperVariant::V2Multilingual,
                PretrainedVocab::WhisperV3 => WhisperVariant::V3Multilingual,
                _ => WhisperVariant::V1Multilingual,
            };
            let first_timestamp = variant.first_timestamp_token_id();
            special
                .values()
                .copied()
                .filter(|&id| id < first_timestamp)
                .collect()
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

/// cl100k_base's reference (`tiktoken`) vocabulary size — one past
/// `<|endofprompt|>`, where splintr's agent tokens begin. Shared with
/// [`base_vocab_size`] so the two cannot drift apart.
const CL100K_BASE_BASE_VOCAB_SIZE: u32 = 100277;

/// o200k_base's reference (`tiktoken`) vocabulary size — one past
/// `<|endofprompt|>`, where splintr's agent tokens begin. Shared with
/// [`base_vocab_size`] so the two cannot drift apart.
const O200K_BASE_BASE_VOCAB_SIZE: u32 = 200019;

/// Get the standard special tokens for cl100k_base encoding (GPT-4, GPT-3.5-turbo).
pub fn cl100k_base_special_tokens() -> FxHashMap<String, u32> {
    let mut special = FxHashMap::default();
    // OpenAI standard special tokens (100257-100276)
    special.insert("<|endoftext|>".to_string(), 100257);
    special.insert("<|fim_prefix|>".to_string(), 100258);
    special.insert("<|fim_middle|>".to_string(), 100259);
    special.insert("<|fim_suffix|>".to_string(), 100260);
    special.insert(
        "<|endofprompt|>".to_string(),
        CL100K_BASE_BASE_VOCAB_SIZE - 1,
    );

    // Agent tokens (100277+)
    insert_agent_tokens(&mut special, CL100K_BASE_BASE_VOCAB_SIZE);

    special
}

/// Get the standard special tokens for o200k_base encoding (GPT-4o).
pub fn o200k_base_special_tokens() -> FxHashMap<String, u32> {
    let mut special = FxHashMap::default();
    // OpenAI standard special tokens (199999-200018)
    special.insert("<|endoftext|>".to_string(), 199999);
    special.insert(
        "<|endofprompt|>".to_string(),
        O200K_BASE_BASE_VOCAB_SIZE - 1,
    );

    // Agent tokens (200019+)
    insert_agent_tokens(&mut special, O200K_BASE_BASE_VOCAB_SIZE);

    special
}

/// Llama 3's reference (`tokenizers`) vocabulary size: 128,000 BPE tokens +
/// 256 reserved special-token slots (128000-128255) = 128,256. splintr only
/// names 11 of those 256 slots (the ones below), leaving the rest as an
/// unnamed gap in the base range; its own multimodal placeholders and agent
/// tokens are appended starting exactly here. Shared with [`base_vocab_size`]
/// so the two cannot drift apart.
const LLAMA3_BASE_VOCAB_SIZE: u32 = 128256;

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
    special.insert("<|image|>".to_string(), LLAMA3_BASE_VOCAB_SIZE);
    special.insert("<|/image|>".to_string(), LLAMA3_BASE_VOCAB_SIZE + 1);
    special.insert("<|audio|>".to_string(), LLAMA3_BASE_VOCAB_SIZE + 2);
    special.insert("<|/audio|>".to_string(), LLAMA3_BASE_VOCAB_SIZE + 3);
    special.insert("<|video|>".to_string(), LLAMA3_BASE_VOCAB_SIZE + 4);
    special.insert("<|/video|>".to_string(), LLAMA3_BASE_VOCAB_SIZE + 5);

    // Agent tokens (128300+)
    insert_agent_tokens_llama3(&mut special, 128300);

    special
}

/// DeepSeek V3's reference (`tokenizers`) vocabulary size: one past
/// `<｜tool▁sep｜>` (128814), the highest id DeepSeek's own tokenizer defines.
/// splintr's agent tokens start higher still, at 128900, leaving
/// 128815-128899 as an unused gap in the base range. Shared with
/// [`base_vocab_size`] so the two cannot drift apart.
const DEEPSEEK_V3_BASE_VOCAB_SIZE: u32 = 128815;

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
    special.insert(
        "<｜tool▁sep｜>".to_string(),
        DEEPSEEK_V3_BASE_VOCAB_SIZE - 1,
    );

    // Agent tokens (128900+)
    insert_agent_tokens(&mut special, 128900);

    special
}

/// Mistral V1's reference (`sentencepiece`) piece count: the bundled
/// `.spm` file has exactly 32,000 pieces (ids 0-31999, including the 0-2
/// native specials below), so splintr's agent tokens start immediately
/// after at 32,000. Shared with [`base_vocab_size`] so the two cannot drift
/// apart.
const MISTRAL_V1_BASE_VOCAB_SIZE: u32 = 32000;

/// Mistral V2's reference (`sentencepiece`) piece count: the bundled
/// `.spm` file has exactly 32,768 pieces (ids 0-32767, including the native
/// specials and V2 control tokens below), so splintr's agent tokens start
/// immediately after at 32,768. Shared with [`base_vocab_size`] so the two
/// cannot drift apart.
const MISTRAL_V2_BASE_VOCAB_SIZE: u32 = 32768;

/// Mistral V3/Tekken's reference (`tokenizers`) vocabulary size: the bundled
/// Tekken vocabulary file has exactly 131,072 tokens (ids 0-131071,
/// including the native specials and control tokens below), so splintr's
/// agent tokens start immediately after at 131,072. Shared with
/// [`base_vocab_size`] so the two cannot drift apart.
const MISTRAL_V3_BASE_VOCAB_SIZE: u32 = 131072;

/// Get the standard special tokens for Mistral V1 encoding.
pub fn mistral_v1_special_tokens() -> FxHashMap<String, u32> {
    let mut special = FxHashMap::default();

    // Mistral SentencePiece special tokens (0-2)
    special.insert("<unk>".to_string(), 0);
    special.insert("<s>".to_string(), 1);
    special.insert("</s>".to_string(), 2);

    // Agent tokens (32000+) - Mistral V1 vocab has ~32000 base tokens
    insert_agent_tokens(&mut special, MISTRAL_V1_BASE_VOCAB_SIZE);

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
    insert_agent_tokens(&mut special, MISTRAL_V2_BASE_VOCAB_SIZE);

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
    insert_agent_tokens(&mut special, MISTRAL_V3_BASE_VOCAB_SIZE);

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
    use crate::core::policy::SpecialDecode;
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

    /// What every bundled vocabulary decodes a marker id to, pinned against the
    /// reference tool that is authoritative for it — the whole point of
    /// [`special_decode_ids`], which is otherwise a table nothing checks.
    ///
    /// Each expectation below was measured, not assumed:
    ///
    /// * `mistral_v2` — `sentencepiece` 0.2.0 and `tokenizers` 0.22.1 on
    ///   `mistral-7b-v0.3`: `decode([3] + hello_ids)` is `'hello'` under both.
    /// * `llama3` — `tokenizers` 0.22.1 on `llama-3.2-1b`: every one of its 256
    ///   added tokens is `special: true`, and `decode([128000] + ids)` is
    ///   `'hello'`.
    /// * `deepseek_v3` — `tokenizers` 0.22.1 on `deepseek-v3-tokenizer`: id 0 is
    ///   `special: true` and dropped, while `<｜User｜>` (128803) is
    ///   `special: false` and `decode([128803] + ids)` is `'<｜User｜>hello'`.
    /// * `whisper` — `tokenizers` 0.22.1 on `whisper-tiny`: `<|startoftranscript|>`
    ///   is `special: true` and dropped; the timestamp tokens are
    ///   `special: false` and rendered.
    /// * `cl100k_base` / `o200k_base` — `tiktoken` 0.8.0, which has no
    ///   `skip_special_tokens` mode and renders: `enc.decode([100257])` is
    ///   `'<|endoftext|>'`. Left rendered rather than overriding the reference.
    #[test]
    fn pretrained_special_decode_ids_follow_the_reference() {
        // (vocab, id, decoded text)
        let dropped: [(&str, u32); 8] = [
            ("mistral_v2", 3),       // [INST]
            ("mistral_v2", 4),       // [/INST]
            ("llama3", 128000),      // <|begin_of_text|>
            ("llama3", 128009),      // <|eot_id|>
            ("deepseek_v3", 0),      // <｜begin▁of▁sentence｜>
            ("deepseek_v3", 128805), // <|EOT|>
            ("whisper", 50258),      // <|startoftranscript|>
            ("whisper", 50259),      // <|en|>
        ];
        for (name, id) in dropped {
            let tokenizer = from_pretrained(name).expect("bundled vocabulary loads");
            assert_eq!(
                tokenizer.decode(&[id]).expect("a skipped id decodes"),
                "",
                "{name}: id {id} must render as nothing, as its reference does"
            );
        }

        let rendered: [(&str, u32, &str); 4] = [
            // `special: false` in DeepSeek's own `tokenizer.json`.
            ("deepseek_v3", 128803, "<｜User｜>"),
            ("deepseek_v3", 128804, "<｜Assistant｜>"),
            // `tiktoken` renders these; splintr does not override it.
            ("cl100k_base", 100257, "<|endoftext|>"),
            ("o200k_base", 199999, "<|endoftext|>"),
        ];
        for (name, id, text) in rendered {
            let tokenizer = from_pretrained(name).expect("bundled vocabulary loads");
            assert_eq!(
                tokenizer.decode(&[id]).expect("a rendered id decodes"),
                text,
                "{name}: id {id} must still render, as its reference does"
            );
        }
    }

    /// `mistral_v3` drops its declared markers too — pinned separately from
    /// [`pretrained_special_decode_ids_follow_the_reference`] because it is the
    /// one arm of [`special_decode_ids`] that no reference on this machine can
    /// answer for.
    ///
    /// It is NOT a measurement. `mistral_common` is not installed and no Tekken
    /// `tokenizer.json` is on the shelf, so this pins the family-consistency
    /// inference the table states: same vendor as V1/V2, same `[INST]`/
    /// `[/INST]` markers, both measured to drop. If a Tekken reference ever
    /// says otherwise, this test is what should fail and be corrected — which
    /// is the reason it exists rather than the vocabulary simply going
    /// unpinned.
    #[test]
    fn mistral_v3_drops_its_markers_by_family_consistency_not_by_measurement() {
        let tokenizer = from_pretrained("mistral_v3").expect("bundled vocabulary loads");
        for (id, spelling) in [
            (3u32, "[INST]"),
            (4, "[/INST]"),
            (1, "<s>"),
            (131072, "<|system|>"),
        ] {
            assert_eq!(
                tokenizer.decode(&[id]).expect("a skipped id decodes"),
                "",
                "mistral_v3: id {id} ({spelling}) must render as nothing, as V1/V2's do"
            );
            assert_eq!(
                tokenizer
                    .decode_with(&[id], SpecialDecode::Render)
                    .expect("a skipped id renders on request"),
                spelling,
                "mistral_v3: id {id} must still be reachable through Render"
            );
        }
    }

    /// Every marker `decode` drops is still reachable, through the explicit
    /// [`SpecialDecode::Render`] mode.
    ///
    /// The point of the default is that a control token is an instruction to the
    /// model rather than text for a user to read — not that its spelling becomes
    /// unreachable. A vocabulary that could only drop them would be strictly
    /// less capable than the reference it is matching, since `tokenizers` offers
    /// `skip_special_tokens=False`.
    #[test]
    fn pretrained_markers_are_still_reachable_with_specials_rendered() {
        for (name, id, spelling) in [
            ("mistral_v2", 3u32, "[INST]"),
            ("mistral_v2", 4, "[/INST]"),
            ("llama3", 128000, "<|begin_of_text|>"),
            ("llama3", 128009, "<|eot_id|>"),
            ("deepseek_v3", 0, "<｜begin▁of▁sentence｜>"),
            ("deepseek_v3", 128805, "<|EOT|>"),
            ("whisper", 50258, "<|startoftranscript|>"),
        ] {
            let tokenizer = from_pretrained(name).expect("bundled vocabulary loads");
            assert_eq!(
                tokenizer.decode(&[id]).expect("a skipped id decodes"),
                "",
                "{name}: id {id} is dropped by default"
            );
            assert_eq!(
                tokenizer
                    .decode_with(&[id], SpecialDecode::Render)
                    .expect("a rendered id decodes"),
                spelling,
                "{name}: id {id} must be reachable with specials rendered"
            );
        }
    }

    /// A marker rendered out of the middle of real text, on both backend
    /// shapes: the explicit mode restores the marker and changes nothing else.
    ///
    /// `[7080, 29477, 2294]` is `sp.encode("hello world")` on
    /// `mistral-7b-v0.3`'s `tokenizer.model`, and `tokenizers` 0.22.1 decodes
    /// `[3, 7080, 29477, 2294, 4]` as `'hello world'` by default and as
    /// `'[INST] hello world[/INST]'` under `skip_special_tokens=False`.
    ///
    /// That space is the point of the "and nothing else": the dummy-prefix strip
    /// is spent by whichever token renders *first*, so with `[INST]` dropped it
    /// eats the space `▁hell` carries, and with `[INST]` rendered it does not.
    /// The reference behaves the same way, and matching it is what says the mode
    /// changes the skip set and nothing about the post-ops.
    #[test]
    fn rendering_specials_restores_the_marker_and_nothing_else() {
        let spm = from_pretrained("mistral_v2").expect("bundled vocabulary loads");
        let ids = [3, 7080, 29477, 2294, 4];
        assert_eq!(spm.decode(&ids).expect("decodes"), "hello world");
        assert_eq!(
            spm.decode_with(&ids, SpecialDecode::Render)
                .expect("decodes"),
            "[INST] hello world[/INST]"
        );

        // The byte-level BPE shape, where the surfaces live in a separate
        // special-token table rather than in the piece vector.
        let bpe = from_pretrained("llama3").expect("bundled vocabulary loads");
        let mut ids = vec![128000];
        ids.extend(bpe.encode("hello"));
        ids.push(128009);
        assert_eq!(bpe.decode(&ids).expect("decodes"), "hello");
        assert_eq!(
            bpe.decode_with(&ids, SpecialDecode::Render)
                .expect("decodes"),
            "<|begin_of_text|>hello<|eot_id|>"
        );
    }

    /// Whisper's timestamp block is `special: false` in `whisper-tiny`'s
    /// `tokenizer.json` and `tokenizers` 0.22.1 renders it, so the skip set must
    /// stop exactly at the first timestamp id — a blanket "skip everything in
    /// `special_tokens`" would silently swallow every transcript timing.
    #[test]
    fn whisper_timestamp_tokens_are_not_decode_skipped() {
        for (name, variant) in [
            ("whisper_v1", WhisperVariant::V1Multilingual),
            ("whisper_v2", WhisperVariant::V2Multilingual),
            ("whisper_v3", WhisperVariant::V3Multilingual),
        ] {
            let tokenizer = from_pretrained(name).expect("bundled vocabulary loads");
            let first = variant.first_timestamp_token_id();
            assert_eq!(
                tokenizer.decode(&[first]).expect("a timestamp id decodes"),
                "<|0.00|>",
                "{name}: the first timestamp token must still render"
            );
            assert_eq!(
                tokenizer
                    .decode(&[first + 1500])
                    .expect("a timestamp id decodes"),
                "<|30.00|>",
                "{name}: the last timestamp token must still render"
            );
            assert_eq!(
                tokenizer
                    .decode(&[variant.notimestamps_token_id()])
                    .expect("a control id decodes"),
                "",
                "{name}: `<|notimestamps|>` is a control token and is dropped"
            );
        }
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

    /// Every bundled vocabulary's base size, pinned against the reference
    /// implementation that defines it (`tiktoken`, `tokenizers`, or
    /// `sentencepiece` — noted per case).
    #[test]
    fn test_base_vocab_size_matches_reference() {
        assert_eq!(base_vocab_size(PretrainedVocab::Cl100kBase), 100277); // tiktoken
        assert_eq!(base_vocab_size(PretrainedVocab::O200kBase), 200019); // tiktoken
        assert_eq!(base_vocab_size(PretrainedVocab::Llama3), 128256); // tokenizers
        assert_eq!(base_vocab_size(PretrainedVocab::DeepseekV3), 128815); // tokenizers
        assert_eq!(base_vocab_size(PretrainedVocab::MistralV1), 32000); // sentencepiece piece count
        assert_eq!(base_vocab_size(PretrainedVocab::MistralV2), 32768); // sentencepiece piece count
        assert_eq!(base_vocab_size(PretrainedVocab::MistralV3), 131072); // tokenizers (Tekken)
        assert_eq!(
            base_vocab_size(PretrainedVocab::WhisperV1),
            WhisperVariant::V1Multilingual.vocab_size() as u32
        );
        assert_eq!(
            base_vocab_size(PretrainedVocab::WhisperV2),
            WhisperVariant::V2Multilingual.vocab_size() as u32
        );
        assert_eq!(
            base_vocab_size(PretrainedVocab::WhisperV3),
            WhisperVariant::V3Multilingual.vocab_size() as u32
        );
    }

    /// The invariant the whole design rests on: a base size can never exceed
    /// the tokenizer's extended `vocab_size`, because agent tokens are purely
    /// additive on top of it.
    #[test]
    fn test_base_vocab_size_never_exceeds_extended_vocab_size() {
        for (name, vocab) in [
            ("cl100k_base", PretrainedVocab::Cl100kBase),
            ("o200k_base", PretrainedVocab::O200kBase),
            ("llama3", PretrainedVocab::Llama3),
            ("deepseek_v3", PretrainedVocab::DeepseekV3),
            ("mistral_v1", PretrainedVocab::MistralV1),
            ("mistral_v2", PretrainedVocab::MistralV2),
            ("mistral_v3", PretrainedVocab::MistralV3),
            ("whisper_v1", PretrainedVocab::WhisperV1),
            ("whisper_v2", PretrainedVocab::WhisperV2),
            ("whisper_v3", PretrainedVocab::WhisperV3),
        ] {
            let extended = from_vocab(vocab).unwrap().vocab_size() as u32;
            let base = base_vocab_size(vocab);
            assert!(
                base <= extended,
                "{name}: base_vocab_size {base} exceeds extended vocab_size {extended}"
            );
        }
    }

    /// The property the whole design rests on: no agent-token id is ever
    /// below `base_vocab_size`. Splintr's agent tokens are safe to bundle by
    /// default *because* they occupy ids strictly above every id the
    /// reference vocabulary uses — this pins that directly against the
    /// actual special-token maps rather than trusting the arithmetic that
    /// builds them.
    #[test]
    fn test_no_agent_token_id_below_base_vocab_size() {
        for (name, vocab) in [
            ("cl100k_base", PretrainedVocab::Cl100kBase),
            ("o200k_base", PretrainedVocab::O200kBase),
            ("llama3", PretrainedVocab::Llama3),
            ("deepseek_v3", PretrainedVocab::DeepseekV3),
            ("mistral_v1", PretrainedVocab::MistralV1),
            ("mistral_v2", PretrainedVocab::MistralV2),
            ("mistral_v3", PretrainedVocab::MistralV3),
        ] {
            let base = base_vocab_size(vocab);
            for name_and_id in agent_token_ids_in(vocab) {
                let (token, id) = name_and_id;
                assert!(
                    id >= base,
                    "{name}: agent token {token:?} has id {id}, below base_vocab_size {base}"
                );
            }
        }
    }

    /// The 54 standard agent-token names, so
    /// `test_no_agent_token_id_below_base_vocab_size` can pick just those out
    /// of a vocabulary's full special-token map (which also holds the
    /// vocabulary's own native specials, at legitimately low ids).
    const AGENT_TOKEN_NAMES: [&str; 54] = [
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|im_start|>",
        "<|im_end|>",
        "<|think|>",
        "<|/think|>",
        "<|plan|>",
        "<|/plan|>",
        "<|step|>",
        "<|/step|>",
        "<|act|>",
        "<|/act|>",
        "<|observe|>",
        "<|/observe|>",
        "<|function|>",
        "<|/function|>",
        "<|result|>",
        "<|/result|>",
        "<|error|>",
        "<|/error|>",
        "<|code|>",
        "<|/code|>",
        "<|output|>",
        "<|/output|>",
        "<|lang|>",
        "<|/lang|>",
        "<|context|>",
        "<|/context|>",
        "<|quote|>",
        "<|/quote|>",
        "<|cite|>",
        "<|/cite|>",
        "<|source|>",
        "<|/source|>",
        "<|memory|>",
        "<|/memory|>",
        "<|recall|>",
        "<|/recall|>",
        "<|pad|>",
        "<|stop|>",
        "<|sep|>",
        "<|image|>",
        "<|/image|>",
        "<|audio|>",
        "<|/audio|>",
        "<|video|>",
        "<|/video|>",
        "<|title|>",
        "<|/title|>",
        "<|section|>",
        "<|/section|>",
        "<|summary|>",
        "<|/summary|>",
    ];

    /// The subset of a vocabulary's special-token map that are agent tokens
    /// (by name), each paired with its id.
    fn agent_token_ids_in(vocab: PretrainedVocab) -> Vec<(String, u32)> {
        let all = special_tokens(vocab);
        AGENT_TOKEN_NAMES
            .iter()
            .filter_map(|name| all.get(*name).map(|&id| (name.to_string(), id)))
            .collect()
    }
}
