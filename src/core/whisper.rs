//! Whisper tokenizer support.
//!
//! Whisper uses GPT-2 byte-level BPE (~50,257 tokens) plus a large set of
//! special tokens for transcription control:
//!
//! - `<|endoftext|>`, `<|startoftranscript|>`
//! - 99 language tokens (v1/v2) or 100 (v3 adds `<|yue|>`)
//! - `<|translate|>`, `<|transcribe|>`
//! - `<|startoflm|>`, `<|startofprev|>`
//! - `<|nospeech|>` (v2/v3) or `<|nocaptions|>` (v1)
//! - `<|notimestamps|>`
//! - 1501 timestamp tokens `<|0.00|>`..`<|30.00|>` in 0.02s increments
//!
//! The base BPE vocabulary is loaded from an HF-style `tokenizer.json` because
//! the GPT-2 BPE is too large to pleasantly embed and it's already the standard
//! Whisper checkpoint artefact on HuggingFace.

use rustc_hash::FxHashMap;
use thiserror::Error;

use super::byte_level::byte_level_decode;
use super::tokenizer::{Tokenizer, TokenizerError, GPT2_PATTERN};

/// Which Whisper vocabulary generation to emit special tokens for.
///
/// Variants differ only in the language-token set and the `<|nospeech|>` /
/// `<|nocaptions|>` naming. The underlying base BPE is the same GPT-2 vocab.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WhisperVariant {
    /// `whisper-{tiny,base,small,medium,large}` (multilingual, v1).
    /// Uses `<|nocaptions|>`. 99 languages. Vocab size 51865.
    V1Multilingual,
    /// `whisper-large-v2` (multilingual). Uses `<|nospeech|>`. 99 languages.
    /// Vocab size 51865.
    V2Multilingual,
    /// `whisper-large-v3` / `distil-whisper-large-v3`.
    /// Adds Cantonese (`<|yue|>`), 100 languages. Vocab size 51866.
    V3Multilingual,
    /// `whisper-{tiny,base,small,medium,large}.en`. No language tokens.
    /// Vocab size 51864.
    EnglishOnly,
}

impl WhisperVariant {
    /// Parse a variant name. Accepts common aliases.
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "whisper_v1" | "whisper-v1" | "whisper-multilingual-v1" => Some(Self::V1Multilingual),
            "whisper_v2" | "whisper-v2" | "whisper" | "whisper-multilingual" => {
                Some(Self::V2Multilingual)
            }
            "whisper_v3" | "whisper-v3" | "whisper-large-v3" => Some(Self::V3Multilingual),
            "whisper_en" | "whisper.en" | "whisper-en" => Some(Self::EnglishOnly),
            _ => None,
        }
    }

    /// Full vocabulary size (base BPE + specials + languages + timestamps).
    ///
    /// Equal to the id of the last timestamp token (`<|30.00|>`) plus one.
    pub fn vocab_size(self) -> usize {
        (self.first_timestamp_token_id() + 1500 + 1) as usize
    }

    /// Start-of-transcript token id (`<|startoftranscript|>`).
    ///
    /// This is the anchor every other special-token id is derived from. The
    /// special-token block always begins with `<|endoftext|>` immediately
    /// followed by `<|startoftranscript|>`, so `eos == sot - 1`. Multilingual
    /// checkpoints use a 50,257-entry base BPE (sot = 50258); the English-only
    /// (`gpt2`) checkpoints use a 50,256-entry base BPE (sot = 50257).
    pub fn sot_token_id(self) -> u32 {
        match self {
            Self::EnglishOnly => 50257,
            _ => 50258,
        }
    }

    /// EOS token id (`<|endoftext|>`) — the entry just before `<|startoftranscript|>`.
    pub fn eos_token_id(self) -> u32 {
        self.sot_token_id() - 1
    }

    /// Token id for `<|translate|>`.
    ///
    /// Sits right after the language block. Present on every variant — the
    /// English-only checkpoints still carry it even though they only transcribe.
    pub fn translate_token_id(self) -> u32 {
        self.sot_token_id() + 1 + self.languages().len() as u32
    }

    /// Token id for `<|transcribe|>`.
    pub fn transcribe_token_id(self) -> u32 {
        self.translate_token_id() + 1
    }

    /// `<|notimestamps|>` token id.
    ///
    /// Layout after `<|transcribe|>`: `<|startoflm|>`, `<|startofprev|>`,
    /// the no-speech token, then `<|notimestamps|>` (4 ids later).
    pub fn notimestamps_token_id(self) -> u32 {
        self.transcribe_token_id() + 4
    }

    /// First timestamp token id (`<|0.00|>`).
    pub fn first_timestamp_token_id(self) -> u32 {
        self.notimestamps_token_id() + 1
    }

    /// The no-speech control token spelling for this variant.
    ///
    /// The original v1 release and the English-only checkpoints spell it
    /// `<|nocaptions|>`; v2/v3 renamed it to `<|nospeech|>`.
    pub fn no_speech_token(self) -> &'static str {
        match self {
            Self::V1Multilingual | Self::EnglishOnly => "<|nocaptions|>",
            Self::V2Multilingual | Self::V3Multilingual => "<|nospeech|>",
        }
    }

    /// Language list in id order. The first language token id is `sot + 1`.
    ///
    /// English-only checkpoints carry the same 99-language table as v1/v2 — the
    /// tokens exist in the vocabulary even though the model only transcribes English.
    pub fn languages(self) -> &'static [&'static str] {
        match self {
            Self::V3Multilingual => WHISPER_LANGUAGES_V3,
            Self::V1Multilingual | Self::V2Multilingual | Self::EnglishOnly => {
                WHISPER_LANGUAGES_V1V2
            }
        }
    }

    /// Get the token id for a language code (e.g. `"en"`, `"yue"`).
    ///
    /// Accepts the deprecated Hebrew code `"iw"` as an alias for `"he"`.
    pub fn language_token_id(self, code: &str) -> Option<u32> {
        let code = if code == "iw" { "he" } else { code };
        let pos = self.languages().iter().position(|&l| l == code)?;
        Some(self.sot_token_id() + 1 + pos as u32)
    }
}

/// Generate the full Whisper special-token map for a variant.
///
/// Includes `<|endoftext|>`, `<|startoftranscript|>`, all language tokens,
/// `<|translate|>`/`<|transcribe|>`, control tokens, `<|notimestamps|>`, and
/// all 1501 timestamp tokens.
pub fn whisper_special_tokens(variant: WhisperVariant) -> FxHashMap<String, u32> {
    let mut m = FxHashMap::default();
    m.insert("<|endoftext|>".to_string(), variant.eos_token_id());
    m.insert("<|startoftranscript|>".to_string(), variant.sot_token_id());

    // Language tokens start at `sot + 1` in id order (all variants, including
    // English-only, carry the language table).
    let lang_base = variant.sot_token_id() + 1;
    for (i, &lang) in variant.languages().iter().enumerate() {
        m.insert(format!("<|{lang}|>"), lang_base + i as u32);
        // Hebrew alias: the canonical OpenAI code is `he`, but some checkpoints
        // (e.g. `whisper-tiny.en`) label it with the deprecated ISO 639-1 code
        // `iw`. Register both spellings at the same id for compatibility.
        if lang == "he" {
            m.insert("<|iw|>".to_string(), lang_base + i as u32);
        }
    }

    // Control tokens after the language block.
    m.insert("<|translate|>".to_string(), variant.translate_token_id());
    m.insert("<|transcribe|>".to_string(), variant.transcribe_token_id());

    let notimestamps = variant.notimestamps_token_id();
    m.insert("<|startoflm|>".to_string(), notimestamps - 3);
    m.insert("<|startofprev|>".to_string(), notimestamps - 2);
    m.insert(variant.no_speech_token().to_string(), notimestamps - 1);
    m.insert("<|notimestamps|>".to_string(), notimestamps);

    // Timestamp tokens: <|0.00|>..<|30.00|> at 0.02s granularity = 1501 tokens.
    let first_ts = variant.first_timestamp_token_id();
    for k in 0..1501u32 {
        let seconds = k as f32 * 0.02;
        let label = format!("<|{seconds:.2}|>");
        m.insert(label, first_ts + k);
    }

    m
}

/// Errors returned by [`from_tokenizer_json_path`] and [`from_tokenizer_json_bytes`].
#[derive(Debug, Error)]
pub enum WhisperTokenizerError {
    #[error("failed to parse tokenizer.json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("tokenizer.json missing field: {0}")]
    MissingField(&'static str),
    #[error("tokenizer.json vocab entry `{0}` is not byte-level decodable")]
    InvalidByteLevel(String),
    #[error(transparent)]
    Tokenizer(#[from] TokenizerError),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Build a Whisper [`Tokenizer`] from an HF-style `tokenizer.json` file.
///
/// Loads `model.vocab` (byte-level-encoded keys → ids), decodes each key back
/// to raw bytes, then installs the programmatically-generated Whisper special
/// token set for the chosen variant.
pub fn from_tokenizer_json_path<P: AsRef<std::path::Path>>(
    path: P,
    variant: WhisperVariant,
) -> Result<Tokenizer, WhisperTokenizerError> {
    let bytes = std::fs::read(path)?;
    from_tokenizer_json_bytes(&bytes, variant)
}

/// Build a Whisper [`Tokenizer`] from tokenizer.json bytes.
pub fn from_tokenizer_json_bytes(
    data: &[u8],
    variant: WhisperVariant,
) -> Result<Tokenizer, WhisperTokenizerError> {
    let root: serde_json::Value = serde_json::from_slice(data)?;

    let vocab = root
        .get("model")
        .and_then(|m| m.get("vocab"))
        .and_then(|v| v.as_object())
        .ok_or(WhisperTokenizerError::MissingField("model.vocab"))?;

    let mut encoder: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
    encoder.reserve(vocab.len());

    for (bl_token, id_val) in vocab {
        let id = id_val
            .as_u64()
            .ok_or(WhisperTokenizerError::MissingField("model.vocab[*] = u32"))?;
        // The vocab keys are byte-level-encoded strings (e.g. "Ġa"). The encoder
        // keys must stay in that encoded form: `encode` byte-level-encodes input
        // before lookup, and `decode` reverses it. We only validate that the key
        // is a well-formed byte-level string here, then store its raw UTF-8 bytes.
        if byte_level_decode(bl_token).is_none() {
            return Err(WhisperTokenizerError::InvalidByteLevel(bl_token.clone()));
        }
        encoder.insert(bl_token.as_bytes().to_vec(), id as u32);
    }

    // Whisper's special tokens are authoritative and deterministic per variant —
    // we generate them rather than trust `added_tokens` (which sometimes omits
    // timestamp tokens depending on the checkpoint).
    let specials = whisper_special_tokens(variant);

    Ok(Tokenizer::new_byte_level(encoder, specials, GPT2_PATTERN)?)
}

// ── language tables ─────────────────────────────────────────────────────────
//
// In id order. First entry = token id 50259 (`<|en|>`).

/// 99 languages for Whisper v1 / v2 multilingual checkpoints.
#[rustfmt::skip]
pub const WHISPER_LANGUAGES_V1V2: &[&str] = &[
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr",
    "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
    "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
    "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
    "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk",
    "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
    "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
    "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
    "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su",
];

/// 100 languages for Whisper v3 (adds `yue` / Cantonese at position 99).
#[rustfmt::skip]
pub const WHISPER_LANGUAGES_V3: &[&str] = &[
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr",
    "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
    "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
    "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
    "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk",
    "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
    "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
    "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
    "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su", "yue",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v1_v2_has_99_languages() {
        assert_eq!(WHISPER_LANGUAGES_V1V2.len(), 99);
        assert_eq!(WhisperVariant::V2Multilingual.languages().len(), 99);
    }

    #[test]
    fn v3_has_100_languages() {
        assert_eq!(WHISPER_LANGUAGES_V3.len(), 100);
        assert_eq!(*WHISPER_LANGUAGES_V3.last().unwrap(), "yue");
    }

    #[test]
    fn multilingual_eos_sot() {
        // Multilingual checkpoints use a 50,257-entry base BPE.
        for v in [
            WhisperVariant::V1Multilingual,
            WhisperVariant::V2Multilingual,
            WhisperVariant::V3Multilingual,
        ] {
            assert_eq!(v.eos_token_id(), 50257);
            assert_eq!(v.sot_token_id(), 50258);
        }
    }

    #[test]
    fn english_only_eos_sot() {
        // English-only (`gpt2`) checkpoints use a 50,256-entry base BPE, so the
        // special-token block starts one id earlier than the multilingual one.
        assert_eq!(WhisperVariant::EnglishOnly.eos_token_id(), 50256);
        assert_eq!(WhisperVariant::EnglishOnly.sot_token_id(), 50257);
    }

    /// Ground-truth ids taken from `openai/whisper-tiny.en` (added_tokens.json).
    #[test]
    fn english_only_special_token_layout() {
        let v = WhisperVariant::EnglishOnly;
        let sp = whisper_special_tokens(v);
        // English-only carries the full 99-language table.
        assert_eq!(v.languages().len(), 99);
        assert_eq!(sp.get("<|startoftranscript|>"), Some(&50257));
        assert_eq!(sp.get("<|en|>"), Some(&50258));
        assert_eq!(sp.get("<|translate|>"), Some(&50357));
        assert_eq!(sp.get("<|transcribe|>"), Some(&50358));
        assert_eq!(sp.get("<|startoflm|>"), Some(&50359));
        assert_eq!(sp.get("<|startofprev|>"), Some(&50360));
        // English-only spells the no-speech token `<|nocaptions|>`.
        assert_eq!(sp.get("<|nocaptions|>"), Some(&50361));
        assert_eq!(sp.get("<|nospeech|>"), None);
        assert_eq!(sp.get("<|notimestamps|>"), Some(&50362));
        assert_eq!(sp.get("<|0.00|>"), Some(&50363));
        assert_eq!(sp.get("<|30.00|>"), Some(&51863));
    }

    #[test]
    fn vocab_size_matches_last_timestamp() {
        // vocab_size == last timestamp id + 1, for every variant.
        for v in [
            WhisperVariant::V1Multilingual,
            WhisperVariant::V2Multilingual,
            WhisperVariant::V3Multilingual,
            WhisperVariant::EnglishOnly,
        ] {
            let sp = whisper_special_tokens(v);
            let max_id = sp.values().copied().max().unwrap();
            assert_eq!(v.vocab_size(), (max_id + 1) as usize);
        }
        assert_eq!(WhisperVariant::V2Multilingual.vocab_size(), 51865);
        assert_eq!(WhisperVariant::V3Multilingual.vocab_size(), 51866);
        assert_eq!(WhisperVariant::EnglishOnly.vocab_size(), 51864);
    }

    #[test]
    fn language_token_id_lookup() {
        assert_eq!(
            WhisperVariant::V2Multilingual.language_token_id("en"),
            Some(50259)
        );
        assert_eq!(
            WhisperVariant::V3Multilingual.language_token_id("yue"),
            Some(50259 + 99)
        );
        assert_eq!(
            WhisperVariant::V2Multilingual.language_token_id("yue"),
            None
        );
    }

    #[test]
    fn special_tokens_v2_coverage() {
        let sp = whisper_special_tokens(WhisperVariant::V2Multilingual);
        assert_eq!(sp.get("<|endoftext|>"), Some(&50257));
        assert_eq!(sp.get("<|startoftranscript|>"), Some(&50258));
        assert_eq!(sp.get("<|en|>"), Some(&50259));
        assert_eq!(sp.get("<|translate|>"), Some(&50358));
        assert_eq!(sp.get("<|transcribe|>"), Some(&50359));
        assert_eq!(sp.get("<|nospeech|>"), Some(&50362));
        assert_eq!(sp.get("<|notimestamps|>"), Some(&50363));
        assert_eq!(sp.get("<|0.00|>"), Some(&50364));
        // Last timestamp token is <|30.00|> = 50364 + 1500 = 51864
        assert_eq!(sp.get("<|30.00|>"), Some(&51864));
    }

    #[test]
    fn special_tokens_v3_offsets() {
        let sp = whisper_special_tokens(WhisperVariant::V3Multilingual);
        assert_eq!(sp.get("<|yue|>"), Some(&(50259 + 99)));
        assert_eq!(sp.get("<|translate|>"), Some(&50359));
        assert_eq!(sp.get("<|notimestamps|>"), Some(&50364));
        assert_eq!(sp.get("<|0.00|>"), Some(&50365));
    }

    #[test]
    fn special_token_count() {
        let sp = whisper_special_tokens(WhisperVariant::V2Multilingual);
        // 2 (eos, sot) + 99 langs + translate + transcribe + startoflm + startofprev
        // + nospeech + notimestamps + 1501 timestamps = 1608, plus the `<|iw|>`
        // Hebrew alias = 1609.
        assert_eq!(sp.len(), 2 + 99 + 4 + 1 + 1 + 1501 + 1);
    }

    #[test]
    fn hebrew_iw_alias_matches_he() {
        for v in [
            WhisperVariant::V2Multilingual,
            WhisperVariant::V3Multilingual,
            WhisperVariant::EnglishOnly,
        ] {
            let sp = whisper_special_tokens(v);
            let he = sp.get("<|he|>").copied();
            assert!(he.is_some());
            assert_eq!(sp.get("<|iw|>").copied(), he);
        }
    }

    #[test]
    fn parses_minimal_tokenizer_json() {
        // A tiny synthetic BPE: one token "a" at id 0, " a" (Ġa) at id 1.
        let json = r#"{
            "model": {
                "type": "BPE",
                "vocab": {
                    "a": 0,
                    "Ġa": 1
                },
                "merges": []
            }
        }"#;
        let _tok = from_tokenizer_json_bytes(json.as_bytes(), WhisperVariant::V2Multilingual)
            .expect("parse ok");
        // Successful construction with the byte-level encoder is the contract;
        // special-token rendering is covered by `special_tokens_v2_coverage`.
    }

    #[test]
    fn rejects_non_byte_level_keys() {
        let json = r#"{"model": {"vocab": {"你好": 0}, "merges": []}}"#;
        let err = from_tokenizer_json_bytes(json.as_bytes(), WhisperVariant::V2Multilingual);
        assert!(matches!(
            err,
            Err(WhisperTokenizerError::InvalidByteLevel(_))
        ));
    }

    #[test]
    fn from_name_parses_aliases() {
        assert_eq!(
            WhisperVariant::from_name("whisper-v1"),
            Some(WhisperVariant::V1Multilingual)
        );
        assert_eq!(
            WhisperVariant::from_name("whisper_v1"),
            Some(WhisperVariant::V1Multilingual)
        );
        // Bare "whisper" and case-insensitivity both resolve to V2.
        assert_eq!(
            WhisperVariant::from_name("Whisper"),
            Some(WhisperVariant::V2Multilingual)
        );
        assert_eq!(
            WhisperVariant::from_name("whisper-v2"),
            Some(WhisperVariant::V2Multilingual)
        );
        assert_eq!(
            WhisperVariant::from_name("whisper-large-v3"),
            Some(WhisperVariant::V3Multilingual)
        );
        assert_eq!(
            WhisperVariant::from_name("whisper.en"),
            Some(WhisperVariant::EnglishOnly)
        );
        assert_eq!(WhisperVariant::from_name("not-a-model"), None);
    }

    #[test]
    fn parses_tokenizer_json_from_path() {
        let json = r#"{
            "model": {
                "type": "BPE",
                "vocab": {
                    "a": 0,
                    "Ġa": 1
                },
                "merges": []
            }
        }"#;
        let dir = std::env::temp_dir();
        let path = dir.join("splintr_whisper_test_tokenizer.json");
        std::fs::write(&path, json).expect("write temp tokenizer.json");

        let result = from_tokenizer_json_path(&path, WhisperVariant::V2Multilingual);

        // Clean up before asserting so a failure doesn't leak the temp file.
        let _ = std::fs::remove_file(&path);
        result.expect("parse from path ok");
    }
}
