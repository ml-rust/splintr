//! Parsers for the shared sections of a HuggingFace `tokenizer.json`:
//! `pre_tokenizer`, `normalizer`, and `added_tokens`. These are backend-agnostic
//! — the family-specific builders in [`super`] consume their output.

use rustc_hash::FxHashMap;
use serde_json::Value;

use super::super::normalizer::NormOp;
use super::super::precompiled::Precompiled;
use super::super::tokenizer::{GPT2_PATTERN, SENTENCEPIECE_PATTERN};
use super::HfJsonError;

/// How input is split before the model runs, distilled to what splintr needs.
#[derive(Debug, Clone)]
pub(super) struct PreTokenization {
    /// Whether tokens are byte-level encoded (GPT-2/Whisper/Llama3 style).
    pub byte_level: bool,
    /// The pre-tokenization split regex.
    pub pattern: String,
    /// Prepend a space to the input (ByteLevel/Metaspace `add_prefix_space`, or
    /// Metaspace `prepend_scheme` != "never").
    pub add_prefix_space: bool,
}

/// Walk a `pre_tokenizer` value (possibly a `Sequence`) and distill it to a
/// byte-level flag plus a split regex.
///
/// - `ByteLevel` anywhere ⇒ byte-level encoding; default to [`GPT2_PATTERN`]
///   unless an explicit `Split` regex is present.
/// - `Split { pattern: Regex|String }` ⇒ use that regex.
/// - `Metaspace` (SentencePiece-style) ⇒ [`SENTENCEPIECE_PATTERN`].
/// - Anything else / absent ⇒ non-byte-level, [`GPT2_PATTERN`] fallback.
pub(super) fn parse_pre_tokenizer(pre: Option<&Value>) -> Result<PreTokenization, HfJsonError> {
    let mut byte_level = false;
    let mut split_regex: Option<String> = None;
    let mut metaspace = false;
    // None until a ByteLevel/Metaspace node sets it; defaulted at the end.
    let mut add_prefix_space: Option<bool> = None;
    // Pre-tokenizer types we neither parse nor handle implicitly via a backend.
    let mut unknown: Vec<String> = Vec::new();

    fn walk(
        v: &Value,
        byte_level: &mut bool,
        split_regex: &mut Option<String>,
        metaspace: &mut bool,
        add_prefix_space: &mut Option<bool>,
        unknown: &mut Vec<String>,
    ) {
        match v.get("type").and_then(Value::as_str) {
            Some("ByteLevel") => {
                *byte_level = true;
                if let Some(b) = v.get("add_prefix_space").and_then(Value::as_bool) {
                    *add_prefix_space = Some(b);
                }
            }
            Some("Metaspace") => {
                *metaspace = true;
                // Newer configs use `prepend_scheme` ("always"/"first"/"never");
                // older ones use `add_prefix_space`.
                if let Some(scheme) = v.get("prepend_scheme").and_then(Value::as_str) {
                    *add_prefix_space = Some(scheme != "never");
                } else if let Some(b) = v.get("add_prefix_space").and_then(Value::as_bool) {
                    *add_prefix_space = Some(b);
                }
            }
            Some("Split") if split_regex.is_none() => {
                // pattern is {"Regex": "..."} or {"String": "..."}.
                if let Some(re) = v.get("pattern").and_then(|p| {
                    p.get("Regex")
                        .and_then(Value::as_str)
                        .or_else(|| p.get("String").and_then(Value::as_str))
                }) {
                    *split_regex = Some(re.to_string());
                }
            }
            // Whitespace-only splitters are subsumed by both our SentencePiece
            // (whitespace-split) and byte-level paths, so they need no pattern.
            Some("Split") | Some("Whitespace") | Some("WhitespaceSplit") => {}
            Some("Sequence") => {
                if let Some(list) = v.get("pretokenizers").and_then(Value::as_array) {
                    for item in list {
                        walk(
                            item,
                            byte_level,
                            split_regex,
                            metaspace,
                            add_prefix_space,
                            unknown,
                        );
                    }
                }
            }
            Some(other) => unknown.push(other.to_string()),
            None => {}
        }
    }

    if let Some(pre) = pre {
        walk(
            pre,
            &mut byte_level,
            &mut split_regex,
            &mut metaspace,
            &mut add_prefix_space,
            &mut unknown,
        );
    }

    let pattern = match (&split_regex, metaspace) {
        (Some(re), _) => re.clone(),
        (None, true) => SENTENCEPIECE_PATTERN.to_string(),
        (None, false) => GPT2_PATTERN.to_string(),
    };

    // If nothing recognizable determined the split (no byte-level, no metaspace,
    // no explicit Split regex) yet the json DID declare pre-tokenizer steps we
    // don't understand, the GPT-2 default would be a silent guess at the split —
    // which changes the tokens. Refuse rather than guess.
    if !byte_level && !metaspace && split_regex.is_none() && !unknown.is_empty() {
        return Err(HfJsonError::UnsupportedPreTokenizer(unknown.join(", ")));
    }

    Ok(PreTokenization {
        byte_level,
        pattern,
        // ByteLevel/Metaspace default `add_prefix_space` to true in HF when the
        // field is absent; real configs set it explicitly.
        add_prefix_space: add_prefix_space.unwrap_or(metaspace || byte_level),
    })
}

/// BERT-family normalizer flags consumed by the WordPiece backend.
#[derive(Debug, Clone)]
pub(super) struct BertNorm {
    pub lowercase: bool,
    /// Isolate CJK ideographs (`handle_chinese_chars`); defaults to true.
    pub handle_chinese_chars: bool,
    /// Strip control/format chars and normalize whitespace (`clean_text`);
    /// defaults to true.
    pub clean_text: bool,
}

/// Extract the WordPiece-relevant flags from a (`BertNormalizer`-shaped)
/// normalizer. WordPiece's `BasicTokenizer` interleaves CJK splitting with
/// casing, so it consumes flags rather than the ordered op pipeline.
pub(super) fn parse_bert_norm(norm: Option<&Value>) -> BertNorm {
    let mut lowercase = false;
    let mut handle_chinese_chars = true;
    let mut clean_text = true;

    fn walk(v: &Value, lc: &mut bool, cjk: &mut bool, clean: &mut bool) {
        match v.get("type").and_then(Value::as_str) {
            Some("Lowercase") => *lc = true,
            Some("BertNormalizer") => {
                if v.get("lowercase").and_then(Value::as_bool).unwrap_or(false) {
                    *lc = true;
                }
                *cjk = v
                    .get("handle_chinese_chars")
                    .and_then(Value::as_bool)
                    .unwrap_or(true);
                *clean = v.get("clean_text").and_then(Value::as_bool).unwrap_or(true);
            }
            Some("Sequence") => {
                if let Some(list) = v.get("normalizers").and_then(Value::as_array) {
                    for item in list {
                        walk(item, lc, cjk, clean);
                    }
                }
            }
            _ => {}
        }
    }
    if let Some(norm) = norm {
        walk(
            norm,
            &mut lowercase,
            &mut handle_chinese_chars,
            &mut clean_text,
        );
    }
    BertNorm {
        lowercase,
        handle_chinese_chars,
        clean_text,
    }
}

/// Parse a `normalizer` value into an ordered list of [`NormOp`]s, flattening
/// `Sequence`s in order.
///
/// A normalizer is an ordered pipeline with no implicit backend fallback, so an
/// unrecognized step (or a `Replace` regex that fails to compile) is a genuine
/// correctness gap: dropping it silently would mis-normalize and produce wrong
/// tokens with no signal. Such cases are surfaced as an error instead.
pub(super) fn parse_norm_ops(norm: Option<&Value>) -> Result<Vec<NormOp>, HfJsonError> {
    let mut ops = Vec::new();
    let mut unknown: Vec<String> = Vec::new();
    let mut bad_regex: Vec<String> = Vec::new();

    fn walk(
        v: &Value,
        ops: &mut Vec<NormOp>,
        unknown: &mut Vec<String>,
        bad_regex: &mut Vec<String>,
    ) {
        match v.get("type").and_then(Value::as_str) {
            Some("NFC") => ops.push(NormOp::Nfc),
            Some("NFD") => ops.push(NormOp::Nfd),
            Some("NFKC") => ops.push(NormOp::Nfkc),
            Some("NFKD") => ops.push(NormOp::Nfkd),
            Some("Lowercase") => ops.push(NormOp::Lowercase),
            Some("StripAccents") => ops.push(NormOp::StripAccents),
            Some("Nmt") => ops.push(NormOp::Nmt),
            Some("Prepend") => {
                if let Some(p) = v.get("prepend").and_then(Value::as_str) {
                    ops.push(NormOp::Prepend(p.to_string()));
                }
            }
            Some("Strip") => ops.push(NormOp::Strip {
                left: v.get("strip_left").and_then(Value::as_bool).unwrap_or(true),
                right: v
                    .get("strip_right")
                    .and_then(Value::as_bool)
                    .unwrap_or(true),
            }),
            Some("Replace") => {
                let content = v
                    .get("content")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                if let Some(p) = v.get("pattern") {
                    if let Some(s) = p.get("String").and_then(Value::as_str) {
                        ops.push(NormOp::ReplaceStr {
                            from: s.to_string(),
                            to: content,
                        });
                    } else if let Some(re) = p.get("Regex").and_then(Value::as_str) {
                        match NormOp::replace_regex(re, content) {
                            Some(op) => ops.push(op),
                            None => bad_regex.push(re.to_string()),
                        }
                    }
                }
            }
            Some("Precompiled") => {
                if let Some(b64) = v.get("precompiled_charsmap").and_then(Value::as_str) {
                    use base64::Engine;
                    if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(b64) {
                        if let Some(pc) = Precompiled::from_bytes(&bytes) {
                            ops.push(NormOp::Precompiled(pc));
                        }
                    }
                }
            }
            // A `BertNormalizer` inside an SP/Unigram graph: expand to its ordered
            // effect (NFD + StripAccents when stripping, then Lowercase).
            Some("BertNormalizer") => {
                let lc = v.get("lowercase").and_then(Value::as_bool).unwrap_or(false);
                let strip = match v.get("strip_accents").and_then(Value::as_bool) {
                    Some(b) => b,
                    None => lc,
                };
                if strip {
                    ops.push(NormOp::Nfd);
                    ops.push(NormOp::StripAccents);
                }
                if lc {
                    ops.push(NormOp::Lowercase);
                }
            }
            Some("Sequence") => {
                if let Some(list) = v.get("normalizers").and_then(Value::as_array) {
                    for item in list {
                        walk(item, ops, unknown, bad_regex);
                    }
                }
            }
            Some(other) => unknown.push(other.to_string()),
            None => {}
        }
    }

    if let Some(norm) = norm {
        walk(norm, &mut ops, &mut unknown, &mut bad_regex);
    }
    if !unknown.is_empty() {
        return Err(HfJsonError::UnsupportedNormalizer(unknown.join(", ")));
    }
    if !bad_regex.is_empty() {
        return Err(HfJsonError::InvalidNormalizerRegex(bad_regex.join(", ")));
    }
    Ok(ops)
}

/// Collect **all** `added_tokens` into a name→id map.
///
/// HuggingFace matches every added token during encoding — both `special` ones
/// (`<|endoftext|>`) and non-special content tokens (e.g. gpt-neox's whitespace
/// runs, deepseek's byte chars) — so the matcher must know all of them, not just
/// the special-flagged ones.
pub(super) fn parse_special_tokens(root: &Value) -> FxHashMap<String, u32> {
    let mut specials = FxHashMap::default();
    if let Some(list) = root.get("added_tokens").and_then(Value::as_array) {
        for t in list {
            if let (Some(content), Some(id)) = (
                t.get("content").and_then(Value::as_str),
                t.get("id").and_then(Value::as_u64),
            ) {
                specials.insert(content.to_string(), id as u32);
            }
        }
    }
    specials
}

/// Ids of `added_tokens` flagged `special: true` — dropped on decode to match
/// HuggingFace's default `skip_special_tokens=true`. Non-special added tokens
/// (e.g. gpt-neox whitespace runs) are kept.
pub(super) fn parse_special_decode_ids(root: &Value) -> rustc_hash::FxHashSet<u32> {
    let mut ids = rustc_hash::FxHashSet::default();
    if let Some(list) = root.get("added_tokens").and_then(Value::as_array) {
        for t in list {
            if t.get("special").and_then(Value::as_bool).unwrap_or(true) {
                if let Some(id) = t.get("id").and_then(Value::as_u64) {
                    ids.insert(id as u32);
                }
            }
        }
    }
    ids
}

/// Find the id of the first matching token content in `added_tokens`.
pub(super) fn find_added_token(root: &Value, candidates: &[&str]) -> Option<u32> {
    let list = root.get("added_tokens").and_then(Value::as_array)?;
    for cand in candidates {
        for t in list {
            if t.get("content").and_then(Value::as_str) == Some(cand) {
                return t.get("id").and_then(Value::as_u64).map(|n| n as u32);
            }
        }
    }
    None
}
