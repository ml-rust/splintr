use serde_json::Value;

use super::pipeline::PreTokenizer;
use super::spec::{PreTokStage, SplitBehavior, SplitPattern};
use crate::core::tokenizer::{TokenizerError, GPT2_PATTERN};

/// Both of these compile a pattern this crate owns, so failure would be a bug
/// here rather than bad caller input. They still surface it as an error instead
/// of panicking: [`PreTokenizer::new`] already returns a `Result`, so carrying
/// it costs nothing and keeps library code panic-free.
pub(super) fn gpt2_regex() -> Result<super::stage::SplitMatcher, TokenizerError> {
    super::stage::SplitMatcher::compile(GPT2_PATTERN)
}

pub(super) fn whitespace_regex() -> Result<super::stage::SplitMatcher, TokenizerError> {
    super::stage::SplitMatcher::compile(r"\w+|[^\w\s]+")
}

/// Build a [`PreTokenizer`] from a `pre_tokenizer` JSON value. Returns `None`
/// when there is no usable pre-tokenizer.
///
/// Deliberately crate-private: the shape it consumes is HuggingFace's internal
/// JSON dialect, which must not become part of splintr's public API. Callers
/// outside the crate build a pipeline with [`PreTokenizer::new`], or load a whole
/// file through [`from_json_bytes`](crate::from_json_bytes) /
/// [`from_json_path`](crate::from_json_path).
///
/// # Errors
/// Returns [`TokenizerError::RegexrError`] if a declared `Split` pattern does
/// not compile, rather than dropping the stage and tokenizing differently.
pub(crate) fn parse(pre: Option<&Value>) -> Result<Option<PreTokenizer>, TokenizerError> {
    let Some(pre) = pre else {
        return Ok(None);
    };
    let mut stages = Vec::new();

    fn walk(v: &Value, stages: &mut Vec<PreTokStage>) {
        match v.get("type").and_then(Value::as_str) {
            Some("Sequence") => {
                if let Some(list) = v.get("pretokenizers").and_then(Value::as_array) {
                    for item in list {
                        walk(item, stages);
                    }
                }
            }
            Some("ByteLevel") => stages.push(PreTokStage::ByteLevel {
                use_regex: v.get("use_regex").and_then(Value::as_bool).unwrap_or(true),
                add_prefix_space: v.get("add_prefix_space").and_then(Value::as_bool) == Some(true),
            }),
            Some("Split") => {
                // HF's `pattern` is either form, and they are not
                // interchangeable: a `String` is matched literally, so its regex
                // metacharacters mean nothing.
                let pat = v.get("pattern").and_then(|p| {
                    p.get("Regex")
                        .and_then(Value::as_str)
                        .map(|s| SplitPattern::Regex(s.to_string()))
                        .or_else(|| {
                            p.get("String")
                                .and_then(Value::as_str)
                                .map(|s| SplitPattern::Literal(s.to_string()))
                        })
                });
                if let Some(pat) = pat {
                    stages.push(PreTokStage::Split {
                        pattern: pat,
                        behavior: SplitBehavior::parse(v.get("behavior").and_then(Value::as_str)),
                        invert: v.get("invert").and_then(Value::as_bool).unwrap_or(false),
                    });
                }
            }
            Some("Digits") => stages.push(PreTokStage::Digits {
                individual: v
                    .get("individual_digits")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
            }),
            Some("Punctuation") => stages.push(PreTokStage::Punctuation {
                behavior: SplitBehavior::parse(v.get("behavior").and_then(Value::as_str)),
            }),
            Some("WhitespaceSplit") => stages.push(PreTokStage::WhitespaceSplit),
            Some("Whitespace") => stages.push(PreTokStage::Whitespace),
            _ => {}
        }
    }

    walk(pre, &mut stages);
    if stages.is_empty() {
        return Ok(None);
    }
    Ok(Some(PreTokenizer::new(stages)?))
}
