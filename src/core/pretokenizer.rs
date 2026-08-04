//! HuggingFace pre-tokenizer pipeline.
//!
//! HF pre-tokenizers form an ordered pipeline: each stage takes the current list
//! of string pieces and splits each one further. A single regex can't model that
//! (e.g. Falcon = `Punctuation → ByteLevel → Digits → Split`), so this engine
//! applies the stages in order and returns the final pre-token pieces.
//!
//! When a `ByteLevel` stage is present the pieces come out byte-level-encoded
//! (each byte mapped to a printable code point), ready for BPE against a
//! byte-level vocab — so the consumer must NOT byte-level-encode again.

use std::borrow::Cow;

use regexr::RegexBuilder;
use serde_json::Value;

use super::byte_level::byte_level_encode;
use super::tokenizer::{TokenizerError, GPT2_PATTERN};

/// What a `Split`/`Punctuation` stage does with the matched delimiter — the full
/// set of HuggingFace `SplitDelimiterBehavior` variants.
///
/// Deliberately *not* `#[non_exhaustive]`: HuggingFace's set is closed and
/// stable, so sealing it would only stop downstream code from matching
/// exhaustively without buying any room to grow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SplitBehavior {
    /// Delimiter becomes its own piece.
    #[default]
    Isolated,
    /// Delimiter is dropped.
    Removed,
    /// Delimiter is appended to the preceding piece.
    MergedWithPrevious,
    /// Delimiter is prepended to the following piece.
    MergedWithNext,
    /// Runs of adjacent delimiters merge into a single piece.
    Contiguous,
}

impl SplitBehavior {
    fn parse(s: Option<&str>) -> Self {
        match s {
            Some("Removed") => SplitBehavior::Removed,
            Some("MergedWithPrevious") => SplitBehavior::MergedWithPrevious,
            Some("MergedWithNext") => SplitBehavior::MergedWithNext,
            Some("Contiguous") => SplitBehavior::Contiguous,
            // "Isolated" and any unknown/absent value.
            _ => SplitBehavior::Isolated,
        }
    }
}

/// How a [`PreTokStage::Split`] pattern is interpreted, mirroring HuggingFace's
/// `pattern` field, which is either a literal string or a regex — the two mean
/// different things and are not interchangeable.
///
/// Deliberately *not* `#[non_exhaustive]`, for the same reason as
/// [`SplitBehavior`]: HuggingFace's set is closed at these two forms, so sealing
/// it would only stop downstream code from matching exhaustively.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SplitPattern {
    /// Matched exactly, character for character. Regex metacharacters carry no
    /// special meaning.
    Literal(String),
    /// Compiled as a regular expression.
    Regex(String),
}

/// One pre-tokenizer stage as a *description*: regexes are given as patterns and
/// compiled by [`PreTokenizer::new`], so a caller never has to name a regex type.
///
/// `#[non_exhaustive]`: this enum tracks HuggingFace's pre-tokenizer spec and
/// grows as new pre-tokenizer types are added there, so adding a variant must
/// not be a breaking change for downstream matchers. The attribute sits on the
/// enum only — putting it on a variant would make that variant unconstructible
/// downstream, defeating the point of the builder.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PreTokStage {
    /// Split on `pattern`, combining the matched delimiters per `behavior`. With
    /// `invert`, the spans *between* matches are the delimiters instead.
    Split {
        pattern: SplitPattern,
        behavior: SplitBehavior,
        invert: bool,
    },
    /// GPT-2 byte-level: optionally split on the GPT-2 regex, then byte-encode.
    /// `add_prefix_space` applies to the whole pipeline, not just this stage.
    ByteLevel {
        use_regex: bool,
        add_prefix_space: bool,
    },
    /// Split digit runs from the rest (optionally each digit individually).
    Digits { individual: bool },
    /// Split punctuation from the rest, honoring the HF delimiter behavior.
    Punctuation { behavior: SplitBehavior },
    /// Split on whitespace, dropping it.
    WhitespaceSplit,
    /// GPT-2 word regex (`\w+|[^\w\s]+`) without byte-encoding.
    Whitespace,
}

/// The compiled counterpart of [`SplitBehavior`], produced by
/// [`PreTokenizer::new`] and stored in a [`Stage`] so `apply` never has to
/// convert on the hot path.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Behavior {
    /// Delimiter becomes its own piece.
    Isolated,
    /// Delimiter is dropped.
    Removed,
    /// Delimiter is appended to the preceding piece.
    MergedWithPrevious,
    /// Delimiter is prepended to the following piece.
    MergedWithNext,
    /// Runs of adjacent delimiters merge into a single piece.
    Contiguous,
}

impl From<SplitBehavior> for Behavior {
    fn from(b: SplitBehavior) -> Self {
        match b {
            SplitBehavior::Isolated => Behavior::Isolated,
            SplitBehavior::Removed => Behavior::Removed,
            SplitBehavior::MergedWithPrevious => Behavior::MergedWithPrevious,
            SplitBehavior::MergedWithNext => Behavior::MergedWithNext,
            SplitBehavior::Contiguous => Behavior::Contiguous,
        }
    }
}

/// A single pre-tokenizer stage.
enum Stage {
    Split {
        re: Box<regexr::Regex>,
        behavior: Behavior,
        invert: bool,
    },
    /// GPT-2 byte-level: optionally split on the GPT-2 regex (pre-compiled), then
    /// byte-encode.
    ByteLevel { re: Option<Box<regexr::Regex>> },
    /// Split digit runs from the rest (optionally each digit individually).
    Digits { individual: bool },
    /// Split punctuation from the rest, honoring the HF delimiter behavior.
    Punctuation { behavior: Behavior },
    /// Split on whitespace, dropping it.
    WhitespaceSplit,
    /// GPT-2 word regex (pre-compiled) without byte-encoding.
    Whitespace { re: Box<regexr::Regex> },
}

/// An ordered pre-tokenizer pipeline.
pub struct PreTokenizer {
    /// The spec this pipeline was built from, kept so [`PreTokenizer::stages`]
    /// can hand it back.
    spec: Vec<PreTokStage>,
    /// The compiled counterpart of `spec`, one entry per stage.
    compiled: Vec<Stage>,
    /// Prepend a space to the whole input before running stages (ByteLevel
    /// `add_prefix_space`).
    add_prefix_space: bool,
    /// Whether a ByteLevel stage byte-encodes the pieces (so BPE skips encoding).
    byte_level: bool,
}

impl PreTokenizer {
    /// Build a pipeline from an ordered list of stage descriptions, compiling
    /// every `Split` pattern.
    ///
    /// # Errors
    /// Returns [`TokenizerError::RegexrError`] if a [`SplitPattern::Regex`] does
    /// not compile. Dropping the stage instead would silently change the split —
    /// and therefore the token ids — with nothing to point at. A
    /// [`SplitPattern::Literal`] is escaped before compiling, so it always
    /// compiles.
    pub fn new(stages: Vec<PreTokStage>) -> Result<Self, TokenizerError> {
        let mut compiled = Vec::with_capacity(stages.len());
        let mut byte_level = false;
        let mut add_prefix_space = false;
        for stage in &stages {
            compiled.push(match stage {
                PreTokStage::Split {
                    pattern,
                    behavior,
                    invert,
                } => Stage::Split {
                    // A literal is compiled as an escaped regex rather than
                    // matched by a separate code path, so both forms share the
                    // delimiter/behavior/invert handling in `emit_segments` and
                    // cannot drift apart. `Cow` avoids cloning the `Regex` arm's
                    // pattern just to unify it with the `Literal` arm's owned,
                    // escaped one.
                    re: Box::new(
                        RegexBuilder::new(&match pattern {
                            SplitPattern::Literal(s) => Cow::Owned(regexr::escape(s)),
                            SplitPattern::Regex(s) => Cow::Borrowed(s.as_str()),
                        })
                        .jit(true)
                        .build()?,
                    ),
                    behavior: (*behavior).into(),
                    invert: *invert,
                },
                PreTokStage::ByteLevel {
                    use_regex,
                    add_prefix_space: prefix,
                } => {
                    byte_level = true;
                    add_prefix_space |= *prefix;
                    Stage::ByteLevel {
                        re: match use_regex {
                            true => Some(Box::new(gpt2_regex()?)),
                            false => None,
                        },
                    }
                }
                PreTokStage::Digits { individual } => Stage::Digits {
                    individual: *individual,
                },
                PreTokStage::Punctuation { behavior } => Stage::Punctuation {
                    behavior: (*behavior).into(),
                },
                PreTokStage::WhitespaceSplit => Stage::WhitespaceSplit,
                PreTokStage::Whitespace => Stage::Whitespace {
                    re: Box::new(whitespace_regex()?),
                },
            });
        }
        Ok(Self {
            spec: stages,
            compiled,
            add_prefix_space,
            byte_level,
        })
    }

    /// Pre-tokenize `text` into the final (BPE-ready) pieces.
    ///
    /// The `add_prefix_space` guard is a literal **space**, matching
    /// `ByteLevel::pre_tokenize`'s own `!normalized.get().starts_with(' ')`:
    /// text opening on any other whitespace still gets the prefix. Measured
    /// against `tokenizers` 0.22.1 on a `ByteLevel { add_prefix_space: true }`
    /// fixture, `"\ta"` pre-tokenizes to `Ġ`/`ĉ`/`a` while `" a"` stays `Ġa`.
    pub fn split(&self, text: &str) -> Vec<String> {
        let mut pieces: Vec<String> = if self.add_prefix_space && !text.starts_with(' ') {
            vec![format!(" {text}")]
        } else {
            vec![text.to_string()]
        };
        for stage in &self.compiled {
            let mut next = Vec::with_capacity(pieces.len());
            for p in &pieces {
                stage.apply(p, &mut next);
            }
            pieces = next;
        }
        pieces.retain(|p| !p.is_empty());
        pieces
    }

    /// Whether a ByteLevel stage byte-encodes the pieces (so BPE skips encoding).
    ///
    /// Derived from the stage list rather than settable: a caller who could set
    /// it independently could desynchronize it from the pipeline.
    pub fn byte_level(&self) -> bool {
        self.byte_level
    }

    /// Whether the pipeline has no stages, in which case it is a no-op.
    pub fn is_empty(&self) -> bool {
        self.spec.is_empty()
    }

    /// The stage descriptions this pipeline was built from, in order.
    pub fn stages(&self) -> &[PreTokStage] {
        &self.spec
    }
}

impl std::fmt::Debug for PreTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The compiled stages hold regexes that aren't printable, so report the
        // spec they came from plus the derived byte-level flag.
        f.debug_struct("PreTokenizer")
            .field("stages", &self.spec)
            .field("byte_level", &self.byte_level)
            .finish()
    }
}

impl Stage {
    fn apply(&self, piece: &str, out: &mut Vec<String>) {
        match self {
            Stage::Split {
                re,
                behavior,
                invert,
            } => split_regex(piece, re, *behavior, *invert, out),
            Stage::ByteLevel { re } => match re {
                Some(re) => {
                    let mut raw = Vec::new();
                    split_regex(piece, re, Behavior::Isolated, false, &mut raw);
                    for r in raw {
                        out.push(byte_level_encode(r.as_bytes()));
                    }
                }
                None => out.push(byte_level_encode(piece.as_bytes())),
            },
            Stage::Digits { individual } => split_digits(piece, *individual, out),
            Stage::Punctuation { behavior } => split_punctuation(piece, *behavior, out),
            Stage::WhitespaceSplit => {
                for w in piece.split_whitespace() {
                    out.push(w.to_string());
                }
            }
            Stage::Whitespace { re } => split_regex(piece, re, Behavior::Isolated, false, out),
        }
    }
}

/// Split `piece` by `re`. The matched spans are the delimiters (or, with
/// `invert`, the spans between matches); everything else is content. The matched
/// delimiters are combined with surrounding content according to `behavior`.
fn split_regex(
    piece: &str,
    re: &regexr::Regex,
    behavior: Behavior,
    invert: bool,
    out: &mut Vec<String>,
) {
    let matches: Vec<(usize, usize)> = re.find_iter(piece).map(|m| (m.start(), m.end())).collect();
    let delims: Vec<(usize, usize)> = if invert {
        let mut d = Vec::new();
        let mut last = 0;
        for &(s, e) in &matches {
            if s > last {
                d.push((last, s));
            }
            last = e;
        }
        if last < piece.len() {
            d.push((last, piece.len()));
        }
        d
    } else {
        matches
    };

    // Flatten into an ordered list of (text, is_delimiter) segments.
    let mut segs: Vec<(&str, bool)> = Vec::with_capacity(delims.len() * 2 + 1);
    let mut last = 0;
    for (s, e) in delims {
        if s > last {
            segs.push((&piece[last..s], false));
        }
        if e > s {
            segs.push((&piece[s..e], true));
        }
        last = e;
    }
    if last < piece.len() {
        segs.push((&piece[last..], false));
    }

    emit_segments(&segs, behavior, out);
}

/// Combine ordered (text, is_delimiter) segments per the HF delimiter behavior,
/// appending the resulting pieces to `out`.
fn emit_segments(segs: &[(&str, bool)], behavior: Behavior, out: &mut Vec<String>) {
    match behavior {
        Behavior::Isolated => {
            for &(t, _) in segs {
                out.push(t.to_string());
            }
        }
        Behavior::Removed => {
            for &(t, d) in segs {
                if !d {
                    out.push(t.to_string());
                }
            }
        }
        Behavior::MergedWithPrevious => {
            // A delimiter attaches to the preceding emitted piece (from this
            // split only); a leading delimiter with no predecessor stands alone.
            let mut local: Vec<String> = Vec::new();
            for &(t, d) in segs {
                if d {
                    if let Some(prev) = local.last_mut() {
                        prev.push_str(t);
                    } else {
                        local.push(t.to_string());
                    }
                } else {
                    local.push(t.to_string());
                }
            }
            out.extend(local);
        }
        Behavior::MergedWithNext => {
            // A delimiter attaches to the following piece; trailing delimiters
            // with no successor stand alone.
            let mut pending = String::new();
            for &(t, d) in segs {
                if d {
                    pending.push_str(t);
                } else {
                    let mut s = std::mem::take(&mut pending);
                    s.push_str(t);
                    out.push(s);
                }
            }
            if !pending.is_empty() {
                out.push(pending);
            }
        }
        Behavior::Contiguous => {
            // Runs of adjacent delimiters merge into one piece; content stays
            // split. (Adjacent delimiters arise from back-to-back matches.)
            let mut run = String::new();
            for &(t, d) in segs {
                if d {
                    run.push_str(t);
                } else {
                    if !run.is_empty() {
                        out.push(std::mem::take(&mut run));
                    }
                    out.push(t.to_string());
                }
            }
            if !run.is_empty() {
                out.push(run);
            }
        }
    }
}

fn split_digits(piece: &str, individual: bool, out: &mut Vec<String>) {
    let mut cur = String::new();
    let mut cur_is_digit = false;
    for c in piece.chars() {
        // HF's Digits pre-tokenizer uses Unicode numericity (`char::is_numeric`,
        // categories Nd/Nl/No), so superscripts/fractions like ²/³/½ count as
        // digits — not just ASCII 0-9.
        let d = c.is_numeric();
        if !cur.is_empty() && (d != cur_is_digit || (d && individual)) {
            out.push(std::mem::take(&mut cur));
        }
        cur.push(c);
        cur_is_digit = d;
        if d && individual {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
}

fn split_punctuation(piece: &str, behavior: Behavior, out: &mut Vec<String>) {
    // Each punctuation char is a delimiter segment; consecutive non-punctuation
    // chars form a content segment. The behavior then combines them.
    let mut segs: Vec<(&str, bool)> = Vec::new();
    let mut content_start = 0;
    let mut i = 0;
    for c in piece.chars() {
        let len = c.len_utf8();
        if is_punctuation(c) {
            if i > content_start {
                segs.push((&piece[content_start..i], false));
            }
            segs.push((&piece[i..i + len], true));
            content_start = i + len;
        }
        i += len;
    }
    if i > content_start {
        segs.push((&piece[content_start..i], false));
    }
    emit_segments(&segs, behavior, out);
}

/// HF punctuation definition (ASCII punctuation + Unicode P* categories).
fn is_punctuation(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_punctuation();
    }
    use unicode_general_category::{get_general_category, GeneralCategory::*};
    matches!(
        get_general_category(c),
        ConnectorPunctuation
            | DashPunctuation
            | ClosePunctuation
            | FinalPunctuation
            | InitialPunctuation
            | OtherPunctuation
            | OpenPunctuation
    )
}

/// Both of these compile a pattern this crate owns, so failure would be a bug
/// here rather than bad caller input. They still surface it as an error instead
/// of panicking: [`PreTokenizer::new`] already returns a `Result`, so carrying
/// it costs nothing and keeps library code panic-free.
fn gpt2_regex() -> Result<regexr::Regex, TokenizerError> {
    Ok(RegexBuilder::new(GPT2_PATTERN).jit(true).build()?)
}

fn whitespace_regex() -> Result<regexr::Regex, TokenizerError> {
    Ok(RegexBuilder::new(r"\w+|[^\w\s]+").jit(true).build()?)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn split(piece: &str, f: impl Fn(&str, &mut Vec<String>)) -> Vec<String> {
        let mut out = Vec::new();
        f(piece, &mut out);
        out
    }

    #[test]
    fn digits_grouped_and_individual() {
        assert_eq!(
            split("abc123def", |p, o| split_digits(p, false, o)),
            vec!["abc", "123", "def"]
        );
        assert_eq!(
            split("abc123", |p, o| split_digits(p, true, o)),
            vec!["abc", "1", "2", "3"]
        );
        // Unicode numerics (superscripts/fractions) count as digits, like HF.
        assert_eq!(
            split("a²½b", |p, o| split_digits(p, true, o)),
            vec!["a", "²", "½", "b"]
        );
    }

    #[test]
    fn punctuation_isolated_and_contiguous() {
        assert_eq!(
            split("a,b!", |p, o| split_punctuation(p, Behavior::Isolated, o)),
            vec!["a", ",", "b", "!"]
        );
        assert_eq!(
            split("a)=b", |p, o| split_punctuation(p, Behavior::Contiguous, o)),
            vec!["a", ")=", "b"]
        );
        assert_eq!(
            split("a,b", |p, o| split_punctuation(p, Behavior::Removed, o)),
            vec!["a", "b"]
        );
    }

    #[test]
    fn split_merge_behaviors() {
        let re = RegexBuilder::new(r"\s+").jit(true).build().unwrap();
        let go = |b| split("a b c", |p, o| split_regex(p, &re, b, false, o));
        assert_eq!(go(Behavior::Isolated), vec!["a", " ", "b", " ", "c"]);
        assert_eq!(go(Behavior::Removed), vec!["a", "b", "c"]);
        assert_eq!(go(Behavior::MergedWithPrevious), vec!["a ", "b ", "c"]);
        assert_eq!(go(Behavior::MergedWithNext), vec!["a", " b", " c"]);
        // Adjacent delimiters merge under Contiguous.
        let re2 = RegexBuilder::new(r"\s").jit(true).build().unwrap();
        assert_eq!(
            split("a  b", |p, o| split_regex(
                p,
                &re2,
                Behavior::Contiguous,
                false,
                o
            )),
            vec!["a", "  ", "b"]
        );
    }

    #[test]
    fn pipeline_digits_then_byte_level() {
        // Sequence[Digits(individual), ByteLevel] like starcoder2: "a 12" → each
        // digit isolated, then byte-encoded (space → Ġ).
        let json = serde_json::json!({
            "type": "Sequence",
            "pretokenizers": [
                {"type": "Digits", "individual_digits": true},
                {"type": "ByteLevel", "add_prefix_space": false, "use_regex": true}
            ]
        });
        let pt = parse(Some(&json)).expect("parses").expect("pipeline");
        assert!(pt.byte_level());
        // "a 12" → ["a"," 1","2"]? Digits first: ["a ","1","2"]; then ByteLevel
        // GPT2-splits "a " → "a"+" "? then byte-encodes. Just assert digits split.
        let pieces = pt.split("a12");
        assert_eq!(pieces, vec!["a", "1", "2"]);
    }

    #[test]
    fn split_invert_keeps_content_between_matches() {
        // invert=true: matches are content; gaps (delimiters) handled by `keep`.
        let re = Box::new(gpt2_regex().expect("GPT2_PATTERN compiles"));
        let mut out = Vec::new();
        // Using a simple digit regex via Whitespace isn't ideal; verify invert
        // path runs without panicking and partitions the string.
        super::split_regex("ab", &re, Behavior::Isolated, true, &mut out);
        assert_eq!(out.concat(), "ab");
    }

    /// HuggingFace's `Split` takes either a literal string or a regex, and they
    /// are not interchangeable. Reference (`tokenizers` package, behavior
    /// `removed`, input `"a.b c"`): `Split(pattern=".")` yields
    /// `[('a', (0,1)), ('b c', (2,5))]`, while `Split(pattern=Regex("."))`
    /// matches every character and yields nothing.
    #[test]
    fn literal_and_regex_split_patterns_are_not_interchangeable() {
        let split = |pattern| {
            PreTokenizer::new(vec![PreTokStage::Split {
                pattern,
                behavior: SplitBehavior::Removed,
                invert: false,
            }])
            .expect("pipeline builds")
            .split("a.b c")
        };
        assert_eq!(
            split(SplitPattern::Literal(".".to_string())),
            vec!["a", "b c"]
        );
        assert!(split(SplitPattern::Regex(".".to_string())).is_empty());
    }

    #[test]
    fn literal_split_pattern_matches_metacharacters_verbatim() {
        let split = |pattern, text: &str| {
            PreTokenizer::new(vec![PreTokStage::Split {
                pattern,
                behavior: SplitBehavior::Removed,
                invert: false,
            }])
            .expect("pipeline builds")
            .split(text)
        };
        // As a regex `a+b` would need one-or-more `a`; as a literal it is the
        // three characters, which appear only in the middle here.
        assert_eq!(
            split(SplitPattern::Literal("a+b".to_string()), "xa+by"),
            vec!["x", "y"]
        );
        // As a regex `|` is an empty alternation matching everywhere.
        assert_eq!(
            split(SplitPattern::Literal("|".to_string()), "a|b"),
            vec!["a", "b"]
        );
    }

    #[test]
    fn split_with_uncompilable_pattern_is_an_error() {
        // Previously such a stage was silently dropped, which changed the split
        // (and so the ids) with nothing to point at.
        let json = serde_json::json!({
            "type": "Split",
            "pattern": {"Regex": "("},
            "behavior": "Isolated"
        });
        assert!(parse(Some(&json)).is_err());
    }

    /// Everything the loader builds from JSON must be expressible through the
    /// public [`PreTokStage`] builder: `parse` reports the spec it used, and
    /// rebuilding from that spec must pre-tokenize identically. Adding a `Stage`
    /// without a `PreTokStage` counterpart fails here.
    #[test]
    fn parsed_stages_round_trip_through_the_public_builder() {
        let probe = "Hello, wörld 42 items!";
        let mut cases: Vec<(Value, Vec<PreTokStage>)> = vec![
            // Nested Sequence, ByteLevel with use_regex:false + add_prefix_space.
            (
                serde_json::json!({
                    "type": "Sequence",
                    "pretokenizers": [
                        {"type": "Sequence", "pretokenizers": [
                            {"type": "Punctuation", "behavior": "Contiguous"},
                            {"type": "Digits", "individual_digits": true}
                        ]},
                        {"type": "ByteLevel", "use_regex": false, "add_prefix_space": true}
                    ]
                }),
                vec![
                    PreTokStage::Punctuation {
                        behavior: SplitBehavior::Contiguous,
                    },
                    PreTokStage::Digits { individual: true },
                    PreTokStage::ByteLevel {
                        use_regex: false,
                        add_prefix_space: true,
                    },
                ],
            ),
            // Bare ByteLevel: use_regex defaults to true, add_prefix_space to false.
            (
                serde_json::json!({"type": "ByteLevel"}),
                vec![PreTokStage::ByteLevel {
                    use_regex: true,
                    add_prefix_space: false,
                }],
            ),
            (
                serde_json::json!({"type": "Whitespace"}),
                vec![PreTokStage::Whitespace],
            ),
            (
                serde_json::json!({"type": "WhitespaceSplit"}),
                vec![PreTokStage::WhitespaceSplit],
            ),
            // A `String` pattern is a literal, a `Regex` pattern is a regex —
            // conflating them changes the split for any metacharacter.
            (
                serde_json::json!({
                    "type": "Split",
                    "pattern": {"String": ","},
                    "behavior": "Removed"
                }),
                vec![PreTokStage::Split {
                    pattern: SplitPattern::Literal(",".to_string()),
                    behavior: SplitBehavior::Removed,
                    invert: false,
                }],
            ),
            // A `String` whose text is regex-significant still matches literally.
            (
                serde_json::json!({
                    "type": "Split",
                    "pattern": {"String": "."},
                    "behavior": "Removed"
                }),
                vec![PreTokStage::Split {
                    pattern: SplitPattern::Literal(".".to_string()),
                    behavior: SplitBehavior::Removed,
                    invert: false,
                }],
            ),
            (
                serde_json::json!({
                    "type": "Split",
                    "pattern": {"Regex": r"\w+"},
                    "invert": true
                }),
                vec![PreTokStage::Split {
                    pattern: SplitPattern::Regex(r"\w+".to_string()),
                    behavior: SplitBehavior::Isolated,
                    invert: true,
                }],
            ),
        ];
        // Every delimiter behavior, spelled as HuggingFace spells it.
        for (name, behavior) in [
            ("Isolated", SplitBehavior::Isolated),
            ("Removed", SplitBehavior::Removed),
            ("MergedWithPrevious", SplitBehavior::MergedWithPrevious),
            ("MergedWithNext", SplitBehavior::MergedWithNext),
            ("Contiguous", SplitBehavior::Contiguous),
        ] {
            cases.push((
                serde_json::json!({
                    "type": "Split",
                    "pattern": {"Regex": r"\s+"},
                    "behavior": name
                }),
                vec![PreTokStage::Split {
                    pattern: SplitPattern::Regex(r"\s+".to_string()),
                    behavior,
                    invert: false,
                }],
            ));
        }

        for (json, expected) in cases {
            let parsed = parse(Some(&json)).expect("parses").expect("pipeline");
            assert_eq!(parsed.stages(), expected.as_slice(), "spec for {json}");
            let built = PreTokenizer::new(expected).expect("builds");
            assert_eq!(built.byte_level(), parsed.byte_level(), "byte_level {json}");
            assert_eq!(built.split(probe), parsed.split(probe), "split for {json}");
        }
    }
}
