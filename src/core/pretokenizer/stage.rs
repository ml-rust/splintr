use super::spec::Behavior;
use super::split::{split_digits, split_punctuation, split_regex};
use crate::core::byte_level::byte_level_encode;

/// Where a `Split` stage's matches come from.
///
/// A `tokenizer.json` may carry, byte for byte, one of the expressions splintr
/// already has a direct scanner for — Qwen 3's file carries Qwen 2's, for
/// instance. Recognising that here is what lets a vocabulary loaded with
/// `from_json` take the same fast path as a bundled one; without it the scanners
/// only ever helped `from_pretrained`.
pub(super) enum SplitMatcher {
    Regex(Box<regexr::Regex>),
    Scanner(crate::core::tokenizer::scanner::SpanScanner),
}

impl SplitMatcher {
    /// A matcher for `pattern`: the direct scanner when one has been proven
    /// against that exact expression, the compiled engine otherwise.
    ///
    /// The single place the choice is made, so every stage that splits by an
    /// expression gets it on the same terms.
    pub(super) fn compile(pattern: &str) -> Result<Self, crate::core::tokenizer::TokenizerError> {
        match crate::core::tokenizer::scanner::for_pattern(pattern) {
            Some(scan) => Ok(SplitMatcher::Scanner(scan)),
            None => Ok(SplitMatcher::Regex(Box::new(
                regexr::RegexBuilder::new(pattern).jit(true).build()?,
            ))),
        }
    }

    /// Appends this matcher's spans over `piece`.
    pub(super) fn matches(&self, piece: &str, out: &mut Vec<(usize, usize)>) {
        match self {
            SplitMatcher::Scanner(scan) => scan(piece, out),
            SplitMatcher::Regex(re) => {
                out.extend(re.find_iter(piece).map(|m| (m.start(), m.end())))
            }
        }
    }
}

/// A single pre-tokenizer stage.
pub(super) enum Stage {
    Split {
        matcher: SplitMatcher,
        behavior: Behavior,
        invert: bool,
    },
    /// GPT-2 byte-level: optionally split on the GPT-2 regex (pre-compiled), then
    /// byte-encode.
    ByteLevel { re: Option<SplitMatcher> },
    /// Split digit runs from the rest (optionally each digit individually).
    Digits { individual: bool },
    /// Split punctuation from the rest, honoring the HF delimiter behavior.
    Punctuation { behavior: Behavior },
    /// Split on whitespace, dropping it.
    WhitespaceSplit,
    /// GPT-2 word regex (pre-compiled) without byte-encoding.
    Whitespace { re: SplitMatcher },
}
impl Stage {
    /// Whether this stage rewrites its input's content rather than only cutting
    /// it, and so cannot hand back subslices of what it was given.
    ///
    /// `ByteLevel` is the only one: it maps every byte to an alphabet
    /// character. Everything else partitions its input.
    pub(super) fn rewrites_content(&self) -> bool {
        matches!(self, Stage::ByteLevel { .. })
    }

    /// Cut `piece` into subslices of itself.
    ///
    /// # Panics
    /// Never called on a stage where [`Stage::rewrites_content`] is true; the
    /// pipeline switches to [`Stage::apply_owned`] at the first such stage.
    pub(super) fn cut<'p>(&self, piece: &'p str, out: &mut Vec<&'p str>) {
        match self {
            Stage::Split {
                matcher,
                behavior,
                invert,
            } => split_regex(piece, matcher, *behavior, *invert, out),
            Stage::Digits { individual } => split_digits(piece, *individual, out),
            Stage::Punctuation { behavior } => split_punctuation(piece, *behavior, out),
            Stage::WhitespaceSplit => out.extend(piece.split_whitespace()),
            Stage::Whitespace { re } => split_regex(piece, re, Behavior::Isolated, false, out),
            Stage::ByteLevel { .. } => {
                unreachable!("ByteLevel rewrites content and is driven through apply_owned")
            }
        }
    }

    /// Apply this stage, producing owned pieces.
    ///
    /// Used from the first content-rewriting stage onwards. Cutting stages
    /// still cut — they just have to copy out of a buffer the caller owns.
    pub(super) fn apply_owned(&self, piece: &str, out: &mut Vec<String>) {
        match self {
            Stage::ByteLevel { re } => match re {
                Some(re) => {
                    let mut raw: Vec<&str> = Vec::new();
                    split_regex(piece, re, Behavior::Isolated, false, &mut raw);
                    out.extend(raw.into_iter().map(|r| byte_level_encode(r.as_bytes())));
                }
                None => out.push(byte_level_encode(piece.as_bytes())),
            },
            cutting => {
                let mut raw: Vec<&str> = Vec::new();
                cutting.cut(piece, &mut raw);
                out.extend(raw.into_iter().map(str::to_owned));
            }
        }
    }
}
