use super::spec::Behavior;
use super::split::{split_digits, split_punctuation, split_regex};
use crate::core::byte_level::byte_level_encode;

/// A single pre-tokenizer stage.
pub(super) enum Stage {
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
                re,
                behavior,
                invert,
            } => split_regex(piece, re, *behavior, *invert, out),
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
