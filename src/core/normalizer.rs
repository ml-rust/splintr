//! Ordered normalizer pipeline mirroring HuggingFace's `normalizer` graph.
//!
//! HuggingFace applies normalizers as an ordered sequence (e.g. albert:
//! `Replace → Replace → NFKD → StripAccents → Lowercase → Precompiled`). A flat
//! set of flags can't reproduce that — order matters (Replace before NFKD, the
//! charsmap last, …). [`NormOp`] models each step; [`Normalizer`] applies them in
//! order so nothing is silently dropped or reordered.

use regexr::RegexBuilder;
use std::borrow::Cow;
use unicode_general_category::{get_general_category, GeneralCategory};
use unicode_normalization::{
    is_nfc_quick, is_nfd_quick, is_nfkc_quick, is_nfkd_quick, IsNormalized, UnicodeNormalization,
};

use super::precompiled::Precompiled;

/// Lowercase `s` one character at a time, which is **not** `str::to_lowercase`.
///
/// `str::to_lowercase` applies Unicode's context-sensitive SpecialCasing rules,
/// most visibly Greek final sigma (word-final `Σ` → `ς`). HuggingFace has no
/// such rule — both its `Lowercase` normalizer and `BertNormalizer` map each
/// `char` through `char::to_lowercase` — so on `bert-base-uncased` `ΟΠΩΣ` is
/// `ο ##π ##ω ##σ` there and was `ο ##π ##ω ##ς` here.
pub(crate) fn lowercase(s: &str) -> String {
    s.chars().flat_map(char::to_lowercase).collect()
}

/// Whether [`lowercase`] would change `s`, asked before allocating.
///
/// Titlecase is tested separately: `ǅ` (U+01C5) is `Lt`, not covered by the
/// `Uppercase` property `char::is_uppercase` reports, and still lowercases.
pub(crate) fn needs_lowercasing(s: &str) -> bool {
    s.chars()
        .any(|c| c.is_uppercase() || get_general_category(c) == GeneralCategory::TitlecaseLetter)
}

/// A single normalization step, matching one HuggingFace normalizer type.
///
/// `#[non_exhaustive]`: this enum tracks HuggingFace's normalizer spec and
/// grows as new normalizer types are added there, so adding a variant must
/// not be a breaking change for downstream matchers.
#[non_exhaustive]
pub enum NormOp {
    Nfc,
    Nfd,
    Nfkc,
    Nfkd,
    Lowercase,
    StripAccents,
    /// Literal string replacement.
    ReplaceStr {
        from: String,
        to: String,
    },
    /// Regex replacement (compiled).
    ReplaceRegex {
        re: Box<regexr::Regex>,
        to: String,
    },
    Prepend(String),
    Strip {
        left: bool,
        right: bool,
    },
    /// SentencePiece NMT normalization (control-char cleanup).
    Nmt,
    /// SentencePiece precompiled charsmap.
    Precompiled(Precompiled),
}

impl NormOp {
    /// Apply this step, borrowing through when it leaves the text unchanged.
    ///
    /// Every arm that can answer "nothing to do" cheaply does so before
    /// allocating. The normalization forms use the crate's quick checks, which
    /// answer `Yes` outright for ASCII and for the already-normalized text that
    /// makes up nearly all real input; the rest test for the one thing they
    /// would change.
    fn apply<'a>(&self, s: Cow<'a, str>) -> Cow<'a, str> {
        match self {
            NormOp::Nfc => match is_nfc_quick(s.chars()) {
                IsNormalized::Yes => s,
                _ => Cow::Owned(s.nfc().collect()),
            },
            NormOp::Nfd => match is_nfd_quick(s.chars()) {
                IsNormalized::Yes => s,
                _ => Cow::Owned(s.nfd().collect()),
            },
            NormOp::Nfkc => match is_nfkc_quick(s.chars()) {
                IsNormalized::Yes => s,
                _ => Cow::Owned(s.nfkc().collect()),
            },
            NormOp::Nfkd => match is_nfkd_quick(s.chars()) {
                IsNormalized::Yes => s,
                _ => Cow::Owned(s.nfkd().collect()),
            },
            NormOp::Lowercase => match needs_lowercasing(&s) {
                true => Cow::Owned(lowercase(&s)),
                false => s,
            },
            // Matches HuggingFace's `StripAccents`: drop only Nonspacing_Mark (Mn)
            // characters. Decomposition (NFD/NFKD) is a separate, preceding op in
            // the sequence. Filtering all combining marks would corrupt spacing
            // marks (Mc) used by scripts like Devanagari/Thai.
            NormOp::StripAccents => {
                let has_mark = s
                    .chars()
                    .any(|c| get_general_category(c) == GeneralCategory::NonspacingMark);
                match has_mark {
                    true => Cow::Owned(
                        s.chars()
                            .filter(|c| get_general_category(*c) != GeneralCategory::NonspacingMark)
                            .collect(),
                    ),
                    false => s,
                }
            }
            NormOp::ReplaceStr { from, to } => {
                if from.is_empty() || !s.contains(from.as_str()) {
                    s
                } else {
                    Cow::Owned(s.replace(from.as_str(), to))
                }
            }
            NormOp::ReplaceRegex { re, to } => match re.replace_all(&s, to) {
                Cow::Borrowed(_) => s,
                Cow::Owned(out) => Cow::Owned(out),
            },
            NormOp::Prepend(p) => {
                let mut out = p.clone();
                out.push_str(&s);
                Cow::Owned(out)
            }
            NormOp::Strip { left, right } => {
                let mut t = s.as_ref();
                if *left {
                    t = t.trim_start();
                }
                if *right {
                    t = t.trim_end();
                }
                if t.len() == s.len() {
                    s
                } else {
                    Cow::Owned(t.to_string())
                }
            }
            NormOp::Nmt => {
                let out = nmt(&s);
                if out == s.as_ref() {
                    s
                } else {
                    Cow::Owned(out)
                }
            }
            NormOp::Precompiled(pc) => {
                let out = pc.normalize(&s);
                if out == s.as_ref() {
                    s
                } else {
                    Cow::Owned(out)
                }
            }
        }
    }

    /// Compile a regex `Replace` step. Returns `None` if the pattern fails to
    /// build — the caller must surface that as an error rather than substitute a
    /// different (literal) operation, which would silently mis-normalize.
    pub fn replace_regex(pattern: &str, to: String) -> Option<Self> {
        RegexBuilder::new(pattern)
            .build()
            .ok()
            .map(|re| NormOp::ReplaceRegex {
                re: Box::new(re),
                to,
            })
    }
}

/// An ordered list of normalization steps.
#[derive(Default)]
pub struct Normalizer {
    ops: Vec<NormOp>,
}

impl Normalizer {
    pub fn new(ops: Vec<NormOp>) -> Self {
        Self { ops }
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Apply every step in order.
    ///
    /// Borrows straight through when nothing changes. That is the common case
    /// and it used to cost a full copy of the input regardless: a `tokenizer.json`
    /// declaring `NFC` — Qwen's does, most HuggingFace files do — paid one
    /// allocation plus a whole normalization pass on every encode, even though
    /// text that is already NFC (all ASCII, and nearly all real input) comes out
    /// byte-identical.
    pub fn normalize<'a>(&self, text: &'a str) -> Cow<'a, str> {
        let mut s = Cow::Borrowed(text);
        for op in &self.ops {
            s = op.apply(s);
        }
        s
    }
}

/// SentencePiece NMT normalization: drop a set of control characters and map
/// several whitespace/format characters to a plain space.
fn nmt(s: &str) -> String {
    s.chars()
        .filter_map(|c| match c as u32 {
            // Removed entirely.
            0x0001..=0x0008 | 0x000B | 0x000E..=0x001F | 0x007F | 0x008F | 0x009F => None,
            // Normalized to space.
            0x0009
            | 0x000A
            | 0x000C
            | 0x000D
            | 0x1680
            | 0x200B..=0x200F
            | 0x2028
            | 0x2029
            | 0x205F
            | 0x3000 => Some(' '),
            _ => Some(c),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Lowercasing is per character, so a word-final `Σ` is `σ` and not the
    /// contextual `ς` that `str::to_lowercase` produces. Every word-final
    /// capital sigma in a Greek corpus is a different id, so the two are pinned
    /// against each other rather than only the expected answer being asserted.
    #[test]
    fn lowercase_does_not_apply_the_greek_final_sigma_rule() {
        assert_eq!(lowercase("ΟΠΩΣ"), "οπωσ");
        assert_eq!("ΟΠΩΣ".to_lowercase(), "οπως");
        assert_eq!(
            NormOp::Lowercase.apply(std::borrow::Cow::Borrowed("ΟΠΩΣ")),
            "οπωσ"
        );
    }

    /// The fast path must not skip a titlecase letter: `ǅ` is `Lt`, which the
    /// Unicode `Uppercase` property does not cover.
    #[test]
    fn lowercase_fast_path_sees_titlecase() {
        assert!(needs_lowercasing("ǅ"));
        assert!(!needs_lowercasing("already lower"));
        assert_eq!(
            NormOp::Lowercase.apply(std::borrow::Cow::Borrowed("ǅ")),
            "ǆ"
        );
    }

    #[test]
    fn nfc_and_nfd_roundtrip() {
        // "é" as base 'e' + combining acute (U+0301).
        let decomposed = "e\u{0301}";
        assert_eq!(
            NormOp::Nfc.apply(std::borrow::Cow::Owned(decomposed.to_string())),
            "é"
        );
        assert_eq!(
            NormOp::Nfd.apply(std::borrow::Cow::Owned("é".to_string())),
            decomposed
        );
    }

    #[test]
    fn nfkc_folds_compatibility_chars() {
        // U+FB01 LATIN SMALL LIGATURE FI -> "fi".
        assert_eq!(
            NormOp::Nfkc.apply(std::borrow::Cow::Owned("\u{FB01}".to_string())),
            "fi"
        );
    }

    #[test]
    fn lowercase_lowercases() {
        assert_eq!(
            NormOp::Lowercase.apply(std::borrow::Cow::Owned("HeLLo".to_string())),
            "hello"
        );
    }

    #[test]
    fn strip_accents_drops_nonspacing_marks() {
        // 'a' + combining macron (U+0304, Mn) -> the mark is removed.
        assert_eq!(
            NormOp::StripAccents.apply(std::borrow::Cow::Owned("a\u{0304}".to_string())),
            "a"
        );
    }

    #[test]
    fn strip_accents_preserves_spacing_marks() {
        // Devanagari "की": KA (U+0915) + vowel sign II (U+0940, a spacing Mc mark)
        // must be preserved, unlike a blanket combining-mark filter.
        let s = "\u{0915}\u{0940}";
        assert_eq!(
            NormOp::StripAccents.apply(std::borrow::Cow::Owned(s.to_string())),
            s
        );
    }

    #[test]
    fn replace_str_is_literal_and_handles_empty() {
        let op = NormOp::ReplaceStr {
            from: " ".to_string(),
            to: "_".to_string(),
        };
        assert_eq!(
            op.apply(std::borrow::Cow::Owned("a b c".to_string())),
            "a_b_c"
        );
        // Empty `from` is a no-op (avoids infinite/unexpected expansion).
        let empty = NormOp::ReplaceStr {
            from: String::new(),
            to: "x".to_string(),
        };
        assert_eq!(empty.apply(std::borrow::Cow::Owned("ab".to_string())), "ab");
    }

    #[test]
    fn replace_regex_compiles_and_applies() {
        let op = NormOp::replace_regex(r"\s+", "_".to_string()).expect("regex builds");
        assert_eq!(
            op.apply(std::borrow::Cow::Owned("a   b".to_string())),
            "a_b"
        );
    }

    #[test]
    fn prepend_prefixes() {
        assert_eq!(
            NormOp::Prepend("▁".to_string()).apply(std::borrow::Cow::Owned("hi".to_string())),
            "▁hi"
        );
    }

    #[test]
    fn strip_respects_sides() {
        let both = NormOp::Strip {
            left: true,
            right: true,
        };
        assert_eq!(
            both.apply(std::borrow::Cow::Owned("  hi  ".to_string())),
            "hi"
        );
        let left_only = NormOp::Strip {
            left: true,
            right: false,
        };
        assert_eq!(
            left_only.apply(std::borrow::Cow::Owned("  hi  ".to_string())),
            "hi  "
        );
    }

    #[test]
    fn nmt_removes_controls_and_maps_whitespace() {
        // U+0008 is removed; tab (U+0009) becomes a space.
        assert_eq!(nmt("a\u{0008}b\tc"), "ab c");
    }

    #[test]
    fn pipeline_applies_in_order() {
        // NFKD must run before StripAccents so accents decompose into removable
        // Mn marks; running them out of order would leave precomposed accents.
        let norm = Normalizer::new(vec![NormOp::Nfkd, NormOp::StripAccents, NormOp::Lowercase]);
        assert_eq!(norm.normalize("ÀÉÎ"), "aei");
    }

    #[test]
    fn empty_normalizer_is_identity() {
        let norm = Normalizer::default();
        assert!(norm.is_empty());
        assert_eq!(norm.normalize("unchanged"), "unchanged");
    }
}
