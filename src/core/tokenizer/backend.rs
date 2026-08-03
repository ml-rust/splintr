use regexr::{Regex as RegexrRegex, RegexBuilder};

#[cfg(feature = "pcre2")]
use pcre2::bytes::Regex as Pcre2Regex;

use super::error::TokenizerError;

/// Regex backend enum for switching between regexr (default) and PCRE2 (optional)
pub(super) enum RegexBackend {
    Regexr(Box<RegexrRegex>),
    #[cfg(feature = "pcre2")]
    Pcre2(Pcre2Regex),
}

impl RegexBackend {
    /// Find all matches in the given text, returning (start, end) byte offsets
    pub(super) fn find_iter<'a>(&'a self, text: &'a str) -> Vec<(usize, usize)> {
        match self {
            RegexBackend::Regexr(regex) => regex
                .find_iter(text)
                .map(|m| (m.start(), m.end()))
                .collect(),
            #[cfg(feature = "pcre2")]
            RegexBackend::Pcre2(regex) => regex
                .find_iter(text.as_bytes())
                .filter_map(|m| m.ok())
                .map(|m| (m.start(), m.end()))
                .collect(),
        }
    }
}

/// Compile one pre-tokenizer expression on the selected backend.
///
/// The single place that knows how each backend is configured (PCRE2 needs
/// `utf`+`ucp` to give `\p{…}` and `\s` their Unicode meanings), so a chained
/// pre-tokenizer's later passes are compiled exactly like its first one.
pub(super) fn compile_pattern(
    pattern: &str,
    use_pcre2: bool,
    use_jit: bool,
) -> Result<RegexBackend, TokenizerError> {
    #[cfg(feature = "pcre2")]
    if use_pcre2 {
        let mut regex_builder = pcre2::bytes::RegexBuilder::new();
        if use_jit {
            regex_builder.jit_if_available(true);
        }
        regex_builder.utf(true);
        regex_builder.ucp(true);
        return Ok(RegexBackend::Pcre2(regex_builder.build(pattern)?));
    }
    #[cfg(not(feature = "pcre2"))]
    let _ = use_pcre2;

    let regex = RegexBuilder::new(pattern).jit(use_jit).build()?;
    Ok(RegexBackend::Regexr(Box::new(regex)))
}

/// One pass of llama.cpp's `unicode_regex_split` (`unicode.cpp:990-1088`).
///
/// Every span produced by the previous pass is re-matched **independently** —
/// the expression sees only that span's text, so `^`, `$` and lookaround treat
/// the span's edges as the edges of the world — and each span is replaced by the
/// ordered sequence of its matches AND the gaps between them
/// (`unicode_regex_split_stl`, `unicode.cpp:486-505`: an unmatched prefix is
/// emitted before every match, and any unmatched tail after the last one).
///
/// So a later expression NEVER re-examines the whole text and never merges
/// anything: it can only cut existing pieces finer. Nothing is exempted from a
/// later pass either — a gap left by pass 1 is an ordinary span that pass 2
/// subdivides like any other. That is why a list of N expressions is not the
/// alternation of those N expressions.
pub(super) fn subdivide(
    re: &RegexBackend,
    text: &str,
    spans: &[(usize, usize)],
) -> Vec<(usize, usize)> {
    let mut out = Vec::with_capacity(spans.len());
    for &(span_start, span_end) in spans {
        let Some(piece) = text.get(span_start..span_end) else {
            continue;
        };
        let mut last = 0;
        for (start, end) in re.find_iter(piece) {
            if start > last {
                out.push((span_start + last, span_start + start));
            }
            // A zero-width match would emit an empty piece upstream too; it
            // carries no bytes, so it is simply not recorded.
            if end > start {
                out.push((span_start + start, span_start + end));
            }
            last = end;
        }
        if last < piece.len() {
            out.push((span_start + last, span_end));
        }
    }
    out
}
