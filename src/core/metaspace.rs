//! SentencePiece metaspace escaping, shared by both SentencePiece backends.
//!
//! SentencePiece does not treat a space as a delimiter to discard. A space is
//! mapped to the word-boundary marker `▁` (U+2581), which is a real vocabulary
//! piece: `" "` alone tokenizes to the `▁` piece, and a trailing space is a
//! trailing `▁`. Splitting the input on whitespace and throwing the whitespace
//! away silently deletes those pieces.
//!
//! The two backends escape with the same rule but differ in *when* the leading
//! marker is added, so [`Prefix`] names the two conventions rather than
//! hard-coding either:
//!
//! - [`Prefix::Always`] — llama.cpp's `llm_tokenizer_spm`, which prepends the
//!   dummy prefix unconditionally (a leading space is never treated as "already
//!   have one"). Used by [`SpmTokenizer`](super::spm::SpmTokenizer).
//! - [`Prefix::WhenAbsent`] — HuggingFace's `Metaspace` pre-tokenizer, which
//!   prepends only `if !normalized.starts_with(replacement)`. Used by
//!   [`SentencePieceTokenizer`](super::sentencepiece::SentencePieceTokenizer),
//!   whose Unigram references (HF `tokenizers`, SentencePiece itself) both
//!   behave that way: `" a "` is `▁a` + `▁`, not `▁` + `▁a` + `▁`.

/// The SentencePiece word-boundary marker (U+2581 LOWER ONE EIGHTH BLOCK).
pub const WORD_BOUNDARY: &str = "\u{2581}";

/// Whether — and on what condition — a leading word-boundary marker is added.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Prefix {
    /// Never prepend (SentencePiece `add_dummy_prefix = false`, HF
    /// `prepend_scheme = "never"`).
    None,
    /// Always prepend, even when the escaped text already starts with a marker.
    Always,
    /// Prepend only when the escaped text does not already start with a marker.
    WhenAbsent,
}

/// Escape `text` the SentencePiece way: every space becomes [`WORD_BOUNDARY`],
/// with an optional leading marker and optional run-merging.
///
/// `collapse_runs` is SentencePiece's `remove_extra_whitespaces` (GGUF
/// `tokenizer.ggml.remove_extra_whitespaces`): a run of spaces becomes one
/// marker instead of one per space. Note that only `' '` is folded — mapping
/// other whitespace (`\n`, `\t`, …) to a space is a *normalizer* step
/// (`NormOp::Nmt`, or SentencePiece's precompiled charsmap), which runs before
/// this and is what both reference implementations rely on.
pub fn escape(text: &str, prefix: Prefix, collapse_runs: bool) -> String {
    let mut out = String::new();
    escape_into(text, prefix, collapse_runs, &mut out);
    out
}

/// [`escape`] into a caller-owned buffer, which the encode path reuses across
/// words rather than allocating one per word of the document.
///
/// The leading marker is decided before the pass rather than inserted at the
/// front afterwards — `insert_str(0, ..)` shifts everything already written, so
/// the old form copied each word twice.
pub fn escape_into(text: &str, prefix: Prefix, collapse_runs: bool, out: &mut String) {
    out.clear();
    out.reserve(text.len() + WORD_BOUNDARY.len());

    // The escaped text begins with a marker exactly when `text` begins with a
    // space (which becomes one) or with a marker already, so the test that used
    // to read the output can be asked of the input.
    let prepend = match prefix {
        Prefix::None => false,
        Prefix::Always => true,
        Prefix::WhenAbsent => !(text.starts_with(' ') || text.starts_with(WORD_BOUNDARY)),
    };
    if prepend {
        out.push_str(WORD_BOUNDARY);
    }

    let mut prev_space = false;
    for ch in text.chars() {
        if ch == ' ' {
            if collapse_runs && prev_space {
                continue;
            }
            prev_space = true;
            out.push_str(WORD_BOUNDARY);
        } else {
            prev_space = false;
            out.push(ch);
        }
    }
}

/// Run `f` on each segment a Unigram model segments independently: one per
/// word-boundary marker, each starting at its marker and running up to the next.
///
/// This is HuggingFace's `Metaspace { split: true }` with `MergedWithNext`
/// behavior. Text before the first marker (when nothing was prepended) is a
/// segment of its own, and a marker with nothing after it — a trailing space —
/// is a segment of its own too, which is precisely the piece the old
/// whitespace-splitting pre-tokenizer discarded.
///
/// Streamed and byte-scanned rather than collected and searched. `match_indices`
/// runs the general two-way string searcher over every word;
/// the marker is three known bytes whose lead byte cannot occur anywhere else
/// in valid UTF-8, so a byte scan finds it and cannot be fooled.
pub fn for_each_segment<'a>(escaped: &'a str, mut f: impl FnMut(&'a str)) {
    let marker = WORD_BOUNDARY.as_bytes();
    let bytes = escaped.as_bytes();
    let mut start = 0;
    let mut at = 0;
    while at + marker.len() <= bytes.len() {
        if bytes[at..at + marker.len()] != *marker {
            at += 1;
            continue;
        }
        if let Some(segment) = escaped.get(start..at).filter(|s| !s.is_empty()) {
            f(segment);
        }
        start = at;
        at += marker.len();
    }
    if let Some(rest) = escaped.get(start..).filter(|s| !s.is_empty()) {
        f(rest);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The segments of `escaped`, collected. Production streams them; the
    /// assertions below read better against a vector.
    fn segments(escaped: &str) -> Vec<&str> {
        let mut out = Vec::new();
        for_each_segment(escaped, |segment| out.push(segment));
        out
    }

    #[test]
    fn spaces_become_markers_and_are_never_dropped() {
        assert_eq!(escape("a b", Prefix::None, false), "a▁b");
        // The defect this module exists to prevent: a standalone or trailing
        // space is a piece, not a delimiter to throw away.
        assert_eq!(escape(" ", Prefix::None, false), "▁");
        assert_eq!(escape("a ", Prefix::None, false), "a▁");
    }

    #[test]
    fn prefix_conventions_differ_on_an_already_marked_start() {
        // llama.cpp SPM: unconditional, so a leading space yields two markers.
        assert_eq!(escape(" a", Prefix::Always, false), "▁▁a");
        assert_eq!(escape("a", Prefix::Always, false), "▁a");
        // HF Metaspace: only when absent.
        assert_eq!(escape(" a", Prefix::WhenAbsent, false), "▁a");
        assert_eq!(escape("a", Prefix::WhenAbsent, false), "▁a");
        // Empty input still gets the marker; the callers guard empty input
        // before escaping.
        assert_eq!(escape("", Prefix::WhenAbsent, false), "▁");
        assert_eq!(escape("", Prefix::None, false), "");
    }

    #[test]
    fn collapsing_runs_merges_only_spaces() {
        assert_eq!(escape("a   b", Prefix::None, true), "a▁b");
        assert_eq!(escape("   ", Prefix::WhenAbsent, true), "▁");
        assert_eq!(escape("   ", Prefix::WhenAbsent, false), "▁▁▁");
        // Other whitespace is a normalizer's job, not this one's.
        assert_eq!(escape("a\n\nb", Prefix::None, true), "a\n\nb");
    }

    #[test]
    fn segments_start_at_each_marker() {
        assert_eq!(segments("▁a▁b"), vec!["▁a", "▁b"]);
        assert_eq!(segments("▁a▁"), vec!["▁a", "▁"]);
        assert_eq!(segments("▁"), vec!["▁"]);
        assert_eq!(segments("▁▁a"), vec!["▁", "▁a"]);
        assert_eq!(segments("a▁b"), vec!["a", "▁b"]);
        assert!(segments("").is_empty());
        assert_eq!(segments("abc"), vec!["abc"]);
    }
}
