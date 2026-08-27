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
//!
//! [`PrependScheme`] answers the other half of the question — which *splits* of
//! one sequence are offered a marker at all — and only added tokens, which cut
//! a sequence into more than one split, make its arms differ.

/// The SentencePiece word-boundary marker (U+2581 LOWER ONE EIGHTH BLOCK).
pub const WORD_BOUNDARY: &str = "\u{2581}";

/// Which splits of one sequence carry a leading word-boundary marker —
/// HuggingFace's `Metaspace.prepend_scheme`.
///
/// A different question from [`Prefix`], and the two are read together.
/// [`Prefix`] says *whether* a marker goes in front of a split that is offered
/// one; this says *which* splits are offered one at all. Added tokens cut a
/// sequence into several splits, and that is where the arms disagree:
/// `"<s>a"` is `<s>`, `▁a` under [`Always`](Self::Always) and `<s>`, `a` under
/// [`First`](Self::First) — a different first content token, not merely a
/// missing marker.
///
/// No `Default`: the two backends that read this were measured under different
/// arms and each states its own — the BPE metaspace fork under
/// [`First`](Self::First), the Unigram fork under [`Always`](Self::Always) —
/// so a defaulted value would silently give one of them the other's.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PrependScheme {
    /// No split is marked (`"never"`, or the legacy
    /// `add_prefix_space: false`).
    Never,
    /// Only the split that opens the sequence (`"first"`). A gap that follows
    /// an added token is not it, and neither is anything after that.
    /// mistral-7b-v0.3's `tokenizer.json` states this.
    First,
    /// Every split, a gap following an added token included (`"always"`, the
    /// legacy `add_prefix_space: true`, and HuggingFace's default when the node
    /// states neither field).
    Always,
}

impl PrependScheme {
    /// The scheme a `tokenizer.json` `prepend_scheme` string names, or `None`
    /// when the file names something this crate cannot represent.
    ///
    /// `None` is for the caller to refuse the file with. Reading an unknown
    /// value as "prepend" (or as "never") loads a tokenizer that is plausible
    /// and wrong, which is the failure this crate refuses everywhere else.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "never" => Some(Self::Never),
            "first" => Some(Self::First),
            "always" => Some(Self::Always),
            _ => None,
        }
    }

    /// Whether a split carries a marker, given whether it opens the sequence.
    pub fn marks(self, is_first: bool) -> bool {
        match self {
            Self::Never => false,
            Self::First => is_first,
            Self::Always => true,
        }
    }
}

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

    /// A `▁`-marked BPE vocabulary carrying the whitespace-run piece `▁▁`, so
    /// the two conventions produce visibly different pieces rather than merely
    /// different counts. `split: false` and `prepend_scheme: "always"` are
    /// VoxCPM2's shape; `merges` is empty, so every input below is one
    /// whole-chunk lookup and the ids name the piece directly.
    const HF_METASPACE_JSON: &str = r#"{
        "added_tokens": [],
        "pre_tokenizer": {"type": "Metaspace", "replacement": "▁",
            "prepend_scheme": "always", "split": false},
        "model": {"type": "BPE", "unk_token": "<unk>",
            "vocab": {"<unk>": 0, "▁double": 1, "▁▁double": 2,
                "▁": 3, "▁▁": 4, "double": 5},
            "merges": []}
    }"#;

    /// A `tokenizer.json` loaded through the real loader, encoding without the
    /// post-processor so the ids are the content pieces alone.
    fn hf(json: &str) -> crate::core::AnyTokenizer {
        crate::core::hf_json::from_json_bytes(json.as_bytes()).expect("the document loads")
    }

    /// A `▁`-marked SPM-BPE vocabulary reaching `▁hello` through the
    /// intermediates a real merge list carries, scored as merge ranks (`-id`).
    /// `▁▁` is deliberately absent: llama.cpp's second marker has nothing to
    /// merge into here, so it stays a piece of its own and is countable.
    fn spm() -> crate::core::spm::SpmTokenizer {
        let tokens: Vec<String> = [
            "<unk>", "▁", "h", "e", "l", "o", "▁h", "▁he", "▁hel", "▁hell", "▁hello",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
        let scores = (0..tokens.len()).map(|i| -(i as f32)).collect();
        crate::core::spm::SpmTokenizer::new(tokens, scores, None, None)
            .expect("the vocabulary is non-empty")
    }

    /// The pieces `spm` produces, by name.
    fn spm_pieces(text: &str) -> Vec<String> {
        let tok = spm();
        crate::core::tokenize::Tokenize::encode(&tok, text)
            .into_iter()
            .filter_map(|id| tok.token_surface(id))
            .collect()
    }

    /// The two ecosystems disagree about a leading space, and both readings are
    /// measured. They must stay apart: converging them moves the first token of
    /// every affected sequence while every id stays in range and decodes back
    /// to the original string, so nothing downstream reports it.
    ///
    /// HuggingFace `tokenizers` 0.22.1, `pre_tokenizer.pre_tokenize_str`:
    ///
    /// | text | result |
    /// |---|---|
    /// | `"double"` | `▁double` |
    /// | `" double"` | `▁double` |
    /// | `"  double"` | `▁▁double` |
    /// | `" "` | `▁` |
    /// | `"  "` | `▁▁` |
    ///
    /// llama.cpp `llm_tokenizer_spm` over `ggml-vocab-llama-spm.gguf`:
    /// `"Hello"` is `[15043]` and `" Hello"` is `[29871, 15043]` — one extra
    /// standalone boundary piece, never a swallowed space.
    #[test]
    fn the_two_prefix_conventions_stay_apart_on_a_leading_space() {
        let tok = hf(HF_METASPACE_JSON);
        // HuggingFace: the marker goes on only when the escaped text does not
        // already open with one, so one space and no space agree.
        assert_eq!(tok.encode_raw("double"), vec![1], "▁double");
        assert_eq!(tok.encode_raw(" double"), vec![1], "▁double");
        // Two spaces escape to two markers, which is the `▁▁double` piece —
        // one marker more than `" double"`, not two more.
        assert_eq!(tok.encode_raw("  double"), vec![2], "▁▁double");
        assert_eq!(tok.encode_raw(" "), vec![3], "▁");
        assert_eq!(tok.encode_raw("  "), vec![4], "▁▁");

        // llama.cpp: the marker goes on without looking, so a leading space is
        // a second boundary rather than the one that was going to be added.
        assert_eq!(spm_pieces("hello"), vec!["▁hello"]);
        assert_eq!(spm_pieces(" hello"), vec!["▁", "▁hello"]);
        assert_eq!(spm_pieces(" "), vec!["▁", "▁"]);
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
