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
    pub(super) fn parse(s: Option<&str>) -> Self {
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
/// compiled by [`PreTokenizer::new`](super::PreTokenizer::new), so a caller
/// never has to name a regex type.
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
/// [`PreTokenizer::new`](super::PreTokenizer::new) and stored in the private
/// `Stage` so `apply` never has to convert on the hot path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum Behavior {
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
