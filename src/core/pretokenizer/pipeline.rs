use std::borrow::Cow;

use regexr::RegexBuilder;

use super::parse::{gpt2_regex, whitespace_regex};
use super::spec::{PreTokStage, SplitPattern};
use super::stage::Stage;
use crate::core::tokenizer::TokenizerError;

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
        self.split_pieces(text)
            .into_iter()
            .map(Cow::into_owned)
            .collect()
    }

    /// [`PreTokenizer::split`] without materializing a `String` per piece.
    ///
    /// Splitting stages only ever *cut* their input, so their output is a set
    /// of subslices of it and needs no allocation at all. Only `ByteLevel`
    /// rewrites content, and it is the last stage of every pipeline that has
    /// one. So the pipeline runs borrowed for as long as it can and switches to
    /// owned pieces at the first rewriting stage — which for the usual
    /// `Split` + `ByteLevel` shape means one allocation per piece instead of
    /// three (the whole-text seed copy, the split piece, the encoded piece).
    ///
    /// The `add_prefix_space` guard is a literal **space**, matching
    /// `ByteLevel::pre_tokenize`'s own `!normalized.get().starts_with(' ')`:
    /// text opening on any other whitespace still gets the prefix. Measured
    /// against `tokenizers` 0.22.1 on a `ByteLevel { add_prefix_space: true }`
    /// fixture, `"\ta"` pre-tokenizes to `Ġ`/`ĉ`/`a` while `" a"` stays `Ġa`.
    pub(crate) fn split_pieces<'a>(&self, text: &'a str) -> Vec<Cow<'a, str>> {
        // The prefix space is the one input the pieces cannot be subslices of
        // `text` for, so that branch runs the pipeline over a local and lifts
        // whatever comes back to owned. It costs nothing in practice: a prefix
        // space is only ever configured by a ByteLevel or Metaspace node, and
        // ByteLevel makes the pieces owned anyway.
        if self.add_prefix_space && !text.starts_with(' ') {
            let prefixed = format!(" {text}");
            return self
                .run(&prefixed)
                .into_iter()
                .map(|piece| Cow::Owned(piece.into_owned()))
                .collect();
        }
        self.run(text)
    }

    /// The stage pipeline over `text`, with pieces borrowed from it for as long
    /// as the stages allow.
    fn run<'p>(&self, text: &'p str) -> Vec<Cow<'p, str>> {
        let rewrite_at = self
            .compiled
            .iter()
            .position(Stage::rewrites_content)
            .unwrap_or(self.compiled.len());

        // Phase 1: cutting stages, entirely in subslices of `text`.
        let mut cut: Vec<&'p str> = vec![text];
        for stage in &self.compiled[..rewrite_at] {
            let mut next = Vec::with_capacity(cut.len());
            for piece in &cut {
                stage.cut(piece, &mut next);
            }
            cut = next;
        }

        if rewrite_at == self.compiled.len() {
            return cut
                .into_iter()
                .filter(|piece| !piece.is_empty())
                .map(Cow::Borrowed)
                .collect();
        }

        // Phase 2: from the first rewriting stage on, pieces are owned.
        let mut owned: Vec<String> = Vec::with_capacity(cut.len());
        for piece in &cut {
            self.compiled[rewrite_at].apply_owned(piece, &mut owned);
        }
        for stage in &self.compiled[rewrite_at + 1..] {
            let mut next: Vec<String> = Vec::with_capacity(owned.len());
            for piece in &owned {
                stage.apply_owned(piece, &mut next);
            }
            owned = next;
        }

        owned
            .into_iter()
            .filter(|piece| !piece.is_empty())
            .map(Cow::Owned)
            .collect()
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
