use thiserror::Error;

use crate::core::vocab::VocabError;

#[derive(Error, Debug)]
pub enum TokenizerError {
    #[error("Regex compilation error (regexr): {0}")]
    RegexrError(#[from] regexr::Error),
    #[cfg(feature = "pcre2")]
    #[error("Regex compilation error (PCRE2): {0}")]
    Pcre2Error(#[from] pcre2::Error),
    #[error("Vocabulary error: {0}")]
    VocabError(#[from] VocabError),
    #[error("SentencePiece BPE error: {0}")]
    SpmError(#[from] crate::core::spm::SpmError),
    #[error("Decoding error: invalid UTF-8")]
    Utf8Error,
    #[error("Decoding error: token ID {0} is not in the vocabulary")]
    InvalidTokenId(u32),
    #[error("Aho-Corasick build error: {0}")]
    AhoCorasickError(#[from] aho_corasick::BuildError),
    #[error("PCRE2 feature not enabled. Compile with --features pcre2")]
    Pcre2NotEnabled,
    #[error("Unknown pretrained model: {0}")]
    UnknownPretrained(String),
    /// The vocabulary name is known, but its data was not compiled in.
    ///
    /// Distinct from [`UnknownPretrained`](Self::UnknownPretrained) so a
    /// stripped build is diagnosable: the name is not a typo and the fix is a
    /// cargo feature, not a different name.
    #[error(
        "Pretrained vocabulary {0} was not bundled in this build. Compile with --features {1}"
    )]
    VocabNotBundled(&'static str, &'static str),
    #[error("Pre-tokenizer pattern list is empty")]
    EmptyPatternList,
    #[error(
        "regex backend options apply only to the byte-level BPE backend; this tokenizer is {0}"
    )]
    NotBpeBackend(&'static str),
}
