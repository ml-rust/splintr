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
    #[error("Aho-Corasick build error: {0}")]
    AhoCorasickError(#[from] aho_corasick::BuildError),
    #[error("PCRE2 feature not enabled. Compile with --features pcre2")]
    Pcre2NotEnabled,
    #[error("Unknown pretrained model: {0}")]
    UnknownPretrained(String),
    #[error("Pre-tokenizer pattern list is empty")]
    EmptyPatternList,
    #[error(
        "regex backend options apply only to the byte-level BPE backend; this tokenizer is {0}"
    )]
    NotBpeBackend(&'static str),
}
