use std::io;

/// Anything that can go wrong while training or writing a vocabulary.
#[derive(Debug, thiserror::Error)]
pub enum TrainError {
    /// A vocabulary smaller than its own seed alphabet was asked for. The
    /// alphabet is the floor — every byte the corpus contains needs an id or the
    /// vocabulary cannot spell the text it was trained on.
    #[error("vocab_size {requested} is below the {alphabet} seed tokens the corpus requires")]
    VocabTooSmall { requested: usize, alphabet: usize },

    /// Training was asked to run over no text at all.
    #[error("the corpus is empty: no words were fed to the trainer")]
    EmptyCorpus,

    /// A piece was handed to a writer that cannot represent it — a
    /// `tokenizer.json` vocabulary is keyed by string, so a piece that is not
    /// valid UTF-8 has no key.
    #[error("token {id} is not valid UTF-8 and cannot be written to a tokenizer.json vocabulary")]
    NotUtf8 { id: u32 },

    #[error("failed to build the pre-tokenizer: {0}")]
    PreTokenizer(#[from] splintr::TokenizerError),

    #[error(transparent)]
    Io(#[from] io::Error),

    #[error(transparent)]
    Json(#[from] serde_json::Error),
}
