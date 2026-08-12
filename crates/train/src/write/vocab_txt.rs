//! `vocab.txt` output: one token per line, id given by position.
//!
//! The BERT-family WordPiece format. It carries nothing but the tokens — no
//! scores, no merges, no configuration — which is all a longest-match segmenter
//! needs.

use std::io::Write;
use std::path::Path;

use crate::error::TrainError;
use crate::wordpiece::WordPieceVocab;

/// The vocabulary as `vocab.txt` text.
///
/// # Errors
/// [`TrainError::NotUtf8`] if a token contains a newline, which the format has
/// no way to escape — a line *is* a token, so one carrying a break would read
/// back as two.
pub fn vocab_txt(vocab: &WordPieceVocab) -> Result<String, TrainError> {
    let mut out = String::with_capacity(vocab.len() * 8);
    for (id, token) in vocab.tokens().iter().enumerate() {
        if token.contains('\n') || token.contains('\r') {
            return Err(TrainError::NotUtf8 { id: id as u32 });
        }
        out.push_str(token);
        out.push('\n');
    }
    Ok(out)
}

/// [`vocab_txt`], written to a file.
pub fn vocab_txt_file(vocab: &WordPieceVocab, path: impl AsRef<Path>) -> Result<(), TrainError> {
    let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
    file.write_all(vocab_txt(vocab)?.as_bytes())?;
    file.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Corpus, WordPieceTrainer};
    use splintr::Tokenize;

    fn trained() -> WordPieceVocab {
        let mut corpus = Corpus::new();
        for word in ["playing", "played", "player", "running", "runner", "reads"] {
            for _ in 0..8 {
                corpus.feed(word);
            }
        }
        WordPieceTrainer::builder()
            .vocab_size(120)
            .special_tokens(["[UNK]", "[CLS]", "[SEP]"])
            .build()
            .train(corpus.counts())
            .expect("training succeeds")
    }

    #[test]
    fn writes_one_token_per_line_in_id_order() {
        let vocab = trained();
        let text = vocab_txt(&vocab).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines, vocab.tokens());
    }

    /// The round trip: the written list drives splintr's own segmenter and its
    /// ids are the positions the file gave them.
    #[test]
    fn the_written_list_drives_the_segmenter() {
        let vocab = trained();
        let text = vocab_txt(&vocab).unwrap();
        let tokens: Vec<String> = text.lines().map(str::to_string).collect();
        let segmenter = splintr::WordPieceTokenizer::new(tokens, 0, 512, false);
        for word in ["playing", "runner"] {
            let ids = segmenter.encode(word);
            assert!(!ids.is_empty());
            assert!(!ids.contains(&0), "{word} fell back to [UNK]: {ids:?}");
        }
    }

    /// A token holding a line break would read back as two tokens and shift
    /// every id after it, so it is refused rather than written.
    #[test]
    fn refuses_a_token_containing_a_newline() {
        let vocab = WordPieceVocab::from_parts(vec!["ok".into(), "bad\ntoken".into()], 0);
        assert!(matches!(
            vocab_txt(&vocab).unwrap_err(),
            TrainError::NotUtf8 { id: 1 }
        ));
    }

    #[test]
    fn writes_to_a_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vocab.txt");
        let vocab = trained();
        vocab_txt_file(&vocab, &path).unwrap();
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            vocab_txt(&vocab).unwrap()
        );
    }
}
