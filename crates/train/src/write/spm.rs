//! `.spm` output: `base64(piece) score [type]` per line, id given by position.
//!
//! splintr's own SentencePiece text format, read by `load_spm_vocab`. The piece
//! is base64 so a `▁` or a control character cannot break the line format, and
//! the score is the log-probability the trainer produced.

use std::io::Write;
use std::path::Path;

use base64::engine::general_purpose::STANDARD;
use base64::Engine;

use crate::error::TrainError;
use crate::unigram::UnigramVocab;

/// `USER_DEFINED` in SentencePiece's piece-type enum: matched verbatim, never
/// merged into. What a special token is.
const USER_DEFINED: u32 = 4;

/// `NORMAL`: an ordinary piece, free to take part in segmentation.
const NORMAL: u32 = 1;

/// The vocabulary as `.spm` text.
///
/// Three columns throughout. The type column is what marks the special tokens
/// as `USER_DEFINED`, and writing it is not optional here even though the
/// loader tolerates its absence: a two-column file reads back as all-`NORMAL`,
/// which would let a segmenter merge *into* a special token.
pub fn spm(vocab: &UnigramVocab) -> String {
    let mut out = String::with_capacity(vocab.len() * 24);
    for (id, (piece, score)) in vocab.tokens().iter().zip(vocab.scores()).enumerate() {
        let kind = if id < vocab.special_count() {
            USER_DEFINED
        } else {
            NORMAL
        };
        out.push_str(&STANDARD.encode(piece));
        out.push(' ');
        // The loader parses this as `f32`, so more precision than that would be
        // written and then silently dropped.
        out.push_str(&format!("{}", *score as f32));
        out.push(' ');
        out.push_str(&kind.to_string());
        out.push('\n');
    }
    out
}

/// [`spm`], written to a file.
pub fn spm_file(vocab: &UnigramVocab, path: impl AsRef<Path>) -> Result<(), TrainError> {
    let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
    file.write_all(spm(vocab).as_bytes())?;
    file.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Corpus, UnigramTrainer};

    fn trained() -> UnigramVocab {
        let mut corpus = Corpus::new().with_metaspace();
        for word in ["playing", "played", "player", "running", "runner", "reads"] {
            for _ in 0..8 {
                corpus.feed(word);
            }
        }
        UnigramTrainer::builder()
            .vocab_size(120)
            .special_tokens(["<unk>", "</s>"])
            .min_frequency(1)
            .build()
            .train(corpus.counts())
            .expect("training succeeds")
    }

    #[test]
    fn writes_one_three_column_line_per_token() {
        let vocab = trained();
        let text = spm(&vocab);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), vocab.len());
        for line in &lines {
            assert_eq!(line.split(' ').count(), 3, "not three columns: {line:?}");
        }
    }

    /// The round trip that matters: what is written is what splintr reads back.
    #[test]
    fn splintr_reads_back_every_piece_and_score() {
        let vocab = trained();
        let loaded =
            splintr::core::load_spm_vocab(spm(&vocab).as_bytes()).expect("splintr loads it");
        assert_eq!(loaded.pieces, vocab.tokens());
        for (read, written) in loaded.scores.iter().zip(vocab.scores()) {
            assert_eq!(*read, *written as f32);
        }
    }

    /// Specials come back marked, so a segmenter cannot merge into them.
    #[test]
    fn specials_read_back_as_user_defined() {
        let vocab = trained();
        let loaded = splintr::core::load_spm_vocab(spm(&vocab).as_bytes()).unwrap();
        for (id, user_defined) in loaded.user_defined.iter().enumerate() {
            assert_eq!(
                *user_defined,
                id < vocab.special_count(),
                "id {id} ({:?}) has the wrong piece type",
                vocab.tokens()[id]
            );
        }
    }

    /// A `▁`-marked piece survives the format — the reason pieces are base64
    /// rather than written literally.
    #[test]
    fn marked_pieces_survive_the_format() {
        let vocab = trained();
        let loaded = splintr::core::load_spm_vocab(spm(&vocab).as_bytes()).unwrap();
        assert!(
            loaded.pieces.iter().any(|p| p.starts_with('\u{2581}')),
            "the corpus was marked, so some piece must carry the marker"
        );
    }

    #[test]
    fn writes_to_a_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.spm");
        let vocab = trained();
        spm_file(&vocab, &path).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), spm(&vocab));
    }
}
