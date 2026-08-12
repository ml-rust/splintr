//! `.tiktoken` output: `base64(token bytes) rank`, one token per line.

use std::io::Write;
use std::path::Path;

use base64::engine::general_purpose::STANDARD;
use base64::Engine;

use crate::error::TrainError;
use crate::vocab::TrainedVocab;

/// The vocabulary as `.tiktoken` text.
///
/// One line per piece, lowest rank first, and the rank is the piece's id — which
/// for a trained vocabulary is its merge order, so the file states the merge
/// priority without a separate merge list.
///
/// Special tokens are **not** written. A `.tiktoken` file carries ranks and
/// nothing else; specials are supplied to `Tokenizer::from_file` alongside the
/// pattern, exactly as they are for every bundled vocabulary.
pub fn tiktoken(vocab: &TrainedVocab) -> String {
    let mut out = String::with_capacity(vocab.pieces().len() * 12);
    for (id, piece) in vocab.pieces().iter().enumerate() {
        out.push_str(&STANDARD.encode(piece));
        out.push(' ');
        out.push_str(&id.to_string());
        out.push('\n');
    }
    out
}

/// [`tiktoken`], written to a file.
pub fn tiktoken_file(vocab: &TrainedVocab, path: impl AsRef<Path>) -> Result<(), TrainError> {
    let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
    file.write_all(tiktoken(vocab).as_bytes())?;
    file.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BpeTrainer, WordCounts};

    fn trained() -> TrainedVocab {
        let counts: WordCounts = [
            (b"lower".to_vec(), 5u64),
            (b"lowest".to_vec(), 4),
            (b"newer".to_vec(), 6),
        ]
        .into_iter()
        .collect();
        BpeTrainer::builder()
            .vocab_size(300)
            .build()
            .train(&counts)
            .expect("training succeeds")
    }

    #[test]
    fn writes_one_line_per_piece_in_rank_order() {
        let vocab = trained();
        let text = tiktoken(&vocab);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), vocab.pieces().len());
        for (id, line) in lines.iter().enumerate() {
            let (encoded, rank) = line.split_once(' ').expect("every line has a rank");
            assert_eq!(rank.parse::<usize>().unwrap(), id);
            assert_eq!(STANDARD.decode(encoded).unwrap(), vocab.pieces()[id]);
        }
    }

    /// The format's own round trip: what is written decodes back to the exact
    /// bytes, including pieces that are not valid UTF-8.
    #[test]
    fn every_piece_survives_base64() {
        let vocab = trained();
        for (id, piece) in vocab.pieces().iter().enumerate() {
            let line = tiktoken(&vocab).lines().nth(id).unwrap().to_string();
            let encoded = line.split_once(' ').unwrap().0.to_string();
            assert_eq!(&STANDARD.decode(encoded).unwrap(), piece);
        }
    }

    #[test]
    fn writes_to_a_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.tiktoken");
        let vocab = trained();
        tiktoken_file(&vocab, &path).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), tiktoken(&vocab));
    }
}
