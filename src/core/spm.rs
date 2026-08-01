//! SentencePiece **BPE** tokenizer (llama.cpp `SPM` / `tokenizer.ggml.model = "llama"`).
//!
//! This is *not* the Unigram algorithm in [`sentencepiece`](super::sentencepiece).
//! The two share a vocabulary format and a word-boundary marker but disagree on
//! what the per-token score means, and therefore on how to segment:
//!
//! | | Unigram (`t5`) | SPM-BPE (`llama`) |
//! |---|---|---|
//! | score | log-probability | **merge rank** (higher = merge earlier) |
//! | algorithm | Viterbi, maximise the *sum* over a segmentation | greedily merge the best-scoring adjacent pair, repeatedly |
//!
//! Running Viterbi over merge-rank scores is not a small inaccuracy — it
//! inverts the objective. In Gemma's vocabulary, scores run roughly `-id`, so
//! short early-id fragments outscore whole words: maximising the sum picks
//! `▁h` + `el` + `lo` (total −431) over `▁hello` (−28610), and
//! `▁sourdough` shatters into `▁s|ou|rd|ou|gh`. The model never saw those
//! pieces during training, so every embedding is computed from out-of-
//! distribution input while the pipeline reports success.
//!
//! The merge loop below reproduces llama.cpp's `llm_tokenizer_spm`, which
//! recovers `▁hello` and `▁sourdough` from the same vocabulary.

use rustc_hash::FxHashMap;
use std::collections::BinaryHeap;
use thiserror::Error;

use super::tokenize::{Tokenize, TokenizeError};

/// The SentencePiece word-boundary marker (U+2581 LOWER ONE EIGHTH BLOCK).
const WORD_BOUNDARY: &str = "\u{2581}";

#[derive(Error, Debug)]
pub enum SpmError {
    #[error("Empty vocabulary")]
    EmptyVocab,
    #[error("Scores length ({scores}) does not match tokens length ({tokens})")]
    ScoreMismatch { scores: usize, tokens: usize },
}

/// One symbol in the working sequence: a slice of the normalized text plus
/// intrusive doubly-linked-list pointers so a merge is O(1).
#[derive(Clone, Copy)]
struct Symbol {
    prev: i64,
    next: i64,
    start: usize,
    len: usize,
}

/// A candidate merge of two adjacent symbols.
///
/// Ordered by score, then by *lower* left index, so `BinaryHeap`'s max-pop
/// yields the highest-scoring merge and breaks ties left-to-right — matching
/// llama.cpp's comparator.
struct Bigram {
    left: i64,
    right: i64,
    score: f32,
    /// Byte length of the merged text, used to detect a stale queue entry.
    len: usize,
}

impl PartialEq for Bigram {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.left == other.left
    }
}
impl Eq for Bigram {}

impl Ord for Bigram {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            // Lower left index wins a tie, so reverse it for a max-heap.
            .then_with(|| other.left.cmp(&self.left))
    }
}
impl PartialOrd for Bigram {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// SentencePiece BPE tokenizer for `tokenizer.ggml.model = "llama"` vocabularies.
pub struct SpmTokenizer {
    token_to_id: FxHashMap<String, u32>,
    id_to_token: Vec<String>,
    /// Merge ranks. Higher merges earlier.
    scores: Vec<f32>,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    unk_id: Option<u32>,
    /// Ids of the 256 `<0xNN>` byte tokens, when the vocab provides them.
    byte_ids: Option<Box<[u32; 256]>>,
    /// Prepend a word boundary to the input (SentencePiece `add_dummy_prefix`).
    add_prefix_space: bool,
}

impl SpmTokenizer {
    /// Build from a GGUF-style vocabulary.
    ///
    /// `scores` are merge ranks, not log-probabilities. When empty, token id
    /// order is used as the merge order (lower id merges earlier), which is the
    /// convention these vocabularies already follow.
    pub fn new(
        tokens: Vec<String>,
        scores: Vec<f32>,
        bos_token_id: Option<u32>,
        eos_token_id: Option<u32>,
    ) -> Result<Self, SpmError> {
        if tokens.is_empty() {
            return Err(SpmError::EmptyVocab);
        }
        let scores = if scores.is_empty() {
            (0..tokens.len()).map(|i| -(i as f32)).collect()
        } else if scores.len() != tokens.len() {
            return Err(SpmError::ScoreMismatch {
                scores: scores.len(),
                tokens: tokens.len(),
            });
        } else {
            scores
        };

        let mut token_to_id = FxHashMap::default();
        token_to_id.reserve(tokens.len());
        for (id, token) in tokens.iter().enumerate() {
            // First id wins: a duplicated piece must resolve to the canonical
            // (lowest) id, matching llama.cpp's vocab construction.
            token_to_id.entry(token.clone()).or_insert(id as u32);
        }

        let unk_id = token_to_id
            .get("<unk>")
            .or_else(|| token_to_id.get("<UNK>"))
            .copied();

        // Byte fallback is all-or-nothing: a partial `<0xNN>` set cannot encode
        // arbitrary input, so fall back to <unk> instead of emitting a hole.
        let mut byte_ids = [0u32; 256];
        let mut complete = true;
        for (b, slot) in byte_ids.iter_mut().enumerate() {
            match token_to_id.get(&format!("<0x{b:02X}>")) {
                Some(&id) => *slot = id,
                None => {
                    complete = false;
                    break;
                }
            }
        }

        Ok(Self {
            token_to_id,
            id_to_token: tokens,
            scores,
            bos_token_id,
            eos_token_id,
            unk_id,
            byte_ids: complete.then(|| Box::new(byte_ids)),
            add_prefix_space: true,
        })
    }

    /// Set SentencePiece `add_dummy_prefix` (GGUF `tokenizer.ggml.add_space_prefix`).
    ///
    /// Defaults to true. Gemma sets it false; prepending a boundary anyway
    /// shifts the very first piece of every input to a different token.
    pub fn with_prefix_space(mut self, add_prefix_space: bool) -> Self {
        self.add_prefix_space = add_prefix_space;
        self
    }

    /// Set the BOS / EOS ids the vocabulary defines (GGUF `add_bos_token` /
    /// `add_eos_token` resolve to `None` here when disabled).
    ///
    /// `encode` never emits them: they are reported through
    /// [`bos_token_id`](Self::bos_token_id) / [`eos_token_id`](Self::eos_token_id)
    /// so the special-token policy can place them.
    pub fn with_special_ids(mut self, bos: Option<u32>, eos: Option<u32>) -> Self {
        self.bos_token_id = bos;
        self.eos_token_id = eos;
        self
    }

    /// SentencePiece normalization: spaces become the boundary marker, with an
    /// optional leading marker.
    fn normalize(&self, text: &str) -> String {
        let mut out = String::with_capacity(text.len() + WORD_BOUNDARY.len());
        if self.add_prefix_space && !text.starts_with(' ') {
            out.push_str(WORD_BOUNDARY);
        }
        for ch in text.chars() {
            if ch == ' ' {
                out.push_str(WORD_BOUNDARY);
            } else {
                out.push(ch);
            }
        }
        out
    }

    /// Merge adjacent symbols, best score first, until nothing merges.
    fn merge(&self, text: &str) -> Vec<Symbol> {
        let mut symbols: Vec<Symbol> = Vec::new();
        for (offset, ch) in text.char_indices() {
            let len = ch.len_utf8();
            let index = symbols.len() as i64;
            symbols.push(Symbol {
                prev: index - 1,
                next: if offset + len == text.len() {
                    -1
                } else {
                    index + 1
                },
                start: offset,
                len,
            });
        }
        if symbols.is_empty() {
            return symbols;
        }

        let mut queue: BinaryHeap<Bigram> = BinaryHeap::new();
        let push = |queue: &mut BinaryHeap<Bigram>, left: i64, right: i64, syms: &[Symbol]| {
            if left < 0 || right < 0 {
                return;
            }
            let (l, r) = (&syms[left as usize], &syms[right as usize]);
            let merged = &text[l.start..r.start + r.len];
            if let Some(&id) = self.token_to_id.get(merged) {
                queue.push(Bigram {
                    left,
                    right,
                    score: self.scores[id as usize],
                    len: merged.len(),
                });
            }
        };

        for i in 1..symbols.len() as i64 {
            push(&mut queue, i - 1, i, &symbols);
        }

        while let Some(bigram) = queue.pop() {
            let (li, ri) = (bigram.left as usize, bigram.right as usize);
            let (left, right) = (symbols[li], symbols[ri]);

            // Either side already absorbed into another merge → stale entry.
            if left.len == 0 || right.len == 0 || left.len + right.len != bigram.len {
                continue;
            }

            // Absorb the right symbol into the left one and unlink it.
            symbols[li].len = left.len + right.len;
            symbols[ri].len = 0;
            symbols[li].next = right.next;
            if right.next >= 0 {
                symbols[right.next as usize].prev = bigram.left;
            }

            push(&mut queue, symbols[li].prev, bigram.left, &symbols);
            push(&mut queue, bigram.left, symbols[li].next, &symbols);
        }

        symbols.into_iter().filter(|s| s.len > 0).collect()
    }

    /// Emit ids for one final symbol, falling back to bytes then `<unk>`.
    fn emit(&self, piece: &str, out: &mut Vec<u32>) {
        if let Some(&id) = self.token_to_id.get(piece) {
            out.push(id);
            return;
        }
        match &self.byte_ids {
            Some(byte_ids) => out.extend(piece.bytes().map(|b| byte_ids[b as usize])),
            None => {
                if let Some(unk) = self.unk_id {
                    out.push(unk);
                }
            }
        }
    }

    /// Encode without any added-token handling.
    ///
    /// Content tokens only: boundary tokens are the
    /// [`SpecialPolicy`](crate::core::SpecialPolicy)'s to add, so that a caller
    /// wrapping two sequences does not get a stray BOS in the middle.
    fn encode_ordinary(&self, text: &str) -> Vec<u32> {
        let normalized = self.normalize(text);
        let mut out = Vec::new();
        for symbol in self.merge(&normalized) {
            self.emit(
                &normalized[symbol.start..symbol.start + symbol.len],
                &mut out,
            );
        }
        out
    }

    /// The raw surface string of a token id (`▁` boundaries and `<0xNN>` byte
    /// tokens are kept as spelled). Used to drive a declared decoder pipeline.
    pub fn token_surface(&self, id: u32) -> Option<String> {
        self.id_to_token.get(id as usize).cloned()
    }

    /// The beginning-of-sequence token id, when the vocabulary defines one.
    pub fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    /// The end-of-sequence token id, when the vocabulary defines one.
    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }
}

impl Tokenize for SpmTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.encode_ordinary(text)
    }

    fn decode(&self, ids: &[u32]) -> Result<String, TokenizeError> {
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            let piece = self
                .id_to_token
                .get(id as usize)
                .ok_or(TokenizeError::InvalidTokenId(id))?;
            // `<0xNN>` byte tokens decode to the raw byte, not to their literal
            // spelling; a multi-byte character is split across several of them
            // and only reassembles as bytes.
            let byte = piece
                .strip_prefix("<0x")
                .and_then(|rest| rest.strip_suffix('>'))
                .and_then(|hex| u8::from_str_radix(hex, 16).ok());
            match byte {
                Some(b) => bytes.push(b),
                None => bytes.extend(piece.replace(WORD_BOUNDARY, " ").as_bytes()),
            }
        }
        String::from_utf8(bytes).map_err(|_| TokenizeError::Utf8Error)
    }

    fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A vocabulary shaped like the ones this tokenizer is for: scores are merge
    /// ranks (`-id`), and it carries the *intermediate* merge results a real BPE
    /// vocabulary contains, not just the fragments and the finished words.
    ///
    /// The ids are arranged so that maximising the summed score would prefer the
    /// cheap fragments — `▁h`(-4) + `el`(-5) + `lo`(-6) = -15 beats
    /// `▁hello`(-24) — which is exactly the trap that shatters Gemma's words
    /// under Viterbi. Merging by best adjacent pair must still reach `▁hello`.
    ///
    /// `▁hell` is deliberately absent so one test can observe a merge chain that
    /// legitimately stops short.
    fn rank_scored_vocab() -> (Vec<String>, Vec<f32>) {
        let tokens: Vec<String> = [
            "<pad>", "<eos>", "<bos>", "<unk>", // 0..3
            "▁h", "el", "lo", "▁w", "or", "ld", // 4..9   fragments, best scores
            "h", "e", "l", "o", "w", "r", "d", "▁", // 10..17 single chars
            "ll", "▁he", // 18..19  intermediates
            "▁hel", "▁wor", // 20..21  intermediates
            "▁hello", "▁world", // 22..23  whole words, worst scores
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
        let scores = (0..tokens.len()).map(|i| -(i as f32)).collect();
        (tokens, scores)
    }

    fn tok() -> SpmTokenizer {
        let (tokens, scores) = rank_scored_vocab();
        SpmTokenizer::new(tokens, scores, None, None).unwrap()
    }

    fn pieces(t: &SpmTokenizer, text: &str) -> Vec<String> {
        t.encode(text)
            .into_iter()
            .map(|id| t.id_to_token[id as usize].clone())
            .collect()
    }

    /// The defect this tokenizer exists to prevent: maximising the sum of
    /// rank-scores prefers many cheap fragments over the whole word. Merging by
    /// best adjacent pair must recover the word.
    #[test]
    fn whole_words_win_over_cheaper_fragment_sequences() {
        let t = tok();
        assert_eq!(pieces(&t, "hello"), vec!["▁hello"]);
        assert_eq!(pieces(&t, "hello world"), vec!["▁hello", "▁world"]);
    }

    #[test]
    fn a_merge_chain_that_stops_short_keeps_every_character() {
        let t = tok();
        // "▁hell" is absent, so merging halts at "▁hel" + "l". It must not
        // invent a token, drop a character, or fall through to <unk>.
        assert_eq!(pieces(&t, "hell"), vec!["▁hel", "l"]);
    }

    #[test]
    fn spaces_become_word_boundaries_and_round_trip() {
        let t = tok();
        let ids = t.encode("hello world");
        assert_eq!(t.decode(&ids).unwrap(), " hello world");
    }

    /// `add_space_prefix = false` (Gemma) must not prepend a boundary — doing so
    /// silently changes the first token of every input.
    #[test]
    fn prefix_space_can_be_disabled() {
        let (tokens, scores) = rank_scored_vocab();
        let t = SpmTokenizer::new(tokens, scores, None, None)
            .unwrap()
            .with_prefix_space(false);
        assert_eq!(pieces(&t, "hello"), vec!["h", "el", "lo"]);
    }

    /// Boundary tokens belong to the special-token policy, not to the model: a
    /// tokenizer that adds them itself gives a caller wrapping two sequences a
    /// stray BOS in the middle, and no way to opt out. `encode` stays raw even
    /// when the vocabulary defines both ids, which remain readable.
    #[test]
    fn bos_and_eos_are_reported_but_never_encoded() {
        let (tokens, scores) = rank_scored_vocab();
        let with = SpmTokenizer::new(tokens.clone(), scores.clone(), Some(2), Some(1)).unwrap();
        assert_eq!(pieces(&with, "hello"), vec!["▁hello"]);
        assert_eq!(with.bos_token_id(), Some(2));
        assert_eq!(with.eos_token_id(), Some(1));

        let without = SpmTokenizer::new(tokens, scores, None, None).unwrap();
        assert_eq!(pieces(&without, "hello"), vec!["▁hello"]);
        assert_eq!(without.bos_token_id(), None);
        assert_eq!(without.eos_token_id(), None);
    }

    /// Unknown characters must become byte tokens when the vocab has the full
    /// `<0xNN>` set, so arbitrary input survives a round trip.
    #[test]
    fn unknown_characters_use_byte_fallback() {
        let mut tokens: Vec<String> = vec!["<unk>".into(), "▁".into()];
        for b in 0..=255u32 {
            tokens.push(format!("<0x{b:02X}>"));
        }
        let n = tokens.len();
        let t = SpmTokenizer::new(tokens, (0..n).map(|i| -(i as f32)).collect(), None, None)
            .unwrap()
            .with_prefix_space(false);

        let ids = t.encode("é");
        assert_eq!(ids.len(), 2, "é is two UTF-8 bytes, so two byte tokens");
        assert_eq!(t.decode(&ids).unwrap(), "é");
    }

    /// Without a complete byte set, an unknown character must map to `<unk>`
    /// rather than emitting a partial or empty result.
    #[test]
    fn unknown_characters_without_byte_fallback_use_unk() {
        let tokens: Vec<String> = ["<unk>", "▁", "a"].iter().map(|s| s.to_string()).collect();
        let t = SpmTokenizer::new(tokens, vec![], None, None)
            .unwrap()
            .with_prefix_space(false);
        assert_eq!(pieces(&t, "z"), vec!["<unk>"]);
    }

    /// With the dummy prefix disabled there is nothing to encode, so empty input
    /// must yield no tokens rather than a stray boundary or an <unk>.
    #[test]
    fn empty_input_produces_no_tokens() {
        let (tokens, scores) = rank_scored_vocab();
        let t = SpmTokenizer::new(tokens, scores, None, None)
            .unwrap()
            .with_prefix_space(false);
        assert!(t.encode("").is_empty());
    }

    /// With the dummy prefix enabled, empty input is the boundary marker alone —
    /// SentencePiece's documented behaviour, asserted so it cannot drift.
    #[test]
    fn empty_input_with_prefix_space_is_the_boundary_marker() {
        let t = tok();
        assert_eq!(pieces(&t, ""), vec!["▁"]);
    }

    /// Merging must be deterministic and must not depend on how many equal
    /// scores are in flight.
    #[test]
    fn repeated_words_tokenize_identically() {
        let t = tok();
        assert_eq!(
            pieces(&t, "hello hello hello"),
            vec!["▁hello", "▁hello", "▁hello"]
        );
    }
}
