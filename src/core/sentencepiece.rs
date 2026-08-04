//! SentencePiece-compatible Unigram tokenizer.
//!
//! Implements the Unigram **Viterbi** algorithm — the maximum total-score
//! segmentation — matching SentencePiece / HuggingFace `tokenizers` (T5, Albert,
//! XLNet, …), with metaspace pre-tokenization, byte-fallback, an ordered
//! normalizer pipeline, and added-token matching.

use rustc_hash::FxHashSet;
use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;
use thiserror::Error;

use super::policy::{PolicyError, SpecialMode};
use super::streaming::{DecodeState, StreamingDecoder};
use super::tokenize::{token_bytes_of, token_text_of, TokenizeError};

#[derive(Error, Debug)]
pub enum SentencePieceError {
    #[error("Empty vocabulary")]
    EmptyVocab,
    #[error("Scores length ({scores}) does not match tokens length ({tokens})")]
    ScoreMismatch { scores: usize, tokens: usize },
    #[error("Decoding error: token ID {0} out of range")]
    InvalidTokenId(u32),
    #[error("Failed to build added-token matcher: {0}")]
    AddedTokensError(#[from] aho_corasick::BuildError),
}

/// SentencePiece-compatible unigram tokenizer.
///
/// Accepts a raw vocabulary (token strings, scores, special token IDs) and
/// performs Viterbi maximum-score segmentation (true SentencePiece Unigram, not
/// greedy) with SentencePiece word boundary markers (▁ U+2581) and byte fallback.
///
/// # Example
///
/// ```
/// use splintr::SentencePieceTokenizer;
///
/// let tokens = vec!["▁Hello".to_string(), "▁world".to_string(), "H".to_string()];
/// let scores = vec![0.0; 3];
/// let tok = SentencePieceTokenizer::new(tokens, scores, None, 2).unwrap();
/// ```
pub struct SentencePieceTokenizer {
    /// Token string -> ID mapping
    token_to_id: HashMap<String, u32>,
    /// ID -> Token string mapping. Behind an `Arc` so decoding — whole-sequence
    /// and streaming alike — can share the piece table rather than copy a
    /// vocabulary-sized vector per decoder.
    id_to_token: Arc<Vec<String>>,
    /// Per-token Unigram scores (log-probs); Viterbi maximizes their sum over the
    /// chosen segmentation.
    ///
    /// `f64`, not `f32`, because the reference implementations are: HuggingFace
    /// `tokenizers` stores `Vec<(String, f64)>` and accumulates `Node::score` /
    /// `Node::backtrace_score` in `f64`, and a `tokenizer.json` score is a JSON
    /// double. Narrowing to `f32` perturbs partial path sums by ~1e-7, which is
    /// enough to reorder two segmentations whose exact scores are equal — see
    /// [`viterbi_piece`](Self::viterbi_piece).
    scores: Vec<f64>,
    /// BOS token ID
    bos_token_id: Option<u32>,
    /// EOS token ID
    eos_token_id: u32,
    /// `<unk>` token ID, auto-detected from the vocab (for OOV in Viterbi).
    unk_id: Option<u32>,
    /// Whether the vocab carries `<0xNN>` byte tokens (byte-fallback for OOV).
    byte_fallback: bool,
    /// Longest token length in chars (bounds the Viterbi inner loop).
    max_piece_chars: usize,
    /// Minimum token score (basis for the unknown-piece penalty).
    min_score: f64,
    /// Ordered normalizer pipeline applied before pre-tokenization.
    normalizer: super::normalizer::Normalizer,
    /// Metaspace `add_prefix_space`: mark the start of the input with `▁` when
    /// it is not already marked.
    add_prefix_space: bool,
    /// SentencePiece `remove_extra_whitespaces`: a run of spaces escapes to a
    /// single `▁` rather than one per space.
    remove_extra_whitespaces: bool,
    /// Added tokens recognized in the input (HF matches these during encoding).
    added: Option<super::added::AddedTokens>,
    /// Ids of `special=true` added tokens dropped on decode (HF default).
    special_decode: rustc_hash::FxHashSet<u32>,
}

impl SentencePieceTokenizer {
    /// Create a tokenizer from raw vocabulary data.
    ///
    /// # Arguments
    /// * `tokens` - Token strings, indexed by token ID
    /// * `scores` - Per-token Unigram score (log-prob) summed and maximized by Viterbi. If empty, defaults to all zeros (uniform).
    /// * `bos_token_id` - Optional beginning-of-sequence token ID
    /// * `eos_token_id` - End-of-sequence token ID
    pub fn new(
        tokens: Vec<String>,
        scores: Vec<f64>,
        bos_token_id: Option<u32>,
        eos_token_id: u32,
    ) -> Result<Self, SentencePieceError> {
        if tokens.is_empty() {
            return Err(SentencePieceError::EmptyVocab);
        }

        let scores = if scores.is_empty() {
            vec![0.0; tokens.len()]
        } else if scores.len() != tokens.len() {
            return Err(SentencePieceError::ScoreMismatch {
                scores: scores.len(),
                tokens: tokens.len(),
            });
        } else {
            scores
        };

        let mut token_to_id = HashMap::with_capacity(tokens.len());
        for (id, token) in tokens.iter().enumerate() {
            token_to_id.insert(token.clone(), id as u32);
        }

        let unk_id = token_to_id
            .get("<unk>")
            .or_else(|| token_to_id.get("<UNK>"))
            .copied();
        let byte_fallback = token_to_id.contains_key("<0x00>");
        let max_piece_chars = tokens.iter().map(|t| t.chars().count()).max().unwrap_or(1);
        let min_score = scores
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
            .min(0.0);

        Ok(Self {
            token_to_id,
            id_to_token: Arc::new(tokens),
            scores,
            bos_token_id,
            eos_token_id,
            unk_id,
            byte_fallback,
            max_piece_chars,
            min_score,
            normalizer: super::normalizer::Normalizer::default(),
            add_prefix_space: true,
            remove_extra_whitespaces: false,
            added: None,
            special_decode: rustc_hash::FxHashSet::default(),
        })
    }

    /// Attach added tokens to recognize in the input during encoding.
    ///
    /// Takes anything convertible into an [`AddedTokenSet`](super::added::AddedTokenSet),
    /// so a caller with no `lstrip`/`rstrip` flags to declare (GGUF, a bundled
    /// vocabulary, a test) can still pass a plain name→id map.
    pub fn with_added_tokens(
        mut self,
        tokens: impl Into<super::added::AddedTokenSet>,
    ) -> Result<Self, SentencePieceError> {
        self.added = super::added::AddedTokens::new(&tokens.into())?;
        Ok(self)
    }

    /// Set ids of `special=true` added tokens to drop on decode (HF default).
    pub fn with_special_decode_ids(mut self, ids: rustc_hash::FxHashSet<u32>) -> Self {
        self.special_decode = ids;
        self
    }

    /// Attach an ordered normalizer pipeline (applied before pre-tokenization).
    /// Returns `self` for chaining.
    pub fn with_normalizer(mut self, normalizer: super::normalizer::Normalizer) -> Self {
        self.normalizer = normalizer;
        self
    }

    /// Set Metaspace `add_prefix_space` (whether the first word gets a leading
    /// `▁`). Defaults to true. Returns `self` for chaining.
    ///
    /// Matches HuggingFace's `Metaspace`: the marker is added only when the
    /// escaped text does not already start with one, so a leading space in the
    /// input is *the* prefix rather than getting a second marker in front of it.
    pub fn with_prefix_space(mut self, add_prefix_space: bool) -> Self {
        self.add_prefix_space = add_prefix_space;
        self
    }

    /// Set SentencePiece `remove_extra_whitespaces` (GGUF
    /// `tokenizer.ggml.remove_extra_whitespaces`): whether a run of spaces
    /// collapses to a single `▁`. Defaults to false. Returns `self` for chaining.
    ///
    /// False is the right default for a HuggingFace `tokenizer.json`, which
    /// declares the collapse as a normalizer step instead (XLM-R and friends
    /// carry `Replace{" {2,}" → " "}` after the precompiled charsmap), and
    /// applying it twice would be harmless but applying it when the file never
    /// asked would not.
    pub fn with_remove_extra_whitespaces(mut self, remove_extra_whitespaces: bool) -> Self {
        self.remove_extra_whitespaces = remove_extra_whitespaces;
        self
    }

    /// Apply the configured normalizer pipeline to an input string.
    fn normalize(&self, text: &str) -> String {
        if self.normalizer.is_empty() {
            text.to_string()
        } else {
            self.normalizer.normalize(text)
        }
    }

    /// Encode text to token IDs using the Unigram **Viterbi** algorithm — the
    /// maximum total-score segmentation, matching SentencePiece / HuggingFace
    /// `tokenizers` (not a greedy longest-match).
    ///
    /// Never emits BOS/EOS: they are reported through
    /// [`bos_token_id`](Self::bos_token_id) / [`eos_token_id`](Self::eos_token_id)
    /// so the special-token policy can place them. Follows the SentencePiece
    /// convention: the input is `▁`-prefixed and spaces become `▁`. Characters
    /// that no token covers fall back to `<0xNN>` byte tokens (if the vocab has
    /// them) or `<unk>`.
    ///
    /// Recognizes added tokens in the input first (when configured), matching
    /// HuggingFace.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        super::added::AddedTokens::dispatch(&self.added, text, |gap| self.encode_ordinary(gap))
    }

    /// Encode text to token IDs under an explicit [`SpecialMode`], governing
    /// whether the added tokens attached via
    /// [`with_added_tokens`](Self::with_added_tokens) are matched in the input
    /// text. Never emits BOS/EOS — see [`encode`](Self::encode); boundary
    /// tokens are [`SpecialPolicy`](crate::core::SpecialPolicy)'s to add via
    /// `AnyTokenizer::encode_with`.
    pub fn encode_with(&self, text: &str, mode: &SpecialMode<'_>) -> Result<Vec<u32>, PolicyError> {
        super::added::AddedTokens::dispatch_with_mode(&self.added, text, mode, |gap| {
            self.encode_ordinary(gap)
        })
    }

    /// Encode without added-token matching (pure Unigram Viterbi). Never emits
    /// BOS/EOS — see [`encode`](Self::encode).
    pub fn encode_ordinary(&self, text: &str) -> Vec<u32> {
        // Empty input has nothing to mark a boundary *of*: HuggingFace and
        // SentencePiece both return no ids. The guard belongs here rather than
        // in the escaping, whose leading marker is correct for every non-empty
        // input — and here rather than in `encode`, so that attaching an
        // added-token matcher (which skips the gap encoder for "") cannot change
        // the answer.
        if text.is_empty() {
            return Vec::new();
        }
        let normalized = self.normalize(text);
        // A normalizer can empty a non-empty input too (a `Strip` over pure
        // whitespace). HF's `prepend` is a no-op on an empty string, so no
        // marker is manufactured there either.
        if normalized.is_empty() {
            return Vec::new();
        }

        // SentencePiece pre-tokenization: spaces become `▁` pieces (they are
        // vocabulary entries, not delimiters to discard) and the text is cut
        // before each marker. Each segment is then Viterbi-segmented
        // independently.
        let escaped = super::metaspace::escape(
            &normalized,
            if self.add_prefix_space {
                super::metaspace::Prefix::WhenAbsent
            } else {
                super::metaspace::Prefix::None
            },
            self.remove_extra_whitespaces,
        );

        let mut tokens = Vec::new();
        let mut chars: Vec<char> = Vec::new();
        for segment in super::metaspace::segments(&escaped) {
            chars.clear();
            chars.extend(segment.chars());
            self.viterbi_piece(&chars, &mut tokens);
        }
        tokens
    }

    /// Append the maximum-score Unigram segmentation of `chars` to `tokens`.
    ///
    /// The lattice sweep mirrors HuggingFace `tokenizers`'
    /// `unigram::Lattice::viterbi` exactly, and the correspondence is load-bearing
    /// on two points:
    ///
    /// * **Candidate order and the strict `>`.** HF relaxes each position over
    ///   `end_nodes[pos]` in *insertion* order — `populate_nodes` walks
    ///   `begin_pos` ascending and the trie yields pieces in ascending length, so
    ///   the incoming edges of a position arrive sorted by start position, with
    ///   the `<unk>` edge last among those from the same start. It keeps the first
    ///   of two equal-scoring predecessors (`if best_node.is_none() || score >
    ///   best_score`). The loops below enumerate `start` ascending, then `end`
    ///   ascending with the unknown-char edge appended after the known ones, and
    ///   likewise update on strictly-greater — so an exact tie resolves to the
    ///   same edge.
    /// * **`f64` accumulation.** HF carries `backtrace_score` in `f64`. Doing the
    ///   same here is not cosmetic: accumulating in `f32` diverges on real input.
    ///   Measured on `BAAI/bge-m3`, `"、hellohellohello"` has two segmentations
    ///   over an identical piece multiset — `h|ello|hel|loh|ello` and
    ///   `hel|loh|ello|h|ello` — hence an exactly equal total (-54.815895557403564
    ///   in `f64`). In `f64` the tie survives to the comparison and the first
    ///   candidate wins, matching HF; in `f32` the ~1e-7 rounding of the partial
    ///   sums makes the later candidate compare strictly greater and the
    ///   segmentation flips. Replaying this DP in Python over 13k fuzz strings,
    ///   `f64` accumulation reproduced HF on every case and `f32` on all but that
    ///   one family.
    fn viterbi_piece(&self, chars: &[char], tokens: &mut Vec<u32>) {
        let n = chars.len();
        if n == 0 {
            return;
        }

        // Viterbi over the character lattice. `best[i]` = best total score to
        // reach position i; `back[i]` = (start, piece) of the chosen edge into i.
        // A piece is Some(id) for a vocab token, or None for an unknown char.
        let unk_penalty = self.min_score - 10.0; // SentencePiece's kUnkPenalty
        let mut best = vec![f64::NEG_INFINITY; n + 1];
        let mut back: Vec<(usize, Option<u32>)> = vec![(0, None); n + 1];
        best[0] = 0.0;

        let mut buf = String::with_capacity(self.max_piece_chars * 4);
        for start in 0..n {
            if best[start] == f64::NEG_INFINITY {
                continue;
            }
            // Known-token edges starting at `start`.
            buf.clear();
            let max_end = (start + self.max_piece_chars).min(n);
            for end in (start + 1)..=max_end {
                buf.push(chars[end - 1]);
                if let Some(&id) = self.token_to_id.get(&buf) {
                    let cand = best[start] + self.scores.get(id as usize).copied().unwrap_or(0.0);
                    if cand > best[end] {
                        best[end] = cand;
                        back[end] = (start, Some(id));
                    }
                }
            }
            // Unknown single-character edge guarantees the lattice is connected.
            let cand = best[start] + unk_penalty;
            if cand > best[start + 1] {
                best[start + 1] = cand;
                back[start + 1] = (start, None);
            }
        }

        // Backtrack into edges, then emit in forward order.
        let mut edges: Vec<(usize, Option<u32>)> = Vec::new();
        let mut pos = n;
        while pos > 0 {
            let (start, piece) = back[pos];
            edges.push((start, piece));
            pos = start;
        }
        edges.reverse();

        let mut prev_unk = false;
        for (start, piece) in edges {
            match piece {
                Some(id) => {
                    tokens.push(id);
                    prev_unk = false;
                }
                None => {
                    // Unknown char: byte-fallback if available, else <unk>. A run
                    // of consecutive unknown chars collapses to a single <unk>,
                    // matching SentencePiece / HuggingFace.
                    if self.byte_fallback {
                        tokens.extend(self.encode_char_as_bytes(chars[start]));
                        prev_unk = false;
                    } else if let Some(unk) = self.unk_id {
                        if !prev_unk {
                            tokens.push(unk);
                            prev_unk = true;
                        }
                    }
                }
            }
        }
    }

    /// Encode a character as individual byte tokens using `<0xNN>` format.
    ///
    /// Each UTF-8 byte of the character is looked up as a token (e.g., `<0xFF>`).
    /// Bytes not present in the vocabulary are silently skipped.
    fn encode_char_as_bytes(&self, c: char) -> Vec<u32> {
        let mut result = Vec::new();
        let mut buf = [0u8; 4];
        let bytes = c.encode_utf8(&mut buf);

        for b in bytes.as_bytes() {
            let byte_token = format!("<0x{:02X}>", b);
            if let Some(&id) = self.token_to_id.get(&byte_token) {
                result.push(id);
            }
        }

        result
    }

    /// The ids dropped when rendering decoded text.
    ///
    /// Built once per [`decode_state`](Self::decode_state) and consulted by
    /// every decode path through it — whole-sequence and streaming alike — so
    /// none of them can drift on which ids they drop. Holds BOS/EOS, `<unk>`,
    /// and any `special=true` added token (`special_decode`), matching
    /// HuggingFace's default decode (skip_special_tokens=True) and the SPM-BPE
    /// sibling's identical rule. Unknown spans were unrecoverable anyway, so the
    /// `<unk>` surface is dropped rather than rendered.
    fn skipped_on_decode(&self) -> FxHashSet<u32> {
        let mut skip = self.special_decode.clone();
        skip.extend(self.bos_token_id);
        skip.insert(self.eos_token_id);
        skip.extend(self.unk_id);
        skip
    }

    /// This tokenizer's decode configuration, as the streaming decoder sees it.
    ///
    /// Whole-sequence decoding and streaming decoding drive the same
    /// [`DecodeState`] through the same cursor, so the two cannot disagree about
    /// what an id means or about what happens to the text it produces. The four
    /// steps `decode` used to spell out inline are exactly the four knobs here:
    /// the skip set, the id-indexed surfaces, `<0xNN>` parsed off the surface,
    /// and the ▁→space substitution followed by the metaspace-prefix strip.
    ///
    /// The substitution is a *rendering* rule, not a post-op over reassembled
    /// text: only a surface may lose its ▁, never a byte a `<0xNN>` token
    /// produced. Measured with the `sentencepiece` package 0.2.0, `decode` of
    /// the ids spelling `<0xE2>`, `<0x96>`, `<0x81>` is `'▁'` — the literal
    /// character — while `decode` of the `▁` piece is `''`; HuggingFace's
    /// declared chain agrees, running `Replace(▁→" ")` *before* `ByteFallback`.
    /// Only a per-surface substitution can tell those apart.
    ///
    /// Cheap to build — the piece vector is shared with this tokenizer rather
    /// than copied — which is what lets `decode` capture one per call instead of
    /// the tokenizer having to cache one that could go stale.
    fn decode_state(&self) -> DecodeState {
        // Shared with the SPM-BPE backend's identically-shaped decode
        // configuration — see `DecodeState::for_piece_vocab`. The strip looks
        // for `' '`, which is what the rendering substitution has already
        // produced from the metaspace prefix's `▁`, so by the time a post-op
        // runs the space is there to remove. It is listed at all only when a
        // prefix was actually added — with `add_prefix_space` off
        // (prepend_scheme = "never") a genuine leading space must survive.
        DecodeState::for_piece_vocab(
            &self.id_to_token,
            self.skipped_on_decode(),
            self.add_prefix_space,
        )
    }

    /// A [`StreamingDecoder`] configured from this tokenizer.
    ///
    /// The only way to build one for this backend: the skipped ids, the
    /// `<0xNN>` byte-fallback resolution, the ▁ substitution and the
    /// metaspace-prefix strip all come from this tokenizer's configuration, so
    /// the stream cannot be pointed at the wrong kind of vocabulary and always
    /// reproduces [`decode`](Self::decode).
    ///
    /// Cheap to call — the piece vector is shared, not copied — and the result
    /// borrows nothing, so it can be moved into a generation task.
    pub fn streaming_decoder(&self) -> StreamingDecoder {
        StreamingDecoder::new(Arc::new(self.decode_state()))
    }

    /// Decode token IDs to text.
    ///
    /// Skips BOS/EOS/`<unk>` and the declared `special=true` ids — see
    /// the internal `skipped_on_decode` set — and converts ▁ back to
    /// spaces as each surface is rendered.
    ///
    /// Strips the single leading space only when the metaspace pre-tokenizer
    /// prepended one (`add_prefix_space` / `prepend_scheme != "never"`). HF's
    /// Metaspace decoder mirrors its prepend behavior; with prefixing disabled a
    /// genuine leading space must be preserved, not eaten. That strip is
    /// position-dependent — it applies to the sequence, not to each token — so
    /// it is the cursor's `at_start` flag that decides which emission it may
    /// touch, and [`streaming_decoder`](Self::streaming_decoder) therefore
    /// reproduces it across chunk boundaries.
    ///
    /// Errors with [`SentencePieceError::InvalidTokenId`] on an id the
    /// vocabulary does not contain — a distinct thing from the skips above,
    /// which are deliberate.
    ///
    /// The degenerate drive of the streaming cursor: one feed of every id, then
    /// a flush. The *lossy* drive, deliberately: this decode has always
    /// assembled its bytes through `String::from_utf8_lossy`, so bytes that
    /// cannot be valid UTF-8 become U+FFFD here rather than an error — unlike
    /// the SPM-BPE sibling, whose whole-sequence decode is strict. Only the
    /// unknown-id decision differs from [`decode_lossy`](Self::decode_lossy).
    pub fn decode(&self, ids: &[u32]) -> Result<String, SentencePieceError> {
        let state = self.decode_state();
        let mut cursor = state.cursor_with_capacity(ids.len() * 4);

        let mut text = cursor
            .feed(ids, |id| Err(SentencePieceError::InvalidTokenId(id)))?
            .unwrap_or_default();
        text.push_str(&cursor.flush());

        Ok(text)
    }

    /// Decode token IDs to text, skipping invalid IDs.
    ///
    /// The lenient half of the pair, over exactly the loop
    /// [`decode`](Self::decode) drives: same pieces, same skips, same
    /// metaspace-prefix strip, same U+FFFD substitution — only an id the
    /// vocabulary does not contain is treated as something to survive rather
    /// than to report. This method never fails, so `on_unknown` is instantiated
    /// with [`Infallible`], letting the compiler prove the `Err` arm away rather
    /// than a runtime assertion claiming it.
    pub fn decode_lossy(&self, ids: &[u32]) -> String {
        let state = self.decode_state();
        let mut cursor = state.cursor_with_capacity(ids.len() * 4);

        let mut text = match cursor.feed(ids, |_| Ok::<(), Infallible>(())) {
            Ok(text) => text.unwrap_or_default(),
            // `Infallible` has no values, so this match has no arms to write.
            Err(never) => match never {},
        };
        text.push_str(&cursor.flush());

        text
    }

    /// The raw surface string of a token id (with metaspace `▁` and `<0xNN>`
    /// byte-fallback markers intact). Used to drive a configuration-declared
    /// decoder pipeline.
    pub fn token_surface(&self, id: u32) -> Option<String> {
        self.id_to_token.get(id as usize).cloned()
    }

    /// Check if a token is the EOS token.
    pub fn is_eos(&self, token_id: u32) -> bool {
        token_id == self.eos_token_id
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    /// Get EOS token ID.
    pub fn eos_token_id(&self) -> u32 {
        self.eos_token_id
    }

    /// Get BOS token ID.
    pub fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }
}

impl super::tokenize::Tokenize for SentencePieceTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.encode(text)
    }

    fn encode_with(&self, text: &str, mode: &SpecialMode<'_>) -> Result<Vec<u32>, PolicyError> {
        self.encode_with(text, mode)
    }

    fn decode(&self, ids: &[u32]) -> Result<String, super::tokenize::TokenizeError> {
        self.decode(ids)
            .map_err(|e| super::tokenize::TokenizeError::Other(e.to_string()))
    }

    /// Skips ids the vocabulary does not contain — the inherent
    /// [`decode_lossy`](SentencePieceTokenizer::decode_lossy), so the trait and
    /// the type can never disagree about what a sequence decodes to.
    fn decode_lossy(&self, ids: &[u32]) -> String {
        SentencePieceTokenizer::decode_lossy(self, ids)
    }

    /// This backend never refuses to stream — the inherent
    /// [`streaming_decoder`](SentencePieceTokenizer::streaming_decoder),
    /// wrapped in the `Ok` the trait's shape needs for
    /// [`AnyTokenizer`](crate::AnyTokenizer)'s sake.
    fn streaming_decoder(&self) -> Result<StreamingDecoder, TokenizeError> {
        Ok(SentencePieceTokenizer::streaming_decoder(self))
    }

    fn decode_token_bytes(&self, id: u32) -> Result<Vec<u8>, TokenizeError> {
        // Rendered through the very rules `decode` drives, so a per-id answer
        // cannot drift from the sequence it emits.
        let state = self.decode_state();
        token_bytes_of(state.render(), id)
    }

    fn decode_token(&self, id: u32) -> Result<String, TokenizeError> {
        let bytes = <Self as super::tokenize::Tokenize>::decode_token_bytes(self, id)?;
        token_text_of(bytes)
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::metaspace::WORD_BOUNDARY;
    use proptest::prelude::*;

    fn make_tokenizer() -> SentencePieceTokenizer {
        // Minimal vocab: ▁Hello, ▁world, ▁, <0x48> (byte fallback for 'H')
        let tokens = vec![
            "<unk>".to_string(),  // 0
            "<s>".to_string(),    // 1 (BOS)
            "</s>".to_string(),   // 2 (EOS)
            "▁Hello".to_string(), // 3
            "▁world".to_string(), // 4
            "▁".to_string(),      // 5
            "H".to_string(),      // 6
            "e".to_string(),      // 7
            "l".to_string(),      // 8
            "o".to_string(),      // 9
        ];
        let scores = vec![0.0; tokens.len()];
        SentencePieceTokenizer::new(tokens, scores, Some(1), 2).unwrap()
    }

    /// Raw `encode` never inserts BOS/EOS — only `AnyTokenizer`'s
    /// `SpecialPolicy` places boundary tokens. The ids remain readable through
    /// the accessors so a policy can be built from them. Mirrors
    /// `spm::tests::bos_and_eos_are_reported_but_never_encoded`.
    #[test]
    fn bos_and_eos_are_reported_but_never_encoded() {
        let with = make_tokenizer();
        assert_eq!(with.encode("Hello world"), vec![3, 4]);
        assert_eq!(with.bos_token_id(), Some(1));
        assert_eq!(with.eos_token_id(), 2);

        let tokens = vec![
            "<unk>".to_string(),
            "▁Hello".to_string(),
            "▁world".to_string(),
        ];
        let scores = vec![0.0; tokens.len()];
        let without = SentencePieceTokenizer::new(tokens, scores, None, 0).unwrap();
        assert_eq!(without.encode("Hello world"), vec![1, 2]);
        assert_eq!(without.bos_token_id(), None);
    }

    #[test]
    fn test_decode_basic() {
        let tok = make_tokenizer();
        let text = tok.decode(&[1, 3, 4]).unwrap();
        assert_eq!(text, "Hello world");
    }

    #[test]
    fn test_decode_skips_bos_eos() {
        let tok = make_tokenizer();
        let text = tok.decode(&[1, 3, 2]).unwrap();
        assert_eq!(text, "Hello");
    }

    #[test]
    fn decode_preserves_leading_space_when_no_prefix() {
        // With add_prefix_space=false (prepend_scheme="never"), a genuine leading
        // space must survive decode rather than being stripped as a ▁ artifact.
        let with_prefix = make_tokenizer();
        assert_eq!(with_prefix.decode(&[3, 4]).unwrap(), "Hello world");

        let no_prefix = make_tokenizer().with_prefix_space(false);
        assert_eq!(no_prefix.decode(&[3, 4]).unwrap(), " Hello world");
        assert_eq!(no_prefix.decode_lossy(&[3, 4]), " Hello world");
    }

    #[test]
    fn test_roundtrip() {
        let tok = make_tokenizer();
        let ids = tok.encode("Hello world");
        let text = tok.decode(&ids).unwrap();
        assert_eq!(text, "Hello world");
    }

    #[test]
    fn test_vocab_size() {
        let tok = make_tokenizer();
        assert_eq!(tok.vocab_size(), 10);
    }

    #[test]
    fn test_is_eos() {
        let tok = make_tokenizer();
        assert!(tok.is_eos(2));
        assert!(!tok.is_eos(1));
    }

    #[test]
    fn test_empty_scores_defaults() {
        let tokens = vec!["▁a".to_string(), "▁b".to_string()];
        let tok = SentencePieceTokenizer::new(tokens, vec![], None, 1).unwrap();
        assert_eq!(tok.vocab_size(), 2);
    }

    #[test]
    fn test_empty_vocab_errors() {
        let result = SentencePieceTokenizer::new(vec![], vec![], None, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_score_mismatch_errors() {
        let tokens = vec!["a".to_string()];
        let result = SentencePieceTokenizer::new(tokens, vec![1.0, 2.0], None, 0);
        assert!(result.is_err());
    }

    /// A space is a vocabulary piece, not a delimiter: `" "` must encode to the
    /// boundary token rather than to nothing. Reference (HF `tokenizers` on
    /// BAAI/bge-m3, `add_special_tokens=False`): `" "` -> `[6]`, the id of `▁`.
    #[test]
    fn a_standalone_space_encodes_to_the_boundary_piece() {
        let tok = make_tokenizer();
        assert_eq!(tok.encode(" "), vec![5]);
        // Not an artifact of the dummy prefix: it is the space itself, so it
        // survives with prefixing disabled too.
        assert_eq!(
            make_tokenizer().with_prefix_space(false).encode(" "),
            vec![5]
        );
    }

    /// Trailing whitespace is a trailing boundary piece. Reference row:
    /// `"trailing whitespace   "` ends in the `▁` id, where whitespace-splitting
    /// pre-tokenization dropped it entirely.
    #[test]
    fn trailing_whitespace_keeps_its_boundary_piece() {
        let tok = make_tokenizer();
        assert_eq!(tok.encode("Hello world "), vec![3, 4, 5]);
        assert_eq!(tok.encode("Hello "), vec![3, 5]);
    }

    /// `remove_extra_whitespaces` merges a run of spaces into one marker — the
    /// reference collapses `" "`, `"  "` and `"   "` to the same single piece —
    /// while leaving it unset keeps one marker per space.
    #[test]
    fn collapsing_makes_a_run_of_spaces_one_boundary_piece() {
        let collapsing = make_tokenizer().with_remove_extra_whitespaces(true);
        assert_eq!(collapsing.encode("   "), collapsing.encode(" "));
        assert_eq!(collapsing.encode("   "), vec![5]);
        assert_eq!(collapsing.encode("Hello   world"), vec![3, 4]);

        let keeping = make_tokenizer();
        assert_eq!(keeping.encode("   "), vec![5, 5, 5]);
    }

    /// A leading space round-trips: with prefixing disabled it is content, so
    /// decode puts it back; with prefixing enabled it *is* the dummy prefix (HF's
    /// Metaspace does not add a second marker in front of one), so encoding is
    /// identical to the unspaced input and decode removes exactly the one marker
    /// the pre-tokenizer is responsible for.
    #[test]
    fn a_leading_space_round_trips_through_decode() {
        let no_prefix = make_tokenizer().with_prefix_space(false);
        let ids = no_prefix.encode(" Hello");
        assert_eq!(ids, vec![3]);
        assert_eq!(no_prefix.decode(&ids).unwrap(), " Hello");

        let with_prefix = make_tokenizer();
        assert_eq!(with_prefix.encode(" Hello"), with_prefix.encode("Hello"));
        assert_eq!(
            with_prefix.decode(&with_prefix.encode(" Hello")).unwrap(),
            "Hello"
        );
    }

    #[test]
    fn test_encode_empty_string() {
        // Empty input has nothing to mark a boundary *of*, so no pieces are
        // emitted (matching HF and SentencePiece) — and, per the raw-backend
        // invariant, no BOS either, whether or not one is configured.
        let tok = make_tokenizer();
        assert!(tok.encode("").is_empty());

        let tokens = vec!["▁a".to_string(), "▁b".to_string()];
        let tok = SentencePieceTokenizer::new(tokens, vec![], None, 1).unwrap();
        assert!(tok.encode("").is_empty());
    }

    #[test]
    fn test_decode_lossy_skips_invalid_tokens() {
        let tok = make_tokenizer();
        // 999 is out of range, should be skipped
        let text = tok.decode_lossy(&[1, 3, 999, 4]);
        assert_eq!(text, "Hello world");
    }

    #[test]
    fn test_decode_lossy_all_invalid() {
        let tok = make_tokenizer();
        let text = tok.decode_lossy(&[999, 1000, 1001]);
        assert_eq!(text, "");
    }

    #[test]
    fn test_decode_invalid_token_id_errors() {
        let tok = make_tokenizer();
        let result = tok.decode(&[1, 999]);
        assert!(result.is_err());
    }

    /// `decode` and `decode_lossy` must agree on which ids they drop: BOS/EOS,
    /// `<unk>`, and any `special=true` added token. Only invalid ids (which
    /// `decode` errors on and `decode_lossy` skips) and UTF-8 handling may
    /// differ between the two paths.
    #[test]
    fn decode_and_decode_lossy_agree_on_skipped_ids() {
        // id 0 is `<unk>`; mark id 6 ("H") as a `special=true` added token to
        // drop on decode, matching the shape of `decode`'s skip set.
        let mut special_decode = rustc_hash::FxHashSet::default();
        special_decode.insert(6u32);
        let tok = make_tokenizer().with_special_decode_ids(special_decode);

        let ids = [1, 0, 3, 6, 4, 2]; // <s> <unk> ▁Hello H ▁world </s>
        let strict = tok.decode(&ids).unwrap();
        let lossy = tok.decode_lossy(&ids);
        assert_eq!(strict, lossy);
        assert_eq!(strict, "Hello world");
    }

    /// The byte-fallback spelling is now resolved by the shared rendering rule
    /// (`ByteFallbackRule::ParseSurface`) rather than by a parser private to
    /// this backend, so it is pinned through `decode` — where it is observable —
    /// instead of through a helper. Replaces the two `parse_byte_fallback` unit
    /// tests the helper had, which is the only behaviour they can still describe.
    ///
    /// One case genuinely moved with the helper: the shared rule accepts any
    /// hex that fits in a byte, so a piece spelled `<0x1>` is byte 0x01 where the
    /// old two-digit-only parser rendered its literal spelling. `sp.decode`,
    /// llama.cpp and HF's `ByteFallback` step all parse the spelling, and no
    /// SentencePiece vocabulary spells a byte token any way but `<0xNN>`.
    #[test]
    fn byte_fallback_spellings_resolve_to_their_byte_through_decode() {
        let tokens = vec![
            "<unk>".to_string(),  // 0
            "<s>".to_string(),    // 1
            "</s>".to_string(),   // 2
            "<0x0A>".to_string(), // 3
            "<0xFF>".to_string(), // 4
            "<0x00>".to_string(), // 5
            "<0x7F>".to_string(), // 6
            "<0xab>".to_string(), // 7  lowercase hex
            "<0xZZ>".to_string(), // 8  not hex at all
            "<0x0A".to_string(),  // 9  unterminated
            "0x0A>".to_string(),  // 10 no opening marker
            "<>".to_string(),     // 11
        ];
        let scores = vec![0.0; tokens.len()];
        let tok = SentencePieceTokenizer::new(tokens, scores, Some(1), 2)
            .unwrap()
            .with_prefix_space(false);

        assert_eq!(tok.decode(&[3]).unwrap(), "\n");
        assert_eq!(tok.decode(&[5]).unwrap(), "\0");
        assert_eq!(tok.decode(&[6]).unwrap(), "\u{7F}");
        // 0xFF and 0xAB are not valid UTF-8 on their own; this decode is lossy
        // over bytes, so they surface as U+FFFD rather than as an error.
        assert_eq!(tok.decode(&[4]).unwrap(), "\u{FFFD}");
        assert_eq!(tok.decode(&[7]).unwrap(), "\u{FFFD}");
        // Anything that is not a byte spelling is ordinary surface text.
        for (id, surface) in [(8u32, "<0xZZ>"), (9, "<0x0A"), (10, "0x0A>"), (11, "<>")] {
            assert_eq!(tok.decode(&[id]).unwrap(), surface);
        }
    }

    /// Two segmentations over an identical piece multiset score exactly equal, so
    /// the answer is decided by which candidate the lattice keeps — and that in
    /// turn depends on the width the partial sums are accumulated at.
    ///
    /// The vocabulary is the 10 pieces of `BAAI/bge-m3` that the failing lattice
    /// actually uses, carrying their real scores (all exactly representable in
    /// `f32`, so only the *accumulation* width is under test). Expectations come
    /// from HuggingFace `tokenizers` 0.22.1 run on a hand-built Unigram
    /// `tokenizer.json` holding exactly these ten entries: `"、hellohellohello"`
    /// -> `['▁','、','h','ello','hel','loh','ello']`. Accumulating in `f32`
    /// instead yields `['▁','、','hel','loh','ello','h','ello']` — same pieces,
    /// different boundaries, different ids reaching the model.
    #[test]
    fn an_exact_score_tie_resolves_the_way_huggingface_resolves_it() {
        let entries: [(&str, f64); 10] = [
            ("<unk>", 0.0),
            ("▁", -3.9299705028533936),
            ("、", -6.610896110534668),
            ("h", -7.701241970062256),
            ("e", -5.701941967010498),
            ("l", -7.762022495269775),
            ("o", -6.417782306671143),
            ("hel", -11.134947776794434),
            ("ello", -11.696972846984863),
            ("loh", -12.585760116577148),
        ];
        let tok = SentencePieceTokenizer::new(
            entries.iter().map(|(t, _)| t.to_string()).collect(),
            entries.iter().map(|(_, s)| *s).collect(),
            None,
            0,
        )
        .unwrap();

        // ▁ 、 h ello hel loh ello
        assert_eq!(tok.encode("、hellohellohello"), vec![1, 2, 3, 8, 7, 9, 8]);
        // Neighbouring lengths never tie, and must stay exactly as they were.
        assert_eq!(tok.encode("、hello"), vec![1, 2, 3, 8]);
        assert_eq!(tok.encode("、hellohello"), vec![1, 2, 7, 9, 8]);
        assert_eq!(
            tok.encode("、hellohellohellohello"),
            vec![1, 2, 7, 9, 8, 7, 9, 8]
        );
    }

    /// Which of two equal-scoring predecessors a position keeps: the one that
    /// starts *earliest* (the longer piece). HuggingFace relaxes a position over
    /// its incoming edges in insertion order — `begin_pos` ascending — and updates
    /// only on strictly greater, so the first one seen survives.
    ///
    /// Expectations measured on `tokenizers` 0.22.1 with a Unigram
    /// `tokenizer.json` carrying exactly this vocabulary (no pre-tokenizer, hence
    /// `with_prefix_space(false)` here): `"aaa"` -> `a|aa`, `"aaaa"` -> `aa|aa`,
    /// `"aaaaa"` -> `a|aa|aa`, `"aabaa"` -> `a|ab|aa`. In `"aaa"` both `a|aa` and
    /// `aa|a` total -3; the edge from position 1 is enumerated before the one from
    /// position 2, so `a|aa` wins.
    #[test]
    fn equal_scoring_predecessors_resolve_to_the_earliest_start() {
        let entries: [(&str, f64); 6] = [
            ("<unk>", 0.0),
            ("a", -1.0),
            ("aa", -2.0),
            ("b", -1.0),
            ("ab", -2.0),
            ("ba", -2.0),
        ];
        let tok = SentencePieceTokenizer::new(
            entries.iter().map(|(t, _)| t.to_string()).collect(),
            entries.iter().map(|(_, s)| *s).collect(),
            None,
            0,
        )
        .unwrap()
        .with_prefix_space(false);

        assert_eq!(tok.encode("aa"), vec![2]);
        assert_eq!(tok.encode("aaa"), vec![1, 2]);
        assert_eq!(tok.encode("aaaa"), vec![2, 2]);
        assert_eq!(tok.encode("aaaaa"), vec![1, 2, 2]);
        assert_eq!(tok.encode("aab"), vec![1, 4]);
        assert_eq!(tok.encode("baa"), vec![3, 2]);
        assert_eq!(tok.encode("aabaa"), vec![1, 4, 2]);
    }

    #[test]
    fn test_decode_byte_fallback_tokens() {
        // Vocab with byte-fallback tokens for UTF-8 encoding of 'é' (0xC3 0xA9)
        let tokens = vec![
            "<unk>".to_string(),  // 0
            "<s>".to_string(),    // 1
            "</s>".to_string(),   // 2
            "<0xC3>".to_string(), // 3
            "<0xA9>".to_string(), // 4
            "▁hi".to_string(),    // 5
        ];
        let scores = vec![0.0; tokens.len()];
        let tok = SentencePieceTokenizer::new(tokens, scores, Some(1), 2).unwrap();

        // Decode: BOS + "▁hi" + byte(0xC3) + byte(0xA9) = "hié"
        // Leading space from ▁ is stripped (multi-token sequence)
        let text = tok.decode(&[1, 5, 3, 4]).unwrap();
        assert_eq!(text, "hié");
    }

    // =========================================================================
    // Streaming: concat(stream) == decode
    // =========================================================================

    /// A Unigram vocabulary shaped like the real ones: sentinels, ▁-prefixed
    /// words, bare characters, and the **complete** `<0xNN>` byte set, so any
    /// character the pieces do not cover falls back to a run of byte tokens.
    ///
    /// Synthetic on purpose: no bundled vocabulary in this crate loads through
    /// this backend (`from_pretrained` serves BPE and SPM-BPE only), so the
    /// streaming tests below build their own in the style of the tests above
    /// rather than pinning behavior to a file that is not here.
    fn stream_vocab() -> Vec<String> {
        let mut tokens: Vec<String> = [
            "<unk>",
            "<s>",
            "</s>",
            "<pad>", // 0..3  sentinels
            "▁",
            "▁hello",
            "▁world",
            "▁a", // 4..7   boundary pieces
            "h",
            "e",
            "l",
            "o",
            "w",
            "r",
            "d",
            "a", // 8..15  bare characters
            "▁世界",
            "é", // 16..17 multi-byte pieces
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
        for b in 0..=255u32 {
            tokens.push(format!("<0x{b:02X}>"));
        }
        tokens
    }

    /// The tokenizer those pieces make: `<pad>` is a declared `special=true` id,
    /// so the skip set holds a non-sentinel too.
    fn stream_tok() -> SentencePieceTokenizer {
        let tokens = stream_vocab();
        let scores = vec![0.0; tokens.len()];
        SentencePieceTokenizer::new(tokens, scores, Some(1), 2)
            .unwrap()
            .with_special_decode_ids([3u32].into_iter().collect())
    }

    /// The id of a piece, looked up by spelling rather than written down, so a
    /// test says which pieces it means instead of which slots they sit in.
    fn ids_of(tokenizer: &SentencePieceTokenizer, pieces: &[&str]) -> Vec<u32> {
        pieces
            .iter()
            .map(|piece| match tokenizer.token_to_id.get(*piece) {
                Some(&id) => id,
                None => panic!("{piece} is a piece of this vocabulary"),
            })
            .collect()
    }

    /// Texts exercising ASCII, leading spaces (the metaspace-prefix trap),
    /// multi-byte scripts and characters this vocabulary can only spell as runs
    /// of `<0xNN>` byte tokens — every shape that can straddle a chunk boundary.
    const STREAM_TEXTS: &[&str] = &[
        "",
        "hello world",
        " hello world",
        "  two leading spaces",
        "Hello, world! 1234567890",
        "こんにちは世界、これはテストです。",
        "Привет, мир!",
        "🎉🚀 emoji 👨‍👩‍👧‍👦 family",
        "héllo — ünïcode, and é as e\u{0301}",
        "def f(x):\n    return x ** 2  # code",
    ];

    /// Feed `ids` through a streaming decoder in the given chunk sizes and
    /// concatenate every emission plus the final flush.
    fn drive_strict(tokenizer: &SentencePieceTokenizer, ids: &[u32], chunk: usize) -> String {
        let mut decoder = tokenizer.streaming_decoder();
        let mut out = String::new();
        for group in ids.chunks(chunk.max(1)) {
            if let Some(text) = decoder.add_tokens(group).expect("ids are all known") {
                out.push_str(&text);
            }
        }
        out.push_str(&decoder.flush());
        out
    }

    /// Same, one id at a time through the lossy entry point.
    fn drive_lossy(tokenizer: &SentencePieceTokenizer, ids: &[u32]) -> String {
        let mut decoder = tokenizer.streaming_decoder();
        let mut out = String::new();
        for &id in ids {
            if let Some(text) = decoder.add_token_lossy(id) {
                out.push_str(&text);
            }
        }
        out.push_str(&decoder.flush());
        out
    }

    /// The point of the factory: streaming a real encoding reproduces `decode`
    /// exactly at every chunk size — including the ▁ substitution, the
    /// metaspace-prefix strip and characters that exist only as `<0xNN>` runs.
    #[test]
    fn stream_matches_decode_on_the_unigram_vocabulary() {
        let tokenizer = stream_tok();
        for text in STREAM_TEXTS {
            let ids = tokenizer.encode(text);
            let expected = tokenizer.decode(&ids).expect("real ids decode");

            for chunk in 1..=ids.len().max(1) {
                assert_eq!(
                    drive_strict(&tokenizer, &ids, chunk),
                    expected,
                    "text: {text:?}, chunk: {chunk}"
                );
            }
            assert_eq!(
                drive_lossy(&tokenizer, &ids),
                tokenizer.decode_lossy(&ids),
                "text: {text:?}"
            );
        }
    }

    /// The `at_start` trap: a skipped id renders nothing, so it must not spend
    /// the metaspace-prefix strip. A leading BOS therefore strips exactly as the
    /// same ids without it do — the failure mode is `" hello world"`, with the
    /// prefix space the encoder added left in. Chunk size 1 is the sharpest
    /// form: feeding the BOS on its own is a push that emits nothing at all.
    #[test]
    fn a_leading_bos_does_not_consume_the_leading_space_strip() {
        let tokenizer = stream_tok();
        let bare = ids_of(&tokenizer, &["▁hello", "▁world"]);
        // `<s>` and `<pad>` (declared special) both render nothing.
        let with_bos = [&[1u32, 3][..], &bare[..]].concat();

        assert_eq!(drive_strict(&tokenizer, &bare, 1), "hello world");
        assert_eq!(drive_strict(&tokenizer, &with_bos, 1), "hello world");
        assert_eq!(
            tokenizer.decode(&with_bos).expect("real ids decode"),
            "hello world"
        );
    }

    /// A ▁ spelled out through byte-fallback ids is the literal character, not a
    /// word boundary — so the substitution must happen per *surface*, never over
    /// reassembled text.
    ///
    /// Ground truth from the `sentencepiece` package 0.2.0: the ids for the
    /// pieces `<0xE2>`, `<0x96>`, `<0x81>` decode to `'▁'`, while the `▁` piece
    /// itself decodes to `''`. HuggingFace's declared chain agrees, running
    /// `Replace(▁→" ")` *before* `ByteFallback`. The same three UTF-8 bytes mean
    /// a character when a `<0xNN>` token produced them and a space when the `▁`
    /// piece did; text that has already been reassembled cannot tell them apart.
    #[test]
    fn a_byte_fallback_metaspace_decodes_to_the_literal_character() {
        let tokenizer = stream_tok();
        let spelled_out = ids_of(&tokenizer, &["<0xE2>", "<0x96>", "<0x81>"]);

        assert_eq!(tokenizer.decode(&spelled_out).unwrap(), WORD_BOUNDARY);
        assert_eq!(tokenizer.decode_lossy(&spelled_out), WORD_BOUNDARY);
        // ...while the `▁` piece itself is the metaspace prefix, and comes off.
        assert_eq!(tokenizer.decode(&ids_of(&tokenizer, &["▁"])).unwrap(), "");
    }

    /// The same three ids through the streaming decoder, under every grouping:
    /// the substitution is a rendering rule, so it cannot depend on where a
    /// chunk boundary fell — and stream and `decode` agree on this case too.
    #[test]
    fn a_byte_fallback_metaspace_streams_as_the_literal_character() {
        let tokenizer = stream_tok();
        let spelled_out = ids_of(&tokenizer, &["<0xE2>", "<0x96>", "<0x81>"]);

        for chunk in 1..=spelled_out.len() {
            assert_eq!(
                drive_strict(&tokenizer, &spelled_out, chunk),
                WORD_BOUNDARY,
                "chunk: {chunk}"
            );
        }
        assert_eq!(drive_lossy(&tokenizer, &spelled_out), WORD_BOUNDARY);
    }

    /// ...and the substitution is not disabled wholesale: an ordinary
    /// ▁-prefixed piece is still a space, the first one being the prefix the
    /// pre-tokenizer added and the second a real one.
    #[test]
    fn an_ordinary_metaspace_piece_still_decodes_to_a_space() {
        let tokenizer = stream_tok();
        let ids = ids_of(&tokenizer, &["▁a", "▁world"]);

        assert_eq!(tokenizer.decode(&ids).unwrap(), "a world");
        assert_eq!(drive_strict(&tokenizer, &ids, 1), "a world");
    }

    /// A character split across several `<0xNN>` byte tokens reassembles across
    /// `add_token` calls: the resolved bytes go through the same UTF-8 buffer
    /// every other byte does, so nothing is emitted until the character is
    /// complete.
    #[test]
    fn a_byte_fallback_char_reassembles_across_add_token_calls() {
        let tokenizer = stream_tok().with_prefix_space(false);

        // 🎉 (U+1F389) is four UTF-8 bytes, none of them a piece of its own.
        let ids = tokenizer.encode("🎉");
        assert_eq!(ids.len(), 4, "four bytes, so four byte-fallback tokens");

        let mut decoder = tokenizer.streaming_decoder();
        for &id in &ids[..3] {
            assert_eq!(decoder.add_token(id).unwrap(), None);
            assert!(decoder.has_pending());
        }
        assert_eq!(decoder.add_token(ids[3]).unwrap(), Some("🎉".to_string()));
        assert!(!decoder.has_pending());
        assert_eq!(decoder.flush(), "");

        assert_eq!(tokenizer.decode(&ids).unwrap(), "🎉");
        for chunk in 1..=ids.len() {
            assert_eq!(drive_strict(&tokenizer, &ids, chunk), "🎉");
        }
        assert_eq!(drive_lossy(&tokenizer, &ids), tokenizer.decode_lossy(&ids));
    }

    proptest! {
        /// Chunk-partition invariance: arbitrary grouping through `add_tokens`
        /// gives what one-at-a-time gives, and both give `decode`.
        #[test]
        fn prop_chunking_matches_decode(
            text in ".{0,120}",
            chunk in 1usize..8,
        ) {
            let tokenizer = stream_tok();
            let ids = tokenizer.encode(&text);
            let expected = tokenizer.decode(&ids).expect("real ids decode");

            prop_assert_eq!(drive_strict(&tokenizer, &ids, 1), expected.clone());
            prop_assert_eq!(drive_strict(&tokenizer, &ids, chunk), expected);
        }

        /// Arbitrary ids — unknown ones, bare byte tokens and mid-character
        /// splits included — stream lossily to exactly `decode_lossy`.
        #[test]
        fn prop_arbitrary_ids_match_decode_lossy(
            ids in prop::collection::vec(0u32..300, 0..48),
        ) {
            let tokenizer = stream_tok();
            prop_assert_eq!(drive_lossy(&tokenizer, &ids), tokenizer.decode_lossy(&ids));
        }

        /// `reset()` purity: a used-then-reset decoder behaves byte-identically
        /// to a freshly built one — `at_start` included, which is why the dirty
        /// prefix is fed before the reset rather than after.
        #[test]
        fn prop_reset_matches_a_fresh_decoder(
            dirty in prop::collection::vec(0u32..300, 0..16),
            ids in prop::collection::vec(0u32..300, 0..32),
        ) {
            let tokenizer = stream_tok();

            let mut reused = tokenizer.streaming_decoder();
            reused.add_tokens_lossy(&dirty);
            reused.reset();
            prop_assert!(!reused.has_pending());
            prop_assert_eq!(reused.pending_bytes(), 0);

            let mut fresh = tokenizer.streaming_decoder();

            let mut from_reused = String::new();
            let mut from_fresh = String::new();
            for &id in &ids {
                let a = reused.add_token_lossy(id);
                let b = fresh.add_token_lossy(id);
                prop_assert_eq!(&a, &b);
                prop_assert_eq!(reused.pending_bytes(), fresh.pending_bytes());
                from_reused.push_str(&a.unwrap_or_default());
                from_fresh.push_str(&b.unwrap_or_default());
            }
            from_reused.push_str(&reused.flush());
            from_fresh.push_str(&fresh.flush());

            prop_assert_eq!(from_reused, from_fresh);
        }
    }

    // =========================================================================
    // Per-id decoding: `Tokenize::decode_token_bytes` / `decode_token`
    // =========================================================================

    /// A vocabulary whose `<0xNN>` entries spell the four bytes of `𐍈`
    /// (U+10348), so a test can ask for one byte of a character on its own.
    /// The metaspace prefix is turned off, which leaves this tokenizer with no
    /// text post-op at all — the shape the agreement test below needs.
    fn byte_fallback_tokenizer() -> SentencePieceTokenizer {
        let tokens: Vec<String> = [
            "<unk>", "<s>", "</s>", "▁Hello", "▁world", "<0xF0>", "<0x90>", "<0x8D>", "<0x88>",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
        let scores = vec![0.0; tokens.len()];
        SentencePieceTokenizer::new(tokens, scores, Some(1), 2)
            .expect("the vocabulary is well formed")
            .with_prefix_space(false)
    }

    /// The three answers the method distinguishes: an ordinary piece renders its
    /// bytes (▁ already substituted, since that is a *rendering* rule), a
    /// skipped id contributes an empty `Vec` rather than an error — it really
    /// does contribute nothing — and an id the vocabulary has no slot for is
    /// reported.
    #[test]
    fn decode_token_bytes_separates_content_skip_and_unknown() {
        use crate::core::tokenize::{Tokenize, TokenizeError};
        let tok = byte_fallback_tokenizer();

        // No sequence-level post-processing: the space the ▁ became is still
        // here, where `decode` of the same id alone would have stripped it on a
        // prefixing tokenizer.
        assert_eq!(tok.decode_token_bytes(3).unwrap(), b" Hello".to_vec());
        assert_eq!(tok.decode_token(3).unwrap(), " Hello");

        // BOS is in `skipped_on_decode`, and so are EOS and `<unk>`.
        for skipped in [0, 1, 2] {
            assert_eq!(tok.decode_token_bytes(skipped).unwrap(), Vec::<u8>::new());
            assert_eq!(tok.decode_token(skipped).unwrap(), "");
        }

        assert!(matches!(
            tok.decode_token_bytes(999),
            Err(TokenizeError::InvalidTokenId(999))
        ));
        assert!(matches!(
            tok.decode_token(999),
            Err(TokenizeError::InvalidTokenId(999))
        ));
    }

    /// The case the pair of methods exists for: a `<0xNN>` id carries one byte
    /// of a four-byte character, so it has bytes but is not text on its own.
    #[test]
    fn a_byte_fallback_id_has_bytes_but_no_text_of_its_own() {
        use crate::core::tokenize::{Tokenize, TokenizeError};
        let tok = byte_fallback_tokenizer();

        for (id, byte) in (5u32..=8).zip([0xF0, 0x90, 0x8D, 0x88]) {
            assert_eq!(tok.decode_token_bytes(id).unwrap(), vec![byte]);
            assert!(matches!(
                tok.decode_token(id),
                Err(TokenizeError::Utf8Error)
            ));
        }
    }

    /// Agreement: concatenating the per-id bytes over a sequence is exactly what
    /// decoding that sequence emits. Exact here because `with_prefix_space(false)`
    /// leaves no post-op, and this backend declares no word separator.
    #[test]
    fn concatenated_token_bytes_equal_the_decoded_sequence() {
        use crate::core::tokenize::Tokenize;
        let tok = byte_fallback_tokenizer();

        for ids in [
            vec![1, 3, 4, 2],       // specials + ordinary pieces
            vec![5, 6, 7, 8],       // one character spelled out byte by byte
            vec![1, 3, 5, 6, 7, 8], // and the two mixed
        ] {
            let joined: Vec<u8> = ids
                .iter()
                .flat_map(|&id| tok.decode_token_bytes(id).expect("every id is known"))
                .collect();
            assert_eq!(joined, tok.decode_lossy(&ids).into_bytes(), "ids: {ids:?}");
        }
    }

    /// The trait's `decode_lossy` and `streaming_decoder` are the inherent ones.
    /// Before this, `decode_lossy` was inherent-only and therefore unreachable
    /// for any caller holding this backend through the trait.
    #[test]
    fn trait_decode_lossy_and_streaming_decoder_match_the_inherent_pair() {
        use crate::core::tokenize::Tokenize;
        let tok = byte_fallback_tokenizer();
        let ids = [1, 3, 4, 999, 2];

        assert_eq!(Tokenize::decode_lossy(&tok, &ids), " Hello world");
        assert_eq!(
            Tokenize::decode_lossy(&tok, &ids),
            SentencePieceTokenizer::decode_lossy(&tok, &ids)
        );

        let mut streamed = Tokenize::streaming_decoder(&tok).expect("Unigram always streams");
        let mut out = streamed.add_tokens_lossy(&ids).unwrap_or_default();
        out.push_str(&streamed.flush());
        assert_eq!(out, " Hello world");
    }
}
