//! SentencePiece-compatible Unigram tokenizer.
//!
//! Implements the Unigram **Viterbi** algorithm — the maximum total-score
//! segmentation — matching SentencePiece / HuggingFace `tokenizers` (T5, Albert,
//! XLNet, …), with metaspace pre-tokenization, byte-fallback, an ordered
//! normalizer pipeline, and added-token matching.

use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SentencePieceError {
    #[error("Empty vocabulary")]
    EmptyVocab,
    #[error("Scores length ({scores}) does not match tokens length ({tokens})")]
    ScoreMismatch { scores: usize, tokens: usize },
    #[error("Decoding error: token ID {0} out of range")]
    InvalidTokenId(u32),
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
    /// ID -> Token string mapping
    id_to_token: Vec<String>,
    /// Per-token Unigram scores (log-probs); Viterbi maximizes their sum over the
    /// chosen segmentation.
    scores: Vec<f32>,
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
    min_score: f32,
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
        scores: Vec<f32>,
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
            .fold(f32::INFINITY, f32::min)
            .min(0.0);

        Ok(Self {
            token_to_id,
            id_to_token: tokens,
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
    pub fn with_added_tokens(mut self, map: &rustc_hash::FxHashMap<String, u32>) -> Self {
        self.added = super::added::AddedTokens::new(map);
        self
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
    fn viterbi_piece(&self, chars: &[char], tokens: &mut Vec<u32>) {
        let n = chars.len();
        if n == 0 {
            return;
        }

        // Viterbi over the character lattice. `best[i]` = best total score to
        // reach position i; `back[i]` = (start, piece) of the chosen edge into i.
        // A piece is Some(id) for a vocab token, or None for an unknown char.
        let unk_penalty = self.min_score - 10.0; // SentencePiece's kUnkPenalty
        let mut best = vec![f32::NEG_INFINITY; n + 1];
        let mut back: Vec<(usize, Option<u32>)> = vec![(0, None); n + 1];
        best[0] = 0.0;

        let mut buf = String::with_capacity(self.max_piece_chars * 4);
        for start in 0..n {
            if best[start] == f32::NEG_INFINITY {
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

    /// Decode token IDs to text.
    ///
    /// Skips BOS/EOS tokens and converts ▁ back to spaces.
    pub fn decode(&self, ids: &[u32]) -> Result<String, SentencePieceError> {
        let mut bytes = Vec::new();

        for &id in ids {
            let token = self
                .id_to_token
                .get(id as usize)
                .ok_or(SentencePieceError::InvalidTokenId(id))?;

            // Skip special tokens (bos/eos/unk), matching HuggingFace's default
            // decode (skip_special_tokens=True). Unknown spans were unrecoverable
            // anyway, so the `<unk>` surface is dropped rather than rendered.
            if Some(id) == self.bos_token_id
                || id == self.eos_token_id
                || Some(id) == self.unk_id
                || self.special_decode.contains(&id)
            {
                continue;
            }

            if let Some(byte_val) = parse_byte_fallback(token) {
                bytes.push(byte_val);
            } else {
                let decoded = token.replace('▁', " ");
                bytes.extend_from_slice(decoded.as_bytes());
            }
        }

        let result = String::from_utf8_lossy(&bytes).into_owned();

        // Strip the single leading space only when the metaspace pre-tokenizer
        // prepended one (add_prefix_space / prepend_scheme != "never"). HF's
        // Metaspace decoder mirrors its prepend behavior; with prefixing disabled
        // a genuine leading space must be preserved, not eaten. (Token-by-token
        // streaming uses the dedicated StreamingDecoder, not this method.)
        if self.add_prefix_space {
            Ok(result
                .strip_prefix(' ')
                .map(str::to_string)
                .unwrap_or(result))
        } else {
            Ok(result)
        }
    }

    /// Decode token IDs to text, skipping invalid IDs.
    pub fn decode_lossy(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();

        for &id in ids {
            if let Some(token) = self.id_to_token.get(id as usize) {
                if Some(id) == self.bos_token_id || id == self.eos_token_id {
                    continue;
                }
                if let Some(byte_val) = parse_byte_fallback(token) {
                    bytes.push(byte_val);
                } else {
                    let decoded = token.replace('▁', " ");
                    bytes.extend_from_slice(decoded.as_bytes());
                }
            }
        }

        let result = String::from_utf8_lossy(&bytes).into_owned();
        // Mirror `decode`: strip the metaspace-induced leading space only when the
        // pre-tokenizer prepended one.
        if self.add_prefix_space {
            if let Some(stripped) = result.strip_prefix(' ') {
                return stripped.to_string();
            }
        }
        result
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

    fn decode(&self, ids: &[u32]) -> Result<String, super::tokenize::TokenizeError> {
        self.decode(ids)
            .map_err(|e| super::tokenize::TokenizeError::Other(e.to_string()))
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size()
    }
}

/// Parse a byte-fallback token like `<0x0A>` into its byte value.
fn parse_byte_fallback(token: &str) -> Option<u8> {
    let inner = token.strip_prefix("<0x")?.strip_suffix('>')?;
    if inner.len() == 2 {
        u8::from_str_radix(inner, 16).ok()
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_parse_byte_fallback_valid() {
        assert_eq!(parse_byte_fallback("<0x0A>"), Some(0x0A));
        assert_eq!(parse_byte_fallback("<0xFF>"), Some(0xFF));
        assert_eq!(parse_byte_fallback("<0x00>"), Some(0x00));
        assert_eq!(parse_byte_fallback("<0x7F>"), Some(0x7F));
        // Lowercase hex
        assert_eq!(parse_byte_fallback("<0xab>"), Some(0xAB));
    }

    #[test]
    fn test_parse_byte_fallback_invalid() {
        assert_eq!(parse_byte_fallback("<0xZZ>"), None);
        assert_eq!(parse_byte_fallback("<0x1>"), None); // single hex digit
        assert_eq!(parse_byte_fallback("<0x123>"), None); // three hex digits
        assert_eq!(parse_byte_fallback("0x0A"), None); // missing angle brackets
        assert_eq!(parse_byte_fallback("<0x0A"), None); // missing closing bracket
        assert_eq!(parse_byte_fallback("0x0A>"), None); // missing opening prefix
        assert_eq!(parse_byte_fallback(""), None);
        assert_eq!(parse_byte_fallback("hello"), None);
        assert_eq!(parse_byte_fallback("<>"), None);
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
}
