use super::super::types::Tokenizer;
use crate::core::added::AddedTokens;
use crate::core::batch;
use crate::core::policy::{PolicyError, SpecialMode};

impl Tokenizer {
    /// Encode text to token IDs.
    ///
    /// By default special tokens in the input are treated as ordinary text. When
    /// the tokenizer was built with added-token matching (HF `tokenizer.json`
    /// loaders), `added_tokens` are recognized first.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if self.match_added_tokens {
            self.encode_with_special(text)
        } else {
            self.encode_ordinary(text)
        }
    }

    /// Encode text to token IDs, always treating special tokens as ordinary text.
    ///
    /// Uses sequential processing, which is faster than parallel for texts up to ~1MB.
    pub fn encode_ordinary(&self, text: &str) -> Vec<u32> {
        self.encode_content(text, false)
    }

    /// Encode text to token IDs using Rayon parallel processing.
    ///
    /// Produces exactly the same ids as [`Tokenizer::encode`] — same
    /// normalizer, same added-token dispatch, same pre-tokenizer/metaspace/
    /// plain-chunk fork — and differs only in execution strategy: the
    /// plain-chunk fork's BPE calls run in parallel via rayon rather than
    /// sequentially. Only beneficial for very large texts (>1MB).
    ///
    /// Metaspace-decoder tokenizers and tokenizers with a multi-stage
    /// pre-tokenizer still run sequentially regardless of this method,
    /// because their per-chunk state (`pending_underscores`, the
    /// pre-tokenizer engine's own iteration) is a left-to-right fold that
    /// cannot be parallelized without changing output — see the internal
    /// `encode_content`.
    pub fn encode_rayon(&self, text: &str) -> Vec<u32> {
        if self.match_added_tokens {
            AddedTokens::dispatch(&self.special_matcher, text, |gap, out| {
                self.encode_content_into(gap, true, opens_input(text, gap), out)
            })
        } else {
            self.encode_content(text, true)
        }
    }

    /// Encode text with special token handling.
    ///
    /// Special tokens in the input are encoded directly without BPE, via the
    /// same `AddedTokens` matcher the SentencePiece/SPM/WordPiece backends
    /// use.
    pub fn encode_with_special(&self, text: &str) -> Vec<u32> {
        AddedTokens::dispatch(&self.special_matcher, text, |gap, out| {
            self.encode_content_into(gap, false, opens_input(text, gap), out)
        })
    }

    /// Encode text to token IDs under an explicit [`SpecialMode`], governing
    /// whether `special_tokens` found in the input text are matched.
    ///
    /// This only concerns added-token matching in the content — it says
    /// nothing about boundary tokens (BOS/EOS/CLS/SEP), which this backend
    /// has no notion of; those come from [`SpecialPolicy`](crate::core::SpecialPolicy)
    /// via [`AnyTokenizer::encode_with`](crate::core::AnyTokenizer::encode_with).
    ///
    /// If this tokenizer was never configured for added-token matching
    /// ([`with_added_token_matching`](Self::with_added_token_matching) is
    /// `false`, [`Tokenizer::encode`]'s default), [`SpecialMode::All`] is read
    /// as "there is no matching to turn on" and falls back to the ordinary
    /// encoding — the same behavior [`Tokenizer::encode`] already gives in
    /// that configuration. [`SpecialMode::Ordinary`] and
    /// [`SpecialMode::Allow`] are the caller stating an explicit choice rather
    /// than asking for this tokenizer's default, so they always take effect
    /// regardless of that flag.
    pub fn encode_with(&self, text: &str, mode: &SpecialMode<'_>) -> Result<Vec<u32>, PolicyError> {
        if matches!(mode, SpecialMode::All) && !self.match_added_tokens {
            return Ok(self.encode_ordinary(text));
        }
        AddedTokens::dispatch_with_mode(&self.special_matcher, text, mode, |gap, out| {
            self.encode_content_into(gap, false, opens_input(text, gap), out)
        })
    }

    /// Batch encode multiple texts.
    ///
    /// Runs across rayon's thread pool when the `rayon` feature is on and the
    /// batch carries enough total input to pay for the hand-off; a small batch
    /// encodes on the calling thread instead, which is faster than waking a
    /// pool for it. The ids and their order do not depend on which path runs.
    pub fn encode_batch(&self, texts: &[String]) -> Vec<Vec<u32>> {
        batch::map(texts, String::len, |text| self.encode(text))
    }

    /// Batch encode multiple texts with special token handling.
    pub fn encode_batch_with_special(&self, texts: &[String]) -> Vec<Vec<u32>> {
        batch::map(texts, String::len, |text| self.encode_with_special(text))
    }
}

/// Whether `gap` is the split that opens `text`.
///
/// Added-token dispatch hands out gaps as subslices of the input, so the one
/// starting at its first byte is the sequence's first split and every other is
/// not — including the gap that follows a leading added token, which is the
/// case that distinguishes this from "the first gap the closure sees".
///
/// Only `Metaspace`'s `prepend_scheme: "first"` reads it; see
/// `Tokenizer::metaspace_transform_at`. Pinned by
/// `metaspace_prefix_is_first_split_only` in the tokenizer tests, so a gap that
/// ever stopped borrowing from the input would fail loudly rather than silently
/// move a token id.
#[inline]
fn opens_input(text: &str, gap: &str) -> bool {
    std::ptr::eq(text.as_ptr(), gap.as_ptr())
}
