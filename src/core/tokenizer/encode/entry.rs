use super::super::types::Tokenizer;
use crate::core::added::AddedTokens;
use crate::core::policy::{PolicyError, SpecialMode};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

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
            AddedTokens::dispatch(&self.special_matcher, text, |gap| {
                self.encode_content(gap, true)
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
        AddedTokens::dispatch(&self.special_matcher, text, |gap| self.encode_ordinary(gap))
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
        AddedTokens::dispatch_with_mode(&self.special_matcher, text, mode, |gap| {
            self.encode_ordinary(gap)
        })
    }

    /// Batch encode multiple texts (parallel when rayon is enabled).
    pub fn encode_batch(&self, texts: &[String]) -> Vec<Vec<u32>> {
        #[cfg(feature = "rayon")]
        {
            texts.par_iter().map(|text| self.encode(text)).collect()
        }
        #[cfg(not(feature = "rayon"))]
        {
            texts.iter().map(|text| self.encode(text)).collect()
        }
    }

    /// Batch encode multiple texts with special token handling.
    pub fn encode_batch_with_special(&self, texts: &[String]) -> Vec<Vec<u32>> {
        #[cfg(feature = "rayon")]
        {
            texts
                .par_iter()
                .map(|text| self.encode_with_special(text))
                .collect()
        }
        #[cfg(not(feature = "rayon"))]
        {
            texts
                .iter()
                .map(|text| self.encode_with_special(text))
                .collect()
        }
    }
}
