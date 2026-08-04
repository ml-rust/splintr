use super::error::TokenizerError;
use super::types::Tokenizer;
use crate::core::byte_level::byte_level_decode_bytes;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use std::convert::Infallible;

impl Tokenizer {
    /// Shared decode loop: renders `tokens` to bytes, deferring the "id not in
    /// the vocabulary" decision to `on_unknown` so strict and lossy decoding
    /// cannot drift into two different notions of "unknown".
    ///
    /// `special=true` added tokens are always dropped (HF default
    /// `skip_special_tokens`), never routed through `on_unknown` — that is a
    /// distinct, intentional skip, not an unknown-id error.
    fn decode_bytes_with<E>(
        &self,
        tokens: &[u32],
        on_unknown: impl Fn(u32) -> Result<(), E>,
    ) -> Result<Vec<u8>, E> {
        let mut result = Vec::with_capacity(tokens.len() * 4);

        for &token in tokens {
            // Drop `special=true` added tokens (HF default skip_special_tokens).
            if self.special_decode_ids.contains(&token) {
                continue;
            }
            if let Some(bytes) = self.decoder.get(&token) {
                if self.use_byte_level {
                    if let Some(decoded) = byte_level_decode_bytes(bytes) {
                        result.extend_from_slice(&decoded);
                    } else {
                        result.extend_from_slice(bytes);
                    }
                } else {
                    result.extend_from_slice(bytes);
                }
            } else if let Some(special) = self.special_tokens_decoder.get(&token) {
                result.extend_from_slice(special.as_bytes());
            } else {
                on_unknown(token)?;
            }
        }

        Ok(result)
    }

    /// Decode token IDs back to bytes.
    ///
    /// Errors with [`TokenizerError::InvalidTokenId`] if `tokens` contains an
    /// id that is not in the vocabulary and not a known special token.
    pub fn decode_bytes(&self, tokens: &[u32]) -> Result<Vec<u8>, TokenizerError> {
        self.decode_bytes_with(tokens, |id| Err(TokenizerError::InvalidTokenId(id)))
    }

    /// Decode token IDs to a string.
    pub fn decode(&self, tokens: &[u32]) -> Result<String, TokenizerError> {
        let bytes = self.decode_bytes(tokens)?;
        let text = String::from_utf8(bytes).map_err(|_| TokenizerError::Utf8Error)?;
        Ok(self.postprocess_decode(text))
    }

    /// Decode token IDs to a string, replacing invalid UTF-8 with replacement character.
    ///
    /// Unknown ids are skipped, mirroring the internal `decode_bytes_with`'s
    /// `special=true` skip — this method never fails, so `on_unknown` is
    /// instantiated with [`Infallible`], letting the compiler prove the `Err`
    /// arm away rather than a runtime assertion claiming it.
    pub fn decode_lossy(&self, tokens: &[u32]) -> String {
        let bytes = match self.decode_bytes_with(tokens, |_| Ok::<(), Infallible>(())) {
            Ok(bytes) => bytes,
            // `Infallible` has no values, so this match has no arms to write.
            Err(never) => match never {},
        };
        let text = String::from_utf8_lossy(&bytes).into_owned();
        self.postprocess_decode(text)
    }

    /// Post-process decoded text for metaspace-decoder tokenizers.
    ///
    /// Converts ▁ (U+2581, lower one eighth block) to space.
    ///
    /// Note: Unlike some tokenizer implementations, we do NOT strip leading spaces.
    /// The ▁ character represents a word boundary and should become a space.
    /// If you need to strip leading space from the very first token in a sequence,
    /// handle that at a higher level (e.g., in your generation loop).
    #[inline]
    fn postprocess_decode(&self, text: String) -> String {
        if self.use_metaspace_decoder {
            // Replace ▁ with space - this preserves word boundaries
            text.replace('\u{2581}', " ")
        } else {
            text
        }
    }

    /// Batch decode multiple token lists.
    pub fn decode_batch(&self, token_lists: &[Vec<u32>]) -> Result<Vec<String>, TokenizerError> {
        #[cfg(feature = "rayon")]
        {
            token_lists
                .par_iter()
                .map(|tokens| self.decode(tokens))
                .collect()
        }
        #[cfg(not(feature = "rayon"))]
        {
            token_lists
                .iter()
                .map(|tokens| self.decode(tokens))
                .collect()
        }
    }

    /// Batch decode multiple token lists, replacing invalid UTF-8.
    pub fn decode_batch_lossy(&self, token_lists: &[Vec<u32>]) -> Vec<String> {
        #[cfg(feature = "rayon")]
        {
            token_lists
                .par_iter()
                .map(|tokens| self.decode_lossy(tokens))
                .collect()
        }
        #[cfg(not(feature = "rayon"))]
        {
            token_lists
                .iter()
                .map(|tokens| self.decode_lossy(tokens))
                .collect()
        }
    }
}
