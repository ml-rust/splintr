use super::error::TokenizerError;
use super::types::Tokenizer;
use crate::core::streaming::{DecodeState, DecodeView, Rendered, StreamingDecoder};
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use std::convert::Infallible;
use std::sync::Arc;

impl Tokenizer {
    /// This tokenizer's decode configuration, as the streaming decoder sees it.
    ///
    /// Whole-sequence decoding and streaming decoding render an id through the
    /// same [`DecodeView`], so the two cannot disagree about what an id means.
    fn decode_view(&self) -> DecodeView<'_> {
        DecodeView {
            decoder: &self.decoder,
            special_tokens_decoder: &self.special_tokens_decoder,
            special_decode_ids: &self.special_decode_ids,
            use_byte_level: self.use_byte_level,
            use_metaspace_decoder: self.use_metaspace_decoder,
        }
    }

    /// A [`StreamingDecoder`] configured from this tokenizer.
    ///
    /// The only way to build one: ByteLevel unmapping, the `special=true` ids
    /// to drop and the metaspace substitution all come from this tokenizer's
    /// configuration, so the stream cannot be pointed at the wrong kind of
    /// vocabulary and always reproduces [`decode`](Tokenizer::decode).
    ///
    /// Cheap to call — the vocabulary map is shared, not copied — and the
    /// result borrows nothing, so it can be moved into a generation task.
    pub fn streaming_decoder(&self) -> StreamingDecoder {
        StreamingDecoder::new(Arc::new(DecodeState::new(
            Arc::clone(&self.decoder),
            self.special_tokens_decoder.clone(),
            self.special_decode_ids.clone(),
            self.use_byte_level,
            self.use_metaspace_decoder,
        )))
    }

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
        let view = self.decode_view();

        for &token in tokens {
            match view.render(token) {
                Rendered::Skipped => {}
                Rendered::Bytes(bytes) => result.extend_from_slice(&bytes),
                Rendered::Unknown => on_unknown(token)?,
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

    /// Post-process decoded text for metaspace-decoder tokenizers — the same
    /// substitution the streaming decoder applies to each emitted chunk (see
    /// [`DecodeView::postprocess`]).
    #[inline]
    fn postprocess_decode(&self, text: String) -> String {
        self.decode_view().postprocess(text)
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
