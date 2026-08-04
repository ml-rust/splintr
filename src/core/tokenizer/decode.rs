use super::error::TokenizerError;
use super::types::{ByteFallback, Tokenizer};
use crate::core::streaming::{
    ByteFallbackRule, DecodePost, DecodeState, Lead, RenderRules, Rendered, StreamingDecoder,
    Surfaces,
};
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use std::convert::Infallible;
use std::sync::Arc;

impl Tokenizer {
    /// This tokenizer's decode configuration, as the streaming decoder sees it.
    ///
    /// Whole-sequence decoding and streaming decoding drive the same
    /// [`DecodeState`] through the same cursor, so the two cannot disagree
    /// about what an id means or about what happens to the text it produces.
    ///
    /// Cheap to build — every table inside is shared with this tokenizer rather
    /// than copied — which is what lets `decode` capture one per call instead
    /// of the tokenizer having to cache one that could go stale.
    pub(crate) fn decode_state(&self) -> DecodeState {
        DecodeState::new(
            RenderRules::new(
                Surfaces::ById(Arc::clone(&self.decoder)),
                Arc::clone(&self.special_tokens_decoder),
                Arc::clone(&self.special_decode_ids),
                match self.decode_byte_fallback() {
                    Some(fallback) => ByteFallbackRule::Table(Arc::clone(fallback.id_bytes())),
                    None => ByteFallbackRule::None,
                },
                self.use_byte_level,
                // The metaspace substitution stays a post-op on this backend:
                // its surfaces are byte strings that can hold half a ▁, so only
                // reassembled text can see the marker.
                false,
            ),
            if self.use_metaspace_decoder {
                vec![DecodePost::MetaspaceToSpace]
            } else {
                Vec::new()
            },
        )
    }

    /// The byte-fallback table decoding resolves `<0xNN>` ids through, or `None`
    /// when this vocabulary has none.
    ///
    /// Gated on `!use_byte_level` exactly as the encode side gates it (see
    /// `Tokenizer::bpe`): the table is keyed by RAW byte value, so on a
    /// ByteLevel vocabulary — whose pieces live in ByteLevel space — encode can
    /// never emit a fallback id, and decode must not resolve one either.
    fn decode_byte_fallback(&self) -> Option<&ByteFallback> {
        (!self.use_byte_level)
            .then_some(self.byte_fallback.as_ref())
            .flatten()
    }

    /// A [`StreamingDecoder`] configured from this tokenizer.
    ///
    /// The only way to build one: ByteLevel unmapping, the `special=true` ids
    /// to drop, the `<0xNN>` byte-fallback resolution and the metaspace
    /// substitution all come from this tokenizer's configuration, so the stream
    /// cannot be pointed at the wrong kind of vocabulary and always reproduces
    /// [`decode`](Tokenizer::decode).
    ///
    /// Cheap to call — the vocabulary map is shared, not copied — and the
    /// result borrows nothing, so it can be moved into a generation task.
    pub fn streaming_decoder(&self) -> StreamingDecoder {
        StreamingDecoder::new(Arc::new(self.decode_state()))
    }

    /// Decode token IDs back to bytes.
    ///
    /// The one decode entry point that wants bytes rather than text, so it
    /// renders through the internal `RenderRules` directly instead of through a cursor —
    /// there is no UTF-8 reassembly and no post-op to run on raw bytes. The
    /// rendering itself is still the same code every other decode path uses.
    ///
    /// `special=true` added tokens are dropped (HF default
    /// `skip_special_tokens`) — a distinct, intentional skip, not an
    /// unknown-id error.
    ///
    /// Errors with [`TokenizerError::InvalidTokenId`] if `tokens` contains an
    /// id that is not in the vocabulary and not a known special token.
    pub fn decode_bytes(&self, tokens: &[u32]) -> Result<Vec<u8>, TokenizerError> {
        let mut result = Vec::with_capacity(tokens.len() * 4);
        let state = self.decode_state();
        let rules = state.render();

        for &token in tokens {
            match rules.render(token) {
                Rendered::Skipped => {}
                Rendered::Bytes { lead, bytes } => {
                    match lead {
                        Lead::None => {}
                    }
                    result.extend_from_slice(&bytes);
                }
                Rendered::Unknown => return Err(TokenizerError::InvalidTokenId(token)),
            }
        }

        Ok(result)
    }

    /// Decode token IDs to a string.
    ///
    /// The degenerate drive of the streaming cursor: one feed of every id, then
    /// a flush. Strict throughout — an id in no table is
    /// [`TokenizerError::InvalidTokenId`] and bytes that are not valid UTF-8
    /// are [`TokenizerError::Utf8Error`], never a U+FFFD substitution — but
    /// *what* an id renders to and what happens to the resulting text is
    /// decided by exactly the code a stream uses.
    pub fn decode(&self, tokens: &[u32]) -> Result<String, TokenizerError> {
        let state = self.decode_state();
        let mut cursor = state.cursor_with_capacity(tokens.len() * 4);

        let emitted = cursor.feed_strict(
            tokens,
            |id| Err(TokenizerError::InvalidTokenId(id)),
            || TokenizerError::Utf8Error,
        )?;

        let mut text = emitted.unwrap_or_default();
        text.push_str(&cursor.finish_strict(|| TokenizerError::Utf8Error)?);

        Ok(text)
    }

    /// Decode token IDs to a string, replacing invalid UTF-8 with replacement character.
    ///
    /// The same degenerate cursor drive as [`decode`](Self::decode), on the
    /// lossy side: unknown ids are skipped, mirroring the `special=true` skip
    /// [`decode_bytes`](Self::decode_bytes) makes, and undecodable bytes become
    /// U+FFFD. This method never fails, so `on_unknown` is instantiated with
    /// [`Infallible`], letting the compiler prove the `Err` arm away rather
    /// than a runtime assertion claiming it.
    pub fn decode_lossy(&self, tokens: &[u32]) -> String {
        let state = self.decode_state();
        let mut cursor = state.cursor_with_capacity(tokens.len() * 4);

        let mut text = match cursor.feed(tokens, |_| Ok::<(), Infallible>(())) {
            Ok(text) => text.unwrap_or_default(),
            // `Infallible` has no values, so this match has no arms to write.
            Err(never) => match never {},
        };
        text.push_str(&cursor.flush());

        text
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
