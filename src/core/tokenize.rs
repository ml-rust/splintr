//! Unified tokenizer trait for all splintr backends.
//!
//! The `Tokenize` trait provides a common interface across BPE, SentencePiece,
//! and WordPiece tokenizers, enabling generic code that works with any backend.

use super::policy::{PolicyError, SpecialDecode, SpecialMode};
use super::streaming::{RenderRules, StreamingDecoder};

/// Common interface for all tokenizer backends.
///
/// Implemented by [`Tokenizer`](super::Tokenizer) (BPE),
/// [`SentencePieceTokenizer`](super::SentencePieceTokenizer) (unigram), and
/// [`WordPieceTokenizer`](super::WordPieceTokenizer) (WordPiece).
pub trait Tokenize: Send + Sync {
    /// Encode text into token IDs.
    fn encode(&self, text: &str) -> Vec<u32>;

    /// Encode text into token IDs under an explicit [`SpecialMode`], governing
    /// whether special/control tokens spelled out in `text` are matched.
    ///
    /// Deliberately not defaulted: every implementor of this trait lives in
    /// this crate, and a default body that ignored `mode` would make the
    /// allow-list/deny-all guarantee silently inert for any future backend
    /// that forgot to override it.
    fn encode_with(&self, text: &str, mode: &SpecialMode<'_>) -> Result<Vec<u32>, PolicyError>;

    /// Decode token IDs back to text.
    ///
    /// Returns an error if any token ID is invalid.
    fn decode(&self, ids: &[u32]) -> Result<String, TokenizeError>;

    /// Decode token IDs back to text under an explicit [`SpecialDecode`],
    /// governing whether the ids the vocabulary declares special are rendered or
    /// dropped.
    ///
    /// [`decode`](Self::decode) is this method under
    /// [`SpecialDecode::Skip`] — HuggingFace's default — so this is the way to
    /// ask for its `skip_special_tokens=False`: the round trip that shows a chat
    /// template's markers, an inspection of what a model actually emitted, a
    /// transcript rendered with its control tokens intact. Errors exactly as
    /// [`decode`](Self::decode) does.
    ///
    /// Deliberately not defaulted, for the reason
    /// [`encode_with`](Self::encode_with) is not: a default body would have to
    /// ignore `specials` and answer as `decode` does, which is precisely the
    /// mode the caller asked *not* to have — and it would do so silently, since
    /// dropped markers read as ordinary text.
    fn decode_with(&self, ids: &[u32], specials: SpecialDecode) -> Result<String, TokenizeError>;

    /// A [`StreamingDecoder`] that reproduces this tokenizer's
    /// [`decode_with`](Self::decode_with) under the same [`SpecialDecode`],
    /// token by token.
    ///
    /// [`streaming_decoder`](Self::streaming_decoder) is this method under
    /// [`SpecialDecode::Skip`]. A stream that could not render control markers
    /// would leave a generation loop unable to see the very tokens it is
    /// watching for, so the two decode entry points offer the mode alike and the
    /// concatenate-equals-decode contract holds in both.
    ///
    /// Fallible for the same reason [`streaming_decoder`](Self::streaming_decoder)
    /// is, and refuses on exactly the same declared pipelines.
    ///
    /// Deliberately not defaulted, for the same reason as above.
    fn streaming_decoder_with(
        &self,
        specials: SpecialDecode,
    ) -> Result<StreamingDecoder, TokenizeError>;

    /// Decode token IDs back to text, surviving whatever [`decode`](Self::decode)
    /// would report: an id in no table is skipped and bytes that cannot be valid
    /// UTF-8 become U+FFFD. Never fails.
    ///
    /// Deliberately not defaulted, for the same reason
    /// [`encode_with`](Self::encode_with) is not: a default body could only be
    /// `decode(...).unwrap_or_default()`, which turns one bad id in a long
    /// sequence into the empty string rather than into the rest of the text —
    /// the exact opposite of what "lossy" promises. Every backend already owns a
    /// real lenient drive of its decode loop, and this makes each of them say so.
    fn decode_lossy(&self, ids: &[u32]) -> String;

    /// A [`StreamingDecoder`] that reproduces this tokenizer's
    /// [`decode`](Self::decode) token by token.
    ///
    /// Fallible on the trait because one implementor is:
    /// [`AnyTokenizer`](super::AnyTokenizer) can
    /// hold a *declared* `decoder` pipeline whose steps cannot be evaluated one
    /// chunk at a time, and refuses with
    /// [`TokenizeError::UnstreamableDecoder`] rather than answering with a
    /// decode the pipeline exists to replace. The four concrete backends are
    /// infallible and always answer `Ok`.
    ///
    /// Each backend also keeps its own **inherent** `streaming_decoder`
    /// returning a bare [`StreamingDecoder`], and that one is what
    /// `tokenizer.streaming_decoder()` resolves to: Rust looks at inherent
    /// methods before trait methods, so a caller holding a concrete backend
    /// never has to unwrap a `Result` that cannot be `Err`. Reach this one
    /// explicitly (`Tokenize::streaming_decoder(&t)`) or through a generic
    /// `T: Tokenize` bound, which is where the fallible shape is needed.
    ///
    /// Deliberately not defaulted: a default body would have to invent a
    /// [`StreamingDecoder`] with no vocabulary behind it, and a backend that
    /// forgot to override it would stream nothing while reporting success.
    fn streaming_decoder(&self) -> Result<StreamingDecoder, TokenizeError>;

    /// The bytes `id` contributes to the decoded stream — ByteLevel alphabet
    /// unmapped and `<0xNN>` byte fallback resolved — and nothing else.
    ///
    /// No sequence-level post-processing runs: no leading-space strip, no
    /// first-token rule, no word separator. What comes back is what the id
    /// *itself* stands for, which is what makes it composable — concatenating it
    /// over a sequence of ids gives the bytes decoding sees before its post-ops,
    /// so a caller can align per-token spans against decoded text.
    ///
    /// An id in the **skip** set returns an **empty** `Vec`, not an error. That
    /// is its true contribution to the stream: every decode path in this crate
    /// treats a skip as a deliberate no-op and reserves errors for ids in no
    /// table at all, and erroring here would both conflate those two and break
    /// the concatenation property above. The skip set is whatever the
    /// implementation's own [`streaming_decoder`](Self::streaming_decoder) skips
    /// — a special that decode drops, and equally an id that *is* in the
    /// vocabulary but carries no surface, which every stream here silently drops
    /// rather than rejects.
    ///
    /// # Errors
    /// [`TokenizeError::InvalidTokenId`] when `id` is outside the vocabulary
    /// altogether — not merely absent from a table the stream would skip past.
    fn decode_token_bytes(&self, id: u32) -> Result<Vec<u8>, TokenizeError>;

    /// [`decode_token_bytes`](Self::decode_token_bytes) as text.
    ///
    /// # Errors
    /// [`TokenizeError::InvalidTokenId`] as above, and
    /// [`TokenizeError::Utf8Error`] when the id's bytes are not valid UTF-8
    /// **standing alone**.
    ///
    /// That second error is the normal case, not an edge case: a single `<0xNN>`
    /// byte-fallback id, or a token holding one byte of a multi-byte character,
    /// spells no complete character by itself and only becomes text once the
    /// neighbouring ids arrive. It is the expected signal to stop decoding
    /// id-at-a-time and use [`streaming_decoder`](Self::streaming_decoder),
    /// which buffers exactly those partial sequences across tokens — that is
    /// what the streaming decoder exists for.
    fn decode_token(&self, id: u32) -> Result<String, TokenizeError>;

    /// Return the vocabulary size (number of distinct tokens).
    fn vocab_size(&self) -> usize;
}

/// The bytes one id renders to under `rules`, as
/// [`Tokenize::decode_token_bytes`] reports them.
///
/// Every implementor renders through the [`RenderRules`] it already builds for
/// its own decode, so a per-id answer cannot drift from what that decode emits;
/// this is the one place the two decisions those rules defer — a skip is empty,
/// an id in no table is an error — are turned into that method's contract.
pub(crate) fn token_bytes_of(rules: &RenderRules, id: u32) -> Result<Vec<u8>, TokenizeError> {
    rules
        .token_bytes(id)
        .ok_or(TokenizeError::InvalidTokenId(id))
}

/// The text `bytes` spell, as [`Tokenize::decode_token`] reports it.
///
/// Shared by every implementor so none of them can decide differently what
/// "these bytes are not text on their own" means.
pub(crate) fn token_text_of(bytes: Vec<u8>) -> Result<String, TokenizeError> {
    String::from_utf8(bytes).map_err(|_| TokenizeError::Utf8Error)
}

/// Error type for the [`Tokenize`] trait's decode method.
#[derive(Debug, thiserror::Error)]
pub enum TokenizeError {
    #[error("Decoding error: invalid UTF-8")]
    Utf8Error,
    #[error("Decoding error: token ID {0} out of range")]
    InvalidTokenId(u32),
    /// The `tokenizer.json` declares a `decoder` pipeline whose named step
    /// cannot be evaluated one chunk at a time, so no streaming decoder can
    /// reproduce [`decode`](Tokenize::decode) for it. Refused rather than
    /// silently answered with the backend's own decode, which renders the raw
    /// pieces the declared pipeline exists to turn into text.
    #[error(
        "the declared decoder pipeline cannot be streamed: its `{0}` step is not incrementally computable — decode the whole sequence instead"
    )]
    UnstreamableDecoder(&'static str),
    #[error("{0}")]
    Other(String),
}
