//! Vocabulary loading utilities for the bundled vocabulary formats.
//!
//! Two formats live here:
//!
//! - **tiktoken** (`.tiktoken`) — `base64(token_bytes) rank` per line, used by
//!   OpenAI's byte-level tokenizers (GPT-3.5, GPT-4, GPT-4o, …).
//! - **SentencePiece** (`.spm`) — `base64(piece) score` per line in id order,
//!   see [`load_spm_vocab`]. A SentencePiece vocabulary cannot be stored in the
//!   tiktoken format without losing its scores and its `<0xNN>` byte-fallback
//!   spellings, which is why it has a format of its own.
//!
//! # Tiktoken Format
//!
//! The tiktoken format is a simple text-based format where each line contains:
//! - A base64-encoded token (the byte sequence)
//! - A space separator
//! - An integer rank (the token's priority in BPE merging)
//!
//! Lower ranks indicate higher priority - tokens with lower ranks are merged
//! first during the BPE encoding process.
//!
//! # Example Format
//!
//! ```text
//! SGVsbG8= 0
//! V29ybGQ= 1
//! IQ== 2
//! ```
//!
//! Where:
//! - `SGVsbG8=` decodes to `Hello` (rank 0, highest priority)
//! - `V29ybGQ=` decodes to `World` (rank 1)
//! - `IQ==` decodes to `!` (rank 2)
//!
//! # Vocabulary Files
//!
//! OpenAI provides vocabulary files for their models:
//! - `cl100k_base.tiktoken`: ~100k tokens for GPT-4, GPT-3.5-turbo
//! - `o200k_base.tiktoken`: ~200k tokens for GPT-4o

use base64::{engine::general_purpose::STANDARD, Engine};
use rustc_hash::{FxHashMap, FxHashSet};
use thiserror::Error;

use super::decode_table::Decoder;
use super::encoder::Encoder;

/// `USER_DEFINED` in SentencePiece's `ModelProto.SentencePiece.Type` enum.
///
/// The only type the loader has to act on. `NORMAL` merges, `CONTROL` is
/// unreachable from text, `BYTE` is reached through byte fallback and `UNKNOWN`
/// through the unk id — none of which needs a flag here.
const USER_DEFINED_PIECE_TYPE: u32 = 4;

/// A SentencePiece vocabulary as [`load_spm_vocab`] returns it: one entry per
/// id, in id order.
///
/// Three parallel vectors rather than one vector of structs, because every
/// consumer wants a whole column — `SpmTokenizer` indexes scores by id in its
/// merge loop, and the pieces go to the decode table untouched.
#[derive(Debug, Clone)]
pub struct SpmVocab {
    /// SentencePiece's own `id_to_piece`, keeping `<0x41>` and `▁` spellings.
    pub pieces: Vec<String>,
    /// `get_score`, which is what merge order is decided by.
    pub scores: Vec<f32>,
    /// Whether each piece is `USER_DEFINED` — matched verbatim, never merged.
    pub user_defined: Vec<bool>,
}

/// Type alias for encoder/decoder pair returned by `load_tiktoken_bpe_with_decoder`.
pub type EncoderDecoderPair = (FxHashMap<Vec<u8>, u32>, FxHashMap<u32, Vec<u8>>);

/// Errors that can occur when loading vocabulary files.
#[derive(Error, Debug)]
pub enum VocabError {
    #[error("Invalid base64 encoding: {0}")]
    Base64Error(#[from] base64::DecodeError),
    #[error("Invalid line format: {0}")]
    ParseError(String),
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Vocabulary is empty")]
    EmptyVocab,
    #[error("Special token {name:?} claims id {id}, which the vocabulary spells {found:?}")]
    SpecialTokenConflict {
        id: u32,
        name: String,
        found: String,
    },
    #[error("SentencePiece vocabulary line for id {id} has no space separating piece from score")]
    SpmMissingScore { id: u32 },
    #[error("SentencePiece piece for id {id} is not valid base64: {source}")]
    SpmBase64 {
        id: u32,
        source: base64::DecodeError,
    },
    #[error("SentencePiece score for id {id} is not a number: {value:?}")]
    SpmScore { id: u32, value: String },
    #[error("SentencePiece piece type for id {id} is not a number: {value:?}")]
    SpmPieceType { id: u32, value: String },
    #[error("SentencePiece piece for id {id} is not valid UTF-8")]
    SpmNonUtf8 { id: u32 },
}

/// Load a bundled SentencePiece vocabulary (`.spm`) as pieces and scores.
///
/// # Format
///
/// One line per token id, **in ascending id order with no gaps**:
///
/// ```text
/// <base64 of the piece, UTF-8 encoded> <score> <type>
/// ```
///
/// The id is the line's position, so it cannot be non-monotonic or duplicated
/// by construction — there is no id field to disagree with the ordering. The
/// piece is SentencePiece's own `id_to_piece`, so byte fallback keeps its real
/// `<0x41>` spelling instead of being reconstructed from a run of raw bytes,
/// and the `▁` word-boundary runs keep theirs.
///
/// The type is SentencePiece's own piece-type enum, and the returned
/// `user_defined` flags are the one thing a caller cannot work out for itself.
/// `USER_DEFINED` pieces are matched **verbatim before merging** and are never
/// merge candidates; `CONTROL` pieces are never matched from text at all. Both
/// score `0.0`, and both are spelled `<...>`, so neither the score nor the
/// spelling separates them — measured with `sentencepiece` 0.2.0 on Gemma 2,
/// `encode("<blockquote>")` (USER_DEFINED) is `[191]` while `encode("<pad>")`
/// (CONTROL) is `[235322, 8939, 235313]`, the piece shattered. Dropping the
/// distinction shatters every user-defined piece: Gemma 2 mistokenizes 3.2% of
/// real documents and Gemma 3 6.3%, because both declare HTML markers
/// (`<blockquote>`, `<table>`, …) and Gemma 3 declares its whitespace runs too.
///
/// A **two-column line carries no type** and is read as `NORMAL`. That is what
/// every `.spm` written before the type column meant implicitly, and it is
/// correct for those files precisely because none of them declares a
/// `USER_DEFINED` piece — Mistral V1/V2 and Llama 2 / Code Llama are
/// `NORMAL`/`CONTROL`/`BYTE`/`UNKNOWN` only.
///
/// # Why not `.tiktoken`
///
/// A `.tiktoken` line is `base64(token_bytes) rank`, which throws away the
/// score. SentencePiece merges by score, not by id order, and the 15 whitespace
/// pieces (`▁`, `▁▁`, …) carry a `-1e9` "never merge" sentinel that id order
/// inverts: with id-order merge ranks, `" Hello world"` comes out as
/// `▁▁` + `Hello` + `▁world` instead of `▁` + `▁Hello` + `▁world`.
///
/// Scores are written as the shortest decimal that round-trips the value, and
/// the vocabularies bundled here hold whole numbers plus the `-1e9` sentinel —
/// all exactly representable in `f32`, so the parse is lossless.
///
/// # Errors
///
/// Returns [`VocabError`] when the data is empty, a line has no separator, a
/// piece is not valid base64 or not valid UTF-8, or a score or type does not
/// parse.
pub fn load_spm_vocab(data: &[u8]) -> Result<SpmVocab, VocabError> {
    let mut pieces = Vec::new();
    let mut scores = Vec::new();
    let mut user_defined = Vec::new();

    for line in data.split(|&b| b == b'\n') {
        // Tolerate a trailing newline and CRLF line endings; a blank line
        // carries no id, so it is skipped rather than filling a slot.
        let line = match line.strip_suffix(b"\r") {
            Some(stripped) => stripped,
            None => line,
        };
        if line.is_empty() {
            continue;
        }
        let id = pieces.len() as u32;

        // Split left-to-right: base64 never contains a space, and neither a
        // score nor a type does, so the first space ends the piece and an
        // optional second ends the score. Splitting from the right instead
        // would read the type as the score on a three-column line.
        let space = line
            .iter()
            .position(|&b| b == b' ')
            .ok_or(VocabError::SpmMissingScore { id })?;
        let (Some(piece_b64), Some(rest)) = (line.get(..space), line.get(space + 1..)) else {
            return Err(VocabError::SpmMissingScore { id });
        };
        let (score_bytes, type_bytes) = match rest.iter().position(|&b| b == b' ') {
            Some(at) => (&rest[..at], Some(&rest[at + 1..])),
            None => (rest, None),
        };

        let bytes = STANDARD
            .decode(piece_b64)
            .map_err(|source| VocabError::SpmBase64 { id, source })?;
        let piece = String::from_utf8(bytes).map_err(|_| VocabError::SpmNonUtf8 { id })?;

        let score_str = std::str::from_utf8(score_bytes)
            .map_err(|_| VocabError::SpmScore {
                id,
                value: String::from_utf8_lossy(score_bytes).into_owned(),
            })?
            .trim();
        let score: f32 = score_str.parse().map_err(|_| VocabError::SpmScore {
            id,
            value: score_str.to_string(),
        })?;

        // Absent (a two-column legacy line) reads as NORMAL, i.e. not
        // user-defined — see this function's docs for why that is safe for the
        // files written before the column existed.
        let is_user_defined = match type_bytes {
            Some(bytes) => {
                let text = std::str::from_utf8(bytes)
                    .map_err(|_| VocabError::SpmPieceType {
                        id,
                        value: String::from_utf8_lossy(bytes).into_owned(),
                    })?
                    .trim();
                let value: u32 = text.parse().map_err(|_| VocabError::SpmPieceType {
                    id,
                    value: text.to_string(),
                })?;
                value == USER_DEFINED_PIECE_TYPE
            }
            None => false,
        };

        pieces.push(piece);
        scores.push(score);
        user_defined.push(is_user_defined);
    }

    if pieces.is_empty() {
        return Err(VocabError::EmptyVocab);
    }
    Ok(SpmVocab {
        pieces,
        scores,
        user_defined,
    })
}

/// Load a tiktoken BPE vocabulary from raw bytes.
///
/// Format: `base64_token rank\n` per line
/// Example: `SGVsbG8= 0` (where "SGVsbG8=" decodes to "Hello")
pub fn load_tiktoken_bpe(data: &[u8]) -> Result<FxHashMap<Vec<u8>, u32>, VocabError> {
    let mut encoder = FxHashMap::default();

    for line in data.split(|&b| b == b'\n') {
        if line.is_empty() {
            continue;
        }

        // Find the space separator
        let space_pos = line
            .iter()
            .rposition(|&b| b == b' ')
            .ok_or_else(|| VocabError::ParseError("Missing space separator".to_string()))?;

        let token_b64 = &line[..space_pos];
        let rank_str = &line[space_pos + 1..];

        // Decode base64 token
        let token = STANDARD.decode(token_b64)?;

        // Parse rank
        let rank_str = std::str::from_utf8(rank_str)
            .map_err(|_| VocabError::ParseError("Invalid UTF-8 in rank".to_string()))?;
        let rank: u32 = rank_str
            .trim()
            .parse()
            .map_err(|_| VocabError::ParseError(format!("Invalid rank: {}", rank_str)))?;

        encoder.insert(token, rank);
    }

    Ok(encoder)
}

/// Magic identifying a packed vocabulary. Bumping the trailing digit is how a
/// format change announces itself, so an old crate refuses a new file instead
/// of misreading it.
const PACKED_MAGIC: &[u8; 8] = b"SPLNTRV1";

/// Load a BPE vocabulary from the packed form `splintr-vocab-pack` writes.
///
/// Each `splintr-vocab-*` crate ships its `.tiktoken` text and packs it in its
/// build script, so this reads something derived at compile time rather than a
/// binary committed to the repository. The format is documented on that crate.
///
/// # Format
///
/// ```text
/// magic    8 bytes   b"SPLNTRV1"
/// count    u32 LE    number of entries
/// entries  count x   varint(rank), varint(len), len raw token bytes
/// ```
///
/// # Why this exists alongside [`load_tiktoken_bpe`]
///
/// The text form spends four base64 characters per three token bytes and writes
/// each rank in decimal, so it is ~47% larger than the ranks it carries. That
/// pushed the published crate past the 10 MiB crates.io limit. This form is what
/// the bundled vocabularies embed; `.tiktoken` remains the interchange format
/// that [`load_tiktoken_bpe`] and `Tokenizer::from_file` read, and
/// `tests/vocab_packed_parity.rs` fails if the two ever disagree.
///
/// Decoding is also strictly less work than the text path — no base64, no
/// decimal parse — which `from_pretrained` pays on every process start.
///
/// This form **copies** each token, for callers whose data is not `'static`.
/// The bundled vocabularies use [`load_packed_bpe_borrowed`] instead and copy
/// nothing.
///
/// Ranks are absolute rather than implied by position. Every bundled vocabulary
/// happens to be contiguous, but the tiktoken format guarantees no such thing,
/// and a positional format would silently renumber a vocabulary with a gap
/// instead of refusing it.
pub fn load_packed_bpe(data: &[u8]) -> Result<Encoder, VocabError> {
    let count = packed_header(data)?;
    packed_into_arena(data, count, &FxHashSet::default())
}

/// Load a packed vocabulary **without copying any token bytes**.
///
/// The zero-copy counterpart to [`load_packed_bpe`], and the reason the packed
/// format exists in the shape it does: every token sits contiguously inside
/// `data`, so a key can point at it instead of owning a copy. `data` must be
/// `'static` — in practice the `include_bytes!` payload in `pretrained.rs`.
///
/// Measured against the copying form on `cl100k_base`, ~3.7x faster. The saving
/// is 100k-200k small allocations and their `memcpy`s, not parsing: both walk
/// the same bytes in the same order.
///
/// The text form has no equivalent, and cannot: base64 tokens do not exist as
/// contiguous bytes anywhere until something decodes them.
pub fn load_packed_bpe_borrowed(data: &'static [u8]) -> Result<Encoder, VocabError> {
    let count = packed_header(data)?;
    packed_into_arena(data, count, &FxHashSet::default())
}

/// Load a packed vocabulary, leaving `skip` out of the encode table.
///
/// For a vocabulary whose merge list proves some entries unreachable — see
/// [`orphan_ids`]. Decoding still needs them, so build the decode table with
/// [`decoder_from_packed`], which omits nothing.
pub fn load_packed_bpe_without(
    data: &'static [u8],
    skip: &FxHashSet<u32>,
) -> Result<Encoder, VocabError> {
    let count = packed_header(data)?;
    packed_into_arena(data, count, skip)
}

/// Build a decode table directly from packed bytes.
///
/// Every id the vocabulary states, including ones the encode table declines.
/// Going through an `Encoder` first would build a hash table this never
/// consults, and — once the encode table starts omitting entries — would give a
/// decoder that cannot spell them.
pub fn decoder_from_packed(data: &[u8]) -> Result<Decoder, VocabError> {
    let count = packed_header(data)?;
    let mut decoder = Decoder::with_capacity(count);
    walk_packed(data, count, |token, id| decoder.insert(id, token))?;
    Ok(decoder)
}

/// Build an encoder whose arena *is* the packed buffer.
///
/// Every token already sits contiguously inside `data`, so the encoder takes
/// one copy of the whole buffer and records where each token is, rather than
/// copying tokens out one at a time. The framing bytes between tokens ride
/// along unused — a few hundred KB against 100k-200k separate copies.
fn packed_into_arena(
    data: &[u8],
    count: usize,
    skip: &FxHashSet<u32>,
) -> Result<Encoder, VocabError> {
    let mut encoder = Encoder::with_arena(data.to_vec(), count - skip.len().min(count));
    let base = data.as_ptr() as usize;
    walk_packed(data, count, |token, rank| {
        if skip.contains(&rank) {
            return;
        }
        let offset = (token.as_ptr() as usize - base) as u32;
        encoder.insert_span(offset, token.len() as u32, rank);
    })?;
    Ok(encoder)
}

/// Validate a packed header and return its entry count.
fn packed_header(data: &[u8]) -> Result<usize, VocabError> {
    if data.len() < 12 || &data[..8] != PACKED_MAGIC {
        return Err(VocabError::ParseError(
            "not a packed vocabulary: bad magic".to_string(),
        ));
    }
    let count = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
    if count == 0 {
        return Err(VocabError::EmptyVocab);
    }
    Ok(count)
}

/// Walk a packed vocabulary's entries, handing each `(token, rank)` to `visit`.
///
/// Shared by the owning and borrowing loaders so the two cannot drift into
/// disagreeing about the format — the only difference between them is what
/// `visit` does with the slice.
fn walk_packed<'a>(
    data: &'a [u8],
    count: usize,
    mut visit: impl FnMut(&'a [u8], u32),
) -> Result<(), VocabError> {
    let mut pos = 12;
    for _ in 0..count {
        let rank = read_varint(data, &mut pos)?;
        let len = read_varint(data, &mut pos)? as usize;
        let end = pos.checked_add(len).ok_or_else(|| {
            VocabError::ParseError("packed vocabulary: token length overflows".to_string())
        })?;
        if end > data.len() {
            return Err(VocabError::ParseError(
                "packed vocabulary: token runs past end of data".to_string(),
            ));
        }
        visit(&data[pos..end], rank);
        pos = end;
    }
    Ok(())
}

/// Magic at the head of a packed merge list.
const PACKED_MERGES_MAGIC: &[u8; 8] = b"SPLNTRM1";

/// One merge rule: what it produces, and where its two operands meet.
#[derive(Clone, Copy, Debug)]
pub struct MergeRule<'a> {
    /// The token the merge produces. Always a vocabulary entry.
    pub result: &'a [u8],
    /// Byte length of the left operand, so the pair is `result[..split]` and
    /// `result[split..]`. Never 0 and never `result.len()`.
    pub split: usize,
}

impl<'a> MergeRule<'a> {
    /// The two tokens this rule joins.
    pub fn operands(&self) -> (&'a [u8], &'a [u8]) {
        self.result.split_at(self.split)
    }
}

/// Load a packed merge list: the token ids a vocabulary merges in, in priority
/// order.
///
/// # Why a vocabulary can need this at all
///
/// A `.tiktoken`-shaped vocabulary states one rank per token, serving as both
/// its id and its merge priority, and [`load_packed_bpe`] is the whole of it.
/// A HuggingFace BPE need not have those coincide. Gemma 4's do not — 465
/// places where a later merge yields a lower id, and 514,906 merges collapsing
/// onto 236,339 distinct tokens — and forcing them into one column mistokenizes
/// 8.1% of real documents. Such a vocabulary ships its merge order separately,
/// which is this.
///
/// # Format
///
/// ```text
/// "SPLNTRM1"                 magic, 8 bytes
/// count                      u32, little-endian
/// (varint(id) varint(split))* count rules, in merge-priority order
/// ```
///
/// Ids rather than token bytes: every merge result is itself a vocabulary
/// entry, so its bytes are already in the companion vocabulary. Reading them
/// back through `pieces` is what ties the two together, and a merge id outside
/// the vocabulary is an error rather than a skipped entry — it would mean the
/// two files were packed from different sources.
///
/// `split` records where the rule's two operands met. It is not needed to rank
/// the merges — the order alone does that — but it is the only way to tell a
/// vocabulary entry that merges build FROM (an atom) from one that no merge
/// touches at all (an orphan), which [`orphan_ids`] needs and encoding
/// correctness depends on.
pub fn load_packed_merge_rules<'a>(
    data: &'a [u8],
    pieces: &'a Decoder,
) -> Result<Vec<MergeRule<'a>>, VocabError> {
    if data.len() < 12 || &data[..8] != PACKED_MERGES_MAGIC {
        return Err(VocabError::ParseError(
            "not a packed merge list: bad magic".to_string(),
        ));
    }
    let count = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
    if count == 0 {
        return Err(VocabError::ParseError("no merges".to_string()));
    }

    let mut rules: Vec<MergeRule<'_>> = Vec::with_capacity(count);
    let mut pos = 12;
    for _ in 0..count {
        let id = read_varint(data, &mut pos)?;
        let split = read_varint(data, &mut pos)? as usize;
        let result = pieces.get(id).ok_or_else(|| {
            VocabError::ParseError(format!(
                "packed merge list: id {id} is not in the vocabulary — the two \
                 were not packed from the same file"
            ))
        })?;
        if split == 0 || split >= result.len() {
            return Err(VocabError::ParseError(format!(
                "packed merge list: id {id} splits at {split}, which leaves an \
                 empty half of a {}-byte token",
                result.len()
            )));
        }
        rules.push(MergeRule { result, split });
    }
    Ok(rules)
}

/// The ids BPE can never produce.
///
/// An entry is reachable exactly when merging its own surface produces it whole,
/// which is what `ranks` is for: the test is a merge, not a scan of the merge
/// list. Naming is the cheaper question and the wrong one — a merge list can
/// name an entry the merge ORDER never reaches, because the operands that would
/// produce it are consumed by earlier merges and never meet. Gemma 4's `▁yyyy`
/// is named by `▁yy ++ yy` and by `▁ ++ yyyy`, and reachable by neither.
///
/// `ranks` is the same bytes→priority map the tokenizer will merge with (see
/// [`crate::core::bpe::merge_ranks_bytes`]), so this asks precisely the question
/// the encode path will answer.
///
/// Unreachable entries are not a defect in the vocabulary; they are reserved
/// ids, markers a caller may emit deliberately, and pieces the trainer kept out
/// of the merge table on purpose. Gemma 4 has 6,326 of them, `<blockquote>` and
/// `<unused0>` among them — 6,322 that no merge names, and four more that the
/// merge order routes around. Decoding must still spell every one.
///
/// What must not happen is *encoding* into one. The whole-chunk fast path — ask
/// the vocabulary whether the entire chunk is a token before merging — answers
/// `<blockquote>` with a single id where BPE, given the same bytes, produces
/// three. Both are ids for the same text, which is precisely the disagreement a
/// tokenizer cannot have. So the encode table is built without them, and this
/// is the set it leaves out.
///
/// Two kinds of entry are never returned, however few merges mention them:
///
/// - **one symbol**. A merge joins two non-empty tokens, so anything BPE could
///   wrongly merge into is at least two symbols, and seeding must keep every
///   one-symbol spelling it might start from.
/// - **`<0xNN>` byte-fallback pieces**. Six characters and often named by no
///   merge at all, but reachable — byte fallback spells a raw byte as one, and
///   dropping them leaves a vocabulary that cannot encode the bytes it declares
///   a fallback for. Recognized by shape rather than by the model's
///   `byte_fallback` flag: keeping an entry encodable is the safe direction,
///   since the failure this guards against is a merge that should not happen,
///   never one that should.
pub fn orphan_ids(
    pieces: &Decoder,
    rules: &[MergeRule<'_>],
    ranks: &crate::core::encoder::Encoder,
) -> FxHashSet<u32> {
    // What the merge can produce at all, and from which pair: an entry no rule
    // results in is one no merge can build, which settles it without running
    // anything, and the pair settles the shortest entries the same way.
    let mut reachable: rustc_hash::FxHashMap<&[u8], usize> = rustc_hash::FxHashMap::default();
    reachable.reserve(rules.len());
    for rule in rules {
        reachable.insert(rule.result, rule.split);
    }
    let unreachable = |bytes: &[u8]| -> bool {
        // A one-symbol entry is what seeding starts from, and a `<0xNN>` piece
        // is how byte fallback spells a raw byte; neither is ever dropped.
        if is_byte_fallback_piece(bytes) || bytes.iter().filter(|b| !is_utf8_tail(**b)).count() <= 1
        {
            return false;
        }
        let Some(&split) = reachable.get(bytes) else {
            return true;
        };
        // Two symbols, one per operand: seeding produces exactly this rule's
        // pair, and the rule ranks it, so the merge fires and the entry is
        // reachable. The commonest shape in a vocabulary, answered without
        // running anything.
        let symbols = |b: &[u8]| b.iter().filter(|b| !is_utf8_tail(**b)).count();
        let (left, right) = MergeRule {
            result: bytes,
            split,
        }
        .operands();
        if symbols(left) == 1 && symbols(right) == 1 {
            return false;
        }
        // A packed vocabulary carrying its own merge order is a HuggingFace
        // model's, which merges over characters.
        !crate::core::bpe::merges_to_whole(bytes, crate::core::bpe::RankLookup::new(ranks), true)
    };

    // One BPE run per entry, and every entry independent — the same parallel
    // pass the json loader's `unreachable_tokens` makes, for the same reason.
    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        let entries: Vec<(u32, &[u8])> = pieces.iter().collect();
        entries
            .par_iter()
            .filter(|(_, bytes)| unreachable(bytes))
            .map(|(id, _)| *id)
            .collect::<Vec<u32>>()
            .into_iter()
            .collect()
    }
    #[cfg(not(feature = "rayon"))]
    {
        pieces
            .iter()
            .filter(|(_, bytes)| unreachable(bytes))
            .map(|(id, _)| id)
            .collect()
    }
}

/// Is this the `<0xNN>` spelling SentencePiece gives a raw byte?
pub(crate) fn is_byte_fallback_piece(token: &[u8]) -> bool {
    token.len() == 6
        && token.starts_with(b"<0x")
        && token[5] == b'>'
        && token[3..5].iter().all(u8::is_ascii_hexdigit)
}

/// Is this a UTF-8 continuation byte? Used to count characters without decoding:
/// a token's symbol count is its non-continuation bytes, and invalid UTF-8 —
/// which a byte-level vocabulary is full of — counts every byte, which is the
/// conservative answer.
fn is_utf8_tail(byte: u8) -> bool {
    byte & 0xC0 == 0x80
}

fn read_varint(data: &[u8], pos: &mut usize) -> Result<u32, VocabError> {
    let mut value: u32 = 0;
    for group in 0..5 {
        let byte = *data.get(*pos).ok_or_else(|| {
            VocabError::ParseError("packed vocabulary: truncated varint".to_string())
        })?;
        *pos += 1;
        value |= u32::from(byte & 0x7F)
            .checked_shl(group * 7)
            .ok_or_else(|| {
                VocabError::ParseError("packed vocabulary: varint too wide".to_string())
            })?;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
    }
    Err(VocabError::ParseError(
        "packed vocabulary: varint too wide".to_string(),
    ))
}

/// Load a tiktoken BPE vocabulary from a file path.
pub fn load_tiktoken_bpe_file(path: &str) -> Result<FxHashMap<Vec<u8>, u32>, VocabError> {
    let data = std::fs::read(path)?;
    load_tiktoken_bpe(&data)
}

/// Load a tiktoken BPE vocabulary and build both encoder and decoder.
///
/// This function preserves all token IDs in the decoder, even if multiple IDs map to the same
/// byte sequence. The encoder will only keep the FIRST occurrence of each byte sequence (lowest ID).
pub fn load_tiktoken_bpe_with_decoder(data: &[u8]) -> Result<EncoderDecoderPair, VocabError> {
    let mut encoder = FxHashMap::default();
    let mut decoder = FxHashMap::default();

    for line in data.split(|&b| b == b'\n') {
        if line.is_empty() {
            continue;
        }

        // Find the space separator
        let space_pos = line
            .iter()
            .rposition(|&b| b == b' ')
            .ok_or_else(|| VocabError::ParseError("Missing space separator".to_string()))?;

        let token_b64 = &line[..space_pos];
        let rank_str = &line[space_pos + 1..];

        // Decode base64 token
        let token = STANDARD.decode(token_b64)?;

        // Parse rank
        let rank_str = std::str::from_utf8(rank_str)
            .map_err(|_| VocabError::ParseError("Invalid UTF-8 in rank".to_string()))?;
        let rank: u32 = rank_str
            .trim()
            .parse()
            .map_err(|_| VocabError::ParseError(format!("Invalid rank: {}", rank_str)))?;

        // Always add to decoder (preserves all token IDs)
        decoder.insert(rank, token.clone());

        // Only add to encoder if this byte sequence isn't already mapped
        // This keeps the FIRST (lowest ID) occurrence
        encoder.entry(token).or_insert(rank);
    }

    Ok((encoder, decoder))
}

/// Place named special tokens into a piece list at the ids they claim.
///
/// Bundled vocabularies carry special tokens the vocabulary *file* does not:
/// splintr's agent tokens sit above the file's last id. Without a slot in the
/// piece list those ids exist only in the matcher — encodable but not
/// decodable, and absent from `vocab_size`. The list is grown to cover them
/// (holes stay empty, which no input can produce and no merge can look up).
///
/// A special that lands on an id the file already spells differently is a
/// disagreement between the two, not something to overwrite: it is reported, so
/// a vocabulary bump that shifts ids fails loudly instead of quietly mapping a
/// chat marker onto a real word.
pub fn place_special_pieces(
    pieces: &mut Vec<String>,
    special: &FxHashMap<String, u32>,
) -> Result<(), VocabError> {
    let Some(&max_id) = special.values().max() else {
        return Ok(());
    };
    if pieces.len() <= max_id as usize {
        pieces.resize(max_id as usize + 1, String::new());
    }
    for (name, &id) in special {
        let Some(slot) = pieces.get_mut(id as usize) else {
            continue;
        };
        if slot.is_empty() {
            *slot = name.clone();
        } else if slot.as_str() != name.as_str() {
            return Err(VocabError::SpecialTokenConflict {
                id,
                name: name.clone(),
                found: slot.clone(),
            });
        }
    }
    Ok(())
}

/// Build a decoder map (token ID → bytes) from an encoder map (bytes → token ID).
///
/// This creates the inverse mapping needed for decoding tokens back to text.
/// The decoder is used during the decode phase to convert token IDs back to
/// their original byte sequences.
///
/// The bytes are copied once into the table's own buffer, which is two
/// allocations for a whole vocabulary where inverting into a map cost one per
/// token.
pub fn build_decoder(encoder: &Encoder) -> Decoder {
    Decoder::from_encoder(encoder)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_tiktoken_bpe() {
        // "Hello" base64 = "SGVsbG8="
        // "World" base64 = "V29ybGQ="
        let data = b"SGVsbG8= 0\nV29ybGQ= 1\n";
        let encoder = load_tiktoken_bpe(data).unwrap();

        assert_eq!(encoder.get(b"Hello".as_slice()), Some(&0));
        assert_eq!(encoder.get(b"World".as_slice()), Some(&1));
        assert_eq!(encoder.len(), 2);
    }

    /// Special tokens above the file's last id must gain a slot, so they decode
    /// and count towards `vocab_size` rather than existing only in the matcher.
    #[test]
    fn place_special_pieces_grows_the_list_to_cover_added_ids() {
        let mut pieces = vec!["<unk>".to_string(), "a".to_string()];
        let mut special = FxHashMap::default();
        special.insert("<unk>".to_string(), 0);
        special.insert("<|pad|>".to_string(), 4);

        place_special_pieces(&mut pieces, &special).unwrap();
        assert_eq!(pieces.len(), 5);
        assert_eq!(pieces[4], "<|pad|>");
        // The hole stays empty — no input produces it and no merge looks it up.
        assert_eq!(pieces[2], "");
        assert_eq!(pieces[3], "");
    }

    /// A special landing on an id the file already spells differently is a
    /// disagreement between the two. Overwriting would silently map a chat
    /// marker onto a real word, so it is reported.
    #[test]
    fn place_special_pieces_reports_a_claimed_id_that_holds_another_token() {
        let mut pieces = vec!["<unk>".to_string(), "▁the".to_string()];
        let mut special = FxHashMap::default();
        special.insert("<|im_start|>".to_string(), 1);

        assert!(matches!(
            place_special_pieces(&mut pieces, &special),
            Err(VocabError::SpecialTokenConflict { id: 1, .. })
        ));
    }

    /// Build a `.spm` blob from `(piece, score)` pairs listed in id order.
    fn spm_blob(entries: &[(&str, &str)]) -> Vec<u8> {
        let mut out = Vec::new();
        for (piece, score) in entries {
            out.extend_from_slice(STANDARD.encode(piece.as_bytes()).as_bytes());
            out.extend_from_slice(format!(" {score}\n").as_bytes());
        }
        out
    }

    /// A three-column blob, i.e. one carrying SentencePiece piece types.
    fn spm_typed_blob(entries: &[(&str, &str, u32)]) -> Vec<u8> {
        let mut out = Vec::new();
        for (piece, score, kind) in entries {
            out.extend_from_slice(STANDARD.encode(piece.as_bytes()).as_bytes());
            out.extend_from_slice(format!(" {score} {kind}\n").as_bytes());
        }
        out
    }

    /// `USER_DEFINED` is the one piece type the loader has to surface, and it
    /// is separable from `CONTROL` only by the type column.
    ///
    /// Both score `0.0` and both are spelled `<...>`, yet SentencePiece matches
    /// one from text and never the other. Measured with `sentencepiece` 0.2.0 on
    /// Gemma 2: `encode("<blockquote>")` (USER_DEFINED, id 191) is `[191]`,
    /// while `encode("<pad>")` (CONTROL, id 0) is `[235322, 8939, 235313]` —
    /// the piece shattered into `<` + `pad` + `>`.
    #[test]
    fn spm_vocab_separates_user_defined_from_control() {
        let data = spm_typed_blob(&[
            ("<pad>", "0.0", 3),        // CONTROL
            ("<unk>", "0.0", 2),        // UNKNOWN
            ("<0x41>", "0.0", 6),       // BYTE
            ("<blockquote>", "0.0", 4), // USER_DEFINED
            ("▁the", "-31.0", 1),       // NORMAL
        ]);
        let v = load_spm_vocab(&data).unwrap();
        assert_eq!(v.pieces.len(), 5);
        assert_eq!(
            v.user_defined,
            vec![false, false, false, true, false],
            "only the USER_DEFINED piece may be flagged — CONTROL and BYTE \
             score 0.0 as well and are spelled the same way"
        );
        // The type column must not be mistaken for the score.
        assert_eq!(v.scores, vec![0.0, 0.0, 0.0, 0.0, -31.0]);
    }

    /// A type that is not a number is an error, not a silent `NORMAL`: reading
    /// it as normal would put a user-defined piece back into the merge loop,
    /// which is exactly the failure the column exists to prevent.
    #[test]
    fn spm_vocab_rejects_an_unparseable_piece_type() {
        let data = b"PHBhZD4= 0.0 control\n".to_vec();
        assert!(matches!(
            load_spm_vocab(&data),
            Err(VocabError::SpmPieceType { id: 0, .. })
        ));
    }

    /// The point of the format: pieces arrive spelled exactly as SentencePiece
    /// spells them — byte fallback as `<0xNN>`, word boundaries as `▁` — and the
    /// scores arrive alongside them instead of being inferred from id order.
    #[test]
    fn spm_vocab_keeps_piece_spelling_and_scores() {
        let data = spm_blob(&[
            ("<unk>", "0.0"),
            ("<0x41>", "0.0"),
            ("▁the", "-31.0"),
            ("▁▁", "-1000000000.0"),
        ]);
        let v = load_spm_vocab(&data).unwrap();

        assert_eq!(v.pieces, vec!["<unk>", "<0x41>", "▁the", "▁▁"]);
        assert_eq!(v.scores, vec![0.0, 0.0, -31.0, -1e9]);
        // Two-column lines carry no type, so nothing is user-defined.
        assert_eq!(v.user_defined, vec![false; 4]);
    }

    /// The `-1e9` "never merge" sentinel is the whole reason scores are stored:
    /// it must survive the text round-trip bit-for-bit, not land near `-1e9`.
    #[test]
    fn spm_vocab_parses_the_never_merge_sentinel_exactly() {
        let data = spm_blob(&[("▁", "-1000000000.0")]);
        let scores = load_spm_vocab(&data).unwrap().scores;
        assert_eq!(scores.first().copied(), Some(-1e9f32));
        assert_eq!(
            scores.first().map(|s| s.to_bits()),
            Some((-1e9f32).to_bits())
        );
    }

    /// A trailing newline and CRLF endings carry no id, so they must not shift
    /// every later piece by one slot.
    #[test]
    fn spm_vocab_ignores_blank_and_carriage_return_line_endings() {
        let mut data = spm_blob(&[("a", "-1.0"), ("b", "-2.0")]);
        data.extend_from_slice(b"\n");
        assert_eq!(load_spm_vocab(&data).unwrap().pieces, vec!["a", "b"]);

        let crlf = b"YQ== -1.0\r\nYg== -2.0\r\n";
        let v = load_spm_vocab(crlf).unwrap();
        assert_eq!(v.pieces, vec!["a", "b"]);
        assert_eq!(v.scores, vec![-1.0, -2.0]);
    }

    /// A line without a separator has no score, and inventing one would put a
    /// piece at an id whose merge priority is a guess.
    #[test]
    fn spm_vocab_rejects_a_line_without_a_score() {
        assert!(matches!(
            load_spm_vocab(b"YQ== -1.0\nYg==\n"),
            Err(VocabError::SpmMissingScore { id: 1 })
        ));
    }

    /// Bad base64, a non-UTF-8 piece, and an unparseable score are each
    /// reported against the id that carries them.
    #[test]
    fn spm_vocab_reports_malformed_fields_with_their_id() {
        assert!(matches!(
            load_spm_vocab(b"YQ== -1.0\n!!!! -2.0\n"),
            Err(VocabError::SpmBase64 { id: 1, .. })
        ));
        // `loE=` decodes to `96 81` — `▁` (`E2 96 81`) with its lead byte lost,
        // which is not valid UTF-8 and so cannot be a piece.
        assert!(matches!(
            load_spm_vocab(b"YQ== -1.0\nloE= -2.0\n"),
            Err(VocabError::SpmNonUtf8 { id: 1 })
        ));
        assert!(matches!(
            load_spm_vocab(b"YQ== -1.0\nYg== rank\n"),
            Err(VocabError::SpmScore { id: 1, .. })
        ));
    }

    /// A `.tiktoken` file fed to the SentencePiece loader must not be accepted
    /// as if it were one: its raw high bytes are not valid UTF-8 pieces.
    #[test]
    fn spm_vocab_rejects_a_tiktoken_file() {
        let mut data = Vec::new();
        data.extend_from_slice(STANDARD.encode([0x80u8]).as_bytes());
        data.extend_from_slice(b" 0\n");
        assert!(matches!(
            load_spm_vocab(&data),
            Err(VocabError::SpmNonUtf8 { id: 0 })
        ));
    }

    #[test]
    fn spm_vocab_rejects_empty_data() {
        assert!(matches!(load_spm_vocab(b""), Err(VocabError::EmptyVocab)));
    }

    #[test]
    fn test_build_decoder() {
        let mut encoder = Encoder::default();
        encoder.insert(b"Hello", 0);
        encoder.insert(b"World", 1);

        let decoder = build_decoder(&encoder);
        assert_eq!(decoder.get(0), Some(&b"Hello"[..]));
        assert_eq!(decoder.get(1), Some(&b"World"[..]));
    }
}
