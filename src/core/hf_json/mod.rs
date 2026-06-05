//! Generic HuggingFace `tokenizer.json` loader.
//!
//! [`from_json_path`] / [`from_json_bytes`] read any HF `tokenizer.json` and
//! dispatch on `model.type` to the matching splintr backend:
//!
//! | `model.type` | Backend                       | Example models                     |
//! |--------------|-------------------------------|------------------------------------|
//! | `BPE`        | [`Tokenizer`] (byte-level/raw)| GPT-2, Whisper, Llama 3, Qwen      |
//! | `Unigram`    | [`SentencePieceTokenizer`]    | T5, Gemma, Albert, XLNet           |
//! | `WordPiece`  | [`WordPieceTokenizer`]         | BERT, DistilBERT, Electra          |
//!
//! Everything needed is read from the file itself — "you supply the json, you
//! supply the tokens": the split regex, byte-level flag, BPE **merge order**
//! (independent of token ids, so RoBERTa-style vocabs work), the normalizer
//! (including SentencePiece's exact `Precompiled` charsmap), and special tokens.
//! Output is verified id-for-id against HuggingFace `tokenizers` across all three
//! families (GPT-2/RoBERTa/Qwen/Whisper; T5/Albert/XLNet; BERT/DistilBERT).
//!
//! For the bundled, zero-config vocabularies (including Whisper multilingual),
//! prefer [`crate::pretrained::from_pretrained`].

mod components;

use rustc_hash::FxHashMap;
use serde_json::Value;
use thiserror::Error;

use super::byte_level::byte_level_decode;
use super::sentencepiece::{SentencePieceError, SentencePieceTokenizer};
use super::tokenize::{Tokenize, TokenizeError};
use super::tokenizer::{Tokenizer, TokenizerError};
use super::wordpiece::WordPieceTokenizer;

use super::normalizer::Normalizer;
use components::{
    find_added_token, parse_bert_norm, parse_norm_ops, parse_pre_tokenizer,
    parse_special_decode_ids, parse_special_tokens,
};

/// Errors from loading a HuggingFace `tokenizer.json`.
#[derive(Debug, Error)]
pub enum HfJsonError {
    #[error("failed to parse tokenizer.json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("tokenizer.json missing field: {0}")]
    MissingField(&'static str),
    #[error("unsupported model.type `{0}` (expected BPE, Unigram, or WordPiece)")]
    UnsupportedModelType(String),
    #[error(
        "unsupported normalizer type(s) `{0}` — refusing to load rather than silently drop them"
    )]
    UnsupportedNormalizer(String),
    #[error("normalizer Replace pattern `{0}` failed to compile as a regex")]
    InvalidNormalizerRegex(String),
    #[error("unsupported pre_tokenizer type(s) `{0}` and no recognized split — refusing to guess the split pattern")]
    UnsupportedPreTokenizer(String),
    #[error("vocab entry `{0}` is not valid byte-level encoding")]
    InvalidByteLevel(String),
    #[error("could not determine the {0} token id from the tokenizer.json")]
    MissingSpecial(&'static str),
    #[error(transparent)]
    Tokenizer(#[from] TokenizerError),
    #[error(transparent)]
    SentencePiece(#[from] SentencePieceError),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// A tokenizer loaded from a `tokenizer.json`, tagged by its backend family.
///
/// Implements [`Tokenize`] so callers can encode/decode generically, or match
/// on the variant for backend-specific APIs.
pub enum Backend {
    /// Byte-pair encoding (byte-level or raw).
    Bpe(Tokenizer),
    /// Unigram / SentencePiece.
    Unigram(SentencePieceTokenizer),
    /// WordPiece (BERT family).
    WordPiece(WordPieceTokenizer),
}

impl Backend {
    /// The raw surface string of a token id, used to feed a declared decoder
    /// pipeline.
    fn token_surface(&self, id: u32) -> Option<String> {
        match self {
            Backend::Bpe(t) => t.token_surface(id),
            Backend::Unigram(t) => t.token_surface(id),
            Backend::WordPiece(t) => t.token_surface(id),
        }
    }
}

/// The `post_processor` template for a single sequence: special-token ids added
/// before and after the content tokens (e.g. BERT's `[CLS]` … `[SEP]`).
#[derive(Default, Clone)]
pub struct PostProcessor {
    prefix: Vec<u32>,
    suffix: Vec<u32>,
}

impl PostProcessor {
    /// Wrap content `ids` with the template's special tokens.
    pub fn apply(&self, ids: Vec<u32>) -> Vec<u32> {
        if self.prefix.is_empty() && self.suffix.is_empty() {
            return ids;
        }
        let mut out = Vec::with_capacity(self.prefix.len() + ids.len() + self.suffix.len());
        out.extend_from_slice(&self.prefix);
        out.extend(ids);
        out.extend_from_slice(&self.suffix);
        out
    }

    /// Whether this template adds no tokens.
    pub fn is_empty(&self) -> bool {
        self.prefix.is_empty() && self.suffix.is_empty()
    }
}

/// A tokenizer loaded from a `tokenizer.json`: a backend family plus the
/// `post_processor` template.
///
/// [`encode`](Tokenize::encode) returns the content tokens (HF's
/// `add_special_tokens=False`); [`encode_with_special_tokens`] additionally
/// applies the post-processor template (HF's default `encode`).
pub struct AnyTokenizer {
    backend: Backend,
    post: PostProcessor,
    /// The `decoder` pipeline declared in the json. When present it drives
    /// decoding (config-driven); when absent the backend's built-in decode runs.
    decoder: Option<super::decoder::Decoder>,
    /// Ids of `special=true` added tokens, skipped before the decoder pipeline.
    special_decode: rustc_hash::FxHashSet<u32>,
}

impl AnyTokenizer {
    /// The `model.type` family name this was built from.
    pub fn family(&self) -> &'static str {
        match &self.backend {
            Backend::Bpe(_) => "BPE",
            Backend::Unigram(_) => "Unigram",
            Backend::WordPiece(_) => "WordPiece",
        }
    }

    /// Borrow the backend tokenizer (to reach backend-specific APIs).
    pub fn backend(&self) -> &Backend {
        &self.backend
    }

    /// Consume into the backend tokenizer.
    pub fn into_backend(self) -> Backend {
        self.backend
    }

    /// The parsed `post_processor` template.
    pub fn post_processor(&self) -> &PostProcessor {
        &self.post
    }

    /// Encode and apply the post-processor template (matching HF's default
    /// `encode`, i.e. `add_special_tokens=True`).
    pub fn encode_with_special_tokens(&self, text: &str) -> Vec<u32> {
        self.post.apply(Tokenize::encode(self, text))
    }
}

impl Tokenize for AnyTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        match &self.backend {
            Backend::Bpe(t) => Tokenize::encode(t, text),
            Backend::Unigram(t) => Tokenize::encode(t, text),
            Backend::WordPiece(t) => Tokenize::encode(t, text),
        }
    }

    fn decode(&self, ids: &[u32]) -> Result<String, TokenizeError> {
        // When the json declares a `decoder`, drive decoding from it: collect the
        // surface strings (skipping special-flagged added tokens, matching HF's
        // default `skip_special_tokens=true`) and run the configured pipeline.
        if let Some(decoder) = &self.decoder {
            let surfaces: Vec<String> = ids
                .iter()
                .filter(|id| !self.special_decode.contains(id))
                .filter_map(|&id| self.backend.token_surface(id))
                .collect();
            return Ok(decoder.decode(surfaces));
        }
        match &self.backend {
            Backend::Bpe(t) => Tokenize::decode(t, ids),
            Backend::Unigram(t) => Tokenize::decode(t, ids),
            Backend::WordPiece(t) => Tokenize::decode(t, ids),
        }
    }

    fn vocab_size(&self) -> usize {
        match &self.backend {
            Backend::Bpe(t) => Tokenize::vocab_size(t),
            Backend::Unigram(t) => Tokenize::vocab_size(t),
            Backend::WordPiece(t) => Tokenize::vocab_size(t),
        }
    }
}

/// Load a tokenizer from a `tokenizer.json` file path.
pub fn from_json_path<P: AsRef<std::path::Path>>(path: P) -> Result<AnyTokenizer, HfJsonError> {
    let bytes = std::fs::read(path)?;
    from_json_bytes(&bytes)
}

/// Load a tokenizer from `tokenizer.json` bytes.
pub fn from_json_bytes(data: &[u8]) -> Result<AnyTokenizer, HfJsonError> {
    let root: Value = serde_json::from_slice(data)?;
    let model = root
        .get("model")
        .ok_or(HfJsonError::MissingField("model"))?;

    let backend = match model_family(model)? {
        "BPE" => build_bpe(&root, model)?,
        "Unigram" => build_unigram(&root, model)?,
        "WordPiece" => build_wordpiece(&root, model)?,
        other => return Err(HfJsonError::UnsupportedModelType(other.to_string())),
    };
    let post = parse_post_processor(&root);
    let decoder = super::decoder::parse(root.get("decoder"));
    let special_decode = parse_special_decode_ids(&root);
    Ok(AnyTokenizer {
        backend,
        post,
        decoder,
        special_decode,
    })
}

/// Parse the `post_processor` (single-sequence template) into the special tokens
/// added before/after the content. Handles `BertProcessing`, `RobertaProcessing`,
/// and `TemplateProcessing`.
fn parse_post_processor(root: &Value) -> PostProcessor {
    let Some(pp) = root.get("post_processor") else {
        return PostProcessor::default();
    };
    match pp.get("type").and_then(Value::as_str) {
        Some("BertProcessing") | Some("RobertaProcessing") => {
            // { cls: [token, id], sep: [token, id] } → [cls] $A [sep]
            let id = |k: &str| pp.get(k).and_then(|p| p.get(1)).and_then(Value::as_u64);
            PostProcessor {
                prefix: id("cls").map(|n| vec![n as u32]).unwrap_or_default(),
                suffix: id("sep").map(|n| vec![n as u32]).unwrap_or_default(),
            }
        }
        Some("TemplateProcessing") => parse_template_processing(pp),
        // Sequence of post-processors: compose their single-sequence effects.
        Some("Sequence") => {
            let mut prefix = Vec::new();
            let mut suffix = Vec::new();
            if let Some(list) = pp.get("processors").and_then(Value::as_array) {
                for sub in list {
                    // Re-wrap so parse_post_processor can recurse on each.
                    let wrapped = serde_json::json!({ "post_processor": sub });
                    let p = parse_post_processor(&wrapped);
                    prefix.extend(p.prefix);
                    // Earlier processors' suffixes sit closest to content.
                    let mut new_suffix = p.suffix;
                    new_suffix.extend(suffix);
                    suffix = new_suffix;
                }
            }
            PostProcessor { prefix, suffix }
        }
        _ => PostProcessor::default(),
    }
}

/// Parse a `TemplateProcessing` single template into prefix/suffix special tokens.
fn parse_template_processing(pp: &Value) -> PostProcessor {
    // Resolve a special-token string to its first id via `special_tokens`.
    let resolve = |tok: &str| -> Option<u32> {
        pp.get("special_tokens")
            .and_then(|m| m.get(tok))
            .and_then(|e| e.get("ids"))
            .and_then(Value::as_array)
            .and_then(|a| a.first())
            .and_then(Value::as_u64)
            .map(|n| n as u32)
    };
    let Some(items) = pp.get("single").and_then(Value::as_array) else {
        return PostProcessor::default();
    };
    let mut prefix = Vec::new();
    let mut suffix = Vec::new();
    let mut seen_sequence = false;
    for item in items {
        if item.get("Sequence").is_some() {
            seen_sequence = true;
        } else if let Some(id) = item
            .get("SpecialToken")
            .and_then(|s| s.get("id"))
            .and_then(Value::as_str)
            .and_then(&resolve)
        {
            if seen_sequence {
                suffix.push(id);
            } else {
                prefix.push(id);
            }
        }
    }
    PostProcessor { prefix, suffix }
}

/// Determine the model family. `model.type` is authoritative when present, but
/// many real `tokenizer.json` files omit it, so we infer from the model's shape:
///
/// - an array `vocab` ⇒ Unigram (token/score pairs)
/// - a `merges` list ⇒ BPE (the decisive BPE marker; note BPE models also carry
///   an empty `continuing_subword_prefix`, so that field alone is not reliable)
/// - `max_input_chars_per_word` ⇒ WordPiece
/// - a non-empty `continuing_subword_prefix` ⇒ WordPiece
/// - otherwise ⇒ BPE
fn model_family(model: &Value) -> Result<&'static str, HfJsonError> {
    if let Some(t) = model.get("type").and_then(Value::as_str) {
        return match t {
            "BPE" => Ok("BPE"),
            "Unigram" => Ok("Unigram"),
            "WordPiece" => Ok("WordPiece"),
            other => Err(HfJsonError::UnsupportedModelType(other.to_string())),
        };
    }
    let nonempty_prefix = model
        .get("continuing_subword_prefix")
        .and_then(Value::as_str)
        .is_some_and(|s| !s.is_empty());
    if model.get("vocab").map(Value::is_array).unwrap_or(false) {
        Ok("Unigram")
    } else if model.get("merges").is_some() {
        Ok("BPE")
    } else if model.get("max_input_chars_per_word").is_some() || nonempty_prefix {
        Ok("WordPiece")
    } else {
        Ok("BPE")
    }
}

fn build_bpe(root: &Value, model: &Value) -> Result<Backend, HfJsonError> {
    let pre = parse_pre_tokenizer(root.get("pre_tokenizer"))?;
    let specials = parse_special_tokens(root);

    let vocab = model
        .get("vocab")
        .and_then(Value::as_object)
        .ok_or(HfJsonError::MissingField("model.vocab"))?;

    let mut encoder: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
    encoder.reserve(vocab.len());
    for (token, id) in vocab {
        let id = id
            .as_u64()
            .ok_or(HfJsonError::MissingField("model.vocab[*] = u32"))? as u32;
        if pre.byte_level && byte_level_decode(token).is_none() {
            return Err(HfJsonError::InvalidByteLevel(token.clone()));
        }
        // Byte-level encoders keep the byte-level-encoded string's bytes (encode
        // byte-level-encodes input before lookup); raw BPE keeps them as-is too.
        encoder.insert(token.as_bytes().to_vec(), id);
    }

    // Merge priority comes from the `merges` list, which is independent of token
    // id (RoBERTa orders them differently from GPT-2). Build a bytes→merge-rank
    // map so BPE merges in the correct order regardless of id assignment.
    let merge_ranks = parse_merge_ranks(model, vocab);

    // Use the multi-stage pre-tokenizer engine when the json declares a pipeline
    // (Digits/Punctuation/Sequence/Split/…). It emits already byte-level-encoded
    // pieces, so the tokenizer itself must not re-encode (plain `new`).
    let engine = super::pretokenizer::parse(root.get("pre_tokenizer"));

    let tok = match engine {
        Some(pt) => {
            // The engine drives splitting + byte-level encoding, so the Tokenizer's
            // own regex is unused (pass a known-good pattern). Keep `use_byte_level`
            // matching the engine so `decode` reverses the byte-level mapping; the
            // encode side skips re-encoding because a pre_tokenizer is attached.
            let t = if pt.byte_level {
                Tokenizer::new_byte_level(encoder, specials, super::tokenizer::GPT2_PATTERN)?
            } else {
                Tokenizer::new(encoder, specials, super::tokenizer::GPT2_PATTERN)?
            };
            let t = match merge_ranks {
                Some(ranks) => t.with_merge_ranks(ranks),
                None => t,
            };
            t.with_pre_tokenizer(pt)
        }
        None => {
            // No pre-tokenizer declared: fall back to the single-regex path.
            let t = if pre.byte_level {
                Tokenizer::new_byte_level(encoder, specials, &pre.pattern)?
            } else {
                Tokenizer::new(encoder, specials, &pre.pattern)?
            };
            let t = match merge_ranks {
                Some(ranks) => t.with_merge_ranks(ranks),
                None => t,
            };
            t.with_prefix_space(pre.byte_level && pre.add_prefix_space)
        }
    };
    // HuggingFace recognizes added tokens in the input during encoding, and drops
    // the special ones on decode. The `normalizer` (e.g. NFC for Qwen/GPT-NeoX)
    // applies to content before splitting.
    let tok = tok
        .with_added_token_matching(true)
        .with_special_decode_ids(parse_special_decode_ids(root))
        .with_normalizer(Normalizer::new(parse_norm_ops(root.get("normalizer"))?));
    Ok(Backend::Bpe(tok))
}

/// Build a bytes → merge-rank map (lower rank = merged first) so BPE merges in
/// the model's true merge order, independent of token id.
///
/// The map covers two groups, ranked so the first group always wins:
/// 1. **Base alphabet** — vocab tokens that are never a merge *result* (the
///    byte-level single chars). Their multi-byte UTF-8 must coalesce before any
///    real merge, so they take the lowest ranks `0..b`.
/// 2. **Merges** — each merged token (`a ++ b`) at rank `b + merge_index`.
///
/// A merge entry is either `[a, b]` or the string `"a b"`. Returns `None` when
/// there is no usable merges list (then BPE uses tiktoken-style id-as-rank).
fn parse_merge_ranks(
    model: &Value,
    vocab: &serde_json::Map<String, Value>,
) -> Option<FxHashMap<Vec<u8>, u32>> {
    let merges = model.get("merges").and_then(Value::as_array)?;

    // Ordered list of merged tokens (and the set, to identify base tokens).
    let mut merged: Vec<String> = Vec::with_capacity(merges.len());
    for m in merges {
        match m {
            Value::Array(p) if p.len() == 2 => {
                if let (Some(a), Some(b)) = (p[0].as_str(), p[1].as_str()) {
                    merged.push(format!("{a}{b}"));
                }
            }
            // String form "a b": split on the first space (byte-level tokens
            // encode real spaces as `Ġ`, never a literal space).
            Value::String(s) => merged.push(s.replacen(' ', "", 1)),
            _ => {}
        }
    }
    if merged.is_empty() {
        return None;
    }
    let merge_set: std::collections::HashSet<&str> = merged.iter().map(String::as_str).collect();

    let mut ranks: FxHashMap<Vec<u8>, u32> = FxHashMap::default();

    // Base alphabet first (vocab tokens that are not a merge result), ordered by
    // id for determinism. They only need ranks below every merge.
    let mut base: Vec<(&String, u64)> = vocab
        .iter()
        .filter(|(k, _)| !merge_set.contains(k.as_str()))
        .filter_map(|(k, v)| v.as_u64().map(|id| (k, id)))
        .collect();
    base.sort_by_key(|&(_, id)| id);
    for (tok, _) in &base {
        ranks.insert(tok.as_bytes().to_vec(), ranks.len() as u32);
    }

    // Then merges, preserving priority order.
    let base_count = ranks.len() as u32;
    for (i, tok) in merged.iter().enumerate() {
        ranks
            .entry(tok.as_bytes().to_vec())
            .or_insert(base_count + i as u32);
    }
    Some(ranks)
}

fn build_unigram(root: &Value, model: &Value) -> Result<Backend, HfJsonError> {
    let vocab = model
        .get("vocab")
        .and_then(Value::as_array)
        .ok_or(HfJsonError::MissingField("model.vocab"))?;

    let mut tokens = Vec::with_capacity(vocab.len());
    let mut scores = Vec::with_capacity(vocab.len());
    for entry in vocab {
        // Each entry is ["token", score].
        let pair = entry
            .as_array()
            .ok_or(HfJsonError::MissingField("model.vocab[*] = [token, score]"))?;
        let token = pair
            .first()
            .and_then(Value::as_str)
            .ok_or(HfJsonError::MissingField("model.vocab[*][0] = token"))?;
        let score = pair.get(1).and_then(Value::as_f64).unwrap_or(0.0) as f32;
        tokens.push(token.to_string());
        scores.push(score);
    }

    // BOS is intentionally None: a HuggingFace Unigram *model* does not prepend
    // BOS (the post_processor does, which we don't replicate), so leaving it off
    // matches `encode(..., add_special_tokens=False)`. EOS only affects decode
    // skipping, so fall back to unk_id/0 rather than failing when there's no </s>.
    let find = |cands: &[&str]| -> Option<u32> {
        find_added_token(root, cands).or_else(|| {
            cands
                .iter()
                .find_map(|c| tokens.iter().position(|t| t == c).map(|i| i as u32))
        })
    };
    let eos = find(&["</s>", "<eos>", "<|endoftext|>", "<|end_of_text|>", "[SEP]"])
        .or_else(|| {
            model
                .get("unk_id")
                .and_then(Value::as_u64)
                .map(|n| n as u32)
        })
        .unwrap_or(0);

    let ops = parse_norm_ops(root.get("normalizer"))?;
    let pre = parse_pre_tokenizer(root.get("pre_tokenizer"))?;
    let tok = SentencePieceTokenizer::new(tokens, scores, None, eos)?
        .with_normalizer(Normalizer::new(ops))
        .with_prefix_space(pre.add_prefix_space)
        .with_added_tokens(&parse_special_tokens(root))
        .with_special_decode_ids(parse_special_decode_ids(root));
    Ok(Backend::Unigram(tok))
}

fn build_wordpiece(root: &Value, model: &Value) -> Result<Backend, HfJsonError> {
    let vocab = model
        .get("vocab")
        .and_then(Value::as_object)
        .ok_or(HfJsonError::MissingField("model.vocab"))?;

    // Build an id-ordered vocab vector; fill any gaps so indices stay aligned.
    let max_id = vocab
        .values()
        .filter_map(Value::as_u64)
        .max()
        .ok_or(HfJsonError::MissingField("model.vocab (empty)"))? as usize;
    let mut id_to_token = vec![String::new(); max_id + 1];
    for (token, id) in vocab {
        let id = id
            .as_u64()
            .ok_or(HfJsonError::MissingField("model.vocab[*] = u32"))? as usize;
        id_to_token[id] = token.clone();
    }

    let unk_token = model
        .get("unk_token")
        .and_then(Value::as_str)
        .unwrap_or("[UNK]");
    let unk_id = vocab
        .get(unk_token)
        .and_then(Value::as_u64)
        .ok_or(HfJsonError::MissingSpecial("unk"))? as u32;

    let max_word_len = model
        .get("max_input_chars_per_word")
        .and_then(Value::as_u64)
        .unwrap_or(100) as usize;

    let norm = parse_bert_norm(root.get("normalizer"));
    // Continuation prefix from the model (default "##"); empty string disables it.
    let prefix = model
        .get("continuing_subword_prefix")
        .and_then(Value::as_str)
        .unwrap_or("##")
        .to_string();

    let tok = WordPieceTokenizer::with_options(
        id_to_token,
        unk_id,
        max_word_len,
        norm.lowercase,
        norm.handle_chinese_chars,
        norm.clean_text,
        prefix,
    )
    .with_added_tokens(&parse_special_tokens(root))
    .with_special_decode_ids(parse_special_decode_ids(root));
    Ok(Backend::WordPiece(tok))
}

#[cfg(test)]
mod tests;
