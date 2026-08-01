//! Construction: parse a `tokenizer.json` into an [`AnyTokenizer`].

use rustc_hash::FxHashMap;
use serde_json::Value;

use super::super::byte_level::byte_level_decode;
use super::super::normalizer::Normalizer;
use super::super::sentencepiece::SentencePieceTokenizer;
use super::super::tokenizer::Tokenizer;
use super::super::wordpiece::WordPieceTokenizer;

use super::super::any_tokenizer::{AnyTokenizer, Backend};
use super::super::policy;
use super::components::{
    find_added_token, parse_bert_norm, parse_norm_ops, parse_pre_tokenizer,
    parse_special_decode_ids, parse_special_tokens,
};
use super::HfJsonError;

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
    let policy = policy::parse(&root)?;
    let decoder = super::super::decoder::parse(root.get("decoder"));
    let special_decode = parse_special_decode_ids(&root);
    Ok(AnyTokenizer {
        backend,
        policy,
        decoder,
        special_decode,
    })
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
    let pre = parse_pre_tokenizer(root.get("pre_tokenizer"));
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
    let engine = super::super::pretokenizer::parse(root.get("pre_tokenizer"));

    // Guess guard: a pre_tokenizer was declared, but neither the multi-stage
    // engine recognized a stage nor the distiller anchored a splitter
    // (ByteLevel/Metaspace/Split). Falling back to the GPT-2 default pattern would
    // silently guess the split and change the tokens, so refuse instead. (Types
    // the engine DOES handle — Digits/Punctuation/Whitespace/… — make `engine`
    // `Some` and never reach here, so this never rejects a supported pipeline.)
    if engine.is_none()
        && !pre.anchored
        && root.get("pre_tokenizer").is_some_and(|v| !v.is_null())
        && !pre.unknown.is_empty()
    {
        return Err(HfJsonError::UnsupportedPreTokenizer(pre.unknown.join(", ")));
    }

    let tok = match engine {
        Some(pt) => {
            // The engine drives splitting + byte-level encoding, so the Tokenizer's
            // own regex is unused (pass a known-good pattern). Keep `use_byte_level`
            // matching the engine so `decode` reverses the byte-level mapping; the
            // encode side skips re-encoding because a pre_tokenizer is attached.
            let t = if pt.byte_level {
                Tokenizer::new_byte_level(encoder, specials, super::super::tokenizer::GPT2_PATTERN)?
            } else {
                Tokenizer::new(encoder, specials, super::super::tokenizer::GPT2_PATTERN)?
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

    // Vocabulary in id order, so the base-alphabet ranks are deterministic.
    let mut base: Vec<(&String, u64)> = vocab
        .iter()
        .filter_map(|(k, v)| v.as_u64().map(|id| (k, id)))
        .collect();
    base.sort_by_key(|&(_, id)| id);

    Some(super::super::bpe::merge_ranks(
        &merged,
        base.iter().map(|(k, _)| k.as_str()),
    ))
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
    let eos = find(policy::EOS_CANDIDATES)
        .or_else(|| {
            model
                .get("unk_id")
                .and_then(Value::as_u64)
                .map(|n| n as u32)
        })
        .unwrap_or(0);

    let ops = parse_norm_ops(root.get("normalizer"))?;
    // The Unigram backend does its own metaspace escaping and splitting, so
    // `pre` is consulted only for `add_prefix_space` — there is no GPT-2 default
    // to silently guess here. Space-run merging stays off: a `tokenizer.json`
    // that wants it declares it as a normalizer step (XLM-R's
    // `Replace{" {2,}" → " "}`), which the pipeline above already applies.
    let pre = parse_pre_tokenizer(root.get("pre_tokenizer"));
    let tok = SentencePieceTokenizer::new(tokens, scores, None, eos)?
        .with_normalizer(Normalizer::new(ops))
        .with_prefix_space(pre.add_prefix_space)
        .with_added_tokens(parse_special_tokens(root))?
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
    .with_added_tokens(parse_special_tokens(root))?
    .with_special_decode_ids(parse_special_decode_ids(root));
    Ok(Backend::WordPiece(tok))
}
