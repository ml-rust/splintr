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
        // A vocab entry that is ALSO declared in `added_tokens` is spelled
        // literally, not byte-level-encoded: HuggingFace matches added tokens
        // against the raw text *before* the model runs, so their vocab spelling
        // is never byte-level material. DeepSeek V3 declares its 818 added
        // tokens in both sections, and 3 of them (`<｜begin▁of▁sentence｜>`,
        // `<｜end▁of▁sentence｜>`, `<｜▁pad▁｜>`, ids 0/1/2) also occupy a
        // `model.vocab` slot — measured over its `tokenizer.json`, those 3 are
        // the *only* non-byte-level entries in its 128000-entry vocab. Decoding
        // them as byte-level fails outright (`｜` is U+FF5C, outside the
        // byte-level alphabet), so the whole file used to be unloadable.
        //
        // The exemption is per entry and driven solely by membership in
        // `added_tokens` — a vocab entry that is NOT an added token and fails to
        // byte-level-decode is still a hard error, because there the failure
        // means a genuinely corrupt vocabulary rather than a literal spelling.
        // Lookup is a single `FxHashMap` probe per entry, so an 818-token added
        // set over a 128k vocab stays O(vocab), not O(vocab × added).
        match specials.get(token) {
            // Both sections claim the token but disagree on its id. Neither can
            // win: the matcher emits the `added_tokens` id while BPE and the
            // decode tables use the `model.vocab` id, so picking either leaves a
            // tokenizer whose encode and decode contradict each other on that
            // token. Report it instead of quietly choosing.
            Some(added) if added.id != id => {
                return Err(HfJsonError::AddedTokenIdConflict {
                    content: token.clone(),
                    vocab_id: id,
                    added_id: added.id,
                });
            }
            // Agreed added token: literal text, so skip the byte-level check.
            Some(_) => {}
            None => {
                if pre.byte_level && byte_level_decode(token).is_none() {
                    return Err(HfJsonError::InvalidByteLevel(token.clone()));
                }
            }
        }
        // Byte-level encoders keep the byte-level-encoded string's bytes (encode
        // byte-level-encodes input before lookup); raw BPE keeps them as-is too.
        // An added token keeps its literal bytes here as well, which is what
        // makes its `model.vocab` id decodable: `decode` finds the literal in
        // the id→bytes table, and both the built-in byte-level decode and the
        // declared `ByteLevel` decoder pass a non-byte-level token through
        // unchanged, so the id renders as the literal string it was declared as.
        encoder.insert(token.as_bytes().to_vec(), id);
    }

    // Merge priority comes from the `merges` list, which is independent of token
    // id (RoBERTa orders them differently from GPT-2). Build a bytes→merge-rank
    // map so BPE merges in the correct order regardless of id assignment.
    let merge_ranks = parse_merge_ranks(model, vocab);

    // `model.byte_fallback: true` declares a full `<0xNN>` byte-fallback set in
    // `model.vocab` (mistral-7b, embeddinggemma, ...): a piece BPE cannot
    // represent should be emitted byte-by-byte through those ids rather than
    // silently dropped. A declared-but-incomplete set is a malformed file, not
    // something to silently degrade from — report it like the other backends'
    // missing-special errors instead of continuing without fallback.
    let byte_fallback = model
        .get("byte_fallback")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let byte_fallback_ids = if byte_fallback {
        Some(
            Tokenizer::byte_fallback_ids_from_encoder(&encoder)
                .ok_or(HfJsonError::MissingSpecial("byte_fallback"))?,
        )
    } else {
        None
    };

    // Use the multi-stage pre-tokenizer engine when the json declares a pipeline
    // (Digits/Punctuation/Sequence/Split/…). It emits already byte-level-encoded
    // pieces, so the tokenizer itself must not re-encode (plain `new`).
    let engine = super::super::pretokenizer::parse(root.get("pre_tokenizer"))?;

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
            let t = if pt.byte_level() {
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
            // No multi-stage pipeline declared: fall back to the single-regex
            // path. `Metaspace` gets its own branch — it is BPE with a
            // `▁`-marked vocab, decoded via `use_metaspace_decoder`, distinct
            // from both plain BPE and ByteLevel.
            let t = if pre.byte_level {
                Tokenizer::new_byte_level(encoder, specials, &pre.pattern)?
            } else if pre.metaspace {
                Tokenizer::new_with_metaspace_decoder(encoder, specials, &pre.pattern)?
            } else {
                Tokenizer::new(encoder, specials, &pre.pattern)?
            };
            let t = match merge_ranks {
                Some(ranks) => t.with_merge_ranks(ranks),
                None => t,
            };
            // Not gated on `pre.byte_level`: `add_prefix_space` is only ever set
            // (non-`None`) by a ByteLevel or Metaspace node (see
            // `parse_pre_tokenizer`), so for the byte-level branch ANDing with
            // `true` was a no-op — this is behavior-preserving there — while for
            // the Metaspace branch it previously force-disabled a prefix the
            // vocab actually needs (`prepend_scheme: "first"` resolves
            // `add_prefix_space` to `true`).
            t.with_prefix_space(pre.add_prefix_space)
        }
    };
    // HuggingFace recognizes added tokens in the input during encoding, and drops
    // the special ones on decode. The `normalizer` (e.g. NFC for Qwen/GPT-NeoX)
    // applies to content before splitting.
    let tok = tok
        .with_added_token_matching(true)
        .with_special_decode_ids(parse_special_decode_ids(root))
        .with_normalizer(Normalizer::new(parse_norm_ops(root.get("normalizer"))?))
        .with_byte_fallback(byte_fallback_ids);
    Ok(Backend::Bpe(tok))
}

/// Build a bytes → merge-rank map (lower rank = merged first) so BPE merges in
/// the model's true merge order, independent of token id.
///
/// The map covers two groups, ranked so the first group always wins:
/// 1. **Base alphabet** — vocab tokens that are never a merge *result* (the
///    byte-level single chars). They take the lowest ranks `0..b` so that a base
///    entry reachable as a merge of two adjacent pieces forms before any real
///    merge. That only reassembles 2-byte UTF-8 characters, whose two bytes
///    concatenate to the whole character; a ≥3-byte character has no rank for
///    its partial prefix, so it is instead never split in the first place —
///    these vocabularies seed BPE by character (see `byte_pair_encode_pieces_seeded`).
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
        // Kept at full `f64` width: HF `tokenizers` reads the same JSON number
        // into an `f64` and its Viterbi compares partial sums at that precision,
        // so narrowing here would reorder equal-scoring segmentations.
        let score = pair.get(1).and_then(Value::as_f64).unwrap_or(0.0);
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
    // Accent stripping is its own setting in the file, already resolved from
    // `strip_accents`' tri-state by `parse_bert_norm` — passing it explicitly is
    // what keeps a `strip_accents: false` cased checkpoint off the unaccented
    // vocabulary entries.
    .with_strip_accents(norm.strip_accents)
    .with_added_tokens(parse_special_tokens(root))?
    .with_special_decode_ids(parse_special_decode_ids(root));
    Ok(Backend::WordPiece(tok))
}
