//! Construction: parse a `tokenizer.json` into an [`AnyTokenizer`].

use rustc_hash::{FxHashMap, FxHashSet};
use serde::de::{MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::value::RawValue;
use serde_json::Value;
use std::borrow::Cow;
use std::fmt;
use std::marker::PhantomData;

use super::super::byte_level::byte_level_decode;
use super::super::decode_table::Decoder;
use super::super::normalizer::Normalizer;
use super::super::sentencepiece::SentencePieceTokenizer;
use super::super::tokenizer::Tokenizer;
use super::super::wordpiece::WordPieceTokenizer;

use super::super::any_tokenizer::{AnyTokenizer, Backend};
use super::super::encoder::Encoder;
use super::super::policy;
use super::components::{
    find_added_token, parse_bert_norm, parse_norm_ops, parse_pre_tokenizer,
    parse_special_decode_ids, parse_special_tokens, parse_unk_id,
};
use super::HfJsonError;

/// A JSON string kept borrowed from the input when it has no escapes.
///
/// `serde`'s own `Cow<str>` always allocates; this borrows whenever the parser
/// can hand back a slice of the original buffer, which for a `tokenizer.json`
/// vocabulary is nearly every one of its 100k-200k tokens.
struct CowStr<'a>(Cow<'a, str>);

impl<'de: 'a, 'a> Deserialize<'de> for CowStr<'a> {
    fn deserialize<D: Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        struct V<'a>(PhantomData<&'a ()>);
        impl<'de: 'a, 'a> Visitor<'de> for V<'a> {
            type Value = CowStr<'a>;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a string")
            }
            fn visit_borrowed_str<E>(self, v: &'de str) -> Result<Self::Value, E> {
                Ok(CowStr(Cow::Borrowed(v)))
            }
            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E> {
                Ok(CowStr(Cow::Owned(v.to_owned())))
            }
            fn visit_string<E>(self, v: String) -> Result<Self::Value, E> {
                Ok(CowStr(Cow::Owned(v)))
            }
        }
        de.deserialize_str(V(PhantomData))
    }
}

/// `model.merges` as the token each entry produces, with where its halves met.
///
/// An entry is `["a", "b"]` or the string `"a b"`. The halves are joined as
/// they are read — parsing into `Vec<Value>` first cost an array `Value` and
/// two `String`s per merge, three allocations each over as many entries as the
/// vocabulary has tokens — and the left half's length rides along so the pair
/// is still recoverable from the join. That is what separates a vocabulary
/// entry merges are built FROM from one no merge mentions; see
/// `vocab::orphan_ids`.
struct MergeList(Vec<(String, usize)>);

impl<'de> Deserialize<'de> for MergeList {
    fn deserialize<D: Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        /// One entry, joined on the spot, with the left half's byte length.
        struct Merged(Option<(String, usize)>);

        impl<'de> Deserialize<'de> for Merged {
            fn deserialize<D: Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
                struct V;
                impl<'de> Visitor<'de> for V {
                    type Value = Merged;
                    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                        f.write_str("a merge pair or its joined string")
                    }
                    /// `["a", "b"]`.
                    fn visit_seq<S: SeqAccess<'de>>(self, mut seq: S) -> Result<Merged, S::Error> {
                        let (a, b) = (seq.next_element::<CowStr>()?, seq.next_element::<CowStr>()?);
                        // A third element means this is not a merge pair.
                        let extra = seq.next_element::<serde::de::IgnoredAny>()?.is_some();
                        Ok(Merged(match (a, b, extra) {
                            // An empty half names no token, so it is not a pair.
                            (Some(a), Some(b), false) if !a.0.is_empty() && !b.0.is_empty() => {
                                let mut s = String::with_capacity(a.0.len() + b.0.len());
                                s.push_str(&a.0);
                                s.push_str(&b.0);
                                let split = a.0.len();
                                Some((s, split))
                            }
                            _ => None,
                        }))
                    }
                    /// `"a b"`: byte-level tokens spell a real space as `Ġ`, so
                    /// the first literal space is the separator.
                    fn visit_str<E>(self, v: &str) -> Result<Merged, E> {
                        let Some(split) = v.find(' ') else {
                            return Ok(Merged(None));
                        };
                        if split == 0 || split + 1 == v.len() {
                            return Ok(Merged(None));
                        }
                        Ok(Merged(Some((v.replacen(' ', "", 1), split))))
                    }
                }
                de.deserialize_any(V)
            }
        }

        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = MergeList;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a merges list")
            }
            fn visit_seq<S: SeqAccess<'de>>(self, mut seq: S) -> Result<MergeList, S::Error> {
                let mut out = Vec::with_capacity(seq.size_hint().unwrap_or(0));
                while let Some(Merged(m)) = seq.next_element::<Merged>()? {
                    out.extend(m);
                }
                Ok(MergeList(out))
            }
        }
        de.deserialize_seq(V)
    }
}

/// `model.vocab` as token/id pairs in file order.
///
/// A `Vec` rather than a map because the loop below reads it once and builds
/// its own tables from it: materializing serde_json's `Map<String, Value>`
/// first meant a `String` and a `Value` per token and a `BTreeMap` insert to
/// put them somewhere, which measured at 13% of load on its own.
struct VocabPairs<'a>(Vec<(Cow<'a, str>, u32)>);

impl<'de: 'a, 'a> Deserialize<'de> for VocabPairs<'a> {
    fn deserialize<D: Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        struct V<'a>(PhantomData<&'a ()>);
        impl<'de: 'a, 'a> Visitor<'de> for V<'a> {
            type Value = VocabPairs<'a>;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a vocabulary object")
            }
            fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Self::Value, M::Error> {
                let mut out = Vec::with_capacity(map.size_hint().unwrap_or(0));
                while let Some((token, id)) = map.next_entry::<CowStr<'a>, u32>()? {
                    out.push((token.0, id));
                }
                Ok(VocabPairs(out))
            }
        }
        de.deserialize_map(V(PhantomData))
    }
}

/// Expand one recorded span into a `Value`.
fn expand(raw: &RawValue) -> Result<Value, HfJsonError> {
    Ok(serde_json::from_str(raw.get())?)
}

/// Rebuild an object `Value` from spans, so code written against `Value` is
/// unchanged. Only ever used for the small fields.
fn object_from(spans: &FxHashMap<&str, &RawValue>) -> Result<Value, HfJsonError> {
    let mut map = serde_json::Map::with_capacity(spans.len());
    for (key, raw) in spans {
        map.insert((*key).to_string(), expand(raw)?);
    }
    Ok(Value::Object(map))
}

/// Load a tokenizer from a `tokenizer.json` file path.
pub fn from_json_path<P: AsRef<std::path::Path>>(path: P) -> Result<AnyTokenizer, HfJsonError> {
    let bytes = std::fs::read(path)?;
    from_json_bytes(&bytes)
}

/// Load a tokenizer from `tokenizer.json` bytes.
pub fn from_json_bytes(data: &[u8]) -> Result<AnyTokenizer, HfJsonError> {
    // Read the file as unparsed spans first. `model.vocab` holds 100k-200k
    // entries and turning it into a `Value` — a `String` key and a `Value` per
    // token, each inserted into a `BTreeMap` — measured at roughly a third of
    // load time, while every other field in the file is small. Recording spans
    // costs a scan, lets the small fields keep the `Value`-based parsing they
    // already have, and lets the vocabulary be read straight into the shape the
    // encoder wants.
    let top: FxHashMap<&str, &RawValue> = serde_json::from_slice(data)?;
    let model_raw = *top.get("model").ok_or(HfJsonError::MissingField("model"))?;
    let mut model_spans: FxHashMap<&str, &RawValue> = serde_json::from_str(model_raw.get())?;
    let vocab_raw = model_spans.remove("vocab");
    let merges_raw = model_spans.remove("merges");

    let mut root_spans = top;
    root_spans.remove("model");
    let root = object_from(&root_spans)?;
    let model = object_from(&model_spans)?;

    let backend = match model_family(&model, vocab_raw, merges_raw.is_some())? {
        "BPE" => {
            let raw = vocab_raw.ok_or(HfJsonError::MissingField("model.vocab"))?;
            let vocab: VocabPairs<'_> = serde_json::from_str(raw.get())?;
            build_bpe(&root, &model, &vocab.0, merges_raw)?
        }
        family => {
            // Unigram and WordPiece read `vocab` off the model as a `Value`, and
            // neither is on a hot path worth restructuring for — put the spans
            // back so those loaders see exactly the model they saw before.
            let mut model = model;
            if let Value::Object(map) = &mut model {
                if let Some(raw) = vocab_raw {
                    map.insert("vocab".to_string(), expand(raw)?);
                }
                if let Some(raw) = merges_raw {
                    map.insert("merges".to_string(), expand(raw)?);
                }
            }
            match family {
                "Unigram" => build_unigram(&root, &model)?,
                "WordPiece" => build_wordpiece(&root, &model)?,
                other => return Err(HfJsonError::UnsupportedModelType(other.to_string())),
            }
        }
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
fn model_family(
    model: &Value,
    vocab: Option<&RawValue>,
    has_merges: bool,
) -> Result<&'static str, HfJsonError> {
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
    // Unigram's vocab is an array of [token, score]; every other family's is an
    // object. Read that off the span rather than expanding it.
    let vocab_is_array = vocab.is_some_and(|v| v.get().trim_start().starts_with('['));
    if vocab_is_array {
        Ok("Unigram")
    } else if has_merges {
        Ok("BPE")
    } else if model.get("max_input_chars_per_word").is_some() || nonempty_prefix {
        Ok("WordPiece")
    } else {
        Ok("BPE")
    }
}

fn build_bpe(
    root: &Value,
    model: &Value,
    vocab: &[(Cow<'_, str>, u32)],
    merges: Option<&RawValue>,
) -> Result<Backend, HfJsonError> {
    let pre = parse_pre_tokenizer(root.get("pre_tokenizer"));
    let specials = parse_special_tokens(root);

    // Parsed before the encode tables are filled, because which entries belong
    // in them depends on it: an entry no merge names is unreachable, and
    // encoding into one contradicts what merging the same bytes produces.
    let merge_rules = parse_merge_rules(merges);
    // `ignore_merges` declares the whole-chunk lookup to BE the semantics: a
    // pre-token found in the vocabulary is emitted whole and the merges are not
    // consulted. Under it an entry no merge can produce is still reachable —
    // `tokenizers` emits it — so nothing is unreachable and nothing is dropped.
    // Llama 3, SmolLM 3, gpt-oss and GLM-4.5 set it; none of them has such an
    // entry, so the two have never yet met, which is exactly why the flag is
    // read here rather than assumed absent.
    let ignore_merges = model
        .get("ignore_merges")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    // `continuing_subword_prefix` marks every symbol of a word except the
    // first, which changes what BPE starts from just as decisively as the
    // end-of-word suffix does. Nothing here implements it, and a vocabulary
    // that sets it would tokenize plausibly and wrongly — the failure mode that
    // silently-ignored model fields keep producing — so it is refused. Empty is
    // the same as absent: CLIP declares `""`, meaning no prefix.
    if let Some(prefix) = model
        .get("continuing_subword_prefix")
        .and_then(Value::as_str)
        .filter(|p| !p.is_empty())
    {
        return Err(HfJsonError::UnsupportedModelField(format!(
            "model.continuing_subword_prefix = {prefix:?}"
        )));
    }
    // `end_of_word_suffix` marks the last symbol of a word, which changes what
    // BPE starts from — and therefore which entries it can reach. Empty is the
    // same as absent, which is how every non-CLIP file in the corpus spells it.
    let suffix = model
        .get("end_of_word_suffix")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty());

    // Merge priority comes from the `merges` list, which is independent of token
    // id (RoBERTa orders them differently from GPT-2). Build a bytes→merge-rank
    // map so BPE merges in the correct order regardless of id assignment.
    //
    // Built here rather than after the encode tables because reachability is
    // decided by running the merge, and running it needs these ranks.
    let merge_ranks = match &merge_rules {
        Some(rules) => parse_merge_ranks(rules, vocab),
        None => None,
    };

    let unreachable = match &merge_rules {
        Some(rules) if !ignore_merges => unreachable_tokens(
            rules,
            vocab,
            suffix,
            merge_ranks.as_ref(),
            // The granularity the encode path will seed at: a HuggingFace merge
            // list operates on characters unless the vocabulary is ByteLevel,
            // whose alphabet is its characters and whose surfaces are already in
            // the mapped space. Same condition as `Tokenizer::bpe_into`'s.
            !pre.byte_level,
        ),
        _ => FxHashSet::default(),
    };

    let mut encoder: Encoder = Encoder::default();
    encoder.reserve(vocab.len() - unreachable.len().min(vocab.len()));
    // `encoder` keyed by the raw bytes each token stands for, filled from the
    // same decode the byte-level validation below already performs. Building it
    // in a second pass cost 11-14% of load; here it is the difference between
    // discarding that decode and keeping it.
    //
    // Partial by design. An added token is spelled literally rather than
    // byte-level-encoded, so it has no raw form to record — and it never
    // reaches this lookup anyway, being matched ahead of BPE. A missing entry
    // only costs the fast path a miss, which falls through to the mapped
    // lookup and the same answer; only a *wrong* entry could change ids, and
    // the mapping is a bijection, so the entries present cannot collide.
    let mut raw_encoder: Encoder = Encoder::default();
    if pre.byte_level {
        raw_encoder.reserve(vocab.len());
    }
    // The id each `<0xNN>` piece carries, for the byte fallback built below. It
    // reads the whole vocabulary rather than the encode table, which declines
    // those pieces, and collecting it here keeps that to no extra pass.
    let mut fallback_ids: [Option<u32>; 256] = [None; 256];
    for (token, id) in vocab {
        let (token, id) = (token.as_ref(), *id);
        if let Some(byte) = byte_fallback_byte(token.as_bytes()) {
            fallback_ids[byte as usize].get_or_insert(id);
        }
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
                    content: token.to_string(),
                    vocab_id: id,
                    added_id: added.id,
                });
            }
            // Agreed added token: literal text, so skip the byte-level check.
            Some(_) => {}
            None => {
                if pre.byte_level {
                    match byte_level_decode(token) {
                        None => return Err(HfJsonError::InvalidByteLevel(token.to_string())),
                        // Validated either way — an entry that fails to
                        // byte-level-decode is a corrupt vocabulary whether or
                        // not merges can reach it — but an unreachable one is
                        // not recorded, because this table is an encode path
                        // too and would reintroduce exactly what the encoder
                        // above declines.
                        Some(raw) => {
                            if !unreachable.contains(token) {
                                raw_encoder.insert(&raw, id);
                            }
                        }
                    }
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
        if unreachable.contains(token) {
            continue;
        }
        encoder.insert(token.as_bytes(), id);
    }

    // Every id the file states, including the ones the encode tables declined.
    // Built only when something was declined: otherwise the encoder covers the
    // same set and the table the tokenizer derives from it is already right.
    let full_decoder = (!unreachable.is_empty()).then(|| {
        let mut decoder = Decoder::with_capacity(vocab.len());
        for (token, id) in vocab {
            decoder.insert(*id, token.as_bytes());
        }
        decoder
    });

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

    // Whether the tokenizer this call builds will be ByteLevel — either via the
    // multi-stage engine's own `ByteLevel` stage or the single-regex `pre.byte_level`
    // path below. `Tokenizer::bpe` discards `byte_fallback` whenever
    // `use_byte_level` is true (the `<0xNN>` table maps RAW byte values, which is
    // the wrong space once input has been byte-level-encoded — see the doc on
    // `Tokenizer::bpe`), so a ByteLevel model's fallback can never fire. Skip
    // building it there entirely rather than constructing a 256-entry `Box` that
    // `has_byte_fallback()` would then report as active despite never being
    // consulted.
    let is_byte_level = engine.as_ref().map_or(pre.byte_level, |pt| pt.byte_level());

    // `model.byte_fallback: true` declares a `<0xNN>` byte-fallback set in
    // `model.vocab` (mistral-7b, embeddinggemma, ...): a piece BPE cannot
    // represent should be emitted through those ids rather than silently
    // dropped. The set need NOT be complete — HuggingFace resolves fallback per
    // character, using `<0xNN>` where the entry exists and `model.unk_token`
    // where it does not, so a file declaring only some of the 256 entries loads
    // and tokenizes fine there (measured against `tokenizers` 0.22.1).
    //
    // The flag governs the `<0xNN>` half ONLY. `model.unk_token` is honored
    // regardless of it: HF's BPE model emits the unk for any piece it cannot
    // represent whether or not `byte_fallback` is set, and consults the
    // `<0xNN>` tokens only when it is — measured against `tokenizers` 0.22.1
    // on a `{"<unk>": 0, "a": 1, "<0x7A>": 2}` vocab with `unk_token: "<unk>"`,
    // where `encode("az")` gives `['a', '<0x7A>']` under `byte_fallback: true`
    // and `['a', '<unk>']` under `byte_fallback: false`, despite `<0x7A>`
    // existing in the vocab both times. So gating construction on the flag
    // would silently DROP unrepresentable pieces in the (common) file that
    // declares an unk without the flag.
    //
    // Neither half being present is still not an error: with nothing to fall
    // back to, `byte_fallback_from` yields `None` and the
    // unrepresentable piece is dropped, exactly as HF does.
    let declares_byte_fallback = model
        .get("byte_fallback")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    // `model.fuse_unk` is a separate knob over the unk half only: it collapses a
    // *run* of unk-resolved characters into a single unk id rather than emitting
    // one per character. It defaults to false when absent, which is
    // `tokenizers`' own default (measured: a file omitting the field encodes
    // `"xyz"` over the vocab above as three `<unk>`s). Every byte-fallback
    // vocabulary on the shelf declares `fuse_unk: true` alongside a *complete*
    // 256-entry `<0xNN>` set, where no character ever reaches the unk branch —
    // so the flag only becomes observable on a partial (or absent) byte table.
    let unk_id = parse_unk_id(
        model,
        |name| {
            vocab
                .iter()
                .find(|(token, _)| token.as_ref() == name)
                .map(|(_, id)| *id)
        },
        None,
    );
    let fuse_unk = model
        .get("fuse_unk")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    // From the whole vocabulary, not from `encoder`: a `<0xNN>` piece is not
    // encodable from its own literal spelling, so the encode table declines it,
    // and the fallback still has to emit it for the raw byte it denotes.
    // `fallback_ids` was collected in the vocabulary pass above, which is the
    // only walk of it this loader makes.
    let byte_fallback = (!is_byte_level)
        .then(|| {
            Tokenizer::byte_fallback_from(
                |spelling| byte_fallback_byte(spelling).and_then(|b| fallback_ids[b as usize]),
                unk_id,
                declares_byte_fallback,
            )
        })
        .flatten()
        .map(|bf| bf.with_fuse_unk(fuse_unk));

    let tok = match engine {
        Some(pt) => {
            // The engine drives splitting + byte-level encoding, so the Tokenizer's
            // own regex is unused (pass a known-good pattern). Keep `use_byte_level`
            // matching the engine so `decode` reverses the byte-level mapping; the
            // encode side skips re-encoding because a pre_tokenizer is attached.
            let t = if pt.byte_level() {
                Tokenizer::from_encoder(
                    encoder,
                    specials,
                    super::super::tokenizer::GPT2_PATTERN,
                    true,
                )?
            } else {
                Tokenizer::from_encoder(
                    encoder,
                    specials,
                    super::super::tokenizer::GPT2_PATTERN,
                    false,
                )?
            };
            let t = match merge_ranks {
                Some(ranks) => t.with_merge_ranks(ranks),
                None => t,
            };
            // Only the pipeline path can hand pre-tokens over unmapped, so it is
            // the only one the raw table can serve.
            let t = t.with_raw_encoder(raw_encoder);
            t.with_pre_tokenizer(pt)
        }
        None => {
            // No multi-stage pipeline declared: fall back to the single-regex
            // path. `Metaspace` gets its own branch — it is BPE with a
            // `▁`-marked vocab, decoded via `use_metaspace_decoder`, distinct
            // from both plain BPE and ByteLevel.
            let t = if pre.byte_level {
                Tokenizer::from_encoder(encoder, specials, &pre.pattern, true)?
            } else if pre.metaspace {
                Tokenizer::from_encoder_with_metaspace_decoder(encoder, specials, &pre.pattern)?
            } else {
                Tokenizer::from_encoder(encoder, specials, &pre.pattern, false)?
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
                .with_metaspace_split(pre.metaspace_split)
        }
    };
    // HuggingFace recognizes added tokens in the input during encoding, and drops
    // the special ones on decode. The `normalizer` (e.g. NFC for Qwen/GPT-NeoX)
    // applies to content before splitting.
    let tok = match full_decoder {
        Some(decoder) => tok.with_decode_table(decoder),
        None => tok,
    };
    let tok = tok
        .with_added_token_matching(true)
        .with_special_decode_ids(parse_special_decode_ids(root))
        .with_normalizer(Normalizer::new(parse_norm_ops(root.get("normalizer"))?))
        .with_byte_fallback(byte_fallback)
        .with_end_of_word_suffix(suffix);
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
fn parse_merge_ranks(rules: &[(String, usize)], vocab: &[(Cow<'_, str>, u32)]) -> Option<Encoder> {
    if rules.is_empty() {
        return None;
    }

    // Vocabulary in id order, so the base-alphabet ranks are deterministic.
    let mut base: Vec<(&str, u32)> = vocab.iter().map(|(k, id)| (k.as_ref(), *id)).collect();
    base.sort_by_key(|&(_, id)| id);

    Some(super::super::bpe::merge_ranks(
        rules.iter().map(|(merged, _)| merged.clone()).collect(),
        base.iter().map(|(k, _)| *k),
    ))
}

/// Read `model.merges` into `(result, split)` rules, or `None` when the file
/// has no usable list (then BPE falls back to tiktoken-style id-as-rank).
fn parse_merge_rules(merges: Option<&RawValue>) -> Option<Vec<(String, usize)>> {
    let MergeList(rules) = serde_json::from_str(merges?.get()).ok()?;
    (!rules.is_empty()).then_some(rules)
}

/// The vocabulary entries BPE can neither produce nor build from.
///
/// The json counterpart of `vocab::orphan_ids`. Such an entry must not be
/// encodable — a whole-chunk lookup would answer with one id where merging the
/// same bytes gives several — while still decoding, which is why only the encode
/// tables drop it.
///
/// The test is the merge itself: an entry is reachable exactly when merging its
/// own surface produces it whole ([`merges_to_whole`]). Asking instead whether
/// some merge *names* the entry is the cheaper question and the wrong one — a
/// merge list can name an entry the merge ORDER never reaches, and Gemma 4 has
/// four such entries and NLLB-200 one. See [`merges_to_whole`] for why the
/// standalone answer settles the question in every context.
///
/// Two kinds of entry skip the test:
///
/// - **Seed spellings.** One symbol, which the merge leaves alone by
///   construction, and which seeding must be able to start from. `suffix`
///   widens this: a model declaring `end_of_word_suffix` seeds the last
///   character of every word as that character *plus* the marker, so `1</w>` is
///   a seed spelling exactly as `1` is — CLIP names 139 of its 256 marked
///   characters in no merge at all.
/// - **`<0xNN>` byte-fallback pieces**, which are six characters and frequently
///   named by no merge, yet are exactly how a raw byte is spelled.
///
/// A model declaring `end_of_word_suffix` keeps the name test outright. Its
/// merge list is keyed by marked spellings, so simulating a merge over a bare
/// vocabulary entry would be simulating a word that model never builds; CLIP is
/// the only such vocabulary in the corpus and is verified id-for-id under the
/// name test.
/// The raw byte a `<0xNN>` piece denotes, or `None` when it is not one.
///
/// Upper-case hex only, which is the spelling SentencePiece writes and the one
/// `ByteFallback` resolution matches elsewhere in the crate.
fn byte_fallback_byte(token: &[u8]) -> Option<u8> {
    let [b'<', b'0', b'x', hi, lo, b'>'] = token else {
        return None;
    };
    let digit = |b: &u8| {
        (*b as char)
            .to_digit(16)
            .filter(|_| !b.is_ascii_lowercase())
    };
    Some((digit(hi)? * 16 + digit(lo)?) as u8)
}

/// Vocabulary size at which the reachability pass is worth a thread pool.
/// Below it the whole pass is a few milliseconds and rayon's spin-up is not.
#[cfg(feature = "rayon")]
const MIN_PARALLEL_VOCAB: usize = 16_384;

fn unreachable_tokens<'v>(
    rules: &[(String, usize)],
    vocab: &'v [(Cow<'v, str>, u32)],
    suffix: Option<&str>,
    ranks: Option<&Encoder>,
    char_granular: bool,
) -> FxHashSet<&'v str> {
    // The name test is kept for a suffixed model, and is the only test
    // available when the ranks could not be built.
    let ranks = suffix.is_none().then_some(ranks).flatten();

    // What the merge can produce at all, and from which pair. An entry no merge
    // results in is one no merge can build — no simulation needed to know that
    // — and the pair settles the shortest entries the same way. Under the name
    // test the operands count as reachable too, which is the older, looser rule
    // kept for the vocabularies that still take it.
    let mut results: FxHashMap<&str, usize> = FxHashMap::default();
    let mut named: FxHashSet<&str> = FxHashSet::default();
    match ranks.is_some() {
        true => {
            results.reserve(rules.len());
            for (merged, split) in rules {
                results.insert(merged.as_str(), *split);
            }
        }
        false => {
            named.reserve(rules.len() * 2);
            for (merged, split) in rules {
                named.insert(merged.as_str());
                named.insert(&merged[..*split]);
                named.insert(&merged[*split..]);
            }
        }
    }

    let is_unreachable = |token: &&'v str| -> bool {
        if is_seed_spelling(token, suffix) {
            return false;
        }
        // A `<0xNN>` piece is reachable only through byte-fallback resolution,
        // which emits it by id. From its own literal spelling it is six
        // characters that merge like any others, so the encode table must
        // decline it — `<0x1D>` typed as text is what HuggingFace spells out.
        if super::super::vocab::is_byte_fallback_piece(token.as_bytes()) {
            return ranks.is_some();
        }
        let Some(ranks) = ranks else {
            return !named.contains(*token);
        };
        // Cheap and decisive in one direction: nothing the merge list ever
        // produces can be reached.
        let Some(&split) = results.get(*token) else {
            return true;
        };
        // Two symbols, one per operand: seeding produces exactly this rule's
        // pair, and the rule ranks it, so the merge fires. The commonest shape
        // in a vocabulary, answered without running anything.
        let symbols = |s: &str| match char_granular {
            true => s.chars().count(),
            false => s.len(),
        };
        if symbols(&token[..split]) == 1 && symbols(&token[split..]) == 1 {
            return false;
        }
        !super::super::bpe::merges_to_whole(
            token.as_bytes(),
            super::super::bpe::RankLookup::new(ranks),
            char_granular,
        )
    };

    // The merge test runs BPE once per entry, which on a large vocabulary is
    // the most expensive thing in the load — and every entry is independent, so
    // it parallelizes exactly. The threshold keeps a small vocabulary off the
    // thread pool, whose spin-up would cost more than the answer.
    #[cfg(feature = "rayon")]
    if ranks.is_some() && vocab.len() >= MIN_PARALLEL_VOCAB {
        use rayon::prelude::*;
        return vocab
            .par_iter()
            .map(|(token, _)| token.as_ref())
            .filter(is_unreachable)
            .collect::<Vec<&'v str>>()
            .into_iter()
            .collect();
    }

    vocab
        .iter()
        .map(|(token, _)| token.as_ref())
        .filter(is_unreachable)
        .collect()
}

/// Whether `token` is something BPE seeds a word from, and so can never be
/// unreachable however the merge list names it: one character, or — under
/// `end_of_word_suffix` — one character carrying the marker.
fn is_seed_spelling(token: &str, suffix: Option<&str>) -> bool {
    let bare = match suffix {
        Some(suffix) => token.strip_suffix(suffix).unwrap_or(token),
        None => token,
    };
    // One character, and not the empty string the marker alone would leave.
    let mut chars = bare.chars();
    chars.next().is_some() && chars.next().is_none()
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
    // The Unigram backend does its own metaspace escaping, so `pre` is consulted
    // only for `add_prefix_space` — there is no GPT-2 default to silently guess
    // here. Space-run merging stays off: a `tokenizer.json` that wants it
    // declares it as a normalizer step (XLM-R's `Replace{" {2,}" → " "}`), which
    // the pipeline above already applies.
    let pre = parse_pre_tokenizer(root.get("pre_tokenizer"));
    let tok = SentencePieceTokenizer::new(tokens, scores, None, eos)?
        .with_normalizer(Normalizer::new(ops))
        .with_prefix_space(pre.add_prefix_space)
        .with_added_tokens(parse_special_tokens(root))?
        .with_special_decode_ids(parse_special_decode_ids(root));
    // Stages the escaping cannot express — `WhitespaceSplit` and friends, which
    // drop what they split on — run ahead of it. The stage parser does not model
    // `Metaspace`, precisely because this backend implements it, so a plain
    // `Metaspace` pre-tokenizer parses to no stages and nothing changes here.
    let tok = match super::super::pretokenizer::parse(root.get("pre_tokenizer"))? {
        // A ByteLevel pipeline rewrites pieces into the GPT-2 byte alphabet,
        // which a `▁`-marked Unigram vocabulary cannot match.
        Some(pt) if !pt.byte_level() => tok.with_word_split(pt),
        _ => tok,
    };
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

    // WordPiece cannot tokenize at all without an unk, so an unresolvable one is
    // a hard error here — unlike BPE, where it just narrows byte fallback.
    let unk_id = parse_unk_id(
        model,
        |name| vocab.get(name).and_then(Value::as_u64).map(|id| id as u32),
        Some("[UNK]"),
    )
    .ok_or(HfJsonError::MissingSpecial("unk"))?;

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
