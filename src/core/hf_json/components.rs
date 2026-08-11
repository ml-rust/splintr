//! Parsers for the shared sections of a HuggingFace `tokenizer.json`:
//! `pre_tokenizer`, `normalizer`, and `added_tokens`. These are backend-agnostic
//! — the family-specific builders in [`super`] consume their output.

use serde_json::Value;

use super::super::added::{AddedToken, AddedTokenSet};
use super::super::normalizer::NormOp;
use super::super::precompiled::{CharsmapDialect, Precompiled};
use super::super::tokenizer::{GPT2_PATTERN, NO_SPLIT_PATTERN, SENTENCEPIECE_PATTERN};
use super::HfJsonError;

/// How input is split before the model runs, distilled to what splintr needs.
#[derive(Debug, Clone)]
pub(super) struct PreTokenization {
    /// Whether tokens are byte-level encoded (GPT-2/Whisper/Llama3 style).
    pub byte_level: bool,
    /// Whether a `Metaspace` node was declared (Mistral/Gemma/Llama-SPM style):
    /// the vocab is `▁`-marked and the tokenizer must be built in metaspace-decoder
    /// mode (`Tokenizer::new_with_metaspace_decoder`) rather than the plain path.
    pub metaspace: bool,
    /// The pre-tokenization split regex.
    pub pattern: String,
    /// Prepend a space to the input (ByteLevel/Metaspace `add_prefix_space`, or
    /// Metaspace `prepend_scheme` != "never").
    pub add_prefix_space: bool,
    /// `Metaspace.split`, defaulting to true as HuggingFace's node does. False
    /// on Mistral's `tokenizer.json`, where the model sees the whole text as one
    /// piece and merges may cross what would otherwise be word boundaries.
    /// Meaningless unless `metaspace`.
    pub metaspace_split: bool,
    /// Whether `pattern` was settled by the file rather than guessed — either a
    /// concrete splitter was recognized (ByteLevel/Metaspace/Split), or the
    /// declared pipeline holds no stages at all and the answer is therefore
    /// [`NO_SPLIT_PATTERN`]. If false, `pattern` is the GPT-2 default, which is
    /// a guess (see the caller's guess guard).
    pub anchored: bool,
    /// Pre-tokenizer `type`s present in the json that this distiller does not
    /// itself model (they may still be handled by the multi-stage engine).
    pub unknown: Vec<String>,
}

/// Walk a `pre_tokenizer` value (possibly a `Sequence`) and distill it to a
/// byte-level flag plus a split regex.
///
/// - `ByteLevel` anywhere ⇒ byte-level encoding; default to [`GPT2_PATTERN`]
///   unless an explicit `Split` regex is present.
/// - `Split { pattern: Regex|String }` ⇒ use that regex.
/// - `Metaspace` (SentencePiece-style) ⇒ [`SENTENCEPIECE_PATTERN`].
/// - Absent, an explicit `null`, or a declared pipeline holding **no stages**
///   (`{"type": "Sequence", "pretokenizers": []}`, however deeply nested) ⇒
///   [`NO_SPLIT_PATTERN`]: HuggingFace splits only where an installed stage
///   splits, so "nothing to run" means "do not split", never "split with a
///   default".
/// - Anything else ⇒ non-byte-level, [`GPT2_PATTERN`] fallback.
///
/// # Declared-and-empty vs declared-and-unrecognized
///
/// The two are told apart by `stages`, the count of non-`Sequence` nodes the
/// walk actually saw — not by whether anything was *understood*. A pipeline with
/// zero nodes has nothing to run and cannot be a guess; a pipeline with nodes
/// this distiller could not use is a guess, so those nodes land in `unknown` and
/// the caller's guess guard refuses the file rather than inventing a split.
///
/// Measured against `tokenizers` 0.22.1 over a BPE fixture, `"a b c"`:
///
/// | `pre_tokenizer` | reference |
/// |---|---|
/// | `null` | `['a', ' ', 'b', ' ', 'c']` |
/// | `{"type":"Sequence","pretokenizers":[]}` | `['a', ' ', 'b', ' ', 'c']` |
/// | nested empty `Sequence`s | `['a', ' ', 'b', ' ', 'c']` |
///
/// So an empty pipeline is byte-identical to `null` there. Every *other* inert
/// shape is a hard load failure in `tokenizers` rather than an alternative
/// splitting — a `Sequence` missing its `pretokenizers` key, a `Split` with no
/// `pattern`, an object with no `type`, an unknown `type`, and a `Sequence`
/// containing one — so refusing them is what matches the reference. The last two
/// already did; the two in between reached [`GPT2_PATTERN`] silently, and are now
/// recorded in `unknown` so the same guard catches them.
pub(super) fn parse_pre_tokenizer(pre: Option<&Value>) -> PreTokenization {
    // A `"pre_tokenizer": null` member is the same thing as no member at all —
    // both deserialize to `Option<PreTokenizerWrapper>::None` in `tokenizers`.
    let pre = pre.filter(|v| !v.is_null());

    /// State threaded through [`walk`], a struct rather than a pile of `&mut`
    /// arguments for the same reason [`BertNormWalk`] is one: the fields are
    /// only interpretable together, and the walk recurses.
    struct Walk {
        byte_level: bool,
        split_regex: Option<String>,
        metaspace: bool,
        /// None until a ByteLevel/Metaspace node sets it; defaulted by the caller.
        add_prefix_space: Option<bool>,
        /// HuggingFace's `Metaspace` defaults `split` to true.
        metaspace_split: bool,
        /// Pre-tokenizer types neither parsed here nor handled implicitly by a
        /// backend.
        unknown: Vec<String>,
        /// Non-`Sequence` nodes seen anywhere in the tree, recognized or not.
        /// Zero of them is the declared-and-empty case; a `Sequence` is a
        /// container and never counts itself.
        stages: usize,
    }

    fn walk(v: &Value, w: &mut Walk) {
        let ty = v.get("type").and_then(Value::as_str);
        if ty != Some("Sequence") {
            w.stages += 1;
        }
        match ty {
            Some("ByteLevel") => {
                w.byte_level = true;
                if let Some(b) = v.get("add_prefix_space").and_then(Value::as_bool) {
                    w.add_prefix_space = Some(b);
                }
            }
            Some("Metaspace") => {
                w.metaspace = true;
                // Newer configs use `prepend_scheme` ("always"/"first"/"never");
                // older ones use `add_prefix_space`.
                if let Some(scheme) = v.get("prepend_scheme").and_then(Value::as_str) {
                    w.add_prefix_space = Some(scheme != "never");
                } else if let Some(b) = v.get("add_prefix_space").and_then(Value::as_bool) {
                    w.add_prefix_space = Some(b);
                }
                if let Some(b) = v.get("split").and_then(Value::as_bool) {
                    w.metaspace_split = b;
                }
            }
            Some("Split") => {
                // pattern is {"Regex": "..."} or {"String": "..."}. The first
                // `Split` in the tree settles the regex; a later one is left to
                // the multi-stage engine, which reads every stage.
                match v.get("pattern").and_then(|p| {
                    p.get("Regex")
                        .and_then(Value::as_str)
                        .or_else(|| p.get("String").and_then(Value::as_str))
                }) {
                    Some(re) => {
                        if w.split_regex.is_none() {
                            w.split_regex = Some(re.to_string());
                        }
                    }
                    // A `Split` with no usable `pattern` splits nothing and is
                    // not a file `tokenizers` will even load (measured: `missing
                    // field 'pattern'`), so it is a shape to refuse, not to
                    // silently replace with the GPT-2 default.
                    None => w.unknown.push("Split (no pattern)".to_string()),
                }
            }
            // Whitespace-only splitters are subsumed by both our SentencePiece
            // (whitespace-split) and byte-level paths, so they need no pattern.
            Some("Whitespace") | Some("WhitespaceSplit") => {}
            Some("Sequence") => {
                if let Some(list) = v.get("pretokenizers").and_then(Value::as_array) {
                    for item in list {
                        walk(item, w);
                    }
                } else {
                    // A `Sequence` with no `pretokenizers` key is not an empty
                    // pipeline, it is an unreadable one (measured: `missing
                    // field 'pretokenizers'`). Count it so it cannot pass for
                    // declared-and-empty, and flag it so the guard fires.
                    w.stages += 1;
                    w.unknown.push("Sequence (no pretokenizers)".to_string());
                }
            }
            Some(other) => w.unknown.push(other.to_string()),
            // No `type` at all: `tokenizers` refuses such a file outright
            // (measured), so this is an unreadable node rather than an inert one.
            None => w.unknown.push("(no type)".to_string()),
        }
    }

    let mut w = Walk {
        byte_level: false,
        split_regex: None,
        metaspace: false,
        add_prefix_space: None,
        metaspace_split: true,
        unknown: Vec::new(),
        stages: 0,
    };
    if let Some(pre) = pre {
        walk(pre, &mut w);
    }
    let Walk {
        byte_level,
        split_regex,
        metaspace,
        add_prefix_space,
        metaspace_split,
        unknown,
        stages,
    } = w;

    // `stages == 0` covers both "no pre_tokenizer member" (the walk never ran)
    // and "a declared pipeline with nothing in it" — HuggingFace treats them
    // identically (measured), so one arm serves both.
    let pattern = match (&split_regex, metaspace, stages) {
        (Some(re), _, _) => re.clone(),
        (None, true, _) => SENTENCEPIECE_PATTERN.to_string(),
        // Nothing to run: run the model over the whole normalized string, as
        // HuggingFace does with no pre-tokenizer stage installed.
        (None, false, 0) => NO_SPLIT_PATTERN.to_string(),
        (None, false, _) => GPT2_PATTERN.to_string(),
    };

    PreTokenization {
        byte_level,
        metaspace,
        pattern,
        // ByteLevel/Metaspace default `add_prefix_space` to true in HF when the
        // field is absent; real configs set it explicitly.
        add_prefix_space: add_prefix_space.unwrap_or(metaspace || byte_level),
        metaspace_split,
        anchored: byte_level || metaspace || split_regex.is_some() || stages == 0,
        unknown,
    }
}

/// BERT-family normalizer flags consumed by the WordPiece backend.
#[derive(Debug, Clone)]
pub(super) struct BertNorm {
    pub lowercase: bool,
    /// Strip accents (`BertNormalizer.strip_accents`), already resolved from the
    /// json's tri-state — see [`parse_bert_norm`]. Independent of `lowercase`.
    pub strip_accents: bool,
    /// Isolate CJK ideographs (`handle_chinese_chars`); defaults to true.
    pub handle_chinese_chars: bool,
    /// Strip control/format chars and normalize whitespace (`clean_text`);
    /// defaults to true.
    pub clean_text: bool,
}

/// State threaded through the [`parse_bert_norm`] walk. A struct rather than a
/// pile of `&mut` arguments because `strip_accents` is only interpretable next
/// to the node that set it.
struct BertNormWalk {
    lowercase: bool,
    /// `None` until a node settles it; `Some` once a `BertNormalizer` (or an
    /// NFD-preceded `StripAccents`) has spoken. Resolved by the caller.
    strip_accents: Option<bool>,
    handle_chinese_chars: bool,
    clean_text: bool,
    /// Whether a decomposing normalizer (NFD/NFKD) has already run in this
    /// sequence — see the `StripAccents` arm for why that matters.
    decomposed: bool,
}

/// Extract the WordPiece-relevant flags from a (`BertNormalizer`-shaped)
/// normalizer. WordPiece's `BasicTokenizer` interleaves CJK splitting with
/// casing, so it consumes flags rather than the ordered op pipeline.
///
/// `strip_accents` is a **tri-state** in the json (`true` / `false` /
/// absent-or-`null`) and is NOT a synonym for `lowercase`. HuggingFace's
/// `BertNormalizer::normalize` computes `strip_accents.unwrap_or(lowercase)`,
/// so the absent form merely *defaults* to `lowercase` while an explicit value
/// wins on its own — cased multilingual BERT ships `strip_accents: false`
/// alongside `lowercase: false`, and a checkpoint whose vocabulary keeps
/// accented forms is mis-tokenized if the two are coupled.
///
/// Measured against `tokenizers` 0.22.1 on a WordPiece fixture holding both
/// `cafe` and `café` (ids 4 and 5), the three cases are:
///
/// | `BertNormalizer` | `"café"` |
/// |---|---|
/// | `lowercase: true,  strip_accents: null`  | `[4]` (`cafe`) |
/// | `lowercase: true,  strip_accents: false` | `[5]` (`café`) |
/// | `lowercase: false, strip_accents: true`  | `[4]` (`cafe`) |
///
/// The `Sequence` walk is deliberately asymmetric about the two sibling node
/// types, and both halves were measured on the same fixture:
///
/// - A standalone `Lowercase` node lowercases and says **nothing** about
///   accents: `Sequence[BertNormalizer{lowercase: false, strip_accents: null},
///   Lowercase]` yields `[5]` (`café` kept). So the `null` default resolves
///   against the `BertNormalizer`'s **own** `lowercase` field, not against
///   whatever else in the sequence happens to lowercase.
/// - A standalone `StripAccents` node does **not** imply BERT-style accent
///   stripping: HF's `StripAccents` only drops nonspacing marks and never
///   decomposes, so on ordinary (NFC) text it is a no-op — `Sequence[StripAccents]`
///   yields `[5]` (`café` kept). Honoring it as a flag would over-strip, because
///   this backend's stripper NFD-decomposes first (as BERT's own does). It is
///   therefore only honored when a decomposing `NFD`/`NFKD` node precedes it,
///   which is the one arrangement HF actually strips under
///   (`Sequence[NFD, StripAccents]` yields `[4]`).
pub(super) fn parse_bert_norm(norm: Option<&Value>) -> BertNorm {
    let mut state = BertNormWalk {
        lowercase: false,
        strip_accents: None,
        handle_chinese_chars: true,
        clean_text: true,
        decomposed: false,
    };

    fn walk(v: &Value, st: &mut BertNormWalk) {
        match v.get("type").and_then(Value::as_str) {
            Some("Lowercase") => st.lowercase = true,
            Some("NFD") | Some("NFKD") => st.decomposed = true,
            Some("StripAccents") if st.decomposed => st.strip_accents = Some(true),
            Some("BertNormalizer") => {
                let lc = v.get("lowercase").and_then(Value::as_bool).unwrap_or(false);
                if lc {
                    st.lowercase = true;
                }
                // Resolved here, against this node's own `lowercase`, because
                // `null` means "follow *my* lowercase" — not the sequence's.
                st.strip_accents = Some(
                    v.get("strip_accents")
                        .and_then(Value::as_bool)
                        .unwrap_or(lc),
                );
                st.handle_chinese_chars = v
                    .get("handle_chinese_chars")
                    .and_then(Value::as_bool)
                    .unwrap_or(true);
                st.clean_text = v.get("clean_text").and_then(Value::as_bool).unwrap_or(true);
            }
            Some("Sequence") => {
                if let Some(list) = v.get("normalizers").and_then(Value::as_array) {
                    for item in list {
                        walk(item, st);
                    }
                }
            }
            _ => {}
        }
    }
    if let Some(norm) = norm {
        walk(norm, &mut state);
    }
    BertNorm {
        lowercase: state.lowercase,
        // Nothing in the file claimed accents either way: no stripping. A bare
        // `Lowercase` normalizer (no `BertNormalizer`) lands here, and HF's
        // `Lowercase` leaves accents intact.
        strip_accents: state.strip_accents.unwrap_or(false),
        handle_chinese_chars: state.handle_chinese_chars,
        clean_text: state.clean_text,
    }
}

/// Parse a `normalizer` value into an ordered list of [`NormOp`]s, flattening
/// `Sequence`s in order.
///
/// A normalizer is an ordered pipeline with no implicit backend fallback, so an
/// unrecognized step (or a `Replace` regex that fails to compile) is a genuine
/// correctness gap: dropping it silently would mis-normalize and produce wrong
/// tokens with no signal. Such cases are surfaced as an error instead.
pub(super) fn parse_norm_ops(norm: Option<&Value>) -> Result<Vec<NormOp>, HfJsonError> {
    let mut ops = Vec::new();
    let mut unknown: Vec<String> = Vec::new();
    let mut bad_regex: Vec<String> = Vec::new();

    fn walk(
        v: &Value,
        ops: &mut Vec<NormOp>,
        unknown: &mut Vec<String>,
        bad_regex: &mut Vec<String>,
    ) {
        match v.get("type").and_then(Value::as_str) {
            Some("NFC") => ops.push(NormOp::Nfc),
            Some("NFD") => ops.push(NormOp::Nfd),
            Some("NFKC") => ops.push(NormOp::Nfkc),
            Some("NFKD") => ops.push(NormOp::Nfkd),
            Some("Lowercase") => ops.push(NormOp::Lowercase),
            Some("StripAccents") => ops.push(NormOp::StripAccents),
            Some("Nmt") => ops.push(NormOp::Nmt),
            Some("Prepend") => {
                if let Some(p) = v.get("prepend").and_then(Value::as_str) {
                    ops.push(NormOp::Prepend(p.to_string()));
                }
            }
            Some("Strip") => ops.push(NormOp::Strip {
                left: v.get("strip_left").and_then(Value::as_bool).unwrap_or(true),
                right: v
                    .get("strip_right")
                    .and_then(Value::as_bool)
                    .unwrap_or(true),
            }),
            Some("Replace") => {
                let content = v
                    .get("content")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                if let Some(p) = v.get("pattern") {
                    if let Some(s) = p.get("String").and_then(Value::as_str) {
                        ops.push(NormOp::ReplaceStr {
                            from: s.to_string(),
                            to: content,
                        });
                    } else if let Some(re) = p.get("Regex").and_then(Value::as_str) {
                        match NormOp::replace_regex(re, content) {
                            Some(op) => ops.push(op),
                            None => bad_regex.push(re.to_string()),
                        }
                    }
                }
            }
            Some("Precompiled") => {
                if let Some(b64) = v.get("precompiled_charsmap").and_then(Value::as_str) {
                    use base64::Engine;
                    if let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(b64) {
                        if let Some(pc) = Precompiled::from_bytes(&bytes) {
                            // A charsmap read out of a `tokenizer.json` is read
                            // back the way `tokenizers` reads it, which is not
                            // the way sentencepiece does.
                            ops.push(NormOp::Precompiled(
                                pc.with_dialect(CharsmapDialect::HuggingFace),
                            ));
                        }
                    }
                }
            }
            // A `BertNormalizer` inside an SP/Unigram graph: expand to its ordered
            // effect (NFD + StripAccents when stripping, then Lowercase).
            Some("BertNormalizer") => {
                let lc = v.get("lowercase").and_then(Value::as_bool).unwrap_or(false);
                let strip = match v.get("strip_accents").and_then(Value::as_bool) {
                    Some(b) => b,
                    None => lc,
                };
                if strip {
                    ops.push(NormOp::Nfd);
                    ops.push(NormOp::StripAccents);
                }
                if lc {
                    ops.push(NormOp::Lowercase);
                }
            }
            Some("Sequence") => {
                if let Some(list) = v.get("normalizers").and_then(Value::as_array) {
                    for item in list {
                        walk(item, ops, unknown, bad_regex);
                    }
                }
            }
            Some(other) => unknown.push(other.to_string()),
            None => {}
        }
    }

    if let Some(norm) = norm {
        walk(norm, &mut ops, &mut unknown, &mut bad_regex);
    }
    if !unknown.is_empty() {
        return Err(HfJsonError::UnsupportedNormalizer(unknown.join(", ")));
    }
    if !bad_regex.is_empty() {
        return Err(HfJsonError::InvalidNormalizerRegex(bad_regex.join(", ")));
    }
    Ok(ops)
}

/// Collect **all** `added_tokens` into a content → [`AddedToken`] set.
///
/// HuggingFace matches every added token during encoding — both `special` ones
/// (`<|endoftext|>`) and non-special content tokens (e.g. gpt-neox's whitespace
/// runs, deepseek's byte chars) — so the matcher must know all of them, not just
/// the special-flagged ones.
///
/// Each entry's `lstrip`/`rstrip` booleans are read here rather than assumed
/// false: XLM-RoBERTa-family vocabularies (bge-m3, bge-reranker-v2-m3, and most
/// multilingual embedding models) declare `<mask>` with `lstrip: true` while
/// leaving it off on their four other added tokens, so the flags are only
/// correct when taken per token from the file. Both default to `false` when
/// absent, which is `tokenizers`' own default for an `AddedToken`.
pub(in crate::core) fn parse_special_tokens(root: &Value) -> AddedTokenSet {
    let mut specials = AddedTokenSet::new();
    if let Some(list) = root.get("added_tokens").and_then(Value::as_array) {
        for t in list {
            if let (Some(content), Some(id)) = (
                t.get("content").and_then(Value::as_str),
                t.get("id").and_then(Value::as_u64),
            ) {
                specials.insert(
                    content,
                    AddedToken {
                        id: id as u32,
                        lstrip: t.get("lstrip").and_then(Value::as_bool).unwrap_or(false),
                        rstrip: t.get("rstrip").and_then(Value::as_bool).unwrap_or(false),
                    },
                );
            }
        }
    }
    specials
}

/// Ids of `added_tokens` flagged `special: true` — dropped on decode to match
/// HuggingFace's default `skip_special_tokens=true`. Non-special added tokens
/// (e.g. gpt-neox whitespace runs) are kept.
pub(super) fn parse_special_decode_ids(root: &Value) -> rustc_hash::FxHashSet<u32> {
    let mut ids = rustc_hash::FxHashSet::default();
    if let Some(list) = root.get("added_tokens").and_then(Value::as_array) {
        for t in list {
            if t.get("special").and_then(Value::as_bool).unwrap_or(true) {
                if let Some(id) = t.get("id").and_then(Value::as_u64) {
                    ids.insert(id as u32);
                }
            }
        }
    }
    ids
}

/// Resolve `model.unk_token` to its `model.vocab` id.
///
/// `default` is the spelling to assume when the model declares none — WordPiece
/// files routinely omit `unk_token` while still relying on `[UNK]`, whereas a
/// BPE file that omits it genuinely has no unk (pass `None` there). Returns
/// `None` when there is no spelling to look up or the declared one is absent
/// from the vocabulary; a backend that cannot work without one turns that into
/// [`HfJsonError::MissingSpecial`].
/// The id of `model.unk_token`, resolved through `lookup`.
///
/// Takes a resolver rather than a vocabulary because the two callers no longer
/// hold the same shape: BPE reads its vocabulary as borrowed token/id pairs and
/// never builds a `Map`, while WordPiece still has one.
pub(super) fn parse_unk_id(
    model: &Value,
    lookup: impl Fn(&str) -> Option<u32>,
    default: Option<&str>,
) -> Option<u32> {
    let unk_token = model.get("unk_token").and_then(Value::as_str).or(default)?;
    lookup(unk_token)
}

/// Find the id of the first matching token content in `added_tokens`.
pub(in crate::core) fn find_added_token(root: &Value, candidates: &[&str]) -> Option<u32> {
    let list = root.get("added_tokens").and_then(Value::as_array)?;
    for cand in candidates {
        for t in list {
            if t.get("content").and_then(Value::as_str) == Some(cand) {
                return t.get("id").and_then(Value::as_u64).map(|n| n as u32);
            }
        }
    }
    None
}
