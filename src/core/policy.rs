//! The special-token policy of a loaded tokenizer: the `post_processor`
//! templates (single **and** pair) plus the named special tokens.
//!
//! Boundary tokens are the one part of tokenization every consumer gets wrong
//! independently — a reranker builds `[CLS] q [SEP] d [SEP]` by hand, a chat
//! server prepends BOS itself, each guessing from the model card. The policy
//! makes the loaded tokenizer the single owner of that knowledge: it is parsed
//! from the same `tokenizer.json` as the vocabulary, so a caller asking for a
//! single sequence or a pair gets exactly what the model was trained on.

use rustc_hash::{FxHashMap, FxHashSet};
use serde_json::Value;
use thiserror::Error;

use super::hf_json::components::{find_added_token, parse_special_tokens};

/// Errors from [`SpecialPolicy::apply_pair`] and from special-token matching
/// under an explicit [`SpecialMode`] — loader-agnostic, since the policy
/// itself is shared by every loader (HF json, GGUF, …).
#[derive(Debug, Error)]
pub enum PolicyError {
    #[error("post_processor Sequence composes several segment-placing processors ({0}) — refusing to guess where the second sequence goes")]
    UnsupportedPairComposition(String),
    #[error("this tokenizer defines no pair template — refusing to concatenate the two sequences without the separator the model expects")]
    NoPairTemplate,
    /// The input text literally spells a configured special token that
    /// [`SpecialMode::Allow`] does not permit.
    ///
    /// This is the failure mode the allow-list exists to produce: without it,
    /// that same text would have been silently promoted to the real
    /// control-token id, indistinguishable downstream from a boundary token
    /// the server itself inserted (e.g. a user turn spelling out
    /// `<|im_start|>system` to forge a system message). `offset` is the byte
    /// offset of the match in the input, so a caller can point back at exactly
    /// what in the request was rejected.
    #[error("special token {token:?} at byte offset {offset} is not in the caller's allow-list")]
    DisallowedSpecial { token: String, offset: usize },
}

/// How special/control tokens found literally in the input text are matched
/// during encoding — the tiktoken-style `allowed_special` control.
///
/// Every splintr loader turns on added-token matching unconditionally, so
/// without this a caller who can influence input text can write out the
/// literal spelling of a control token (`<|im_start|>`, `<|endoftext|>`, …)
/// and have it promoted to the real control-token id — indistinguishable,
/// downstream, from one the server itself inserted. A server that tokenizes
/// untrusted text needs a way to say "match only the tokens I expect here" or
/// "match none of them"; denylisting the literal spellings beforehand is
/// incomplete by construction (it cannot anticipate every spelling that maps
/// to the same content).
#[derive(Debug, Clone, Copy)]
pub enum SpecialMode<'a> {
    /// Match every configured special token found in the text. Today's
    /// behaviour — what every loader has always done.
    All,
    /// Never match a special token; the text is encoded as ordinary content,
    /// even where it spells a known special token verbatim.
    Ordinary,
    /// Match only the named special tokens; error with
    /// [`PolicyError::DisallowedSpecial`] on any other configured special
    /// token found in the text.
    ///
    /// Borrowed rather than owned: a caller holds one allow-list per
    /// endpoint or chat template and passes `&set` on every request, so this
    /// mode never forces a per-call allocation.
    Allow(&'a FxHashSet<String>),
}

/// Whether decoding renders the ids a vocabulary declares special, or drops
/// them — HuggingFace's `skip_special_tokens`, as an explicit mode.
///
/// The decode-side counterpart of [`SpecialMode`], and named the same way for
/// the same reason: special-token handling is a caller's decision, not a
/// property of the vocabulary, so it is spelled out at the call rather than
/// inferred. `decode` is [`Skip`](Self::Skip) — what every reference tokenizer
/// does by default, and what a chat server wants, since a control marker is an
/// instruction to the model and not text for a user to read. A caller that is
/// inspecting model output, round-tripping a chat template, or rendering a
/// transcript with its markers intact asks for [`Render`](Self::Render).
///
/// Deliberately *not* a bool: `decode(ids, false)` at a call site says nothing
/// about which way round the flag runs, and this is exactly the decision whose
/// two answers differ silently — both produce plausible text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecialDecode {
    /// Drop every id the vocabulary declares special, rendering nothing in its
    /// place. HuggingFace's `skip_special_tokens=True`, and what
    /// [`Tokenize::decode`](crate::Tokenize::decode) does.
    Skip,
    /// Render each declared-special id's own spelling, exactly as any ordinary
    /// token is rendered. HuggingFace's `skip_special_tokens=False`.
    ///
    /// Only the *declared special* ids come back. An id that carries no surface
    /// at all is still dropped — there is nothing to render — which is the same
    /// decision every decode path here already makes for it.
    Render,
}

/// Candidate contents for the end-of-sequence token, most specific first.
///
/// Shared with the Unigram backend's own eos fallback in `mod.rs`, which needs
/// the same candidates for its internal (never-`None`) eos.
pub(super) const EOS_CANDIDATES: &[&str] =
    &["</s>", "<eos>", "<|endoftext|>", "<|end_of_text|>", "[SEP]"];

/// One element of a post-processor template: a literal special token, or the
/// slot where the first (`A`) / second (`B`) sequence's content tokens go.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Segment {
    Special(u32),
    A,
    B,
}

/// The pair-composition state of a `Template`.
///
/// `Ambiguous` defers a `Sequence`'s multi-contributor conflict from load time
/// to first pair-encode: the `single` template still composes soundly, so a
/// tokenizer that only ever sees single sequences must keep loading.
#[derive(Debug, Clone)]
enum PairTemplate {
    Defined(Vec<Segment>),
    /// The json declares no pair template.
    Absent,
    /// A `Sequence` had multiple segment-placing processors, so where `B` goes
    /// would be a guess. Holds the competing processor names for the error.
    Ambiguous(String),
}

/// The single-sequence template and, when the json defines one, the pair
/// template. `pair` is `Absent` rather than a guess: silently reusing `single`
/// twice would drop the separator BERT-family models expect between segments.
#[derive(Debug, Clone)]
struct Template {
    single: Vec<Segment>,
    pair: PairTemplate,
}

impl Default for Template {
    /// No post-processor: the content tokens, untouched.
    fn default() -> Self {
        Self {
            single: vec![Segment::A],
            pair: PairTemplate::Absent,
        }
    }
}

impl Template {
    /// Whether this template does anything beyond emitting `A` — used to detect
    /// a `Sequence` whose pair composition would be a guess.
    fn contributes(&self) -> bool {
        self.single != [Segment::A] || matches!(self.pair, PairTemplate::Defined(_))
    }
}

/// How a loaded tokenizer wraps content tokens with special tokens, plus the
/// special tokens it knows by name.
#[derive(Debug, Clone, Default)]
pub struct SpecialPolicy {
    template: Template,
    eos_id: Option<u32>,
    named: FxHashMap<String, u32>,
}

impl SpecialPolicy {
    /// Wrap one sequence's content `ids` using the single-sequence template.
    pub fn apply_single(&self, ids: Vec<u32>) -> Vec<u32> {
        if self.template.single == [Segment::A] {
            return ids;
        }
        let mut out = Vec::with_capacity(ids.len() + self.template.single.len());
        render(&self.template.single, &ids, &[], &mut out);
        out
    }

    /// Wrap two sequences' content ids using the pair template.
    ///
    /// Errors when the json defined no pair template — concatenating `a` and `b`
    /// without the model's separator would feed it a sequence unlike anything it
    /// saw in training, and silently.
    pub fn apply_pair(&self, a: &[u32], b: &[u32]) -> Result<Vec<u32>, PolicyError> {
        let pair = match &self.template.pair {
            PairTemplate::Defined(pair) => pair,
            PairTemplate::Absent => return Err(PolicyError::NoPairTemplate),
            PairTemplate::Ambiguous(names) => {
                return Err(PolicyError::UnsupportedPairComposition(names.clone()))
            }
        };
        let mut out = Vec::with_capacity(a.len() + b.len() + pair.len());
        render(pair, a, b, &mut out);
        Ok(out)
    }

    /// The end-of-sequence token id, when the json names one.
    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_id
    }

    /// Whether `id` is the end-of-sequence token.
    pub fn is_eos(&self, id: u32) -> bool {
        self.eos_id == Some(id)
    }

    /// The id of an added token by its content (e.g. `"[CLS]"`, `"<|im_end|>"`).
    pub fn special_token_id(&self, name: &str) -> Option<u32> {
        self.named.get(name).copied()
    }

    /// Build a policy that only wraps a single sequence in boundary tokens.
    ///
    /// For sources that state their boundaries as flags plus ids rather than as
    /// a template — a GGUF's `add_bos_token` / `add_eos_token` — so the loader
    /// resolves the flags and hands over the ids that survive them. Passing
    /// `None` for both yields the identity template, for a vocabulary that
    /// states no boundary tokens at all; a source whose boundaries are a
    /// cls/sep pair uses [`cls_sep`](Self::cls_sep) instead.
    ///
    /// `eos_id` is the vocabulary's end-of-sequence token, which exists
    /// independently of whether one is *appended*: a generation loop must still
    /// be able to stop on it.
    ///
    /// There is deliberately no pair template. These sources never declare one,
    /// and synthesizing one would concatenate two sequences without the
    /// separator the model expects — [`apply_pair`](Self::apply_pair) refuses
    /// instead.
    pub(super) fn boundary(
        bos: Option<u32>,
        eos: Option<u32>,
        eos_id: Option<u32>,
        named: FxHashMap<String, u32>,
    ) -> Self {
        let mut single = Vec::with_capacity(3);
        single.extend(bos.map(Segment::Special));
        single.push(Segment::A);
        single.extend(eos.map(Segment::Special));
        Self {
            template: Template {
                single,
                pair: PairTemplate::Absent,
            },
            eos_id,
            named,
        }
    }

    /// Build the BERT-family `[CLS] A [SEP]` / `[CLS] A [SEP] B [SEP]` policy
    /// from the two ids directly.
    ///
    /// The sibling of [`boundary`](Self::boundary) for a source that states its
    /// boundaries as a cls/sep *pair* rather than as BOS/EOS flags: a GGUF
    /// `bert` vocabulary. llama.cpp's WPM path prepends `[CLS]` and appends
    /// `[SEP]` whenever `add_special` is set, and HuggingFace's
    /// `all-MiniLM-L6-v2` `tokenizer.json` declares exactly this template
    /// (measured with `tokenizers` 0.22.1: `"hello world"` →
    /// `[101, 7592, 2088, 102]`, and as a pair with `"goodbye moon"` →
    /// `[101, 7592, 2088, 102, 9119, 4231, 102]`). Both containers of one
    /// checkpoint must therefore answer `encode` identically, so both build the
    /// template through the one `cls_sep_segments` shape.
    ///
    /// Both ids are required, not `Option`: a vocabulary that names no `[CLS]`
    /// or no `[SEP]` has no boundary to place, and the caller keeps the
    /// identity [`boundary`](Self::boundary) policy rather than having tokens
    /// invented for it.
    pub(super) fn cls_sep(
        cls: u32,
        sep: u32,
        eos_id: Option<u32>,
        named: FxHashMap<String, u32>,
    ) -> Self {
        Self {
            template: cls_sep_segments(Some(cls), Some(sep), PairShape::Bert),
            eos_id,
            named,
        }
    }

    /// How many special-token slots the single-sequence template adds around
    /// the content — 2 for `[CLS] A [SEP]`, 1 for a lone BOS, 0 for identity.
    ///
    /// For a caller that must fit a sequence into a fixed model length: the
    /// content has to be truncated to `max_len - single_overhead()` *before*
    /// [`apply_single`](Self::apply_single) runs, because truncating after it
    /// would cut the trailing `[SEP]`/EOS off — the exact position a
    /// last-token-pooling model reads. The count is otherwise only obtainable
    /// as `apply_single(vec![]).len()`, which is a trick rather than an answer.
    pub fn single_overhead(&self) -> usize {
        self.template
            .single
            .iter()
            .filter(|s| matches!(s, Segment::Special(_)))
            .count()
    }
}

/// Expand a template into ids, substituting the content of both sequences.
fn render(segments: &[Segment], a: &[u32], b: &[u32], out: &mut Vec<u32>) {
    for segment in segments {
        match segment {
            Segment::Special(id) => out.push(*id),
            Segment::A => out.extend_from_slice(a),
            Segment::B => out.extend_from_slice(b),
        }
    }
}

/// Parse the whole policy out of a `tokenizer.json` root value.
pub(super) fn parse(root: &Value) -> Result<SpecialPolicy, PolicyError> {
    Ok(SpecialPolicy {
        template: parse_template(root.get("post_processor"))?,
        // Deliberately a genuine `Option`: the Unigram backend's internal eos
        // falls back to `unk_id`/0 because it only drives decode-skipping, which
        // is the wrong answer for a caller asking what EOS *is*.
        eos_id: find_added_token(root, EOS_CANDIDATES),
        // Name → id only: the `lstrip`/`rstrip` flags belong to the matcher that
        // splits the input, not to a lookup that answers "what id is `[CLS]`?".
        named: parse_special_tokens(root).into_id_map(),
    })
}

/// Parse a `post_processor` node into its single and pair templates.
fn parse_template(pp: Option<&Value>) -> Result<Template, PolicyError> {
    let Some(pp) = pp else {
        return Ok(Template::default());
    };
    match pp.get("type").and_then(Value::as_str) {
        // `{ cls: [token, id], sep: [token, id] }`. Neither json type carries a
        // pair array, so both templates are synthesized from the two ids.
        Some("BertProcessing") => Ok(cls_sep_template(pp, PairShape::Bert)),
        Some("RobertaProcessing") => Ok(cls_sep_template(pp, PairShape::Roberta)),
        Some("TemplateProcessing") => Ok(parse_template_processing(pp)),
        Some("Sequence") => parse_sequence(pp),
        _ => Ok(Template::default()),
    }
}

/// How a cls/sep processor joins two sequences.
enum PairShape {
    /// `[CLS] A [SEP] B [SEP]`
    Bert,
    /// `<s> A </s> </s> B </s>` — RoBERTa doubles the separator.
    Roberta,
}

fn cls_sep_template(pp: &Value, shape: PairShape) -> Template {
    let id = |k: &str| {
        pp.get(k)
            .and_then(|p| p.get(1))
            .and_then(Value::as_u64)
            .map(|n| n as u32)
    };
    cls_sep_segments(id("cls"), id("sep"), shape)
}

/// The `[CLS] A [SEP]` / `[CLS] A [SEP] B [SEP]` shape, built from the two ids
/// alone.
///
/// Split out of [`cls_sep_template`] so a source that states its cls/sep as
/// plain ids rather than as a json node — a GGUF `bert` vocabulary, whose
/// boundaries llama.cpp's WPM path takes straight from `[CLS]`/`[SEP]` — lands
/// on the *same* segments as the `tokenizer.json` of the same model. One shape,
/// one place: the two carriers cannot drift into disagreeing about what
/// `encode` returns for one checkpoint.
fn cls_sep_segments(cls: Option<u32>, sep: Option<u32>, shape: PairShape) -> Template {
    let mut single = Vec::with_capacity(3);
    single.extend(cls.map(Segment::Special));
    single.push(Segment::A);
    single.extend(sep.map(Segment::Special));

    // A pair needs both ids to be separable at all; without them, refuse rather
    // than emit a template that runs the two sequences together.
    let pair = match (cls, sep) {
        (Some(cls), Some(sep)) => PairTemplate::Defined(match shape {
            PairShape::Bert => vec![
                Segment::Special(cls),
                Segment::A,
                Segment::Special(sep),
                Segment::B,
                Segment::Special(sep),
            ],
            PairShape::Roberta => vec![
                Segment::Special(cls),
                Segment::A,
                Segment::Special(sep),
                Segment::Special(sep),
                Segment::B,
                Segment::Special(sep),
            ],
        }),
        _ => PairTemplate::Absent,
    };
    Template { single, pair }
}

/// Parse a `TemplateProcessing`'s `single` and `pair` arrays.
fn parse_template_processing(pp: &Value) -> Template {
    // Resolve a special-token content string to its first id.
    let resolve = |tok: &str| -> Option<u32> {
        pp.get("special_tokens")
            .and_then(|m| m.get(tok))
            .and_then(|e| e.get("ids"))
            .and_then(Value::as_array)
            .and_then(|a| a.first())
            .and_then(Value::as_u64)
            .map(|n| n as u32)
    };
    let segments = |key: &str| -> Option<Vec<Segment>> {
        let items = pp.get(key).and_then(Value::as_array)?;
        let mut out = Vec::with_capacity(items.len());
        for item in items {
            if let Some(seq) = item.get("Sequence") {
                // `{"Sequence": {"id": "A"|"B", "type_id": n}}`.
                match seq.get("id").and_then(Value::as_str) {
                    Some("B") => out.push(Segment::B),
                    _ => out.push(Segment::A),
                }
            } else if let Some(id) = item
                .get("SpecialToken")
                .and_then(|s| s.get("id"))
                .and_then(Value::as_str)
                .and_then(&resolve)
            {
                out.push(Segment::Special(id));
            }
        }
        Some(out)
    };
    Template {
        single: segments("single").unwrap_or_else(|| vec![Segment::A]),
        pair: match segments("pair") {
            Some(segs) => PairTemplate::Defined(segs),
            None => PairTemplate::Absent,
        },
    }
}

/// Compose a `Sequence` of post-processors.
///
/// Singles nest: each processor wraps the result of the ones before it, so the
/// earlier processor's tokens sit closest to the content.
fn parse_sequence(pp: &Value) -> Result<Template, PolicyError> {
    let mut single = vec![Segment::A];
    let mut pair = PairTemplate::Absent;
    let mut contributors: Vec<String> = Vec::new();

    if let Some(list) = pp.get("processors").and_then(Value::as_array) {
        for sub in list {
            let t = parse_template(Some(sub))?;
            if t.contributes() {
                contributors.push(
                    sub.get("type")
                        .and_then(Value::as_str)
                        .unwrap_or("<untyped>")
                        .to_string(),
                );
                pair = t.pair.clone();
            }
            single = substitute(&single, &t.single);
        }
    }

    // Two processors that each place segments give no sound answer for where
    // `B` goes. `single` still composed correctly above, so the tokenizer must
    // keep loading — only a pair encode needs to fail, and only then.
    if contributors.len() > 1 {
        pair = PairTemplate::Ambiguous(contributors.join(", "));
    }
    Ok(Template { single, pair })
}

/// Replace every `A` slot in `outer` with the whole of `inner`.
fn substitute(outer: &[Segment], inner: &[Segment]) -> Vec<Segment> {
    let mut out = Vec::with_capacity(outer.len() + inner.len());
    for segment in outer {
        match segment {
            Segment::A => out.extend_from_slice(inner),
            other => out.push(other.clone()),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(json: &str) -> Result<SpecialPolicy, PolicyError> {
        let root: Value = serde_json::from_str(json).expect("valid json");
        parse(&root)
    }

    #[test]
    fn no_post_processor_leaves_content_untouched() {
        let p = policy("{}").expect("parses");
        assert_eq!(p.apply_single(vec![7, 8]), vec![7, 8]);
        assert!(matches!(
            p.apply_pair(&[7], &[8]),
            Err(PolicyError::NoPairTemplate)
        ));
    }

    #[test]
    fn bert_processing_synthesizes_both_templates() {
        let p = policy(
            r#"{"post_processor": {"type": "BertProcessing",
                "cls": ["[CLS]", 1], "sep": ["[SEP]", 2]}}"#,
        )
        .expect("parses");
        assert_eq!(p.apply_single(vec![3]), vec![1, 3, 2]);
        assert_eq!(p.apply_pair(&[3], &[4]).expect("pair"), vec![1, 3, 2, 4, 2]);
    }

    /// RoBERTa doubles the separator between the two segments; collapsing it to
    /// one would shift every position after it.
    #[test]
    fn roberta_processing_doubles_the_separator() {
        let p = policy(
            r#"{"post_processor": {"type": "RobertaProcessing",
                "cls": ["<s>", 0], "sep": ["</s>", 2]}}"#,
        )
        .expect("parses");
        assert_eq!(p.apply_single(vec![5]), vec![0, 5, 2]);
        assert_eq!(
            p.apply_pair(&[5], &[6]).expect("pair"),
            vec![0, 5, 2, 2, 6, 2]
        );
    }

    #[test]
    fn template_processing_reads_the_pair_array() {
        let p = policy(
            r#"{"post_processor": {"type": "TemplateProcessing",
                "single": [
                    {"SpecialToken": {"id": "<s>", "type_id": 0}},
                    {"Sequence": {"id": "A", "type_id": 0}}
                ],
                "pair": [
                    {"SpecialToken": {"id": "<s>", "type_id": 0}},
                    {"Sequence": {"id": "A", "type_id": 0}},
                    {"SpecialToken": {"id": "</s>", "type_id": 0}},
                    {"Sequence": {"id": "B", "type_id": 1}}
                ],
                "special_tokens": {
                    "<s>": {"id": "<s>", "ids": [1], "tokens": ["<s>"]},
                    "</s>": {"id": "</s>", "ids": [2], "tokens": ["</s>"]}
                }}}"#,
        )
        .expect("parses");
        assert_eq!(p.apply_single(vec![9]), vec![1, 9]);
        assert_eq!(p.apply_pair(&[9], &[10]).expect("pair"), vec![1, 9, 2, 10]);
    }

    /// A `Sequence` of a no-op processor and a real one composes to the real
    /// one — the common `[ByteLevel, TemplateProcessing]` shape.
    #[test]
    fn sequence_with_one_contributor_composes() {
        let p = policy(
            r#"{"post_processor": {"type": "Sequence", "processors": [
                {"type": "ByteLevel", "add_prefix_space": true},
                {"type": "BertProcessing", "cls": ["[CLS]", 1], "sep": ["[SEP]", 2]}
            ]}}"#,
        )
        .expect("parses");
        assert_eq!(p.apply_single(vec![3]), vec![1, 3, 2]);
        assert_eq!(p.apply_pair(&[3], &[4]).expect("pair"), vec![1, 3, 2, 4, 2]);
    }

    /// Two segment-placing processors give no sound answer for where `B` goes,
    /// but that ambiguity must not fail the whole tokenizer load — `single`
    /// still composes fine, so only a pair encode should refuse.
    #[test]
    fn sequence_with_two_contributors_loads_and_refuses_only_on_pair() {
        let p = policy(
            r#"{"post_processor": {"type": "Sequence", "processors": [
                {"type": "BertProcessing", "cls": ["[CLS]", 1], "sep": ["[SEP]", 2]},
                {"type": "RobertaProcessing", "cls": ["<s>", 3], "sep": ["</s>", 4]}
            ]}}"#,
        )
        .expect("parses despite pair ambiguity");
        // Singles nest: the earlier (Bert) template stays the outer frame and
        // the later (Roberta) processor's template is substituted into its `A`
        // slot, so Roberta's cls/sep (3, 4) end up closest to the content.
        assert_eq!(p.apply_single(vec![99]), vec![1, 3, 99, 4, 2]);
        assert!(matches!(
            p.apply_pair(&[99], &[100]),
            Err(PolicyError::UnsupportedPairComposition(_))
        ));
    }

    #[test]
    fn eos_and_named_tokens_come_from_added_tokens() {
        let p = policy(
            r#"{"added_tokens": [
                {"id": 1, "content": "[CLS]", "special": true},
                {"id": 2, "content": "[SEP]", "special": true}
            ]}"#,
        )
        .expect("parses");
        assert_eq!(p.special_token_id("[CLS]"), Some(1));
        assert_eq!(p.eos_token_id(), Some(2));
        assert!(p.is_eos(2));
        assert!(!p.is_eos(1));
    }

    /// The two carriers of one checkpoint's boundaries — a `tokenizer.json`
    /// `BertProcessing` node and a GGUF's bare `[CLS]`/`[SEP]` ids — must build
    /// the same policy. all-MiniLM-L6-v2 states cls=101/sep=102 in both, and
    /// `tokenizers` 0.22.1 encodes `"hello world"` there as
    /// `[101, 7592, 2088, 102]`.
    #[test]
    fn cls_sep_from_ids_matches_the_json_path() {
        let from_json = policy(
            r#"{"post_processor": {"type": "BertProcessing",
                "cls": ["[CLS]", 101], "sep": ["[SEP]", 102]}}"#,
        )
        .expect("parses");
        let from_ids = SpecialPolicy::cls_sep(101, 102, Some(102), FxHashMap::default());

        assert_eq!(
            from_ids.apply_single(vec![7592, 2088]),
            vec![101, 7592, 2088, 102]
        );
        assert_eq!(
            from_json.apply_single(vec![7592, 2088]),
            from_ids.apply_single(vec![7592, 2088])
        );
        assert_eq!(
            from_json.apply_pair(&[7592], &[9119]).expect("pair"),
            from_ids.apply_pair(&[7592], &[9119]).expect("pair")
        );
    }

    /// The count a caller reserves before truncating: special slots only, never
    /// the content slot.
    #[test]
    fn single_overhead_counts_only_the_special_slots() {
        assert_eq!(SpecialPolicy::default().single_overhead(), 0);
        let bos_only = SpecialPolicy::boundary(Some(1), None, None, FxHashMap::default());
        assert_eq!(bos_only.single_overhead(), 1);
        assert_eq!(
            SpecialPolicy::cls_sep(101, 102, None, FxHashMap::default()).single_overhead(),
            2
        );
    }

    /// A vocabulary with no end-of-sequence token must report `None`, not a
    /// fabricated id — a caller stopping generation on token 0 would be wrong.
    #[test]
    fn absent_eos_is_none() {
        let p = policy(r#"{"added_tokens": [{"id": 0, "content": "<unk>"}]}"#).expect("parses");
        assert_eq!(p.eos_token_id(), None);
        assert!(!p.is_eos(0));
    }
}
