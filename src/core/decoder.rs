//! HuggingFace `decoder` pipeline.
//!
//! A `tokenizer.json` `decoder` is an ordered chain of operations applied to the
//! list of token *surface strings* (after special tokens are skipped) to turn
//! them back into text. This module parses that chain and applies it faithfully,
//! so decoding is driven by the file's declared decoder rather than inferred from
//! the backend type.
//!
//! Each op transforms a `Vec<String>` of tokens into another `Vec<String>`; the
//! final result is the concatenation. This mirrors `tokenizers`'
//! `Decoder::decode_chain`. Every HF decoder variant is handled
//! (`ByteLevel`, `Replace`, `Strip`, `Fuse`, `ByteFallback`, `Metaspace`,
//! `WordPiece`, `BPEDecoder`, and `Sequence`), so no configured step is silently
//! dropped or approximated.

use serde_json::Value;

use super::byte_level::byte_level_decode;
use super::streaming::{ByteFallbackRule, DecodePost, RenderRules, WordSeparator};

/// A `Replace` decoder pattern: a literal string or a regex.
enum ReplacePattern {
    Str(String),
    Regex(Box<regexr::Regex>),
}

/// A single decoder operation.
enum DecodeOp {
    /// Reverse GPT-2 byte-level encoding (mapped code points → bytes). Tokens
    /// that aren't valid byte-level (e.g. literal-content added tokens) are kept
    /// as-is; bytes are concatenated across tokens before one UTF-8 decode so
    /// multi-byte characters split across tokens reassemble correctly.
    ByteLevel,
    /// Replace each occurrence of a pattern with `to`, per token.
    Replace { pattern: ReplacePattern, to: String },
    /// Strip up to `start` leading and `stop` trailing `content` chars per token.
    Strip {
        content: char,
        start: usize,
        stop: usize,
    },
    /// Concatenate all tokens into a single token.
    Fuse,
    /// Combine runs of `<0xNN>` byte tokens into UTF-8 (invalid runs → one
    /// replacement char per byte), passing other tokens through.
    ByteFallback,
    /// SentencePiece metaspace: replace `replacement` with a space and, when
    /// `add_prefix_space`, drop the single leading space of the first token.
    Metaspace {
        replacement: char,
        add_prefix_space: bool,
    },
    /// WordPiece: strip the continuation `prefix` from non-initial tokens (or add
    /// a leading space), optionally cleaning whitespace before punctuation.
    WordPiece { prefix: String, cleanup: bool },
    /// BPE word-suffix decoder: replace `suffix` (e.g. `</w>`) with a space,
    /// except on the final token where it is removed.
    Bpe { suffix: String },
}

/// An ordered decoder pipeline parsed from a `tokenizer.json` `decoder` value.
pub struct Decoder {
    ops: Vec<DecodeOp>,
}

impl Decoder {
    /// Apply the chain to the (already special-skipped) surface tokens and
    /// concatenate into the decoded string.
    pub fn decode(&self, tokens: Vec<String>) -> String {
        let mut toks = tokens;
        for op in &self.ops {
            toks = op.apply(toks);
        }
        toks.concat()
    }

    /// The declared pipeline as per-id [`RenderRules`] plus ordered
    /// [`DecodePost`]s — the same machinery every backend already decodes
    /// through — or `None` when the pipeline cannot be driven that way.
    ///
    /// Lowering is what gives the declared pipeline a *streaming* form. Every op
    /// here transforms a token list, so it is only expressible against a cursor
    /// when it distributes over that list: a rule about one token, or a rule
    /// about text that a held piece of state can carry across a chunk boundary.
    ///
    /// `None` is returned for three declared shapes that are provably not
    /// incrementally computable, and they are not faked:
    ///
    /// * [`DecodeOp::Bpe`] — it branches on whether a token is the *last* one,
    ///   which a stream does not know until it has ended.
    /// * A `Strip` with `stop > 0` after a `Fuse` — a strip off the end of the
    ///   whole text, which is the same unknowable last token in another form.
    /// * A `Replace` after a `Fuse` — an arbitrary pattern over the whole text,
    ///   which can match across a chunk boundary. (Both a `Regex` and a literal
    ///   pattern can; neither is lowered.)
    ///
    /// It is also returned for the ops that *are* incrementally computable but
    /// that the rendering rules cannot express: a per-token `Regex` replace, a
    /// per-token `Strip`, and any chain that orders its surface-level ops after
    /// a `ByteLevel` or a `ByteFallback` — a rendering rule only ever sees a
    /// surface, never the text a resolved byte run decoded to. None of those
    /// shapes appears in a shipping `tokenizer.json`; the shapes that do —
    /// `Metaspace`, `WordPiece`, `ByteLevel`, and
    /// `Sequence[Replace, ByteFallback, Fuse, Strip]` — all lower.
    ///
    /// The returned rules carry no vocabulary: a pipeline is declared
    /// independently of the tokenizer it runs against, so the caller supplies the
    /// tables with [`RenderRules::with_vocabulary`].
    pub(crate) fn lower(&self) -> Option<(RenderRules, Vec<DecodePost>)> {
        self.lower_or_reason().ok()
    }

    /// The declared type name of the op that stopped [`lower`](Self::lower), or
    /// `None` when the pipeline lowers.
    ///
    /// The reason and the lowering come out of the same pass, so a caller that
    /// refuses to stream can name the step it refused on without a second,
    /// drifting description of which shapes are refusable.
    pub(crate) fn unstreamable_op(&self) -> Option<&'static str> {
        self.lower_or_reason().err()
    }

    /// [`lower`](Self::lower)'s one implementation, reporting the declared type
    /// name of the op it refused on instead of a bare `None`.
    fn lower_or_reason(&self) -> Result<(RenderRules, Vec<DecodePost>), &'static str> {
        let mut byte_fallback = ByteFallbackRule::None;
        let mut use_byte_level = false;
        let mut replaces: Vec<(String, String)> = Vec::new();
        let mut separator: Option<WordSeparator> = None;
        let mut unit_cleanup = false;
        let mut post = Vec::new();
        // Whether the token list has already been collapsed to a single token.
        // `Fuse` does that outright; `ByteLevel` does it too, since it decodes
        // every token's bytes into one string. Past that point an op sees whole
        // text rather than a token, and the index-dependent ops (`Metaspace`'s
        // first token, `WordPiece`'s non-first, `Bpe`'s last) would mean
        // something else entirely — so none of them is lowered past it.
        let mut fused = false;
        // Whether a `ByteFallback` has run. Past that point a surface-level op
        // would also see the text that byte-fallback runs decoded to, which a
        // *rendering* rule — which only ever sees a surface — cannot reproduce.
        // The shipping chain declares its `Replace` ahead of the fallback, which
        // is the order the rules do express.
        let mut bytes_resolved = false;

        for op in &self.ops {
            match op {
                DecodeOp::ByteLevel => {
                    // Only as the whole spelling rule, never layered over one:
                    // the surface a later op would see is not the surface the
                    // rules render from.
                    if fused || bytes_resolved || !replaces.is_empty() || separator.is_some() {
                        return Err("ByteLevel");
                    }
                    use_byte_level = true;
                    fused = true;
                }
                DecodeOp::Replace { pattern, to } => {
                    let from = match pattern {
                        ReplacePattern::Str(from) => from,
                        // An arbitrary regex over a surface: incrementally
                        // computable per token, but not a rule the renderer has.
                        ReplacePattern::Regex(_) => return Err("Replace"),
                    };
                    if fused || bytes_resolved {
                        return Err("Replace");
                    }
                    replaces.push((from.clone(), to.clone()));
                }
                DecodeOp::Strip {
                    content,
                    start,
                    stop,
                } => {
                    // Per-token strips are not a rendering rule; a strip over
                    // fused text is a position, which the cursor already tracks.
                    if !fused || *stop > 0 {
                        return Err("Strip");
                    }
                    match *start {
                        0 => {}
                        1 if *content == ' ' => post.push(DecodePost::StripLeadingSpace),
                        _ => return Err("Strip"),
                    }
                }
                // Nothing to lower: concatenation is what the cursor does with
                // every token anyway. It only matters to the ops that follow.
                DecodeOp::Fuse => fused = true,
                DecodeOp::ByteFallback => {
                    if fused {
                        return Err("ByteFallback");
                    }
                    byte_fallback = ByteFallbackRule::DeclaredRun;
                    bytes_resolved = true;
                }
                DecodeOp::Metaspace {
                    replacement,
                    add_prefix_space,
                } => {
                    if fused || bytes_resolved {
                        return Err("Metaspace");
                    }
                    replaces.push((replacement.to_string(), " ".to_string()));
                    // HF strips the single leading space of the *first* token;
                    // the cursor strips it off the first text it emits, which is
                    // that same space whenever the first token renders anything.
                    if *add_prefix_space {
                        post.push(DecodePost::StripLeadingSpace);
                    }
                }
                DecodeOp::WordPiece { prefix, cleanup } => {
                    if fused || bytes_resolved || separator.is_some() {
                        return Err("WordPiece");
                    }
                    // An empty prefix is stripped from *every* surface, so no
                    // token ever carries a separator — which is
                    // `WordSeparator::None`, not `EveryToken`. (The GGUF
                    // WordPiece backend reads an empty prefix the opposite way,
                    // because its vocabulary had real `##` markers removed; a
                    // declared empty prefix says there were never any.)
                    separator = Some(if prefix.is_empty() {
                        WordSeparator::None
                    } else {
                        WordSeparator::Continuation(prefix.clone())
                    });
                    unit_cleanup = *cleanup;
                }
                // Branches on the last token, which a stream cannot know.
                DecodeOp::Bpe { .. } => return Err("BPEDecoder"),
            }
        }

        let mut rules = RenderRules::declared(byte_fallback, use_byte_level);
        for (from, to) in replaces {
            rules = rules.with_surface_replace(from, to);
        }
        if let Some(separator) = separator {
            rules = rules.with_word_separator(separator);
        }
        if unit_cleanup {
            rules = rules.with_unit_cleanup();
        }
        Ok((rules, post))
    }
}

impl DecodeOp {
    fn apply(&self, tokens: Vec<String>) -> Vec<String> {
        match self {
            DecodeOp::ByteLevel => {
                let mut bytes = Vec::new();
                for t in &tokens {
                    match byte_level_decode(t) {
                        Some(b) => bytes.extend_from_slice(&b),
                        None => bytes.extend_from_slice(t.as_bytes()),
                    }
                }
                vec![String::from_utf8_lossy(&bytes).into_owned()]
            }
            DecodeOp::Replace { pattern, to } => tokens
                .into_iter()
                .map(|t| match pattern {
                    ReplacePattern::Str(from) => t.replace(from.as_str(), to),
                    ReplacePattern::Regex(re) => re.replace_all(&t, to).into_owned(),
                })
                .collect(),
            DecodeOp::Strip {
                content,
                start,
                stop,
            } => tokens
                .into_iter()
                .map(|t| strip_token(&t, *content, *start, *stop))
                .collect(),
            DecodeOp::Fuse => vec![tokens.concat()],
            DecodeOp::ByteFallback => byte_fallback(tokens),
            DecodeOp::Metaspace {
                replacement,
                add_prefix_space,
            } => tokens
                .into_iter()
                .enumerate()
                .map(|(i, t)| {
                    let replaced: String = t
                        .chars()
                        .map(|c| if c == *replacement { ' ' } else { c })
                        .collect();
                    if i == 0 && *add_prefix_space {
                        replaced
                            .strip_prefix(' ')
                            .map(str::to_string)
                            .unwrap_or(replaced)
                    } else {
                        replaced
                    }
                })
                .collect(),
            DecodeOp::WordPiece { prefix, cleanup } => tokens
                .into_iter()
                .enumerate()
                .map(|(i, t)| {
                    let mut s = if i != 0 {
                        match t.strip_prefix(prefix.as_str()) {
                            Some(rest) => rest.to_string(),
                            None => format!(" {t}"),
                        }
                    } else {
                        t
                    };
                    if *cleanup {
                        s = wordpiece_cleanup(&s);
                    }
                    s
                })
                .collect(),
            DecodeOp::Bpe { suffix } => {
                let n = tokens.len();
                tokens
                    .into_iter()
                    .enumerate()
                    .map(|(i, t)| t.replace(suffix.as_str(), if i + 1 == n { "" } else { " " }))
                    .collect()
            }
        }
    }
}

/// Strip up to `start` leading and `stop` trailing `content` chars from a token.
fn strip_token(token: &str, content: char, start: usize, stop: usize) -> String {
    let chars: Vec<char> = token.chars().collect();
    let mut lo = 0;
    while lo < start && lo < chars.len() && chars[lo] == content {
        lo += 1;
    }
    let mut hi = chars.len();
    let mut removed = 0;
    while removed < stop && hi > lo && chars[hi - 1] == content {
        hi -= 1;
        removed += 1;
    }
    chars[lo..hi].iter().collect()
}

/// Combine runs of `<0xNN>` byte tokens into bytes, decoding each maximal run as
/// UTF-8 (one replacement char per byte when the run isn't valid UTF-8).
fn byte_fallback(tokens: Vec<String>) -> Vec<String> {
    let mut out = Vec::with_capacity(tokens.len());
    let mut run: Vec<u8> = Vec::new();
    let flush = |run: &mut Vec<u8>, out: &mut Vec<String>| {
        if run.is_empty() {
            return;
        }
        match std::str::from_utf8(run) {
            Ok(s) => out.push(s.to_string()),
            Err(_) => out.extend(run.iter().map(|_| "\u{fffd}".to_string())),
        }
        run.clear();
    };
    for t in tokens {
        match parse_byte_token(&t) {
            Some(b) => run.push(b),
            None => {
                flush(&mut run, &mut out);
                out.push(t);
            }
        }
    }
    flush(&mut run, &mut out);
    out
}

/// Parse a `<0xNN>` byte-fallback token into its byte value.
///
/// Shared with the lowered form's rendering
/// ([`ByteFallbackRule::DeclaredRun`]), so a declared `ByteFallback` step reads
/// exactly the same spellings whichever way it is driven.
pub(crate) fn parse_byte_token(token: &str) -> Option<u8> {
    let hex = token.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() == 2 {
        u8::from_str_radix(hex, 16).ok()
    } else {
        None
    }
}

/// HuggingFace WordPiece decoder cleanup, applied per token. Matches
/// `tokenizers`' `cleanup`: tightens spacing before punctuation and around a few
/// English contractions. Because it runs per token (not on the joined string),
/// the contraction rules rarely fire — e.g. `don ' t` is preserved — which is the
/// observed HF behavior.
///
/// Shared with the lowered form, which applies it to the same unit — one token
/// plus the separator it carries — through
/// [`RenderRules`](crate::core::streaming::RenderRules)' unit cleanup.
pub(crate) fn wordpiece_cleanup(s: &str) -> String {
    s.replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
        .replace(" ' ", "'")
        .replace(" n't", "n't")
        .replace(" 'm", "'m")
        .replace(" 's", "'s")
        .replace(" 've", "'ve")
        .replace(" 're", "'re")
}

/// Parse a `tokenizer.json` `decoder` value into a [`Decoder`]. Returns `None`
/// when there is no decoder (then the backend's built-in decode is used).
pub fn parse(decoder: Option<&Value>) -> Option<Decoder> {
    let decoder = decoder?;
    let mut ops = Vec::new();
    walk(decoder, &mut ops);
    if ops.is_empty() {
        None
    } else {
        Some(Decoder { ops })
    }
}

fn walk(v: &Value, ops: &mut Vec<DecodeOp>) {
    match v.get("type").and_then(Value::as_str) {
        Some("Sequence") => {
            if let Some(list) = v.get("decoders").and_then(Value::as_array) {
                for item in list {
                    walk(item, ops);
                }
            }
        }
        Some("ByteLevel") => ops.push(DecodeOp::ByteLevel),
        Some("Replace") => {
            let to = v
                .get("content")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            if let Some(p) = v.get("pattern") {
                if let Some(s) = p.get("String").and_then(Value::as_str) {
                    ops.push(DecodeOp::Replace {
                        pattern: ReplacePattern::Str(s.to_string()),
                        to,
                    });
                } else if let Some(re) = p.get("Regex").and_then(Value::as_str) {
                    if let Ok(compiled) = regexr::RegexBuilder::new(re).build() {
                        ops.push(DecodeOp::Replace {
                            pattern: ReplacePattern::Regex(Box::new(compiled)),
                            to,
                        });
                    }
                }
            }
        }
        Some("Strip") => {
            let content = v
                .get("content")
                .and_then(Value::as_str)
                .and_then(|s| s.chars().next())
                .unwrap_or(' ');
            ops.push(DecodeOp::Strip {
                content,
                start: v.get("start").and_then(Value::as_u64).unwrap_or(0) as usize,
                stop: v.get("stop").and_then(Value::as_u64).unwrap_or(0) as usize,
            });
        }
        Some("Fuse") => ops.push(DecodeOp::Fuse),
        Some("ByteFallback") => ops.push(DecodeOp::ByteFallback),
        Some("Metaspace") => {
            let replacement = v
                .get("replacement")
                .and_then(Value::as_str)
                .and_then(|s| s.chars().next())
                .unwrap_or('▁');
            // Newer configs use `prepend_scheme`; older ones `add_prefix_space`.
            let add_prefix_space = match v.get("prepend_scheme").and_then(Value::as_str) {
                Some(scheme) => scheme != "never",
                None => v
                    .get("add_prefix_space")
                    .and_then(Value::as_bool)
                    .unwrap_or(true),
            };
            ops.push(DecodeOp::Metaspace {
                replacement,
                add_prefix_space,
            });
        }
        Some("WordPiece") => ops.push(DecodeOp::WordPiece {
            prefix: v
                .get("prefix")
                .and_then(Value::as_str)
                .unwrap_or("##")
                .to_string(),
            cleanup: v.get("cleanup").and_then(Value::as_bool).unwrap_or(true),
        }),
        Some("BPEDecoder") => ops.push(DecodeOp::Bpe {
            suffix: v
                .get("suffix")
                .and_then(Value::as_str)
                .unwrap_or("</w>")
                .to_string(),
        }),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::streaming::{DecodeState, Surfaces};
    use rustc_hash::{FxHashMap, FxHashSet};
    use std::convert::Infallible;
    use std::sync::Arc;

    fn dec(json: serde_json::Value, tokens: &[&str]) -> String {
        let d = parse(Some(&json)).unwrap();
        d.decode(tokens.iter().map(|s| s.to_string()).collect())
    }

    /// Drive the *lowered* pipeline over the same surfaces [`dec`] decodes, in
    /// chunks of `chunk` ids, concatenating every emission plus the final flush.
    ///
    /// The surfaces are handed over as the vocabulary itself — id `i` is
    /// `tokens[i]` — which is what a `Surfaces::ByIndex` backend is, so the
    /// token list the declared pipeline takes and the id list a cursor takes are
    /// the same list.
    fn lowered(json: &serde_json::Value, tokens: &[&str], chunk: usize) -> String {
        let decoder = parse(Some(json)).unwrap();
        let (rules, post) = decoder.lower().expect("the pipeline lowers");
        let surfaces: Vec<String> = tokens.iter().map(|s| s.to_string()).collect();
        let state = DecodeState::new(
            rules.with_vocabulary(
                Surfaces::ByIndex(Arc::new(surfaces)),
                Arc::new(FxHashMap::default()),
                Arc::new(FxHashSet::default()),
            ),
            post,
        );

        let ids: Vec<u32> = (0..tokens.len() as u32).collect();
        let mut cursor = state.cursor_with_capacity(tokens.len() * 4);
        let mut out = String::new();
        for group in ids.chunks(chunk.max(1)) {
            let emitted = match cursor.feed(group, |_| Ok::<(), Infallible>(())) {
                Ok(text) => text,
                // `Infallible` has no values, so this match has no arms to write.
                Err(never) => match never {},
            };
            out.push_str(&emitted.unwrap_or_default());
        }
        out.push_str(&cursor.flush());
        out
    }

    /// The unit's proof obligation: a lowered pipeline is `Decoder::decode`, at
    /// every chunking of the token list. Returns what both produced, so a caller
    /// can also pin the exact text.
    fn same(json: serde_json::Value, tokens: &[&str]) -> String {
        let expected = dec(json.clone(), tokens);
        for chunk in 1..=tokens.len().max(1) {
            assert_eq!(
                lowered(&json, tokens, chunk),
                expected,
                "lowered drive in chunks of {chunk} over {tokens:?}"
            );
        }
        expected
    }

    #[test]
    fn byte_level_reassembles_cross_token_utf8_and_keeps_literals() {
        // "café" byte-level: 'c','a','f','Ã','©' — the two bytes of 'é' split
        // across tokens must reassemble. A literal-content token is kept as-is.
        let j = serde_json::json!({"type": "ByteLevel"});
        assert_eq!(dec(j.clone(), &["c", "a", "f", "Ã", "©"]), "café");
        assert_eq!(dec(j, &["a", "    ", "b"]), "a    b");
    }

    #[test]
    fn metaspace_strips_leading_only_when_prefixing() {
        let always = serde_json::json!({"type": "Metaspace", "prepend_scheme": "always"});
        assert_eq!(dec(always, &["▁Hello", "▁world"]), "Hello world");
        let never = serde_json::json!({"type": "Metaspace", "prepend_scheme": "never"});
        assert_eq!(dec(never, &["▁Hello", "▁world"]), " Hello world");
    }

    #[test]
    fn wordpiece_prefix_and_cleanup() {
        let j = serde_json::json!({"type": "WordPiece", "prefix": "##", "cleanup": true});
        assert_eq!(dec(j.clone(), &["hello", "##world"]), "helloworld");
        assert_eq!(dec(j.clone(), &["hello", ",", "world"]), "hello, world");
        // Contractions are preserved (per-token cleanup never sees " ' ").
        assert_eq!(dec(j, &["don", "'", "t"]), "don ' t");
    }

    #[test]
    fn metaspace_byte_fallback_fuse_sequence() {
        // Llama-style SP decoder: Replace ▁→space, ByteFallback, Fuse, Strip 1 left.
        let j = serde_json::json!({"type": "Sequence", "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"},
            {"type": "Fuse"},
            {"type": "Strip", "content": " ", "start": 1, "stop": 0},
        ]});
        // "▁Hi" + bytes of '€' (E2 82 AC) → " Hi€" → strip leading space → "Hi€".
        assert_eq!(dec(j, &["▁Hi", "<0xE2>", "<0x82>", "<0xAC>"]), "Hi€");
    }

    #[test]
    fn bpe_word_suffix_decoder() {
        let j = serde_json::json!({"type": "BPEDecoder", "suffix": "</w>"});
        assert_eq!(dec(j, &["hello</w>", "world</w>"]), "hello world");
    }

    /// The declared `ByteFallback` emits one U+FFFD **per byte** of an invalid
    /// run, which is not std's maximal-subpart rule. Measured against
    /// `tokenizers` 0.22.1 on `mistral-7b-v0.3/tokenizer.json`; pinned here for
    /// both the whole-sequence chain and its lowered form.
    #[test]
    fn declared_byte_fallback_is_one_replacement_char_per_byte() {
        let j = mistral_chain();
        // std agrees on these two.
        assert_eq!(same(j.clone(), &["<0x80>"]), "\u{fffd}");
        assert_eq!(same(j.clone(), &["<0x80>", "<0x80>"]), "\u{fffd}\u{fffd}");
        // ...and not on these two: std would give "\u{FFFD}A" (0x41 is 'A') and
        // a single "\u{FFFD}" respectively.
        assert_eq!(same(j.clone(), &["<0xE2>", "<0x41>"]), "\u{fffd}\u{fffd}");
        assert_eq!(same(j, &["<0xF0>", "<0x9F>"]), "\u{fffd}\u{fffd}");
    }

    /// The Llama/Mistral-style SentencePiece chain, which four of the shipping
    /// `tokenizer.json` files declare.
    fn mistral_chain() -> serde_json::Value {
        serde_json::json!({"type": "Sequence", "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"},
            {"type": "Fuse"},
            {"type": "Strip", "content": " ", "start": 1, "stop": 0},
        ]})
    }

    /// Every shipping decoder shape, lowered and driven at every chunking, must
    /// come out as [`Decoder::decode`] does.
    #[test]
    fn every_shipping_pipeline_lowers_to_the_same_text() {
        let byte_level = serde_json::json!({"type": "ByteLevel"});
        assert_eq!(same(byte_level.clone(), &["c", "a", "f", "Ã", "©"]), "café");
        assert_eq!(same(byte_level, &["a", "    ", "b"]), "a    b");

        let always = serde_json::json!({"type": "Metaspace", "prepend_scheme": "always"});
        assert_eq!(same(always, &["▁Hello", "▁world"]), "Hello world");
        let never = serde_json::json!({"type": "Metaspace", "prepend_scheme": "never"});
        assert_eq!(same(never, &["▁Hello", "▁world"]), " Hello world");

        let wordpiece = serde_json::json!({"type": "WordPiece", "prefix": "##", "cleanup": true});
        assert_eq!(same(wordpiece.clone(), &["hello", "##world"]), "helloworld");
        assert_eq!(
            same(wordpiece.clone(), &["hello", ",", "world"]),
            "hello, world"
        );
        // The cleanup is per token, so a bare apostrophe keeps its spaces while
        // a whole contraction token loses its separator.
        assert_eq!(same(wordpiece.clone(), &["don", "'", "t"]), "don ' t");
        assert_eq!(same(wordpiece, &["it", "'s", "fine"]), "it's fine");

        let plain = serde_json::json!({"type": "WordPiece", "prefix": "##", "cleanup": false});
        assert_eq!(same(plain, &["hello", ",", "world"]), "hello , world");

        // An empty declared prefix comes off every surface, so nothing carries a
        // separator at all.
        let unmarked = serde_json::json!({"type": "WordPiece", "prefix": "", "cleanup": false});
        assert_eq!(same(unmarked, &["hello", "world"]), "helloworld");

        let mistral = mistral_chain();
        assert_eq!(
            same(mistral.clone(), &["▁Hi", "<0xE2>", "<0x82>", "<0xAC>"]),
            "Hi€"
        );
        // A run that ends on a surface token, with the dummy prefix stripped
        // from the very first emitted text and nowhere else.
        assert_eq!(
            same(mistral.clone(), &["▁a", "<0x80>", "▁b"]),
            "a\u{fffd} b"
        );
        // The strip is spent by the first token even when that token renders
        // nothing but the space it takes.
        assert_eq!(same(mistral, &["▁", "▁a"]), " a");
    }

    /// The three declared shapes that are not incrementally computable are
    /// refused outright rather than approximated.
    #[test]
    fn pipelines_that_cannot_stream_do_not_lower() {
        let lowers = |json: serde_json::Value| parse(Some(&json)).unwrap().lower().is_some();

        // Branches on the last token.
        assert!(!lowers(
            serde_json::json!({"type": "BPEDecoder", "suffix": "</w>"})
        ));
        // A trailing strip over the fused text: the last token again.
        assert!(!lowers(
            serde_json::json!({"type": "Sequence", "decoders": [
                {"type": "ByteFallback"},
                {"type": "Fuse"},
                {"type": "Strip", "content": " ", "start": 0, "stop": 1},
            ]})
        ));
        // An arbitrary regex over the fused text, which can match across a
        // chunk boundary.
        assert!(!lowers(
            serde_json::json!({"type": "Sequence", "decoders": [
                {"type": "ByteFallback"},
                {"type": "Fuse"},
                {"type": "Replace", "pattern": {"Regex": " +"}, "content": " "},
            ]})
        ));

        // ...and the shipping chain, whose leading strip *is* a position, still
        // lowers.
        assert!(lowers(mistral_chain()));
    }
}
