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
fn parse_byte_token(token: &str) -> Option<u8> {
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
fn wordpiece_cleanup(s: &str) -> String {
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

    fn dec(json: serde_json::Value, tokens: &[&str]) -> String {
        let d = parse(Some(&json)).unwrap();
        d.decode(tokens.iter().map(|s| s.to_string()).collect())
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
}
