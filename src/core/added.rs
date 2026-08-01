//! Added-token matching shared by the SentencePiece and WordPiece backends.
//!
//! HuggingFace always recognizes `added_tokens` in the input during encoding
//! (e.g. `[CLS]`, `</s>`, chat markers) — even with `add_special_tokens=False`.
//! The BPE backend already does this via its special-token matcher; this gives
//! the other backends the same behavior.

use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use rustc_hash::FxHashMap;

/// An Aho-Corasick matcher over a set of added-token strings → ids.
pub struct AddedTokens {
    matcher: AhoCorasick,
    ids: Vec<u32>,
}

impl AddedTokens {
    /// Build a matcher from a name→id map. Returns `None` when empty.
    pub fn new(map: &FxHashMap<String, u32>) -> Option<Self> {
        if map.is_empty() {
            return None;
        }
        let entries: Vec<(&String, u32)> = map.iter().map(|(k, v)| (k, *v)).collect();
        let patterns: Vec<&str> = entries.iter().map(|(k, _)| k.as_str()).collect();
        let ids: Vec<u32> = entries.iter().map(|(_, v)| *v).collect();
        // Leftmost-longest so a longer added token (e.g. a 24-space run) wins over
        // a shorter one (a 2-space run) starting at the same position, matching
        // HuggingFace. Default (Standard) reports the earliest-ending match,
        // which would split the run into several short tokens.
        let matcher = AhoCorasickBuilder::new()
            .match_kind(MatchKind::LeftmostLongest)
            .build(&patterns)
            .ok()?;
        Some(Self { matcher, ids })
    }

    /// Split `text` on added tokens, emitting their ids and encoding the gaps via
    /// `encode_gap`.
    pub fn encode_with<F>(&self, text: &str, mut encode_gap: F) -> Vec<u32>
    where
        F: FnMut(&str) -> Vec<u32>,
    {
        let mut out = Vec::new();
        let mut last = 0;
        for m in self.matcher.find_iter(text) {
            if m.start() > last {
                out.extend(encode_gap(&text[last..m.start()]));
            }
            out.push(self.ids[m.pattern().as_usize()]);
            last = m.end();
        }
        if last < text.len() {
            out.extend(encode_gap(&text[last..]));
        }
        out
    }

    /// Shared `Tokenize::encode` dispatch for the SPM/Unigram/WordPiece backends:
    /// recognize added tokens first (HF behavior), falling back to `encode_gap`
    /// when none are configured.
    pub fn dispatch<F>(added: &Option<Self>, text: &str, mut encode_gap: F) -> Vec<u32>
    where
        F: FnMut(&str) -> Vec<u32>,
    {
        match added {
            Some(added) => added.encode_with(text, encode_gap),
            None => encode_gap(text),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefers_longest_overlapping_added_token() {
        // Tokens for 2- and 4-space runs (gpt-neox style). A 4-space input must
        // match the single 4-space token, not the 2-space token twice.
        let mut map = FxHashMap::default();
        map.insert("  ".to_string(), 10);
        map.insert("    ".to_string(), 20);
        let at = AddedTokens::new(&map).unwrap();
        // gap encoder marks any leftover text so we'd notice a bad split.
        let ids = at.encode_with("a    b", |gap| gap.bytes().map(u32::from).collect());
        assert_eq!(ids, vec![u32::from(b'a'), 20, u32::from(b'b')]);
    }
}
