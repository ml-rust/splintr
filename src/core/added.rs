//! Added-token matching shared by all four backends (BPE, SentencePiece, SPM,
//! WordPiece).
//!
//! HuggingFace always recognizes `added_tokens` in the input during encoding
//! (e.g. `[CLS]`, `</s>`, chat markers) — even with `add_special_tokens=False`.
//! This module provides one matcher implementation so every backend gets the
//! same behavior instead of each reimplementing its own.

use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use rustc_hash::FxHashMap;

/// An Aho-Corasick matcher over a set of added-token strings → ids.
#[derive(Clone)]
pub struct AddedTokens {
    matcher: AhoCorasick,
    ids: Vec<u32>,
}

impl AddedTokens {
    /// Build a matcher from a name→id map.
    ///
    /// Returns `Ok(None)` only when `map` is empty — there is nothing to match,
    /// so added-token matching is legitimately off. Returns `Err` if
    /// Aho-Corasick fails to build the automaton from a *non-empty* map. These
    /// two cases must stay distinguishable: collapsing a build failure into
    /// `None` (as a prior version did via `.ok()`) would silently disable
    /// added-token matching. Every special/control token in subsequent input
    /// would then fall through to ordinary encoding — ids stay in range, text
    /// still round-trips, and nothing surfaces the loss. An allow-list
    /// enforcement mode relies on this same matcher, so a silently-absent
    /// matcher would also silently skip that check.
    pub fn new(map: &FxHashMap<String, u32>) -> Result<Option<Self>, aho_corasick::BuildError> {
        if map.is_empty() {
            return Ok(None);
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
            .build(&patterns)?;
        Ok(Some(Self { matcher, ids }))
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

    /// Shared `Tokenize::encode` dispatch for all backends (BPE, SPM,
    /// Unigram/SentencePiece, WordPiece): recognize added tokens first (HF
    /// behavior), falling back to `encode_gap` when none are configured.
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
        let at = AddedTokens::new(&map).unwrap().unwrap();
        // gap encoder marks any leftover text so we'd notice a bad split.
        let ids = at.encode_with("a    b", |gap| gap.bytes().map(u32::from).collect());
        assert_eq!(ids, vec![u32::from(b'a'), 20, u32::from(b'b')]);
    }

    #[test]
    fn empty_map_yields_no_matcher() {
        // An empty map is a legitimate "matching is off" state, distinct from a
        // build failure — both must not be conflated into the same `None`.
        let map = FxHashMap::default();
        assert!(AddedTokens::new(&map).unwrap().is_none());
    }
}
