//! Added-token matching shared by all four backends (BPE, SentencePiece, SPM,
//! WordPiece).
//!
//! HuggingFace always recognizes `added_tokens` in the input during encoding
//! (e.g. `[CLS]`, `</s>`, chat markers) — even with `add_special_tokens=False`.
//! This module provides one matcher implementation so every backend gets the
//! same behavior instead of each reimplementing its own.

use std::convert::Infallible;

use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use rustc_hash::FxHashMap;

use super::policy::{PolicyError, SpecialMode};

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
    /// `encode_gap`. Equivalent to [`encode_with_mode`](Self::encode_with_mode)
    /// under [`SpecialMode::All`], which admits every match and therefore cannot
    /// fail — expressed here by instantiating the shared loop's error type as
    /// [`Infallible`], so the compiler proves the `Err` arm away rather than a
    /// runtime assertion claiming it.
    pub fn encode_with<F>(&self, text: &str, encode_gap: F) -> Vec<u32>
    where
        F: FnMut(&str) -> Vec<u32>,
    {
        match self.encode_matched(text, encode_gap, |_, _| Ok::<(), Infallible>(())) {
            Ok(ids) => ids,
            // `Infallible` has no values, so this match has no arms to write.
            Err(never) => match never {},
        }
    }

    /// Split `text` on added tokens per `mode`, emitting their ids (or
    /// refusing per [`SpecialMode::Allow`]) and encoding the gaps via
    /// `encode_gap`.
    ///
    /// [`SpecialMode::Ordinary`] never consults the matcher at all — the whole
    /// text goes straight to `encode_gap` — rather than matching and then
    /// discarding the match, which would double the matching work and could
    /// disagree with the matcher used elsewhere about where a boundary falls.
    pub fn encode_with_mode<F>(
        &self,
        text: &str,
        mode: &SpecialMode<'_>,
        mut encode_gap: F,
    ) -> Result<Vec<u32>, PolicyError>
    where
        F: FnMut(&str) -> Vec<u32>,
    {
        match mode {
            SpecialMode::Ordinary => Ok(encode_gap(text)),
            SpecialMode::All => Ok(self.encode_with(text, encode_gap)),
            SpecialMode::Allow(allowed) => {
                self.encode_matched(text, encode_gap, |matched, offset| {
                    if allowed.contains(matched) {
                        Ok(())
                    } else {
                        Err(PolicyError::DisallowedSpecial {
                            token: matched.to_owned(),
                            offset,
                        })
                    }
                })
            }
        }
    }

    /// The one shared match/gap loop. `admit` is consulted for every matched
    /// added token, with the matched text and its byte offset in `text`, and
    /// short-circuits the whole encode by returning `Err`.
    ///
    /// Taking the per-match decision as a closure rather than a `SpecialMode`
    /// keeps [`SpecialMode::Ordinary`] — which must never reach this loop —
    /// unrepresentable here, instead of a runtime arm asserting it cannot happen.
    fn encode_matched<F, A, E>(
        &self,
        text: &str,
        mut encode_gap: F,
        mut admit: A,
    ) -> Result<Vec<u32>, E>
    where
        F: FnMut(&str) -> Vec<u32>,
        A: FnMut(&str, usize) -> Result<(), E>,
    {
        let mut out = Vec::new();
        let mut last = 0;
        for m in self.matcher.find_iter(text) {
            if m.start() > last {
                out.extend(encode_gap(&text[last..m.start()]));
            }
            admit(&text[m.start()..m.end()], m.start())?;
            out.push(self.ids[m.pattern().as_usize()]);
            last = m.end();
        }
        if last < text.len() {
            out.extend(encode_gap(&text[last..]));
        }
        Ok(out)
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

    /// Mode-aware form of [`dispatch`](Self::dispatch), shared by every
    /// backend's `Tokenize::encode_with`.
    pub fn dispatch_with_mode<F>(
        added: &Option<Self>,
        text: &str,
        mode: &SpecialMode<'_>,
        mut encode_gap: F,
    ) -> Result<Vec<u32>, PolicyError>
    where
        F: FnMut(&str) -> Vec<u32>,
    {
        match added {
            Some(added) => added.encode_with_mode(text, mode, encode_gap),
            None => Ok(encode_gap(text)),
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

    fn special_map() -> FxHashMap<String, u32> {
        let mut map = FxHashMap::default();
        map.insert("<|im_start|>".to_string(), 100);
        map.insert("<|im_end|>".to_string(), 101);
        map
    }

    #[test]
    fn all_mode_matches_every_configured_special() {
        let at = AddedTokens::new(&special_map()).unwrap().unwrap();
        let ids = at
            .encode_with_mode("<|im_start|>hi<|im_end|>", &SpecialMode::All, |gap| {
                gap.bytes().map(u32::from).collect()
            })
            .unwrap();
        assert_eq!(ids, vec![100, u32::from(b'h'), u32::from(b'i'), 101]);
    }

    #[test]
    fn allow_mode_permits_a_listed_token() {
        let at = AddedTokens::new(&special_map()).unwrap().unwrap();
        let mut allowed = rustc_hash::FxHashSet::default();
        allowed.insert("<|im_start|>".to_string());
        let ids = at
            .encode_with_mode("<|im_start|>hi", &SpecialMode::Allow(&allowed), |gap| {
                gap.bytes().map(u32::from).collect()
            })
            .unwrap();
        assert_eq!(ids, vec![100, u32::from(b'h'), u32::from(b'i')]);
    }

    #[test]
    fn allow_mode_refuses_an_unlisted_token_with_the_right_token_and_offset() {
        let at = AddedTokens::new(&special_map()).unwrap().unwrap();
        // Empty allow-list: nothing is permitted, so the match at byte offset 2
        // (after "hi") must be refused, carrying both the exact token text and
        // its byte offset — not just any error.
        let allowed = rustc_hash::FxHashSet::default();
        let err = at
            .encode_with_mode("hi<|im_end|>", &SpecialMode::Allow(&allowed), |gap| {
                gap.bytes().map(u32::from).collect()
            })
            .unwrap_err();
        match err {
            PolicyError::DisallowedSpecial { token, offset } => {
                assert_eq!(token, "<|im_end|>");
                assert_eq!(offset, 2);
            }
            other => panic!("expected DisallowedSpecial, got {other:?}"),
        }
    }

    #[test]
    fn ordinary_mode_never_promotes_the_literal_text() {
        let at = AddedTokens::new(&special_map()).unwrap().unwrap();
        // The gap encoder must see the *whole* text, including the special
        // token's literal spelling — the matcher must not be consulted at all.
        let mut gap_calls = Vec::new();
        let ids = at
            .encode_with_mode("<|im_start|>hi", &SpecialMode::Ordinary, |gap| {
                gap_calls.push(gap.to_string());
                gap.bytes().map(u32::from).collect()
            })
            .unwrap();
        assert_eq!(gap_calls, vec!["<|im_start|>hi".to_string()]);
        assert_eq!(
            ids,
            "<|im_start|>hi".bytes().map(u32::from).collect::<Vec<_>>()
        );
    }
}
