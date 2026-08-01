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

/// One added token: the id it encodes to, plus HuggingFace's `lstrip`/`rstrip`
/// flags, which decide how much of the surrounding whitespace the token eats.
///
/// The flags are declared per token in a `tokenizer.json`'s `added_tokens`
/// array, and they genuinely differ *within* one vocabulary — bge-m3 declares
/// `<mask>` with `lstrip: true` while its four other added tokens (`<s>`,
/// `<pad>`, `</s>`, `<unk>`) leave both flags off — so they can never be a
/// per-tokenizer setting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AddedToken {
    /// The token id this content encodes to.
    pub id: u32,
    /// Absorb the whitespace immediately *preceding* a match into the token.
    ///
    /// Measured with `tokenizers` 0.22.1 against bge-m3's `tokenizer.json`
    /// (`add_special_tokens=False`): `"end. <mask>x"` is
    /// `[3564, 5, 250001, 1022]`. Without this flag the space before `<mask>`
    /// survives as the lone `▁` piece (id 6) and the sequence gains a token the
    /// model never saw there.
    pub lstrip: bool,
    /// Absorb the whitespace immediately *following* a match into the token —
    /// the mirror of [`lstrip`](Self::lstrip).
    pub rstrip: bool,
}

impl AddedToken {
    /// An added token with both strip flags off.
    ///
    /// This is the shape of every added token that does not come from a
    /// `tokenizer.json`: GGUF vocabularies and the bundled tiktoken-style
    /// vocabularies have no place to declare the flags, so "no flags" is their
    /// only correct reading — unlike the `tokenizer.json` loader, which must
    /// read whatever the file says.
    pub const fn plain(id: u32) -> Self {
        Self {
            id,
            lstrip: false,
            rstrip: false,
        }
    }
}

/// An owned content → [`AddedToken`] set: what a backend is handed to build its
/// matcher from.
///
/// A bare `FxHashMap<String, u32>` cannot express the strip flags, and every
/// caller that *has* none (GGUF, the bundled vocabularies, tests) still gets to
/// pass one — the `From` impls below turn it into a set of
/// [`plain`](AddedToken::plain) tokens — so only the `tokenizer.json` loader
/// pays the cost of spelling the flags out.
#[derive(Clone, Debug, Default)]
pub struct AddedTokenSet {
    tokens: FxHashMap<String, AddedToken>,
}

impl AddedTokenSet {
    /// An empty set: no added token is recognized in the input.
    pub fn new() -> Self {
        Self::default()
    }

    /// Declare `content` as an added token with explicit flags.
    pub fn insert(&mut self, content: impl Into<String>, token: AddedToken) {
        self.tokens.insert(content.into(), token);
    }

    /// Declare `content` as an added token with both strip flags off.
    pub fn insert_plain(&mut self, content: impl Into<String>, id: u32) {
        self.insert(content, AddedToken::plain(id));
    }

    /// The token declared for `content`, if any.
    pub fn get(&self, content: &str) -> Option<AddedToken> {
        self.tokens.get(content).copied()
    }

    /// Number of declared added tokens.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Whether nothing is declared — matching is legitimately off.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Iterate the declared tokens as `(content, token)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, AddedToken)> + '_ {
        self.tokens.iter().map(|(k, v)| (k.as_str(), *v))
    }

    /// Consume the set into the plain name→id map the rest of the crate speaks
    /// (decode tables, `SpecialPolicy`'s named lookups). Consuming rather than
    /// borrowing moves the token strings instead of cloning every one of them.
    pub fn into_id_map(self) -> FxHashMap<String, u32> {
        self.tokens.into_iter().map(|(k, v)| (k, v.id)).collect()
    }
}

impl From<FxHashMap<String, u32>> for AddedTokenSet {
    fn from(map: FxHashMap<String, u32>) -> Self {
        Self {
            tokens: map
                .into_iter()
                .map(|(k, id)| (k, AddedToken::plain(id)))
                .collect(),
        }
    }
}

impl From<&FxHashMap<String, u32>> for AddedTokenSet {
    fn from(map: &FxHashMap<String, u32>) -> Self {
        Self {
            tokens: map
                .iter()
                .map(|(k, id)| (k.clone(), AddedToken::plain(*id)))
                .collect(),
        }
    }
}

impl FromIterator<(String, AddedToken)> for AddedTokenSet {
    fn from_iter<T: IntoIterator<Item = (String, AddedToken)>>(iter: T) -> Self {
        Self {
            tokens: iter.into_iter().collect(),
        }
    }
}

/// An Aho-Corasick matcher over a set of added-token strings → [`AddedToken`]s.
#[derive(Clone)]
pub struct AddedTokens {
    matcher: AhoCorasick,
    tokens: Vec<AddedToken>,
}

impl AddedTokens {
    /// Build a matcher from a declared added-token set.
    ///
    /// Returns `Ok(None)` only when `set` is empty — there is nothing to match,
    /// so added-token matching is legitimately off. Returns `Err` if
    /// Aho-Corasick fails to build the automaton from a *non-empty* set. These
    /// two cases must stay distinguishable: collapsing a build failure into
    /// `None` (as a prior version did via `.ok()`) would silently disable
    /// added-token matching. Every special/control token in subsequent input
    /// would then fall through to ordinary encoding — ids stay in range, text
    /// still round-trips, and nothing surfaces the loss. An allow-list
    /// enforcement mode relies on this same matcher, so a silently-absent
    /// matcher would also silently skip that check.
    pub fn new(set: &AddedTokenSet) -> Result<Option<Self>, aho_corasick::BuildError> {
        if set.is_empty() {
            return Ok(None);
        }
        let entries: Vec<(&str, AddedToken)> = set.iter().collect();
        let patterns: Vec<&str> = entries.iter().map(|(k, _)| *k).collect();
        let tokens: Vec<AddedToken> = entries.iter().map(|(_, t)| *t).collect();
        // Leftmost-longest so a longer added token (e.g. a 24-space run) wins over
        // a shorter one (a 2-space run) starting at the same position, matching
        // HuggingFace. Default (Standard) reports the earliest-ending match,
        // which would split the run into several short tokens.
        let matcher = AhoCorasickBuilder::new()
            .match_kind(MatchKind::LeftmostLongest)
            .build(&patterns)?;
        Ok(Some(Self { matcher, tokens }))
    }

    /// The id of the added token occupying byte 0 of `text`, if any.
    ///
    /// The SentencePiece backends need this because their dummy prefix belongs
    /// to the whole input and is applied *before* the split: when an added token
    /// starts the input the prefix has nothing to attach to, and whether it then
    /// surfaces as a standalone piece depends on *which* token that is. Asking
    /// the same matcher that performs the split keeps the two answers from
    /// disagreeing about where the first boundary falls.
    ///
    /// Strictly positional: a token whose [`lstrip`](AddedToken::lstrip) would
    /// absorb leading whitespace does *not* count as occupying byte 0. The
    /// reference cases this answer feeds are stated positionally (`" <s>x"` ->
    /// `▁▁`, `<s>`, `x` — the marker is not standalone because whitespace
    /// precedes the sentinel), and no loader that builds an SPM backend (GGUF,
    /// the bundled SentencePiece vocabularies) can declare strip flags at all,
    /// so the two readings never disagree on a reachable configuration.
    pub fn id_at_start(&self, text: &str) -> Option<u32> {
        self.matcher
            .find(text)
            .filter(|m| m.start() == 0)
            .map(|m| self.tokens[m.pattern().as_usize()].id)
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
    ///
    /// This is also where [`lstrip`](AddedToken::lstrip) / [`rstrip`](AddedToken::rstrip)
    /// are applied — on the *gap*, never on the matched token itself, so that a
    /// flagged token eats the neighbouring whitespace instead of leaving it to
    /// be encoded as a piece of its own. Both are bounded by the surrounding
    /// matches: `lstrip` trims only back to the end of the previous match, so
    /// when the previous token already claimed that whitespace with `rstrip`
    /// there is nothing left to take. Measured with `tokenizers` 0.22.1 on a
    /// vocabulary declaring `[R]` (rstrip) and `[L]` (lstrip): `"a [R] [L] b"`
    /// gives spans `[R] ` then `[L]` — the earlier match wins the one space.
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
            // A previous token's `rstrip` can reach past this match: the matcher
            // runs over the whole text, and whitespace itself can be an added
            // token (gpt-neox declares whole space runs). Such a match no longer
            // exists — its text was absorbed — so it is dropped rather than
            // emitted a second time, and `last` never walks backwards over text
            // that was already encoded.
            //
            // This is a deliberate divergence, on the one configuration where
            // the reference contradicts itself: with `[R]` (rstrip) and `"  "`
            // both added, `tokenizers` 0.22.1 encodes `"[R]  x"` as `[R]  `
            // spanning bytes 0..5 *and* `"  "` spanning 3..5 — the same two
            // spaces counted twice, with overlapping offsets. No real vocabulary
            // declares an rstrip token alongside a whitespace token; emitting the
            // text once is the coherent reading, and it is what keeps this loop
            // from slicing a reversed range.
            if m.end() <= last {
                continue;
            }
            let token = self.tokens[m.pattern().as_usize()];
            // Clamped for the partial-overlap case (the strip ate the match's
            // first bytes but not all of them): the gap is then empty, never a
            // reversed range that would panic on slicing.
            let match_start = m.start().max(last);
            // `trim_end`/`trim_start` cut exactly the chars `char::is_whitespace`
            // accepts (Unicode `White_Space`), which is what HuggingFace strips —
            // measured with `tokenizers` 0.22.1 over a flagged token surrounded by
            // one candidate char at a time: U+000B, U+0085, U+00A0, U+1680,
            // U+2000, U+2028, U+2029, U+202F, U+205F and U+3000 are all absorbed,
            // while U+001C..U+001F (whitespace to Python's `str.isspace`), U+180E,
            // U+200B and U+FEFF are not. That set is `White_Space` exactly —
            // neither ASCII-only nor Python's notion of whitespace.
            let gap_end = if token.lstrip {
                last + text[last..match_start].trim_end().len()
            } else {
                match_start
            };
            // Still guarded by `>`: a gap that strips away entirely is never
            // handed to the gap encoder, preserving this loop's standing promise
            // that it never emits an empty gap (the SPM backend spends its
            // single dummy prefix on the first gap it is asked to encode, so an
            // empty one would spend it on nothing).
            if gap_end > last {
                out.extend(encode_gap(&text[last..gap_end]));
            }
            admit(&text[m.start()..m.end()], m.start())?;
            out.push(token.id);
            last = m.end();
            if token.rstrip {
                let tail = &text[last..];
                last += tail.len() - tail.trim_start().len();
            }
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

    /// A set of plain (unflagged) tokens, the shape every non-json loader builds.
    fn plain_set(entries: &[(&str, u32)]) -> AddedTokenSet {
        let mut set = AddedTokenSet::new();
        for (content, id) in entries {
            set.insert_plain(*content, *id);
        }
        set
    }

    /// Encode gaps as their raw bytes so any leftover text is visible in the ids.
    fn bytes(gap: &str) -> Vec<u32> {
        gap.bytes().map(u32::from).collect()
    }

    #[test]
    fn prefers_longest_overlapping_added_token() {
        // Tokens for 2- and 4-space runs (gpt-neox style). A 4-space input must
        // match the single 4-space token, not the 2-space token twice.
        let at = AddedTokens::new(&plain_set(&[("  ", 10), ("    ", 20)]))
            .unwrap()
            .unwrap();
        // gap encoder marks any leftover text so we'd notice a bad split.
        let ids = at.encode_with("a    b", bytes);
        assert_eq!(ids, vec![u32::from(b'a'), 20, u32::from(b'b')]);
    }

    #[test]
    fn empty_map_yields_no_matcher() {
        // An empty map is a legitimate "matching is off" state, distinct from a
        // build failure — both must not be conflated into the same `None`.
        assert!(AddedTokens::new(&AddedTokenSet::new()).unwrap().is_none());
    }

    #[test]
    fn plain_id_map_round_trips_without_flags() {
        // The ergonomic path every non-json loader takes: a bare name→id map
        // must arrive as tokens with both flags off, and come back out as the
        // same map for the decode tables.
        let mut map = FxHashMap::default();
        map.insert("<pad>".to_string(), 1);
        let set = AddedTokenSet::from(&map);
        assert_eq!(set.get("<pad>"), Some(AddedToken::plain(1)));
        assert_eq!(AddedTokenSet::from(map.clone()).into_id_map(), map);
    }

    fn special_map() -> AddedTokenSet {
        plain_set(&[("<|im_start|>", 100), ("<|im_end|>", 101)])
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

    /// A set holding one flagged token plus one plain neighbour, so every strip
    /// test also proves the flags are per token rather than per matcher.
    fn strip_set(lstrip: bool, rstrip: bool) -> AddedTokenSet {
        let mut set = AddedTokenSet::new();
        set.insert(
            "<mask>",
            AddedToken {
                id: 250_001,
                lstrip,
                rstrip,
            },
        );
        set.insert_plain("<pad>", 1);
        set
    }

    /// Record what the gap encoder is actually handed, so an empty or
    /// unstripped gap is visible rather than inferred from the ids.
    fn record(calls: &mut Vec<String>) -> impl FnMut(&str) -> Vec<u32> + '_ {
        move |gap: &str| {
            calls.push(gap.to_string());
            bytes(gap)
        }
    }

    #[test]
    fn lstrip_absorbs_the_preceding_whitespace() {
        // Reference (`tokenizers` 0.22.1, bge-m3, add_special_tokens=False):
        // "end. <mask>x" -> [3564, 5, 250001, 1022] — the space before <mask>
        // never becomes a piece of its own.
        let at = AddedTokens::new(&strip_set(true, false)).unwrap().unwrap();
        let mut calls = Vec::new();
        let ids = at.encode_with("end. <mask>x", record(&mut calls));
        assert_eq!(calls, vec!["end.".to_string(), "x".to_string()]);
        let mut expect = bytes("end.");
        expect.push(250_001);
        expect.extend(bytes("x"));
        assert_eq!(ids, expect);
    }

    #[test]
    fn rstrip_absorbs_the_following_whitespace() {
        let at = AddedTokens::new(&strip_set(false, true)).unwrap().unwrap();
        let mut calls = Vec::new();
        let ids = at.encode_with("a <mask> b", record(&mut calls));
        assert_eq!(calls, vec!["a ".to_string(), "b".to_string()]);
        let mut expect = bytes("a ");
        expect.push(250_001);
        expect.extend(bytes("b"));
        assert_eq!(ids, expect);
    }

    #[test]
    fn both_flags_absorb_whitespace_on_both_sides() {
        let at = AddedTokens::new(&strip_set(true, true)).unwrap().unwrap();
        let mut calls = Vec::new();
        // A whole run goes, not just one char, and non-ASCII Unicode
        // `White_Space` counts: U+3000 is absorbed by the reference too.
        let ids = at.encode_with("a  \t<mask>\u{3000} b", record(&mut calls));
        assert_eq!(calls, vec!["a".to_string(), "b".to_string()]);
        let mut expect = bytes("a");
        expect.push(250_001);
        expect.extend(bytes("b"));
        assert_eq!(ids, expect);
    }

    #[test]
    fn flags_off_leave_both_gaps_untouched() {
        // The pre-flag behaviour, which the four unflagged bge-m3 tokens (<s>,
        // <pad>, </s>, <unk>) already matched and must keep matching byte for byte.
        let at = AddedTokens::new(&strip_set(false, false)).unwrap().unwrap();
        let mut calls = Vec::new();
        let ids = at.encode_with("a <mask> b", record(&mut calls));
        assert_eq!(calls, vec!["a ".to_string(), " b".to_string()]);
        let mut expect = bytes("a ");
        expect.push(250_001);
        expect.extend(bytes(" b"));
        assert_eq!(ids, expect);
    }

    #[test]
    fn a_gap_that_strips_to_empty_is_never_handed_to_the_gap_encoder() {
        // The SPM backend spends its single dummy prefix on the first gap it is
        // asked to encode, so handing it "" would spend it on nothing.
        let at = AddedTokens::new(&strip_set(true, true)).unwrap().unwrap();
        let mut calls = Vec::new();
        let ids = at.encode_with("   <mask>   ", record(&mut calls));
        assert!(calls.is_empty(), "gap encoder was called with {calls:?}");
        assert_eq!(ids, vec![250_001]);
    }

    #[test]
    fn rstrip_reaching_over_a_whitespace_token_encodes_the_text_once() {
        // The one case where a strip can swallow a *match*: an rstrip token
        // followed by a whitespace added token. The absorbed match is dropped —
        // never emitted a second time, and never left to slice a reversed range
        // (see `encode_matched`, which also records where the reference differs).
        let mut set = AddedTokenSet::new();
        set.insert(
            "[R]",
            AddedToken {
                id: 5,
                lstrip: false,
                rstrip: true,
            },
        );
        set.insert_plain("  ", 6);
        let at = AddedTokens::new(&set).unwrap().unwrap();
        let mut calls = Vec::new();
        let ids = at.encode_with("[R]  x", record(&mut calls));
        assert_eq!(calls, vec!["x".to_string()]);
        let mut expect = vec![5];
        expect.extend(bytes("x"));
        assert_eq!(ids, expect);
    }

    #[test]
    fn only_the_flagged_one_of_two_adjacent_added_tokens_strips() {
        // <pad> is plain, <mask> is lstrip: the space after <pad> stays, the one
        // before <mask> goes. Reference (bge-m3): "café<pad>a <mask>a" ->
        // [26216, 1, 10, 250001, 10], with no lone `▁` piece before <mask>.
        let at = AddedTokens::new(&strip_set(true, false)).unwrap().unwrap();
        let mut calls = Vec::new();
        let ids = at.encode_with("x<pad> a <mask>a", record(&mut calls));
        assert_eq!(
            calls,
            vec!["x".to_string(), " a".to_string(), "a".to_string()]
        );
        let mut expect = bytes("x");
        expect.push(1);
        expect.extend(bytes(" a"));
        expect.push(250_001);
        expect.extend(bytes("a"));
        assert_eq!(ids, expect);
    }
}
