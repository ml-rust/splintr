//! SentencePiece **BPE** tokenizer (llama.cpp `SPM` / `tokenizer.ggml.model = "llama"`).
//!
//! This is *not* the Unigram algorithm in [`sentencepiece`](super::sentencepiece).
//! The two share a vocabulary format and a word-boundary marker but disagree on
//! what the per-token score means, and therefore on how to segment:
//!
//! | | Unigram (`t5`) | SPM-BPE (`llama`) |
//! |---|---|---|
//! | score | log-probability | **merge rank** (higher = merge earlier) |
//! | algorithm | Viterbi, maximise the *sum* over a segmentation | greedily merge the best-scoring adjacent pair, repeatedly |
//!
//! Running Viterbi over merge-rank scores is not a small inaccuracy — it
//! inverts the objective. In Gemma's vocabulary, scores run roughly `-id`, so
//! short early-id fragments outscore whole words: maximising the sum picks
//! `▁h` + `el` + `lo` (total −431) over `▁hello` (−28610), and
//! `▁sourdough` shatters into `▁s|ou|rd|ou|gh`. The model never saw those
//! pieces during training, so every embedding is computed from out-of-
//! distribution input while the pipeline reports success.
//!
//! The merge loop below reproduces llama.cpp's `llm_tokenizer_spm`, which
//! recovers `▁hello` and `▁sourdough` from the same vocabulary.

use rustc_hash::FxHashMap;
use std::collections::BinaryHeap;
use std::convert::Infallible;
use thiserror::Error;

use super::metaspace::{self, Prefix, WORD_BOUNDARY};
use super::policy::{PolicyError, SpecialMode};
use super::tokenize::{Tokenize, TokenizeError};

/// SentencePiece's "never merge" score sentinel.
///
/// A trainer writes this on pieces it refuses to let the merger build — in
/// Mistral's vocabularies, the 15 whitespace runs `▁`, `▁▁`, … Since scores are
/// merge ranks here (higher merges earlier), it loses to every real merge.
/// It is also the right score for a slot added after the vocabulary file ends,
/// such as an added token that must be matched verbatim rather than merged into.
pub const NEVER_MERGE: f32 = -1e9;

/// Where the dummy prefix (`add_dummy_prefix` / `add_space_prefix`) is placed
/// once the input is split on added tokens.
///
/// The two reference implementations of this same vocabulary format genuinely
/// disagree, and both were measured rather than inferred — so neither is "the"
/// behavior and the loader that built the tokenizer has to say which one its
/// vocabulary was produced for. Encoding `"[INST]Write"` and `"a[INST]b"`:
///
/// | | [`Once`](Self::Once) (HF) | [`AfterEachSpecial`](Self::AfterEachSpecial) (llama.cpp) |
/// |---|---|---|
/// | `[INST]Write` | `▁`, `[INST]`, `Write` | `[INST]`, `▁Write` |
/// | `a[INST]b` | `▁a`, `[INST]`, `b` | `▁a`, `[INST]`, `▁b` |
///
/// Getting this wrong is invisible from the outside — every id stays in range
/// and decodes back to the original string — while every chat prompt (which is
/// exactly a text/marker alternation) reaches the model as pieces it was not
/// trained on.
///
/// The default is [`AfterEachSpecial`](Self::AfterEachSpecial), because
/// [`SpmTokenizer::new`] takes a GGUF-style vocabulary and llama.cpp is the
/// reference for those. A vocabulary lifted out of a HuggingFace
/// `tokenizer.model` is the case that has to be declared, and its one loader
/// does declare it.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum SpmPrefixScheme {
    /// Prefix the whole text **once**, before splitting on added tokens
    /// (HuggingFace / `sentencepiece`).
    ///
    /// SentencePiece normalizes — and therefore prefixes — the input and only
    /// then splits, so only the stretch beginning at byte 0 can carry a marker.
    /// A leading added token leaves the marker standing alone, with no text to
    /// attach to. Measured with
    /// `AutoTokenizer.from_pretrained("mistral-7b-v0.3", use_fast=False)` and
    /// `add_special_tokens=False`: `"[INST]Write"` -> `[29473, 3, 6006]`
    /// (`▁`, `[INST]`, bare `Write`) and `"a[INST]b"` -> `[1032, 3, 29494]`
    /// (`▁a`, `[INST]`, bare `b`).
    ///
    /// This is HuggingFace's *corrected* behavior (`legacy = false` in
    /// `tokenizer_config.json`). Which scheme a bundled `.spm` vocabulary needs
    /// is not determined by the fact that it came from a `tokenizer.model` —
    /// it is that per-checkpoint `legacy` flag: Mistral V2 sets `legacy = false`
    /// and needs `Once`, but Mistral V1 sets `legacy = true` and needs
    /// [`AfterEachSpecial`](Self::AfterEachSpecial) despite also being extracted
    /// from a `tokenizer.model` — see `spm_prefix_scheme` in `pretrained.rs`,
    /// which reads that flag off per vocabulary rather than assuming it.
    Once,
    /// Prefix the first stretch **and every stretch that follows an added
    /// token** (llama.cpp's `is_prev_special`).
    ///
    /// `llama-vocab.cpp`'s `LLAMA_VOCAB_TYPE_SPM` arm walks the fragment buffer
    /// with `bool is_prev_special = true` ("prefix with space if first token"),
    /// prepending `' '` to a raw-text fragment whenever the flag is set and
    /// re-arming it on every special-token fragment. A special token at the very
    /// start therefore emits **no** standalone marker — there is no text
    /// fragment before it to prefix.
    ///
    /// Correct for every GGUF-loaded vocabulary, because llama.cpp is what
    /// actually runs those files — and so the default, see the type's docs.
    /// Also correct for a bundled `.spm` vocabulary whose checkpoint declares
    /// `legacy = true` (Mistral V1) — see [`Once`](Self::Once)'s docs.
    #[default]
    AfterEachSpecial,
}

#[derive(Error, Debug)]
pub enum SpmError {
    #[error("Empty vocabulary")]
    EmptyVocab,
    #[error("Scores length ({scores}) does not match tokens length ({tokens})")]
    ScoreMismatch { scores: usize, tokens: usize },
    #[error("Failed to build added-token matcher: {0}")]
    AddedTokensError(#[from] aho_corasick::BuildError),
}

/// One symbol in the working sequence: a slice of the normalized text plus
/// intrusive doubly-linked-list pointers so a merge is O(1).
#[derive(Clone, Copy)]
struct Symbol {
    prev: i64,
    next: i64,
    start: usize,
    len: usize,
}

/// A candidate merge of two adjacent symbols.
///
/// Ordered by score, then by *lower* left index, so `BinaryHeap`'s max-pop
/// yields the highest-scoring merge and breaks ties left-to-right — matching
/// llama.cpp's comparator.
struct Bigram {
    left: i64,
    right: i64,
    score: f32,
    /// Byte length of the merged text, used to detect a stale queue entry.
    len: usize,
}

impl PartialEq for Bigram {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.left == other.left
    }
}
impl Eq for Bigram {}

impl Ord for Bigram {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            // Lower left index wins a tie, so reverse it for a max-heap.
            .then_with(|| other.left.cmp(&self.left))
    }
}
impl PartialOrd for Bigram {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// SentencePiece BPE tokenizer for `tokenizer.ggml.model = "llama"` vocabularies.
pub struct SpmTokenizer {
    token_to_id: FxHashMap<String, u32>,
    id_to_token: Vec<String>,
    /// Merge ranks. Higher merges earlier.
    scores: Vec<f32>,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    unk_id: Option<u32>,
    /// Ids of the 256 `<0xNN>` byte tokens, when the vocab provides them.
    byte_ids: Option<Box<[u32; 256]>>,
    /// Prepend a word boundary to the input (SentencePiece `add_dummy_prefix`).
    add_prefix_space: bool,
    /// *Where* that word boundary goes once the input is split on added tokens.
    prefix_scheme: SpmPrefixScheme,
    /// Control/added tokens to recognize verbatim in the input during encoding.
    added: Option<super::added::AddedTokens>,
    /// Ids of `special=true` added tokens dropped on decode (HF default).
    ///
    /// The vocabulary's own BOS/EOS/UNK are *not* in here — they have fields of
    /// their own and are dropped unconditionally — so this set holds exactly
    /// what a loader declares on top of them.
    special_decode: rustc_hash::FxHashSet<u32>,
}

impl SpmTokenizer {
    /// Build from a GGUF-style vocabulary.
    ///
    /// `scores` are merge ranks, not log-probabilities. When empty, token id
    /// order is used as the merge order (lower id merges earlier), which is the
    /// convention these vocabularies already follow.
    pub fn new(
        tokens: Vec<String>,
        scores: Vec<f32>,
        bos_token_id: Option<u32>,
        eos_token_id: Option<u32>,
    ) -> Result<Self, SpmError> {
        if tokens.is_empty() {
            return Err(SpmError::EmptyVocab);
        }
        let scores = if scores.is_empty() {
            (0..tokens.len()).map(|i| -(i as f32)).collect()
        } else if scores.len() != tokens.len() {
            return Err(SpmError::ScoreMismatch {
                scores: scores.len(),
                tokens: tokens.len(),
            });
        } else {
            scores
        };

        let mut token_to_id = FxHashMap::default();
        token_to_id.reserve(tokens.len());
        for (id, token) in tokens.iter().enumerate() {
            // First id wins: a duplicated piece must resolve to the canonical
            // (lowest) id, matching llama.cpp's vocab construction.
            token_to_id.entry(token.clone()).or_insert(id as u32);
        }

        let unk_id = token_to_id
            .get("<unk>")
            .or_else(|| token_to_id.get("<UNK>"))
            .copied();

        // Byte fallback is all-or-nothing: a partial `<0xNN>` set cannot encode
        // arbitrary input, so fall back to <unk> instead of emitting a hole.
        let mut byte_ids = [0u32; 256];
        let mut complete = true;
        for (b, slot) in byte_ids.iter_mut().enumerate() {
            match token_to_id.get(&format!("<0x{b:02X}>")) {
                Some(&id) => *slot = id,
                None => {
                    complete = false;
                    break;
                }
            }
        }

        Ok(Self {
            token_to_id,
            id_to_token: tokens,
            scores,
            bos_token_id,
            eos_token_id,
            unk_id,
            byte_ids: complete.then(|| Box::new(byte_ids)),
            add_prefix_space: true,
            prefix_scheme: SpmPrefixScheme::default(),
            added: None,
            special_decode: rustc_hash::FxHashSet::default(),
        })
    }

    /// Attach added tokens to recognize in the input during encoding.
    ///
    /// Without this, a control token spliced into the prompt text (`<start_of_turn>`)
    /// is normalized and merged like ordinary content, so the model sees a handful
    /// of fragments where its chat template promised one token — silently, since the
    /// ids stay in range and decode back to the same string.
    ///
    /// Takes anything convertible into an [`AddedTokenSet`](super::added::AddedTokenSet),
    /// so a caller with no `lstrip`/`rstrip` flags to declare (GGUF, a bundled
    /// vocabulary, a test) can still pass a plain name→id map.
    pub fn with_added_tokens(
        mut self,
        tokens: impl Into<super::added::AddedTokenSet>,
    ) -> Result<Self, SpmError> {
        self.added = super::added::AddedTokens::new(&tokens.into())?;
        Ok(self)
    }

    /// Set ids of `special=true` added tokens to drop on decode (HF default).
    ///
    /// Replaces rather than unions, unlike
    /// [`WordPieceTokenizer::with_special_decode_ids`](super::wordpiece::WordPieceTokenizer::with_special_decode_ids):
    /// that constructor resolves the `[CLS]`/`[SEP]`/… names its vocabulary
    /// spells into the same set, so a caller stating what its *file* declares is
    /// adding to knowledge already there. This constructor puts nothing in here —
    /// the vocabulary's own BOS/EOS/UNK live in their own fields and are skipped
    /// by [`is_skipped_on_decode`](Self::is_skipped_on_decode) regardless — so
    /// there is nothing to union with, and replacing keeps the field meaning
    /// exactly "what the loader declared", as on the Unigram sibling.
    pub fn with_special_decode_ids(mut self, ids: rustc_hash::FxHashSet<u32>) -> Self {
        self.special_decode = ids;
        self
    }

    /// Set SentencePiece `add_dummy_prefix` (GGUF `tokenizer.ggml.add_space_prefix`).
    ///
    /// Defaults to true. Gemma sets it false; prepending a boundary anyway
    /// shifts the very first piece of every input to a different token.
    pub fn with_prefix_space(mut self, add_prefix_space: bool) -> Self {
        self.add_prefix_space = add_prefix_space;
        self
    }

    /// Select where the dummy prefix lands relative to added tokens — see
    /// [`SpmPrefixScheme`], whose two arms are the two references' measured,
    /// mutually incompatible behaviors.
    ///
    /// The loader that read the vocabulary is the only place that knows which
    /// reference the file runs under, so it is the only place that can answer
    /// this. Every GGUF file runs under llama.cpp
    /// ([`AfterEachSpecial`](SpmPrefixScheme::AfterEachSpecial)). A vocabulary
    /// extracted from a HuggingFace `tokenizer.model` is *not* determined by
    /// that fact alone — it depends on the checkpoint's `legacy` flag in
    /// `tokenizer_config.json` (`false` -> [`Once`](SpmPrefixScheme::Once),
    /// `true` -> `AfterEachSpecial` too) — see [`SpmPrefixScheme::Once`]'s docs
    /// and `spm_prefix_scheme` in `pretrained.rs`, which reads that flag off
    /// per vocabulary.
    ///
    /// Inert when [`with_prefix_space`](Self::with_prefix_space) is off: with no
    /// marker to place, the two schemes agree everywhere.
    pub fn with_prefix_scheme(mut self, prefix_scheme: SpmPrefixScheme) -> Self {
        self.prefix_scheme = prefix_scheme;
        self
    }

    /// Set the BOS / EOS ids the vocabulary defines (GGUF `add_bos_token` /
    /// `add_eos_token` resolve to `None` here when disabled).
    ///
    /// `encode` never emits them: they are reported through
    /// [`bos_token_id`](Self::bos_token_id) / [`eos_token_id`](Self::eos_token_id)
    /// so the special-token policy can place them.
    pub fn with_special_ids(mut self, bos: Option<u32>, eos: Option<u32>) -> Self {
        self.bos_token_id = bos;
        self.eos_token_id = eos;
        self
    }

    /// The dummy-prefix convention this vocabulary encodes under.
    ///
    /// [`Prefix::Always`] — SentencePiece prepends the marker even when the
    /// input already starts with a space, which is what the 46/46 llama.cpp
    /// reference agreement rests on — unless `add_dummy_prefix` is off, in
    /// which case no marker is ever added anywhere.
    ///
    /// This answers *whether* a stretch handed to
    /// [`encode_segment`](Self::encode_segment) gets a marker;
    /// [`SpmPrefixScheme`] answers *which* stretches are handed one.
    fn prefix(&self) -> Prefix {
        if self.add_prefix_space {
            Prefix::Always
        } else {
            Prefix::None
        }
    }

    /// Escape one stretch of the input and merge it into pieces.
    ///
    /// Which stretches get a `prefix` other than [`Prefix::None`] is the
    /// [`SpmPrefixScheme`]'s decision — see [`gap_encoder`](Self::gap_encoder).
    /// Spaces themselves are never collapsed: this backend's vocabularies carry
    /// the whitespace-run pieces (`▁▁`, …) and merge them themselves.
    fn encode_segment(&self, text: &str, prefix: Prefix) -> Vec<u32> {
        let escaped = metaspace::escape(text, prefix, false);
        let mut out = Vec::new();
        for symbol in self.merge(&escaped) {
            self.emit(&escaped[symbol.start..symbol.start + symbol.len], &mut out);
        }
        out
    }

    /// Merge adjacent symbols, best score first, until nothing merges.
    fn merge(&self, text: &str) -> Vec<Symbol> {
        let mut symbols: Vec<Symbol> = Vec::new();
        for (offset, ch) in text.char_indices() {
            let len = ch.len_utf8();
            let index = symbols.len() as i64;
            symbols.push(Symbol {
                prev: index - 1,
                next: if offset + len == text.len() {
                    -1
                } else {
                    index + 1
                },
                start: offset,
                len,
            });
        }
        if symbols.is_empty() {
            return symbols;
        }

        let mut queue: BinaryHeap<Bigram> = BinaryHeap::new();
        let push = |queue: &mut BinaryHeap<Bigram>, left: i64, right: i64, syms: &[Symbol]| {
            if left < 0 || right < 0 {
                return;
            }
            let (l, r) = (&syms[left as usize], &syms[right as usize]);
            let merged = &text[l.start..r.start + r.len];
            if let Some(&id) = self.token_to_id.get(merged) {
                queue.push(Bigram {
                    left,
                    right,
                    score: self.scores[id as usize],
                    len: merged.len(),
                });
            }
        };

        for i in 1..symbols.len() as i64 {
            push(&mut queue, i - 1, i, &symbols);
        }

        while let Some(bigram) = queue.pop() {
            let (li, ri) = (bigram.left as usize, bigram.right as usize);
            let (left, right) = (symbols[li], symbols[ri]);

            // Either side already absorbed into another merge → stale entry.
            if left.len == 0 || right.len == 0 || left.len + right.len != bigram.len {
                continue;
            }

            // Absorb the right symbol into the left one and unlink it.
            symbols[li].len = left.len + right.len;
            symbols[ri].len = 0;
            symbols[li].next = right.next;
            if right.next >= 0 {
                symbols[right.next as usize].prev = bigram.left;
            }

            push(&mut queue, symbols[li].prev, bigram.left, &symbols);
            push(&mut queue, bigram.left, symbols[li].next, &symbols);
        }

        symbols.into_iter().filter(|s| s.len > 0).collect()
    }

    /// Emit ids for one final symbol, falling back to bytes then `<unk>`.
    fn emit(&self, piece: &str, out: &mut Vec<u32>) {
        if let Some(&id) = self.token_to_id.get(piece) {
            out.push(id);
            return;
        }
        match &self.byte_ids {
            Some(byte_ids) => out.extend(piece.bytes().map(|b| byte_ids[b as usize])),
            None => {
                if let Some(unk) = self.unk_id {
                    out.push(unk);
                }
            }
        }
    }

    /// Encode without any added-token handling.
    ///
    /// Content tokens only: boundary tokens are the
    /// [`SpecialPolicy`](crate::core::SpecialPolicy)'s to add, so that a caller
    /// wrapping two sequences does not get a stray BOS in the middle.
    pub fn encode_ordinary(&self, text: &str) -> Vec<u32> {
        // Empty input has nothing to mark a boundary *of*: `sp.encode("")` is
        // `[]`, and so is llama.cpp's `ggml-vocab-llama-spm` fixture. The guard
        // belongs here rather than in `encode_segment`, whose unconditional
        // prefix is correct for every non-empty input (verified 46/46 against
        // llama.cpp) and is what deliberately produces the standalone marker
        // from an empty stretch in `standalone_prefix`.
        if text.is_empty() {
            return Vec::new();
        }
        self.encode_segment(text, self.prefix())
    }

    /// Whether the whole-input dummy prefix must surface as a standalone `▁`
    /// piece because an added token occupies byte 0.
    ///
    /// Measured against `AutoTokenizer.from_pretrained("mistral-7b-v0.3",
    /// use_fast=False)` with `add_special_tokens=False`, i.e. Mistral's own
    /// SentencePiece-backed tokenizer: `"[INST]Write"` -> `[29473, 3, 6006]` =
    /// `▁`, `[INST]`, `Write`. The prefix belongs to the *input*, not to a gap,
    /// so when the input opens with an added token the prefix has nothing to
    /// attach to and is emitted on its own.
    ///
    /// The exception is the vocabulary's own sentinels (BOS / EOS / UNK), which
    /// swallow it: `"<s>x"` -> `[1, 29512]`, not `[29473, 1, 29512]`, while the
    /// otherwise identical `"[INST]x"` -> `[29473, 3, 29512]`. HuggingFace's
    /// `LlamaTokenizer.tokenize` drops a leading lone `▁` exactly when the piece
    /// after it is one of `all_special_tokens`, which for these vocabularies is
    /// precisely `<s>` / `</s>` / `<unk>` — the three ids named here.
    ///
    /// Byte 0 only, and only a *lone* marker: `"[INST]<s>x"` keeps the prefix
    /// (`[29473, 3, 1, 29512]`) because the sentinel is not first, and `" <s>x"`
    /// never had a lone marker to drop (`[1027, 1, 29512]` — `▁▁` then `<s>`).
    ///
    /// All of this is [`SpmPrefixScheme::Once`]'s alone. Under
    /// [`AfterEachSpecial`](SpmPrefixScheme::AfterEachSpecial) a standalone
    /// marker never exists to begin with — llama.cpp prefixes *text* fragments,
    /// and a leading added token has none before it — so the sentinel rule has
    /// nothing to suppress and needs no second code path here.
    fn prefix_stands_alone(&self, text: &str) -> bool {
        if !self.add_prefix_space || self.prefix_scheme != SpmPrefixScheme::Once {
            return false;
        }
        self.added
            .as_ref()
            .and_then(|added| added.id_at_start(text))
            .is_some_and(|id| {
                let leading = Some(id);
                leading != self.bos_token_id
                    && leading != self.eos_token_id
                    && leading != self.unk_id
            })
    }

    /// Whether an added token occupies byte 0, so that no gap begins the input.
    ///
    /// This — not "a standalone marker was emitted" — is what spends the single
    /// [`Once`](SpmPrefixScheme::Once) prefix: a leading sentinel swallows it
    /// (`"<s>x"` -> `['<s>', 'x']`, bare `x`) while any other leading added
    /// token leaves it standing alone (`"[INST]Write"` -> `['▁', '[INST]',
    /// 'Write']`). Both spend it; only one of them emits anything.
    fn starts_with_added_token(&self, text: &str, split: bool) -> bool {
        split
            && self
                .added
                .as_ref()
                .is_some_and(|added| added.id_at_start(text).is_some())
    }

    /// The ids that precede the added-token split: the lone `▁` piece when the
    /// whole-input dummy prefix has nothing to attach to, otherwise nothing.
    ///
    /// `split` is false for [`SpecialMode::Ordinary`], which never consults the
    /// matcher: the whole text is then one stretch starting at byte 0 and
    /// carries the prefix itself.
    fn standalone_prefix(&self, text: &str, split: bool) -> Vec<u32> {
        if split && self.prefix_stands_alone(text) {
            // An empty stretch escaped *with* the prefix is exactly the lone
            // boundary piece, so no separate id lookup is needed and a
            // vocabulary that spells the marker differently still agrees with
            // itself.
            self.encode_segment("", self.prefix())
        } else {
            Vec::new()
        }
    }

    /// The gap encoder to hand to [`AddedTokens`](super::added::AddedTokens),
    /// applying the dummy prefix where this tokenizer's [`SpmPrefixScheme`] says
    /// it goes.
    ///
    /// A gap is by construction either the stretch beginning at byte 0 or the
    /// stretch immediately following an added token, so the two schemes reduce
    /// to two lines here:
    ///
    /// - [`Once`](SpmPrefixScheme::Once) — SentencePiece normalizes, and
    ///   therefore prefixes, *before* it splits, so only the stretch beginning
    ///   at byte 0 can carry the prefix. Prefixing every gap instead changes the
    ///   ids of every Mistral chat prompt, since those all embed
    ///   `[INST]`/`[/INST]` mid-text: `"a[INST]b"` came out `▁a`, `[INST]`, `▁b`
    ///   (`[1032, 3, 1055]`) where the HF reference is `▁a`, `[INST]`, `b`
    ///   (`[1032, 3, 29494]`).
    /// - [`AfterEachSpecial`](SpmPrefixScheme::AfterEachSpecial) — llama.cpp's
    ///   `is_prev_special` is set before the loop and re-set by every special
    ///   fragment, so *every* text fragment is prefixed. That is `"a[INST]b"` ->
    ///   `▁a`, `[INST]`, `▁b`, which is what the GGUF path must produce.
    ///
    /// The matcher runs over the **original** text, not the escaped text.
    /// Escaping rewrites every space as a three-byte `▁`, which shifts byte
    /// offsets and would stop an added token that contains a space (a
    /// whitespace-run token, a multi-word chat marker) from ever matching. The
    /// added-token strings are unescaped surface forms, so matching them against
    /// unescaped input is the only self-consistent choice — and it costs
    /// nothing, because escaping is per-character and therefore identical
    /// whether it happens before or after the split. Only the prefix is
    /// position-dependent, and that is what this function places.
    ///
    /// `prefix_spent` says whether the single whole-input prefix is already
    /// accounted for; if so, no gap carries one — which can only arise under
    /// [`Once`](SpmPrefixScheme::Once), the one scheme that has a single prefix
    /// to spend. Note "spent" is not "emitted": when a sentinel leads the input
    /// the prefix is *swallowed* rather than emitted, and it must not then
    /// reappear on the following gap — reference `"<s>x"` -> `['<s>', 'x']`,
    /// with a bare `x`. Handing the encoder back
    /// rather than driving the split here lets the infallible
    /// ([`Tokenize::encode`]) and mode-aware ([`encode_with`](Self::encode_with))
    /// paths share this placement without either inventing an error it cannot
    /// produce.
    fn gap_encoder(&self, prefix_spent: bool) -> impl FnMut(&str) -> Vec<u32> + '_ {
        // Under `Once`: whichever stretch begins at byte 0 carries the prefix —
        // and when an added token begins the input instead, no stretch does.
        // `AddedTokens` never hands out an empty gap, so this cannot be spent on
        // nothing. Unused under `AfterEachSpecial`, where every gap is prefixed
        // and `standalone_prefix` is always empty.
        let mut carries_prefix = !prefix_spent;
        move |gap: &str| {
            let carries = match self.prefix_scheme {
                // Every gap either begins the input or follows an added token,
                // which is exactly llama.cpp's `is_prev_special` condition.
                SpmPrefixScheme::AfterEachSpecial => true,
                // One prefix for the whole input, spent on the first gap.
                SpmPrefixScheme::Once => std::mem::take(&mut carries_prefix),
            };
            let prefix = if carries { self.prefix() } else { Prefix::None };
            self.encode_segment(gap, prefix)
        }
    }

    /// The raw surface string of a token id (`▁` boundaries and `<0xNN>` byte
    /// tokens are kept as spelled). Used to drive a declared decoder pipeline.
    pub fn token_surface(&self, id: u32) -> Option<String> {
        self.id_to_token.get(id as usize).cloned()
    }

    /// The beginning-of-sequence token id, when the vocabulary defines one.
    pub fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    /// The end-of-sequence token id, when the vocabulary defines one.
    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }

    /// Encode text to token IDs under an explicit [`SpecialMode`], governing
    /// whether the added tokens attached via
    /// [`with_added_tokens`](Self::with_added_tokens) are matched in the input
    /// text. Never emits BOS/EOS — see [`Tokenize::encode`]; boundary tokens
    /// are [`SpecialPolicy`](crate::core::SpecialPolicy)'s to add via
    /// `AnyTokenizer::encode_with`.
    pub fn encode_with(&self, text: &str, mode: &SpecialMode<'_>) -> Result<Vec<u32>, PolicyError> {
        // See `encode_ordinary`: empty input has nothing to mark a boundary
        // *of*, and the guard must sit ahead of the split so that attaching a
        // matcher cannot change the answer.
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let split = !matches!(mode, SpecialMode::Ordinary);
        let mut out = self.standalone_prefix(text, split);
        let mut encode_gap = self.gap_encoder(self.starts_with_added_token(text, split));
        out.extend(super::added::AddedTokens::dispatch_with_mode(
            &self.added,
            text,
            mode,
            &mut encode_gap,
        )?);
        Ok(out)
    }

    /// Whether a token id is dropped when rendering decoded text.
    ///
    /// Shared by [`decode`](Self::decode) and
    /// [`decode_lossy`](Self::decode_lossy) — through the one loop both drive —
    /// so the two paths cannot drift on which ids they drop. Skips BOS/EOS,
    /// `<unk>`, and any `special=true` added token (`special_decode`), matching
    /// HuggingFace's default decode (`skip_special_tokens=True`) and the Unigram
    /// sibling's identical rule.
    ///
    /// Measured with the `sentencepiece` Python package 0.2.0 on Mistral's own
    /// `tokenizer.model`: `decode([1, 7080, 29477, 2294, 2])` is `'hello world'`
    /// — the boundary tokens produce nothing. Left unskipped, every generated
    /// sequence carried a literal `<s>`/`</s>` into the decoded text.
    ///
    /// `<unk>` goes with them rather than becoming SentencePiece's `unk_surface`
    /// (`' ⁇ '`): that is `sp.decode`'s own API, not the HF
    /// `skip_special_tokens` semantics this crate follows, and an unknown span
    /// was unrecoverable anyway.
    fn is_skipped_on_decode(&self, id: u32) -> bool {
        Some(id) == self.bos_token_id
            || Some(id) == self.eos_token_id
            || Some(id) == self.unk_id
            || self.special_decode.contains(&id)
    }

    /// Render ids to bytes, deciding through `on_unknown` what an id the
    /// vocabulary does not contain means.
    ///
    /// The single decode loop. Strict decoding instantiates `E` with
    /// [`TokenizeError`] and returns from `on_unknown`; lossy decoding
    /// instantiates it with [`Infallible`] and skips — so the two cannot
    /// disagree about what an id renders to, and the compiler proves the lossy
    /// path's `Err` arm away rather than an assertion claiming it.
    ///
    /// Bytes rather than text because a `<0xNN>` run is only valid UTF-8 once
    /// reassembled: what happens to a byte sequence that is not (an error, or
    /// U+FFFD) is the caller's decision, and the only one they differ on.
    fn decode_to_bytes<E>(
        &self,
        ids: &[u32],
        on_unknown: impl Fn(u32) -> Result<(), E>,
    ) -> Result<Vec<u8>, E> {
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            let piece = match self.id_to_token.get(id as usize) {
                Some(piece) => piece,
                None => {
                    on_unknown(id)?;
                    continue;
                }
            };
            if self.is_skipped_on_decode(id) {
                continue;
            }
            // `<0xNN>` byte tokens decode to the raw byte, not to their literal
            // spelling; a multi-byte character is split across several of them
            // and only reassembles as bytes.
            let byte = piece
                .strip_prefix("<0x")
                .and_then(|rest| rest.strip_suffix('>'))
                .and_then(|hex| u8::from_str_radix(hex, 16).ok());
            match byte {
                Some(b) => bytes.push(b),
                None => bytes.extend(piece.replace(WORD_BOUNDARY, " ").as_bytes()),
            }
        }
        Ok(bytes)
    }

    /// Remove the dummy prefix from rendered text — see [`decode`](Self::decode),
    /// which documents why exactly one space comes off and only when one was
    /// added.
    fn strip_dummy_prefix(&self, text: String) -> String {
        if !self.add_prefix_space {
            return text;
        }
        match text.strip_prefix(' ') {
            Some(rest) => rest.to_string(),
            None => text,
        }
    }

    /// Render the pieces, then strip the dummy prefix.
    ///
    /// BOS/EOS/`<unk>` and the declared `special=true` ids produce nothing —
    /// see [`is_skipped_on_decode`](Self::is_skipped_on_decode).
    ///
    /// SentencePiece's `add_dummy_prefix` puts a boundary before the first piece
    /// on encode, so rendering `▁` back to a space leaves one space that was
    /// never in the input. The reference pipelines both remove exactly one:
    /// `sp.decode`, and HuggingFace's declared decoder chain
    /// `Replace(▁→" ") → ByteFallback → Fuse → Strip{content: " ", start: 1}`.
    ///
    /// Exactly one — never all leading whitespace. `"  Hello"` encodes to the
    /// two-space piece `▁▁` plus `▁Hello`, which renders to three spaces; only
    /// the dummy one comes off, leaving the two the caller wrote.
    ///
    /// And only when a dummy prefix was actually added: with
    /// `add_dummy_prefix` off (Gemma) encoding never inserts one, so removing a
    /// space here would eat one the caller wrote. llama.cpp gates its
    /// detokenizer on the same flag.
    ///
    /// Errors with [`TokenizeError::InvalidTokenId`] on an id the vocabulary
    /// does not contain — a distinct thing from the skips above, which are
    /// deliberate — and with [`TokenizeError::Utf8Error`] when the rendered
    /// bytes are not valid UTF-8.
    pub fn decode(&self, ids: &[u32]) -> Result<String, TokenizeError> {
        let bytes = self.decode_to_bytes(ids, |id| Err(TokenizeError::InvalidTokenId(id)))?;
        let text = String::from_utf8(bytes).map_err(|_| TokenizeError::Utf8Error)?;
        Ok(self.strip_dummy_prefix(text))
    }

    /// Decode ids to text, skipping ids the vocabulary does not contain and
    /// replacing undecodable bytes with U+FFFD.
    ///
    /// The lenient half of the pair, over exactly the loop
    /// [`decode`](Self::decode) drives: same pieces, same skips, same dummy-prefix
    /// strip — only an unknown id and a broken byte sequence are treated as
    /// something to survive rather than to report. This method never fails, so
    /// `on_unknown` is instantiated with [`Infallible`], letting the compiler
    /// prove the `Err` arm away rather than a runtime assertion claiming it.
    pub fn decode_lossy(&self, ids: &[u32]) -> String {
        let bytes = match self.decode_to_bytes(ids, |_| Ok::<(), Infallible>(())) {
            Ok(bytes) => bytes,
            // `Infallible` has no values, so this match has no arms to write.
            Err(never) => match never {},
        };
        self.strip_dummy_prefix(String::from_utf8_lossy(&bytes).into_owned())
    }
}

impl Tokenize for SpmTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        // Recognize added tokens in the input first (HF behavior), then SPM-BPE.
        // See `encode_ordinary` for why empty input is guarded ahead of the split.
        if text.is_empty() {
            return Vec::new();
        }
        let mut out = self.standalone_prefix(text, true);
        let mut encode_gap = self.gap_encoder(self.starts_with_added_token(text, true));
        out.extend(super::added::AddedTokens::dispatch(
            &self.added,
            text,
            &mut encode_gap,
        ));
        out
    }

    fn encode_with(&self, text: &str, mode: &SpecialMode<'_>) -> Result<Vec<u32>, PolicyError> {
        self.encode_with(text, mode)
    }

    /// Render the pieces, then strip the dummy prefix — the inherent
    /// [`decode`](SpmTokenizer::decode), which documents both, so the trait and
    /// the type can never disagree about what an id decodes to.
    fn decode(&self, ids: &[u32]) -> Result<String, TokenizeError> {
        self.decode(ids)
    }

    fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A vocabulary shaped like the ones this tokenizer is for: scores are merge
    /// ranks (`-id`), and it carries the *intermediate* merge results a real BPE
    /// vocabulary contains, not just the fragments and the finished words.
    ///
    /// The ids are arranged so that maximising the summed score would prefer the
    /// cheap fragments — `▁h`(-4) + `el`(-5) + `lo`(-6) = -15 beats
    /// `▁hello`(-24) — which is exactly the trap that shatters Gemma's words
    /// under Viterbi. Merging by best adjacent pair must still reach `▁hello`.
    ///
    /// `▁hell` is deliberately absent so one test can observe a merge chain that
    /// legitimately stops short.
    fn rank_scored_vocab() -> (Vec<String>, Vec<f32>) {
        let tokens: Vec<String> = [
            "<pad>", "<eos>", "<bos>", "<unk>", // 0..3
            "▁h", "el", "lo", "▁w", "or", "ld", // 4..9   fragments, best scores
            "h", "e", "l", "o", "w", "r", "d", "▁", // 10..17 single chars
            "ll", "▁he", // 18..19  intermediates
            "▁hel", "▁wor", // 20..21  intermediates
            "▁hello", "▁world", // 22..23  whole words, worst scores
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
        let scores = (0..tokens.len()).map(|i| -(i as f32)).collect();
        (tokens, scores)
    }

    fn tok() -> SpmTokenizer {
        let (tokens, scores) = rank_scored_vocab();
        SpmTokenizer::new(tokens, scores, None, None).unwrap()
    }

    fn pieces(t: &SpmTokenizer, text: &str) -> Vec<String> {
        t.encode(text)
            .into_iter()
            .map(|id| t.id_to_token[id as usize].clone())
            .collect()
    }

    /// The defect this tokenizer exists to prevent: maximising the sum of
    /// rank-scores prefers many cheap fragments over the whole word. Merging by
    /// best adjacent pair must recover the word.
    #[test]
    fn whole_words_win_over_cheaper_fragment_sequences() {
        let t = tok();
        assert_eq!(pieces(&t, "hello"), vec!["▁hello"]);
        assert_eq!(pieces(&t, "hello world"), vec!["▁hello", "▁world"]);
    }

    #[test]
    fn a_merge_chain_that_stops_short_keeps_every_character() {
        let t = tok();
        // "▁hell" is absent, so merging halts at "▁hel" + "l". It must not
        // invent a token, drop a character, or fall through to <unk>.
        assert_eq!(pieces(&t, "hell"), vec!["▁hel", "l"]);
    }

    /// Spaces survive as boundary markers and come back as spaces — and the
    /// dummy prefix does not leak into the decoded text. Reference:
    /// `"Hello, world!"` -> `[22557, 28725, 1526, 28808]` -> `"Hello, world!"`.
    #[test]
    fn spaces_become_word_boundaries_and_round_trip() {
        let t = tok();
        let ids = t.encode("hello world");
        assert_eq!(t.decode(&ids).unwrap(), "hello world");
    }

    /// Only the dummy prefix comes off, so leading spaces the caller actually
    /// wrote are preserved. Reference rows, on the Mistral vocabulary:
    /// `" Hello world"` -> `[28705, 22557, 1526]`, `"  Hello"` -> `[259, 22557]`
    /// and `" "` -> `[259]` all decode back to themselves. Here `▁` (id 17) is
    /// the standalone boundary this vocabulary emits for a leading space, so
    /// `" hello"` renders as two spaces and keeps one.
    #[test]
    fn a_leading_space_survives_decoding() {
        let t = tok();
        let ids = t.encode(" hello");
        assert_eq!(t.decode(&ids).unwrap(), " hello");
        assert_eq!(t.decode(&t.encode(" hello world")).unwrap(), " hello world");
        assert_eq!(t.decode(&t.encode("")).unwrap(), "");
    }

    /// `add_space_prefix = false` (Gemma) must not prepend a boundary — doing so
    /// silently changes the first token of every input.
    #[test]
    fn prefix_space_can_be_disabled() {
        let (tokens, scores) = rank_scored_vocab();
        let t = SpmTokenizer::new(tokens, scores, None, None)
            .unwrap()
            .with_prefix_space(false);
        assert_eq!(pieces(&t, "hello"), vec!["h", "el", "lo"]);
        // Decoding must stay symmetric: no prefix was added, so none is
        // removed, and a space the caller wrote survives untouched.
        assert_eq!(t.decode(&t.encode(" hello")).unwrap(), " hello");
    }

    /// Boundary tokens belong to the special-token policy, not to the model: a
    /// tokenizer that adds them itself gives a caller wrapping two sequences a
    /// stray BOS in the middle, and no way to opt out. `encode` stays raw even
    /// when the vocabulary defines both ids, which remain readable.
    #[test]
    fn bos_and_eos_are_reported_but_never_encoded() {
        let (tokens, scores) = rank_scored_vocab();
        let with = SpmTokenizer::new(tokens.clone(), scores.clone(), Some(2), Some(1)).unwrap();
        assert_eq!(pieces(&with, "hello"), vec!["▁hello"]);
        assert_eq!(with.bos_token_id(), Some(2));
        assert_eq!(with.eos_token_id(), Some(1));

        let without = SpmTokenizer::new(tokens, scores, None, None).unwrap();
        assert_eq!(pieces(&without, "hello"), vec!["▁hello"]);
        assert_eq!(without.bos_token_id(), None);
        assert_eq!(without.eos_token_id(), None);
    }

    /// Unknown characters must become byte tokens when the vocab has the full
    /// `<0xNN>` set, so arbitrary input survives a round trip.
    #[test]
    fn unknown_characters_use_byte_fallback() {
        let mut tokens: Vec<String> = vec!["<unk>".into(), "▁".into()];
        for b in 0..=255u32 {
            tokens.push(format!("<0x{b:02X}>"));
        }
        let n = tokens.len();
        let t = SpmTokenizer::new(tokens, (0..n).map(|i| -(i as f32)).collect(), None, None)
            .unwrap()
            .with_prefix_space(false);

        let ids = t.encode("é");
        assert_eq!(ids.len(), 2, "é is two UTF-8 bytes, so two byte tokens");
        assert_eq!(t.decode(&ids).unwrap(), "é");
    }

    /// Without a complete byte set, an unknown character must map to `<unk>`
    /// rather than emitting a partial or empty result.
    #[test]
    fn unknown_characters_without_byte_fallback_use_unk() {
        let tokens: Vec<String> = ["<unk>", "▁", "a"].iter().map(|s| s.to_string()).collect();
        let t = SpmTokenizer::new(tokens, vec![], None, None)
            .unwrap()
            .with_prefix_space(false);
        assert_eq!(pieces(&t, "z"), vec!["<unk>"]);
    }

    /// With the dummy prefix disabled there is nothing to encode, so empty input
    /// must yield no tokens rather than a stray boundary or an <unk>.
    #[test]
    fn empty_input_produces_no_tokens() {
        let (tokens, scores) = rank_scored_vocab();
        let t = SpmTokenizer::new(tokens, scores, None, None)
            .unwrap()
            .with_prefix_space(false);
        assert!(t.encode("").is_empty());
    }

    /// The dummy prefix must not manufacture a token out of nothing: with it
    /// enabled, empty input still encodes to no tokens at all. `sp.encode("")`
    /// is `[]`, as is llama.cpp's `ggml-vocab-llama-spm` empty-string fixture.
    ///
    /// Asserted for both matcher states because they take different paths —
    /// `AddedTokens::encode_with` never invokes the gap encoder for `""` — and
    /// attaching a matcher must not change what the tokenizer means.
    #[test]
    fn empty_input_with_prefix_space_still_produces_no_tokens() {
        assert!(tok().encode("").is_empty());

        let (tokens, scores) = rank_scored_vocab();
        let mut map = FxHashMap::default();
        map.insert("<bos>".to_string(), 2);
        let with_matcher = SpmTokenizer::new(tokens, scores, None, None)
            .unwrap()
            .with_added_tokens(&map)
            .unwrap();
        assert!(with_matcher.encode("").is_empty());
    }

    /// llama.cpp prepends the dummy prefix **unconditionally** when
    /// `add_space_prefix` is on — a leading space in the input is never
    /// treated as "already have one". A real llama.cpp vocabulary later
    /// merges `▁▁` into a single piece (e.g. `ggml-vocab-llama-spm.gguf`
    /// maps `" "` to id `259`, spelled `▁▁`), but this synthetic vocab has
    /// no such merged token, so the two boundary symbols simply stay
    /// unmerged. What must hold regardless of vocabulary is the boundary
    /// *count*: a leading space must produce one more boundary symbol than
    /// no leading space at all, never fewer or the same.
    #[test]
    fn leading_space_is_never_swallowed_by_the_dummy_prefix() {
        let t = tok();
        // Single space: two independent boundary pieces, not one.
        assert_eq!(pieces(&t, " "), vec!["▁", "▁"]);
        // The dummy prefix stands alone (nothing merges "▁▁"), and the rest
        // of the input still tokenizes exactly as it would with no leading
        // space at all.
        assert_eq!(pieces(&t, " hello"), vec!["▁", "▁hello"]);
        assert_eq!(pieces(&t, " hello world"), vec!["▁", "▁hello", "▁world"]);
    }

    /// Structural property that must hold for any vocabulary, not just this
    /// synthetic one: with `add_prefix_space` on, encoding text with a
    /// leading space must yield exactly one more leading boundary symbol
    /// than encoding the same text without it. This is the guarantee
    /// llama.cpp's reference outputs rely on (verified separately against
    /// `ggml-vocab-llama-spm.gguf` / `ggml-vocab-phi-3.gguf`, where e.g.
    /// `" Hello"` -> `[29871, 15043]` but `"Hello"` -> `[15043]`: the extra
    /// leading id is exactly one standalone boundary token).
    #[test]
    fn leading_space_yields_exactly_one_extra_leading_boundary_piece() {
        let t = tok();
        let without = pieces(&t, "hello");
        let with = pieces(&t, " hello");
        assert_eq!(with.len(), without.len() + 1);
        assert_eq!(with[0], WORD_BOUNDARY);
        assert_eq!(&with[1..], &without[..]);
    }

    /// Attach `<bos>`/`<eos>` as ordinary added tokens — *not* as the
    /// vocabulary's BOS/EOS sentinels, which behave differently at byte 0 (see
    /// `sentinel_added_tokens_swallow_the_standalone_prefix`).
    fn tok_with_added_scheme(scheme: SpmPrefixScheme) -> SpmTokenizer {
        let (tokens, scores) = rank_scored_vocab();
        let mut map = FxHashMap::default();
        map.insert("<bos>".to_string(), 2);
        map.insert("<eos>".to_string(), 1);
        SpmTokenizer::new(tokens, scores, None, None)
            .unwrap()
            .with_prefix_scheme(scheme)
            .with_added_tokens(&map)
            .unwrap()
    }

    /// The HuggingFace / `sentencepiece` scheme, which every reference row in
    /// the tests below was measured under.
    fn tok_with_added() -> SpmTokenizer {
        tok_with_added_scheme(SpmPrefixScheme::Once)
    }

    /// A control token spliced into the prompt text must survive as its own id.
    /// Without matching it is normalized and merged like content — `<bos>hello`
    /// silently becomes a run of `<unk>`s, in range and reversible, so nothing
    /// downstream notices the chat template was destroyed.
    #[test]
    fn added_tokens_in_the_input_encode_to_their_own_id() {
        let t = tok_with_added();
        assert_eq!(
            pieces(&t, "<bos>hello world<eos>"),
            // The dummy prefix is the whole input's, applied before the split:
            // `<bos>` sits at byte 0, so the prefix stands alone, and the gap
            // that *follows* an added token is escaped bare — hence `h|el|lo`
            // rather than `▁hello`. This test previously asserted
            // `[2, 22, 23, 1]` (`<bos>`, `▁hello`, `▁world`, `<eos>`), which
            // pinned the per-gap prefixing bug.
            vec!["▁", "<bos>", "h", "el", "lo", "▁world", "<eos>"],
            "the markers stay whole and the gaps still merge"
        );
    }

    /// Under [`SpmPrefixScheme::Once`] the dummy prefix belongs to the *input*,
    /// not to each gap between added tokens (llama.cpp differs — see
    /// `the_two_prefix_schemes_disagree_exactly_here`). Measured against
    /// Mistral's own SentencePiece tokenizer
    /// (`AutoTokenizer.from_pretrained("mistral-7b-v0.3", use_fast=False)`,
    /// `add_special_tokens=False`):
    ///
    /// | input | reference ids | pieces |
    /// |---|---|---|
    /// | `a[INST]b` | `[1032, 3, 29494]` | `▁a`, `[INST]`, `b` |
    /// | `x[/INST]y` | `[2086, 4, 29492]` | `▁x`, `[/INST]`, `y` |
    ///
    /// Prefixing each gap instead produced `▁b` / `▁y`, changing the ids of
    /// every Mistral chat prompt.
    #[test]
    fn only_the_stretch_at_byte_zero_carries_the_dummy_prefix() {
        let t = tok_with_added();
        assert_eq!(
            pieces(&t, "hello<eos>world"),
            vec!["▁hello", "<eos>", "w", "or", "ld"],
            "the leading gap is prefixed; the gap after the marker is not"
        );
        // Three gaps, still exactly one prefix — the first one.
        assert_eq!(
            pieces(&t, "hello<eos>world<eos>hello"),
            vec!["▁hello", "<eos>", "w", "or", "ld", "<eos>", "h", "el", "lo"]
        );
    }

    /// Under [`SpmPrefixScheme::Once`], when an added token occupies byte 0 the
    /// prefix has nothing to attach to, and SentencePiece emits it as a
    /// standalone `▁` piece before the token.
    /// Reference: `"[INST]Write"` -> `[29473, 3, 6006]` and `"[INST]"` ->
    /// `[29473, 3]`, where `29473` is the lone `▁` piece.
    #[test]
    fn a_leading_added_token_leaves_the_dummy_prefix_standing_alone() {
        let t = tok_with_added();
        assert_eq!(pieces(&t, "<bos>"), vec!["▁", "<bos>"]);
        assert_eq!(
            pieces(&t, "<bos>hello"),
            vec!["▁", "<bos>", "h", "el", "lo"]
        );
        // A leading *space* means the marker is no longer alone — it is part of
        // the first gap, which carries the prefix as usual. Reference:
        // `" <s>x"` -> `[1027, 1, 29512]`, whose first id is the merged `▁▁`
        // piece; this synthetic vocabulary has no `▁▁`, so the two markers stay
        // unmerged. What must hold either way is that the gap is escaped with
        // the prefix rather than a bare marker being emitted beside it.
        assert_eq!(
            pieces(&t, " <bos>hello"),
            vec!["▁", "▁", "<bos>", "h", "el", "lo"]
        );
    }

    /// Under [`SpmPrefixScheme::Once`], a leading BOS/EOS/UNK *swallows* the
    /// standalone prefix, unlike any other added token. It is an HF-path rule:
    /// under [`SpmPrefixScheme::AfterEachSpecial`] no standalone marker is
    /// produced for any leading token, sentinel or not — pinned by
    /// `the_sentinel_rule_is_inert_under_the_llama_cpp_scheme`.
    ///
    /// Reference: `"<s>x"` -> `[1, 29512]` but `"[INST]x"` ->
    /// `[29473, 3, 29512]`; HuggingFace's `LlamaTokenizer.tokenize` drops a
    /// leading lone `▁` exactly when the next piece is one of
    /// `all_special_tokens` (`<s>`, `</s>`, `<unk>` for these vocabularies).
    ///
    /// And only at byte 0: `"[INST]<s>x"` -> `[29473, 3, 1, 29512]` keeps it.
    #[test]
    fn sentinel_added_tokens_swallow_the_standalone_prefix() {
        let (tokens, scores) = rank_scored_vocab();
        let mut map = FxHashMap::default();
        map.insert("<bos>".to_string(), 2);
        map.insert("<eos>".to_string(), 1);
        // BOS = `<bos>` (2); EOS left unset so `<eos>` stays an ordinary added
        // token and the two cases are visible side by side in one tokenizer.
        let t = SpmTokenizer::new(tokens, scores, Some(2), None)
            .unwrap()
            .with_prefix_scheme(SpmPrefixScheme::Once)
            .with_added_tokens(&map)
            .unwrap();

        assert_eq!(pieces(&t, "<bos>hello"), vec!["<bos>", "h", "el", "lo"]);
        assert_eq!(
            pieces(&t, "<eos>hello"),
            vec!["▁", "<eos>", "h", "el", "lo"]
        );
        // Byte 0 only: the sentinel sitting second does not reach back and drop
        // a prefix that a non-sentinel already left standing.
        assert_eq!(
            pieces(&t, "<eos><bos>hello"),
            vec!["▁", "<eos>", "<bos>", "h", "el", "lo"]
        );
    }

    /// The two schemes, on the same inputs, in one place — so the difference is
    /// documented in code and neither can silently drift into the other. Both
    /// columns are the measured behavior of their own reference:
    ///
    /// | input | [`Once`] (HF, `use_fast=False`) | [`AfterEachSpecial`] (llama.cpp) |
    /// |---|---|---|
    /// | `<M>hello` | `▁`, `<M>`, `h`,`el`,`lo` | `<M>`, `▁hello` |
    /// | `hello<M>world` | `▁hello`, `<M>`, `w`,`or`,`ld` | `▁hello`, `<M>`, `▁world` |
    /// | `<M>` | `▁`, `<M>` | `<M>` |
    ///
    /// HF rows follow `"[INST]Write"` -> `[29473, 3, 6006]` and `"a[INST]b"` ->
    /// `[1032, 3, 29494]` (a bare gap after the marker). llama.cpp rows follow
    /// `llama-vocab.cpp`'s `is_prev_special`, which is armed before the loop and
    /// re-armed by every special fragment, so every *text* fragment is prefixed
    /// and a leading special has no fragment before it to prefix.
    ///
    /// [`Once`]: SpmPrefixScheme::Once
    /// [`AfterEachSpecial`]: SpmPrefixScheme::AfterEachSpecial
    #[test]
    fn the_two_prefix_schemes_disagree_exactly_here() {
        let hf = tok_with_added_scheme(SpmPrefixScheme::Once);
        let cpp = tok_with_added_scheme(SpmPrefixScheme::AfterEachSpecial);

        // A leading added token: HF strands the marker, llama.cpp emits none and
        // prefixes the text that follows instead.
        assert_eq!(
            pieces(&hf, "<bos>hello"),
            vec!["▁", "<bos>", "h", "el", "lo"]
        );
        assert_eq!(pieces(&cpp, "<bos>hello"), vec!["<bos>", "▁hello"]);

        // A mid-text added token: HF has already spent its one prefix on the
        // leading stretch, llama.cpp prefixes the following stretch too.
        assert_eq!(
            pieces(&hf, "hello<eos>world"),
            vec!["▁hello", "<eos>", "w", "or", "ld"]
        );
        assert_eq!(
            pieces(&cpp, "hello<eos>world"),
            vec!["▁hello", "<eos>", "▁world"]
        );

        // Nothing but a marker.
        assert_eq!(pieces(&hf, "<bos>"), vec!["▁", "<bos>"]);
        assert_eq!(pieces(&cpp, "<bos>"), vec!["<bos>"]);

        // With no added token in the input there is one stretch, which begins at
        // byte 0 and is prefixed either way — the schemes must agree here, or
        // one of them is prefixing something other than a fragment boundary.
        for text in ["hello world", " hello", "hello"] {
            assert_eq!(pieces(&hf, text), pieces(&cpp, text), "input {text:?}");
        }
    }

    /// The BOS/EOS/UNK sentinel rule is HuggingFace's: `LlamaTokenizer.tokenize`
    /// drops a leading lone `▁` when the next piece is in `all_special_tokens`.
    /// Under [`SpmPrefixScheme::AfterEachSpecial`] there is no standalone marker
    /// for it to drop, so the rule must be inert — a sentinel and an ordinary
    /// added token in the same position encode identically, and neither needs a
    /// second code path.
    #[test]
    fn the_sentinel_rule_is_inert_under_the_llama_cpp_scheme() {
        let (tokens, scores) = rank_scored_vocab();
        let mut map = FxHashMap::default();
        map.insert("<bos>".to_string(), 2);
        map.insert("<eos>".to_string(), 1);
        // BOS = `<bos>`; `<eos>` stays an ordinary added token, exactly as in
        // `sentinel_added_tokens_swallow_the_standalone_prefix` — where the two
        // differ.
        let t = SpmTokenizer::new(tokens, scores, Some(2), None)
            .unwrap()
            .with_prefix_scheme(SpmPrefixScheme::AfterEachSpecial)
            .with_added_tokens(&map)
            .unwrap();

        assert_eq!(pieces(&t, "<bos>hello"), vec!["<bos>", "▁hello"]);
        assert_eq!(
            pieces(&t, "<eos>hello"),
            vec!["<eos>", "▁hello"],
            "the sentinel and the ordinary marker must be indistinguishable here"
        );
    }

    /// With `add_space_prefix = false` (Gemma) there is no dummy prefix to place
    /// at all, so added-token splitting must add nothing anywhere — neither a
    /// standalone marker before a leading token nor one inside any gap. With no
    /// marker to place, the scheme has nothing to choose between, so both must
    /// give the same answer.
    #[test]
    fn prefix_disabled_adds_no_marker_around_added_tokens() {
        for scheme in [SpmPrefixScheme::Once, SpmPrefixScheme::AfterEachSpecial] {
            let (tokens, scores) = rank_scored_vocab();
            let mut map = FxHashMap::default();
            map.insert("<bos>".to_string(), 2);
            let t = SpmTokenizer::new(tokens, scores, None, None)
                .unwrap()
                .with_prefix_space(false)
                .with_prefix_scheme(scheme)
                .with_added_tokens(&map)
                .unwrap();

            assert_eq!(
                pieces(&t, "<bos>hello"),
                vec!["<bos>", "h", "el", "lo"],
                "{scheme:?}"
            );
            assert_eq!(
                pieces(&t, "hello<bos>"),
                vec!["h", "el", "lo", "<bos>"],
                "{scheme:?}"
            );
        }
    }

    /// [`SpecialMode::Ordinary`] never splits, so the whole text is one stretch
    /// beginning at byte 0 and carries the prefix — the marker's literal
    /// spelling is content, and no standalone prefix piece appears.
    #[test]
    fn ordinary_mode_prefixes_the_whole_text_once() {
        let t = tok_with_added();
        let ids = t.encode_with("<bos>hello", &SpecialMode::Ordinary).unwrap();
        assert_eq!(ids, t.encode_ordinary("<bos>hello"));
        assert!(!ids.contains(&2), "the marker is content, not its own id");
    }

    /// A tokenizer built without added tokens must behave exactly as it did
    /// before matching existed — same words, and a marker string left as content.
    #[test]
    fn without_added_tokens_encoding_is_unchanged() {
        let t = tok();
        assert_eq!(pieces(&t, "hello world"), vec!["▁hello", "▁world"]);
        assert!(
            !t.encode("<bos>hello").contains(&2),
            "no matcher configured, so `<bos>` is ordinary text"
        );
    }

    /// Boundary tokens must not leak into decoded text. Ground truth is the
    /// `sentencepiece` Python package, version 0.2.0, reading Mistral's own
    /// `tokenizer.model` — the file splintr bundles as `mistral` / `mistral_v2`:
    ///
    /// ```text
    /// decode([1, 7080, 29477, 2294, 2]) -> 'hello world'
    /// id 1 '<s>'   -> ''
    /// id 2 '</s>'  -> ''
    /// ```
    ///
    /// Left unskipped, the same ids came back as `"<s> hello world</s>"`, so
    /// every generated sequence carried its own boundary markers into the text
    /// a user reads.
    ///
    /// `<unk>` (id 0) goes with them. `sp.decode([0])` is `' ⁇ '` — its
    /// `unk_surface` setting — but that is `sp.decode`'s own API rather than the
    /// HuggingFace `skip_special_tokens=True` semantics this crate follows, and
    /// which the Unigram sibling already drops `<unk>` under.
    #[test]
    fn boundary_tokens_decode_to_nothing() {
        let tok = crate::core::pretrained::from_pretrained("mistral_v2")
            .expect("mistral_v2 vocabulary loads");

        // `[7080, 29477, 2294]` is `sp.encode("hello world")` on this file.
        assert_eq!(
            tok.decode(&[1, 7080, 29477, 2294, 2]).unwrap(),
            "hello world"
        );
        assert_eq!(tok.decode(&[0]).unwrap(), "");
        assert_eq!(tok.decode(&[1]).unwrap(), "");
        assert_eq!(tok.decode(&[2]).unwrap(), "");
    }

    /// Ids a loader declares `special = true` are dropped too, on top of the
    /// vocabulary's own sentinels — that is the whole point of the set, since a
    /// chat marker's id is not spelled `<s>` and no name test would find it.
    #[test]
    fn declared_special_ids_are_skipped_on_decode() {
        let (tokens, scores) = rank_scored_vocab();
        // `<pad>` (0) declared special; `▁hello` (22) deliberately not, so the
        // set is shown to drop what it holds rather than everything.
        let t = SpmTokenizer::new(tokens, scores, None, None)
            .unwrap()
            .with_special_decode_ids([0u32].into_iter().collect());

        assert_eq!(t.decode(&[0, 22, 23]).unwrap(), "hello world");
        assert_eq!(t.decode(&[0]).unwrap(), "");
        // Replacing rather than unioning: a second call states the whole set,
        // so `▁world` starts being dropped and `<pad>` stops.
        let t = t.with_special_decode_ids([23u32].into_iter().collect());
        assert_eq!(t.decode(&[22, 23]).unwrap(), "hello");
        assert_eq!(t.decode(&[0, 22]).unwrap(), "<pad> hello");
    }

    /// The lenient decode is the strict one everywhere the strict one succeeds,
    /// and skips exactly what it refuses — an id the vocabulary does not
    /// contain, which `decode` reports as
    /// [`TokenizeError::InvalidTokenId`].
    #[test]
    fn decode_lossy_agrees_with_decode_and_skips_what_it_rejects() {
        let t = tok();
        for text in ["hello world", " hello", "hell", ""] {
            let ids = t.encode(text);
            assert_eq!(
                t.decode_lossy(&ids),
                t.decode(&ids).unwrap(),
                "input {text:?}"
            );
        }

        let unknown = t.vocab_size() as u32;
        let ids = [22, unknown, 23];
        assert!(matches!(
            t.decode(&ids),
            Err(TokenizeError::InvalidTokenId(id)) if id == unknown
        ));
        assert_eq!(t.decode_lossy(&ids), "hello world");
    }

    /// The skip rule is shared, so it applies on the lossy side too — a lossy
    /// decoder is not a way around `skip_special_tokens`.
    #[test]
    fn decode_lossy_skips_the_same_ids_decode_does() {
        let (tokens, scores) = rank_scored_vocab();
        // BOS = `<bos>` (2), EOS = `<eos>` (1), and `<unk>` (3) resolves itself.
        let t = SpmTokenizer::new(tokens, scores, Some(2), Some(1))
            .unwrap()
            .with_special_decode_ids([0u32].into_iter().collect());

        assert_eq!(t.decode_lossy(&[2, 22, 3, 23, 0, 1]), "hello world");
        assert_eq!(t.decode(&[2, 22, 3, 23, 0, 1]).unwrap(), "hello world");
    }

    /// Merging must be deterministic and must not depend on how many equal
    /// scores are in flight.
    #[test]
    fn repeated_words_tokenize_identically() {
        let t = tok();
        assert_eq!(
            pieces(&t, "hello hello hello"),
            vec!["▁hello", "▁hello", "▁hello"]
        );
    }
}
