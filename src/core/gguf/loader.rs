//! Construction: turn a [`GgufVocab`] into an [`AnyTokenizer`], dispatching on
//! the algorithm the file declares.

use rustc_hash::FxHashMap;

use super::super::any_tokenizer::{AnyTokenizer, Backend};
use super::super::bpe;
use super::super::normalizer::{NormOp, Normalizer};
use super::super::policy::SpecialPolicy;
use super::super::precompiled::Precompiled;
use super::super::sentencepiece::SentencePieceTokenizer;
use super::super::spm::{SpmPrefixScheme, SpmTokenizer};
use super::super::tokenizer::{Tokenizer, GPT2_PATTERN, LLAMA3_PATTERN, QWEN2_PATTERN};
use super::super::wordpiece::WordPieceTokenizer;
use super::error::GgufVocabError;
use super::vocab::GgufVocab;

/// Build a tokenizer from the `tokenizer.ggml.*` metadata of a GGUF file.
///
/// `tokenizer.ggml.model` names which tokenization *algorithm* the vocabulary
/// was built with, and the four values in circulation are genuinely different
/// algorithms over superficially similar data:
///
/// | `model` | scores | merges | algorithm |
/// |---|---|---|---|
/// | `bert` | ignored | — | WordPiece, greedy longest match with `##` |
/// | `t5` | log-probabilities | — | Unigram, Viterbi maximum-sum segmentation |
/// | `llama` | **merge ranks** | — | SentencePiece BPE, best adjacent pair first |
/// | `gpt2` | — | **required** | byte-level BPE over an explicit merge list |
///
/// Collapsing these is not a rounding error. Running Unigram Viterbi over a
/// `llama` vocabulary maximises the wrong objective — its scores are ranks, so
/// cheap early-id fragments outscore whole words and `▁sourdough` becomes
/// `▁s|ou|rd|ou|gh`. Running it over a `gpt2` vocabulary is worse still: those
/// files carry no scores at all, so every token scores equally and the `merges`
/// list that defines the tokenizer is never read.
///
/// Neither failure is visible downstream. The ids are in range, the embedding
/// shapes are right, and retrieval quietly degrades to near-noise. So the
/// routing below dispatches on the declared model and rejects what it cannot
/// honour rather than guessing.
///
/// The returned tokenizer's [`SpecialPolicy`] owns the boundary tokens: the
/// backends are built with no BOS/EOS of their own, so `add_bos_token` /
/// `add_eos_token` are honoured in exactly one place, whichever algorithm the
/// file declares.
pub fn from_gguf_vocab(vocab: GgufVocab) -> Result<AnyTokenizer, GgufVocabError> {
    if vocab.tokens.is_empty() {
        return Err(GgufVocabError::EmptyVocab);
    }
    match vocab.model.as_str() {
        "bert" => build_wordpiece(vocab),
        "t5" => build_unigram(vocab),
        "llama" => build_spm(vocab),
        "gpt2" => build_byte_level_bpe(vocab),
        other => Err(GgufVocabError::UnsupportedModel(other.to_owned())),
    }
}

/// `bert`: WordPiece, wrapped in the `[CLS]`/`[SEP]` the vocabulary names.
fn build_wordpiece(mut vocab: GgufVocab) -> Result<AnyTokenizer, GgufVocabError> {
    // Control and user-defined tokens are read from the ORIGINAL strings, before
    // normalization: `normalize_wordpiece_vocab` rewrites unbracketed pieces to
    // `##X`, which would spell one of these as something no input ever
    // contains. The rewrite is index-preserving, so these surface strings and
    // the normalized vocab below agree on every id.
    let mut named = special_token_map(&vocab, &vocab.tokens);

    // Convert a SentencePiece-marked vocab to WordPiece convention first —
    // see `normalize_wordpiece_vocab`. Everything below (the [UNK] lookup
    // and the uncased heuristic) matches against plain strings like "the",
    // so it MUST run on the normalized vocab or it silently misfires.
    let tokens = normalize_wordpiece_vocab(std::mem::take(&mut vocab.tokens));

    let unk_token_id = find_special_token_id(&tokens, &vocab, "[UNK]", 0);

    // GGUF has no standard key for casing, so heuristic: a vocab holding
    // lowercase "the" but not "The" is uncased.
    let do_lower_case = tokens.iter().any(|t| t == "the") && !tokens.iter().any(|t| t == "The");
    // Accent stripping is likewise absent from GGUF metadata, so it is left at
    // the constructor's seed (= `do_lower_case`). That is exactly HuggingFace's
    // own rule for a `BertNormalizer` whose `strip_accents` is absent/`null`
    // (`strip_accents.unwrap_or(lowercase)`), which is the shape BERT-family
    // checkpoints ship — so an unspecified GGUF lands on the same behavior its
    // `tokenizer.json` would have produced.

    // The named map is how a caller asks for `[CLS]` by name; BERT-family models
    // need those ids to assemble the pairs their heads were trained on. These are
    // merged over the control/user-defined tokens rather than replacing them: `lookup_special`
    // can resolve an id from metadata that the type array never flags, so dropping
    // it would lose a lookup that works today.
    for name in ["[UNK]", "[PAD]", "[CLS]", "[SEP]"] {
        if let Some(id) = lookup_special(&tokens, &vocab, name) {
            named.insert(name.to_owned(), id);
        }
    }

    // Read before `named` is moved into the policy below.
    let (cls, sep) = (named.get("[CLS]").copied(), named.get("[SEP]").copied());

    // Which ids decode drops. The backend tests ids, not surface strings, so the
    // file's OWN declared specials go in — otherwise a vocabulary that spells
    // them `<s>`/`</s>`/`<unk>` leaks them into decoded text purely for not
    // being bracketed. The selection matches what the other dialects drop:
    // `t5` and `llama` skip bos/eos/unk (`SentencePieceTokenizer` /
    // `SpmTokenizer`), and BERT's pad/cls/sep join them because those are the
    // ids a BERT file states its boundaries with. Deliberately NOT every
    // `token_type == 3` CONTROL id: that array drives added-token *matching*
    // (see `special_token_map`), and no other dialect silences everything it
    // flags, so doing it here would invent a broader policy for one backend.
    // The four bracketed lookups come from `named`, which the loop above filled
    // with `lookup_special` — so an id stated only in the metadata counts too.
    let mut special_decode: rustc_hash::FxHashSet<u32> = [vocab.bos_token_id, vocab.eos_token_id]
        .into_iter()
        .flatten()
        .collect();
    special_decode.extend(
        ["[UNK]", "[PAD]", "[CLS]", "[SEP]"]
            .into_iter()
            .filter_map(|name| named.get(name).copied()),
    );

    let eos_token_id = vocab.eos_token_id.unwrap_or(0);
    let backend = Backend::WordPiece(
        WordPieceTokenizer::new(tokens, unk_token_id, 200, do_lower_case)
            .with_added_tokens(&named)?
            .with_special_decode_ids(special_decode),
    );

    // `add_bos_token` / `add_eos_token` are not how a BERT file states its
    // boundaries — the `[CLS]`/`[SEP]` ids are — so the template is built from
    // those two ids, through the same constructor the `tokenizer.json` path
    // uses. Without it, `encode` on a GGUF returned bare content tokens while
    // the *same model's* `tokenizer.json` wrapped them, and a `Pooling::Cls`
    // consumer silently read a content token at position 0 as the sentence
    // vector. Measured on all-MiniLM-L6-v2 (`tokenizers` 0.22.1 and llama.cpp's
    // WPM path with `add_special` set): `"hello world"` → `[101, 7592, 2088, 102]`.
    //
    // A vocabulary that names neither keeps the identity policy: there is no id
    // to place, and inventing one would be worse than placing none.
    let policy = match (cls, sep) {
        (Some(cls), Some(sep)) => SpecialPolicy::cls_sep(cls, sep, Some(eos_token_id), named),
        _ => SpecialPolicy::boundary(None, None, Some(eos_token_id), named),
    };
    Ok(AnyTokenizer::new(backend, policy))
}

/// `t5`: true Unigram. Scores are log-probabilities and Viterbi is correct.
fn build_unigram(mut vocab: GgufVocab) -> Result<AnyTokenizer, GgufVocabError> {
    // One map, two uses: matched in the input by the backend, and resolvable by
    // name through the policy. Neither substitutes for the other.
    let specials = special_token_map(&vocab, &vocab.tokens);
    let tokens = std::mem::take(&mut vocab.tokens);
    // GGUF stores scores as `f32`; the widening is exact and lets the Viterbi
    // compare partial sums at the `f64` precision HF `tokenizers` uses.
    let scores: Vec<f64> = vocab
        .scores
        .take()
        .unwrap_or_default()
        .into_iter()
        .map(f64::from)
        .collect();
    let eos_token_id = vocab.eos_token_id.unwrap_or(2);
    let prefix_space = unigram_prefix_space(&vocab);
    let normalizer = unigram_normalizer(vocab.precompiled_charsmap.as_deref());

    // Which ids decode drops, chosen exactly as the `llama` arm below chooses:
    // the three ids the FILE states, and nothing broader. The backend already
    // skips its own BOS/EOS/`<unk>` fields, but two of those three are not the
    // same thing as what the file declared — BOS is passed as `None` here
    // (boundaries are the policy's), and the backend resolves `<unk>` by
    // spelling, so a vocabulary that states an `unknown_token_id` for a piece
    // spelled anything else is not covered. Both leaked into `decode()` output.
    //
    // Deliberately NOT every `token_type == 3` (CONTROL) id in `specials`: that
    // array drives added-token *matching*, not decode skipping, and both sibling
    // arms decline the same broadening for the same reason. The `bert` arm adds
    // `[PAD]`/`[CLS]`/`[SEP]` on top because those are the ids a BERT file states
    // its boundaries with; a `t5` file states none of them, so the `llama`
    // precedent — bos/eos/unk — is the right one to follow here.
    let special_decode: rustc_hash::FxHashSet<u32> = [
        vocab.bos_token_id,
        vocab.eos_token_id,
        vocab.unknown_token_id,
    ]
    .into_iter()
    .flatten()
    .collect();

    // `None` for BOS: boundary tokens are placed by the policy, so the backend
    // must not also prepend one. The backend's `eos` is not a boundary — it only
    // drives decode-skipping and `is_eos` — so it takes the resolved id.
    let backend = Backend::Unigram(
        SentencePieceTokenizer::new(tokens, scores, None, eos_token_id)?
            .with_special_decode_ids(special_decode)
            .with_normalizer(normalizer)
            .with_prefix_space(prefix_space)
            .with_remove_extra_whitespaces(remove_extra_whitespaces(&vocab))
            .with_added_tokens(&specials)?,
    );

    Ok(AnyTokenizer::new(
        backend,
        boundary_policy(&vocab, eos_token_id, true, true, specials),
    ))
}

/// `llama`: SentencePiece BPE. Scores are merge ranks, so segmentation is
/// repeated best-adjacent-pair merging, not Viterbi.
fn build_spm(mut vocab: GgufVocab) -> Result<AnyTokenizer, GgufVocabError> {
    // One map, two uses: matched in the input by the backend, and resolvable by
    // name through the policy. A chat template needs both — the marker spliced
    // into the prompt text must survive, and its id must be reachable by name.
    let specials = special_token_map(&vocab, &vocab.tokens);
    let tokens = std::mem::take(&mut vocab.tokens);
    let scores = vocab.scores.take().unwrap_or_default();
    let eos_token_id = vocab.eos_token_id.unwrap_or(2);

    // `None` for both special ids: the policy places boundaries, so the backend
    // must not insert any of its own.
    // llama.cpp is the reference for a GGUF file — it is what actually runs
    // these vocabularies — so the dummy prefix follows its `is_prev_special`
    // rule: prepended to the first text fragment and to every fragment after a
    // control token, with no standalone marker before a leading one. Encoding
    // `"<start_of_turn>hi"` must be `[<start_of_turn>, ▁hi]`; under the
    // HuggingFace scheme it would be `[▁, <start_of_turn>, hi]`, three pieces
    // the model never saw in that arrangement.
    // Which ids decode drops. The backend resolves `<unk>` by name itself, but
    // its BOS/EOS fields stay `None` here — they are the policy's — so the ids
    // the file states are declared as decode-skipped instead, which is
    // decode-only and cannot reach the encode path. Without them a generated
    // sequence carried a literal `<s>`/`</s>` into the decoded text. The file's
    // `unknown_token_id` joins them for a vocabulary that spells its unknown
    // piece as something other than `<unk>`. Deliberately NOT every CONTROL /
    // USER_DEFINED id in `specials`: that array drives added-token *matching*,
    // and the `bert` arm above declines the same broadening for the same reason.
    let special_decode: rustc_hash::FxHashSet<u32> = [
        vocab.bos_token_id,
        vocab.eos_token_id,
        vocab.unknown_token_id,
    ]
    .into_iter()
    .flatten()
    .collect();

    let backend = Backend::Spm(
        SpmTokenizer::new(tokens, scores, None, None)?
            .with_prefix_space(add_space_prefix(&vocab, true))
            .with_prefix_scheme(SpmPrefixScheme::AfterEachSpecial)
            .with_special_decode_ids(special_decode)
            .with_added_tokens(&specials)?,
    );

    Ok(AnyTokenizer::new(
        backend,
        boundary_policy(&vocab, eos_token_id, true, false, specials),
    ))
}

/// `gpt2`: byte-level BPE. The `merges` list defines the tokenizer, so a
/// file without one cannot be tokenized correctly and is refused.
fn build_byte_level_bpe(mut vocab: GgufVocab) -> Result<AnyTokenizer, GgufVocabError> {
    let merges = vocab.merges.take().ok_or(GgufVocabError::MissingMerges)?;
    let tokens = std::mem::take(&mut vocab.tokens);

    // Token strings are already byte-level-encoded ("Ġhello"); the encoder is
    // keyed on those bytes because encode byte-level-encodes before lookup.
    let mut encoder: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
    encoder.reserve(tokens.len());
    for (id, token) in tokens.iter().enumerate() {
        encoder.insert(token.as_bytes().to_vec(), id as u32);
    }

    let merge_ranks = build_merge_ranks(&merges, &tokens);
    // Two distinct uses of the same control/user-defined tokens: `specials`
    // teaches the encoder to match them in the input, `named` lets a caller
    // look one up by name. Neither substitutes for the other.
    let specials = special_token_map(&vocab, &tokens);
    let named = specials.clone();
    let patterns = byte_level_pattern(vocab.pre.as_deref())?;

    let eos_token_id = vocab.eos_token_id.unwrap_or(0);

    let backend = Backend::Bpe(
        Tokenizer::new_byte_level_chain(encoder, specials, patterns)?
            .with_merge_ranks(merge_ranks)
            .with_added_token_matching(true),
    );

    Ok(AnyTokenizer::new(
        backend,
        boundary_policy(&vocab, eos_token_id, false, false, named),
    ))
}

/// Resolve the file's `add_bos_token` / `add_eos_token` flags into a policy.
///
/// Applied here rather than inside each backend because the backends disagree:
/// the Unigram tokenizer would prepend BOS and never append EOS, the BPE
/// tokenizer neither. Reading the file's own flags in one place makes the file
/// authoritative for every architecture instead of inheriting whichever
/// convention a backend happened to implement.
///
/// This matters beyond tidiness for last-token pooling: Qwen3-Embedding is
/// trained to summarise a sequence into its final `<|endoftext|>` position, so a
/// dropped EOS means pooling reads a content token instead of the summary.
///
/// A flag that is set but whose id the file never states adds nothing — there is
/// no id to add — while `eos_id` still reports the vocabulary's end-of-sequence
/// token so generation can stop on it.
fn boundary_policy(
    vocab: &GgufVocab,
    eos_token_id: u32,
    bos_default: bool,
    eos_default: bool,
    named: FxHashMap<String, u32>,
) -> SpecialPolicy {
    let bos = vocab
        .add_bos_token
        .unwrap_or(bos_default)
        .then_some(vocab.bos_token_id)
        .flatten();
    let eos = vocab
        .add_eos_token
        .unwrap_or(eos_default)
        .then_some(vocab.eos_token_id)
        .flatten();
    SpecialPolicy::boundary(bos, eos, Some(eos_token_id), named)
}

// ── Multi-pass pre-tokenizer expressions ─────────────────────────────────────
//
// Each constant below is one entry of a llama.cpp `regex_exprs` list, copied
// character for character from `llm_tokenizer_bpe`'s constructor. They are only
// ever used as part of an ordered list — see [`byte_level_pattern`] — because a
// list is applied pass by pass, each pass subdividing the previous pass's
// pieces, and no single alternation reproduces that.

/// `LLAMA_VOCAB_PRE_TYPE_FALCON` pass 1, llama-vocab.cpp:344.
///
/// Punctuation and the ASCII symbols. llama.cpp reaches this through its
/// collapsed-text path (`unicode.cpp:1012-1066`), where `\p{P}` expands to the
/// PUNCTUATION category plus its sub-128 members `!-#%-*,-/:-;?-@[-]_{}` — the
/// same set a Unicode-aware engine matches for `\p{P}` directly. The explicit
/// `$+<=>^~|` and `` ` `` are the ASCII half of `\p{S}`, spelled out because the
/// list deliberately does NOT take non-ASCII symbols.
const FALCON_PUNCT_PATTERN: &str = r"[\p{P}\$\+<=>\^~\|`]+";

/// `LLAMA_VOCAB_PRE_TYPE_FALCON` pass 3, llama-vocab.cpp:346.
///
/// Cuts a digit run into groups of three from the LEFT (the pieces pass 2 left
/// behind are re-matched left to right, and the remainder trails as a gap).
const FALCON_DIGIT_TRIPLE_PATTERN: &str = r"[0-9][0-9][0-9]";

/// A single Unicode digit — `LLAMA_VOCAB_PRE_TYPE_STARCODER` and friends' pass 1
/// (llama-vocab.cpp:357) and `LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_CODER`'s pass 5
/// (llama-vocab.cpp:339).
const SINGLE_DIGIT_PATTERN: &str = r"\p{N}";

/// A lone CR or LF — `DEEPSEEK_LLM` pass 1 (llama-vocab.cpp:310) and
/// `DEEPSEEK_CODER` pass 1 (llama-vocab.cpp:335).
///
/// Running first, it isolates every line break into a span of its own, so no
/// later pass in either list ever sees a span that contains one.
const LINE_BREAK_PATTERN: &str = r"[\r\n]";

/// CJK/Hangul block runs — `DEEPSEEK_LLM` pass 5 (llama-vocab.cpp:314) and
/// `DEEPSEEK_CODER` pass 4 (llama-vocab.cpp:338), byte-identical to each other.
///
/// Transcribed verbatim, including the `\u{0800}`-`\u{4E00}` range that spans far
/// more than the CJK blocks the name suggests. Narrowing it to what it "means"
/// would change the split.
const DEEPSEEK_CJK_PATTERN: &str = r"[一-龥ࠀ-一가-퟿]+";

/// `LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_CODER` pass 2, llama-vocab.cpp:336.
const DEEPSEEK_CODER_LETTER_PATTERN: &str = r"\s?\p{L}+";

/// `LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_CODER` pass 3, llama-vocab.cpp:337.
const DEEPSEEK_CODER_PUNCT_PATTERN: &str = r"\s?\p{P}+";

/// `LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM` pass 2, llama-vocab.cpp:311.
///
/// An explicit enumeration of letter ranges rather than `\p{L}`, and NOT
/// interchangeable with it: it omits every script outside the list (Arabic,
/// Hebrew, Devanagari, Thai, Hiragana, Han …), which the pass therefore leaves
/// to fall through as gaps.
const DEEPSEEK_LLM_LETTER_PATTERN: &str = r"\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-\u{1F7D}ᾀ-ᾴᾶ-ᾼ\u{1FBE}ῂ-ῄῆ-ῌῐ-\u{1FD3}ῖ-\u{1FDB}ῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤ\u{2126}ℨ\u{212A}-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+";

/// `LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM` pass 3, llama-vocab.cpp:312.
///
/// ASCII punctuation/symbols plus their fullwidth and CJK counterparts, again
/// enumerated rather than expressed as `\p{P}`.
const DEEPSEEK_LLM_PUNCT_PATTERN: &str = r"\s?[!-/:-~！-／：-～‘-‟　-。]+";

/// `LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM` pass 4, llama-vocab.cpp:313.
///
/// The `$` is end-of-span, not end-of-line: llama.cpp matches each pass against
/// the span in isolation (`unicode.cpp:487`) with no multiline flag. Pass 1 of
/// this same list already isolated every CR/LF, so by the time this runs no span
/// holds a line break and the two readings of `$` cannot diverge.
const DEEPSEEK_LLM_TRAILING_SPACE_PATTERN: &str = r"\s+$";

/// `LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM` pass 6, llama-vocab.cpp:315.
const DEEPSEEK_LLM_DIGITS_PATTERN: &str = r"\p{N}+";

/// The ordered pre-tokenizer expressions a byte-level BPE vocabulary was built
/// with — llama.cpp's `regex_exprs` list for the named pre-tokenizer.
///
/// `tokenizer.ggml.pre` names the pre-tokenizer, and the choice is not
/// cosmetic: it decides where text is cut before any merge is applied, so two
/// patterns over the same vocabulary and merge list produce different ids.
/// llama.cpp keeps the same table (`llama_vocab::impl::load`, the
/// `LLAMA_VOCAB_PRE_TYPE_*` mapping); this mirrors the subset that reaches an
/// embedding model.
///
/// Concretely, `qwen2` splits digits one at a time and keeps letter runs
/// whole, while the GPT-2 family splits ` ?\p{N}+` runs and has no `(?i:)`
/// contraction handling — so tokenizing jina-v2-code's vocabulary with Qwen's
/// pattern silently mis-segments every number and contraction in the corpus.
///
/// Most names yield a list of ONE expression, which is the ordinary single-regex
/// split. The rest yield several, applied in sequence — each pass re-matching
/// the pieces the previous pass produced and cutting them finer, never merging
/// and never re-reading the whole text (`unicode_regex_split`, unicode.cpp:990).
/// A list of N expressions is therefore NOT their alternation: `falcon`'s
/// three-expression list first isolates punctuation runs, then applies the GPT-2
/// split inside each remaining piece, then chops digit runs into threes.
///
/// An unrecognised name is refused rather than defaulted: a wrong split is
/// invisible downstream, and every id it produces is still in range.
pub(super) fn byte_level_pattern(
    pre: Option<&str>,
) -> Result<&'static [&'static str], GgufVocabError> {
    // Every name below was traced through llama.cpp twice: the `pre` string to a
    // `LLAMA_VOCAB_PRE_TYPE_*` value in `llama_vocab::impl::load`, and that value
    // to the literal `regex_exprs` list in `llm_tokenizer_bpe`'s constructor. A
    // name is listed only when the full list its enum value yields is reproduced
    // here expression for expression, in order.
    //
    // `default` is llama.cpp's fallback pre-tokenizer, which is the GPT-2 split.
    match pre.unwrap_or("default") {
        // ── QWEN2_PATTERN ────────────────────────────────────────────────────
        // All of these reach a `regex_exprs` list of one expression, at
        // llama-vocab.cpp:371-379 (`STABLELM2`/`QWEN2`/`HUNYUAN`/`SOLAR_OPEN`
        // share one `case` label) or llama-vocab.cpp:471-476 (`GROK_2`, whose
        // string is byte-identical to the former's). llama.cpp writes the
        // contraction group case-expanded as `(?:'[sS]|'[tT]|…)`; the comment
        // directly above each list records the tokenizer.json original as
        // `(?i:'s|'t|…)`, which is this constant character for character.
        //
        //   qwen2            → QWEN2       llama-vocab.cpp:1953 → :371
        //   deepseek-r1-qwen → QWEN2       llama-vocab.cpp:1954 → :371
        //   kormo            → QWEN2       llama-vocab.cpp:1955 → :371
        //   megrez           → QWEN2       llama-vocab.cpp:2027 → :371
        //   stablelm2        → STABLELM2   llama-vocab.cpp:1963 → :371
        //   hunyuan          → HUNYUAN     llama-vocab.cpp:2062 → :371
        //   solar-open       → SOLAR_OPEN  llama-vocab.cpp:2090 → :371
        //   grok-2           → GROK_2      llama-vocab.cpp:2078 → :471
        "qwen2" | "deepseek-r1-qwen" | "kormo" | "megrez" | "stablelm2" | "hunyuan"
        | "solar-open" | "grok-2" => Ok(&[QWEN2_PATTERN]),

        // ── GPT2_PATTERN ─────────────────────────────────────────────────────
        // `GPT2`/`MPT`/`OLMO`/`JAIS`/`TRILLION`/`GRANITE_DOCLING` share one `case`
        // label at llama-vocab.cpp:361-369 whose list is the single expression
        // `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)`
        // — this constant minus its trailing `|\s+`. That is not a difference in
        // splitting: llama.cpp matches this exact string in
        // `unicode_regex_split_custom` (unicode.cpp:759) and hands it to the
        // hand-written `unicode_regex_split_custom_gpt2`, whose whitespace
        // fallthrough (unicode.cpp:317-322, commented `// regex: \s+`) emits the
        // bare run the written alternation would drop. The `\s+(?!\S)` branch
        // above it (unicode.cpp:311-315) fires only for a run of >1 whitespace
        // followed by more text, exactly as the lookahead requires.
        //
        //   default (and an absent key) → DEFAULT, kept from the pre-existing
        //   mapping; llama-vocab.cpp:1883.
        //
        //   gpt-2           → GPT2             llama-vocab.cpp:1925 → :361
        //   phi-2           → GPT2             llama-vocab.cpp:1926 → :361
        //   jina-es         → GPT2             llama-vocab.cpp:1927 → :361
        //   jina-de         → GPT2             llama-vocab.cpp:1928 → :361
        //   gigachat        → GPT2             llama-vocab.cpp:1929 → :361
        //   jina-v2-es      → GPT2             llama-vocab.cpp:1930 → :361
        //   jina-v2-de      → GPT2             llama-vocab.cpp:1931 → :361
        //   a.x-4.0         → GPT2             llama-vocab.cpp:1932 → :361
        //   mellum          → GPT2             llama-vocab.cpp:1933 → :361
        //   modern-bert     → GPT2             llama-vocab.cpp:1934 → :361
        //   jina-v1-en      → GPT2             llama-vocab.cpp:1940 → :361
        //   jina-v2-code    → GPT2             llama-vocab.cpp:1941 → :361
        //   roberta-bpe     → GPT2             llama-vocab.cpp:1942 → :361
        //   exaone4         → GPT2             llama-vocab.cpp:2013 → :361
        //   mpt             → MPT              llama-vocab.cpp:1919 → :362
        //   olmo            → OLMO             llama-vocab.cpp:1966 → :363
        //   jais            → JAIS             llama-vocab.cpp:1988 → :364
        //   trillion        → TRILLION         llama-vocab.cpp:2044 → :365
        //   granite-docling → GRANITE_DOCLING  llama-vocab.cpp:2048 → :366
        //
        // `jina-v2-en` has no entry in llama.cpp's table; it is kept from the
        // pre-existing mapping rather than re-derived.
        "default" | "gpt-2" | "phi-2" | "roberta-bpe" | "jina-v1-en" | "jina-v2-en"
        | "jina-v2-es" | "jina-v2-de" | "jina-v2-code" | "jina-es" | "jina-de" | "gigachat"
        | "a.x-4.0" | "mellum" | "modern-bert" | "exaone4" | "mpt" | "olmo" | "jais"
        | "trillion" | "granite-docling" => Ok(&[GPT2_PATTERN]),

        // ── LLAMA3_PATTERN ───────────────────────────────────────────────────
        // `LLAMA3`, `DBRX`/`SMAUG` (one `case` label) and `CHATGLM4` each reach a
        // `regex_exprs` list of one expression, and all three strings are
        // byte-identical to each other. llama.cpp writes the contraction group
        // case-expanded as `(?:'[sS]|'[tT]|…)`; the comment above the `LLAMA3`
        // list records the tokenizer.json original as `(?i:'s|'t|…)`, which is
        // this constant character for character.
        //
        //   llama-bpe → LLAMA3    llama-vocab.cpp:1894 → :283
        //   llama3    → LLAMA3    (splintr alias for the same vocabulary)
        //   dbrx      → DBRX      llama-vocab.cpp:1970 → :301
        //   smaug-bpe → SMAUG     llama-vocab.cpp:1973 → :302
        //   glm4      → CHATGLM4  llama-vocab.cpp:1981 → :395
        "llama-bpe" | "llama3" | "dbrx" | "smaug-bpe" | "glm4" => Ok(&[LLAMA3_PATTERN]),

        // ── Multi-pass lists ─────────────────────────────────────────────────
        // From here down the enum value yields several expressions, applied in
        // order, each subdividing the previous pass's pieces.
        //
        // Where a list's entry is llama.cpp's GPT-2 string
        // (`'s|'t|…|\s+(?!\S)`, no trailing `|\s+`), [`GPT2_PATTERN`] is the
        // expression used, for the reason recorded above the GPT-2 arm:
        // llama.cpp intercepts that exact string in `unicode_regex_split_custom`
        // (unicode.cpp:759) and runs `unicode_regex_split_custom_gpt2`, whose
        // whitespace fallthrough (unicode.cpp:317-322, commented `// regex:
        // \s+`) supplies the bare run the written alternation omits.

        //   falcon → FALCON   llama-vocab.cpp:1917 → :342-347
        //
        // Pass 1 punctuation/symbol runs, pass 2 the GPT-2 split, pass 3 digit
        // triples. The order matters: pass 3 only ever sees the ` ?\p{N}+` runs
        // pass 2 produced, so `1234` becomes `123` + `4` rather than being cut
        // from the right.
        "falcon" => Ok(&[
            FALCON_PUNCT_PATTERN,
            GPT2_PATTERN,
            FALCON_DIGIT_TRIPLE_PATTERN,
        ]),

        //   starcoder  → STARCODER  llama-vocab.cpp:1923 → :349-359
        //   refact     → REFACT     llama-vocab.cpp:1947 → :350
        //   command-r  → COMMAND_R  llama-vocab.cpp:1950 → :351
        //   smollm     → SMOLLM     llama-vocab.cpp:1998 → :352
        //   codeshell  → CODESHELL  llama-vocab.cpp:2002 → :353
        //   exaone     → EXAONE     llama-vocab.cpp:2011 → :354
        //   minerva-7b → MINERVA    llama-vocab.cpp:2025 → :355
        //
        // Seven enum values sharing one `case` label at llama-vocab.cpp:349-359.
        // Isolating single digits FIRST is what makes this differ from the plain
        // GPT-2 split: pass 2's ` ?\p{N}+` can then never span two digits, and a
        // leading space stays with the first digit only.
        "starcoder" | "refact" | "command-r" | "smollm" | "codeshell" | "exaone" | "minerva-7b" => {
            Ok(&[SINGLE_DIGIT_PATTERN, GPT2_PATTERN])
        }

        //   deepseek-llm → DEEPSEEK_LLM  llama-vocab.cpp:1900 → :308-317
        //
        // Six passes. Note there is no catch-all expression anywhere in the
        // list: text no pass matches (Cyrillic-adjacent scripts, emoji, …)
        // survives as an unmatched gap and is handed to BPE whole, which is
        // exactly what llama.cpp does with it.
        "deepseek-llm" => Ok(&[
            LINE_BREAK_PATTERN,
            DEEPSEEK_LLM_LETTER_PATTERN,
            DEEPSEEK_LLM_PUNCT_PATTERN,
            DEEPSEEK_LLM_TRAILING_SPACE_PATTERN,
            DEEPSEEK_CJK_PATTERN,
            DEEPSEEK_LLM_DIGITS_PATTERN,
        ]),

        //   deepseek-coder → DEEPSEEK_CODER  llama-vocab.cpp:1904 → :333-341
        //
        // Five passes, and the digit pass is `\p{N}` (one digit at a time), not
        // `deepseek-llm`'s `\p{N}+`.
        "deepseek-coder" => Ok(&[
            LINE_BREAK_PATTERN,
            DEEPSEEK_CODER_LETTER_PATTERN,
            DEEPSEEK_CODER_PUNCT_PATTERN,
            DEEPSEEK_CJK_PATTERN,
            SINGLE_DIGIT_PATTERN,
        ]),

        other => Err(GgufVocabError::UnsupportedPreTokenizer(other.to_owned())),
    }
}

/// SentencePiece `add_dummy_prefix` (`tokenizer.ggml.add_space_prefix`).
fn add_space_prefix(vocab: &GgufVocab, default: bool) -> bool {
    vocab.add_space_prefix.unwrap_or(default)
}

/// Whether a Unigram (`t5`) vocabulary escapes a word-boundary marker before
/// the FIRST word of the input.
///
/// Not simply `add_space_prefix`. llama.cpp's `llm_tokenizer_ugm::normalize`
/// emits the marker at the start of every non-whitespace run when EITHER the
/// dummy prefix is requested OR `remove_extra_whitespaces` is set:
///
/// ```text
/// if ((shall_prepend_space && !is_space_prepended) || shall_merge_spaces) { … }
/// ```
///
/// `jina-embeddings-v3` is the case that separates the two: it declares
/// `add_space_prefix = false` with `remove_extra_whitespaces = true`, so the
/// reference marks its first word after all. Honouring only the first flag
/// leaves the leading word unmarked, and since a Unigram vocabulary stores
/// `▁Rust` but not bare `Rust`, Viterbi shatters it into fragments. Measured
/// against llama.cpp on the same file: 20 tokens instead of 19, and the pooled
/// embedding drifts to cosine 0.940 — wrong, yet nowhere near broken enough to
/// fail a retrieval check.
///
/// The rule is Unigram-only. The `llama` (SentencePiece BPE) path is a
/// different tokenizer upstream and reads `add_space_prefix` alone.
pub(super) fn unigram_prefix_space(vocab: &GgufVocab) -> bool {
    add_space_prefix(vocab, true) || remove_extra_whitespaces(vocab)
}

/// The normalizer a `t5` (Unigram) vocabulary runs before pre-tokenization.
///
/// `tokenizer.ggml.precompiled_charsmap` is SentencePiece's own normalization
/// table, and llama.cpp applies it at the top of `llm_tokenizer_ugm::normalize`
/// — before the dummy prefix, before space-run merging, before Viterbi. Skipping
/// it does not merely change spacing: the table folds tab, newline, NBSP, ZWJ
/// and the fullwidth punctuation block onto forms the vocabulary actually
/// contains, so an unnormalized `，` or `\t` matches no piece and comes out as
/// `<unk>`. Measured on bge-m3, that is 10 of 40 reference cases wrong, every
/// one of them an `<unk>` where the reference has a real token.
///
/// This is the same [`NormOp::Precompiled`] step the `tokenizer.json` loader
/// builds from the base64 `precompiled_charsmap` of a `Precompiled` normalizer —
/// one decoder, two carriers of the same blob.
///
/// A blob that does not parse yields an empty pipeline rather than an error: the
/// table is an optimization of the vocabulary's own coverage, and refusing to
/// build a tokenizer that is otherwise complete would be worse than normalizing
/// nothing. The other dialects get no charsmap step — llama.cpp applies this
/// table only in its `ugm` (Unigram) tokenizer.
fn unigram_normalizer(charsmap: Option<&[u8]>) -> Normalizer {
    let ops = match charsmap.and_then(Precompiled::from_bytes) {
        Some(pc) => vec![NormOp::Precompiled(pc)],
        None => Vec::new(),
    };
    Normalizer::new(ops)
}

/// `tokenizer.ggml.remove_extra_whitespaces`, defaulting to false as upstream does.
///
/// Two things read it, both on the `t5` path: [`unigram_prefix_space`] above,
/// and the backend's own space-run merging — llama.cpp's `shall_merge_spaces`,
/// which turns a run of spaces into one boundary marker instead of one per
/// space. Set, `"   "` is the same single `▁` piece as `" "`; unset, each space
/// is its own marker. Either way the spaces are *pieces*, never discarded.
fn remove_extra_whitespaces(vocab: &GgufVocab) -> bool {
    vocab.remove_extra_whitespaces.unwrap_or(false)
}

/// Build a bytes → merge-rank map from the GGUF `merges` list.
///
/// Ranks are assigned so the base alphabet (vocab entries that are never a merge
/// result) always merges before any real merge, then merges follow in list
/// order — the shared construction in [`bpe::merge_ranks`], which the
/// HuggingFace `tokenizer.json` loader also uses, because merge priority is
/// independent of token id in both formats.
pub(super) fn build_merge_ranks(
    merges: &[String],
    tokens: &[String],
) -> crate::core::token_bytes::Encoder {
    // Each entry is "a b"; byte-level tokens encode real spaces as `Ġ`, so the
    // first space is always the separator.
    let merged: Vec<String> = merges.iter().map(|s| s.replacen(' ', "", 1)).collect();
    bpe::merge_ranks(merged, tokens.iter().map(String::as_str))
}

/// CONTROL in the GGUF `tokenizer.ggml.token_type` enum.
const CONTROL_TOKEN_TYPE: u32 = 3;

/// USER_DEFINED in the GGUF `tokenizer.ggml.token_type` enum.
const USER_DEFINED_TOKEN_TYPE: u32 = 4;

/// Map of special/control/user-defined token strings to ids, for added-token
/// matching.
///
/// llama.cpp partitions both CONTROL and USER_DEFINED tokens out of the input
/// text as literal strings before merging even begins — neither ever
/// participates in the merge loop. Selecting CONTROL alone misses vocabularies
/// that spell their added tokens as USER_DEFINED, e.g. Gemma's whitespace-run
/// pieces (`"  "`, `"   "`, ...), which then never match and silently fall
/// through to be re-merged from single-space pieces instead.
fn special_token_map(vocab: &GgufVocab, tokens: &[String]) -> FxHashMap<String, u32> {
    let mut specials = FxHashMap::default();

    if let Some(types) = vocab.token_type.as_ref() {
        // Driven from the tokens so an id the type array covers but the vocab
        // does not is skipped rather than indexed.
        for (id, token) in tokens.iter().enumerate() {
            if matches!(
                types.get(id),
                Some(&CONTROL_TOKEN_TYPE) | Some(&USER_DEFINED_TOKEN_TYPE)
            ) {
                specials.insert(token.clone(), id as u32);
            }
        }
    }
    specials
}

/// The SentencePiece word-boundary marker (U+2581 LOWER ONE EIGHTH BLOCK).
const SP_WORD_BOUNDARY: char = '\u{2581}';

/// Rewrite a GGUF BERT vocab that uses SentencePiece word-boundary markers into
/// the WordPiece convention [`WordPieceTokenizer`] expects.
///
/// Some GGUF converters store a WordPiece vocab with SentencePiece marking:
/// word-INITIAL pieces get a leading `▁` and continuation pieces are bare,
/// instead of BERT's bare-initial / `##`-continuation. `nomic-embed-text-v1.5`
/// is one such file — 23695 of its 30522 tokens carry `▁` and **zero** carry
/// `##`, so `vocab[1996]` is `"▁the"` where bert-base-uncased has `"the"`, and
/// `vocab[2015]` is `"s"` where bert-base-uncased has `"##s"`.
///
/// Handing those strings to `WordPieceTokenizer` unchanged means no word ever
/// matches its own vocab entry, so greedy longest-match shatters every word
/// into stray fragments: `"hello the quick brown fox"` round-tripped as
/// `"hell o the qui ck bro wn fo x"`. The resulting ids are near-random, and
/// mean-pooled embeddings of ANY two texts collapse onto the corpus average —
/// measured cosine distance between unrelated sentences was ~0.0005, which
/// makes dense retrieval pure noise while still looking healthy end to end.
///
/// The mapping is total and lossless for this convention:
/// - `▁X` → `X`      (word-initial)
/// - `[SPECIAL]` → unchanged
/// - `X` → `##X`     (continuation)
///
/// Punctuation and digits carry `▁` too (`"▁!"`, `"▁1"`), so they land on the
/// word-initial branch exactly as BERT expects.
///
/// A vocab that already uses `##`, or that has no `▁` at all, is returned
/// untouched — detection is on the vocab's own contents, never on the model
/// name, so a correctly-marked file is never rewritten.
pub(super) fn normalize_wordpiece_vocab(tokens: Vec<String>) -> Vec<String> {
    let has_sp_marker = tokens.iter().any(|t| t.starts_with(SP_WORD_BOUNDARY));
    let has_wordpiece_marker = tokens.iter().any(|t| t.starts_with("##"));
    if !has_sp_marker || has_wordpiece_marker {
        return tokens;
    }

    tokens
        .into_iter()
        .map(|t| {
            if let Some(stripped) = t.strip_prefix(SP_WORD_BOUNDARY) {
                stripped.to_owned()
            } else if t.starts_with('[') && t.ends_with(']') {
                // [PAD], [CLS], [SEP], [UNK], [unusedN] — never continuations.
                t
            } else {
                format!("##{t}")
            }
        })
        .collect()
}

/// Find a special token ID, checking the vocab for the token string first, then
/// falling back to the metadata field. `None` when neither states one.
///
/// The vocab comes first because the string is the ground truth: a file whose
/// `[UNK]` sits at a different id than its `unknown_token_id` claims would
/// otherwise emit an id that decodes to some other token.
fn lookup_special(tokens: &[String], vocab: &GgufVocab, token_str: &str) -> Option<u32> {
    for (id, t) in tokens.iter().enumerate() {
        if t == token_str {
            return Some(id as u32);
        }
    }

    match token_str {
        "[UNK]" => vocab.unknown_token_id,
        "[PAD]" => vocab.padding_token_id,
        "[CLS]" => vocab.cls_token_id,
        "[SEP]" => vocab.sep_token_id,
        _ => None,
    }
}

/// [`lookup_special`] with a fallback for the ids a backend cannot do without.
pub(super) fn find_special_token_id(
    tokens: &[String],
    vocab: &GgufVocab,
    token_str: &str,
    default: u32,
) -> u32 {
    lookup_special(tokens, vocab, token_str).unwrap_or(default)
}
