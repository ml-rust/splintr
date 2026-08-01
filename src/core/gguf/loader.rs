//! Construction: turn a [`GgufVocab`] into an [`AnyTokenizer`], dispatching on
//! the algorithm the file declares.

use rustc_hash::FxHashMap;

use super::super::any_tokenizer::{AnyTokenizer, Backend};
use super::super::bpe;
use super::super::normalizer::{NormOp, Normalizer};
use super::super::policy::SpecialPolicy;
use super::super::precompiled::Precompiled;
use super::super::sentencepiece::SentencePieceTokenizer;
use super::super::spm::SpmTokenizer;
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

/// `bert`: WordPiece. Boundaries come from `[CLS]`/`[SEP]` in the vocabulary, so
/// no boundary template is synthesized.
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

    let eos_token_id = vocab.eos_token_id.unwrap_or(0);
    let backend = Backend::WordPiece(
        WordPieceTokenizer::new(tokens, unk_token_id, 200, do_lower_case).with_added_tokens(&named),
    );
    // No boundary template: BERT wraps with `[CLS]`/`[SEP]`, and `add_bos_token`
    // / `add_eos_token` are not what these files use to say so.
    Ok(AnyTokenizer::new(
        backend,
        SpecialPolicy::boundary(None, None, Some(eos_token_id), named),
    ))
}

/// `t5`: true Unigram. Scores are log-probabilities and Viterbi is correct.
fn build_unigram(mut vocab: GgufVocab) -> Result<AnyTokenizer, GgufVocabError> {
    // One map, two uses: matched in the input by the backend, and resolvable by
    // name through the policy. Neither substitutes for the other.
    let specials = special_token_map(&vocab, &vocab.tokens);
    let tokens = std::mem::take(&mut vocab.tokens);
    let scores = vocab.scores.take().unwrap_or_default();
    let eos_token_id = vocab.eos_token_id.unwrap_or(2);
    let prefix_space = unigram_prefix_space(&vocab);
    let normalizer = unigram_normalizer(vocab.precompiled_charsmap.as_deref());

    // `None` for BOS: boundary tokens are placed by the policy, so the backend
    // must not also prepend one. The backend's `eos` is not a boundary — it only
    // drives decode-skipping and `is_eos` — so it takes the resolved id.
    let backend = Backend::Unigram(
        SentencePieceTokenizer::new(tokens, scores, None, eos_token_id)?
            .with_normalizer(normalizer)
            .with_prefix_space(prefix_space)
            .with_remove_extra_whitespaces(remove_extra_whitespaces(&vocab))
            .with_added_tokens(&specials),
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
    let backend = Backend::Spm(
        SpmTokenizer::new(tokens, scores, None, None)?
            .with_prefix_space(add_space_prefix(&vocab, true))
            .with_added_tokens(&specials),
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
    let pattern = byte_level_pattern(vocab.pre.as_deref())?;

    let eos_token_id = vocab.eos_token_id.unwrap_or(0);

    let backend = Backend::Bpe(
        Tokenizer::new_byte_level(encoder, specials, pattern)?
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

/// The pre-tokenizer split regex a byte-level BPE vocabulary was built with.
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
/// An unrecognised name is refused rather than defaulted: a wrong split is
/// invisible downstream, and every id it produces is still in range.
pub(super) fn byte_level_pattern(pre: Option<&str>) -> Result<&'static str, GgufVocabError> {
    // Every name below was traced through llama.cpp twice: the `pre` string to a
    // `LLAMA_VOCAB_PRE_TYPE_*` value in `llama_vocab::impl::load`, and that value
    // to the literal `regex_exprs` list in `llm_tokenizer_bpe`'s constructor. A
    // name is listed only when its enum value yields a SINGLE expression equal to
    // the constant it is mapped to. Enum values that yield several expressions are
    // deliberately absent: llama.cpp runs those passes in sequence, each one
    // subdividing the previous pass's pieces, which no single alternation
    // reproduces.
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
        | "solar-open" | "grok-2" => Ok(QWEN2_PATTERN),

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
        | "trillion" | "granite-docling" => Ok(GPT2_PATTERN),

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
        "llama-bpe" | "llama3" | "dbrx" | "smaug-bpe" | "glm4" => Ok(LLAMA3_PATTERN),
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
pub(super) fn build_merge_ranks(merges: &[String], tokens: &[String]) -> FxHashMap<Vec<u8>, u32> {
    // Each entry is "a b"; byte-level tokens encode real spaces as `Ġ`, so the
    // first space is always the separator.
    let merged: Vec<String> = merges.iter().map(|s| s.replacen(' ', "", 1)).collect();
    bpe::merge_ranks(&merged, tokens.iter().map(String::as_str))
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
