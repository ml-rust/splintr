# Training Your Own Vocabulary

You can train your own vocabulary with splintr — BPE, WordPiece or Unigram — and write it in a format splintr loads straight back. Training lives in a separate crate, `splintr-train`, so the tokenizer itself never carries training dependencies.

**Why train inside splintr.** A vocabulary is only meaningful against the word boundaries it was trained on. `splintr-train` cuts the corpus with splintr's _own_ `Normalizer` and `PreTokenizer` — the same types `Tokenizer::with_normalizer` and `Tokenizer::with_pre_tokenizer` take — so the boundaries a vocabulary is trained on and the boundaries it is later encoded against come from one implementation and cannot drift apart.

---

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [Choosing a trainer](#choosing-a-trainer)
- [Library use](#library-use)
- [Pre-tokenization and the recipe](#pre-tokenization-and-the-recipe)
- [Output formats](#output-formats)
- [Options that change the vocabulary](#options-that-change-the-vocabulary)
- [Scale and memory](#scale-and-memory)
- [Compared with HuggingFace `tokenizers`](#compared-with-huggingface-tokenizers)
- [What is not measured](#what-is-not-measured)

---

## Install

```bash
cargo install splintr-train      # the CLI
```

```toml
# or as a library
[dependencies]
splintr-train = "0.1"
```

The CLI lives behind the `cli` feature, which is on by default so `cargo install` works. A library consumer can switch it off to avoid pulling in an argument parser:

```toml
splintr-train = { version = "0.1", default-features = false }
```

---

## Quick start

Each input file is read as one document per line.

```bash
# BPE, tiktoken-style — the GPT-family shape
splintr-train bpe corpus.txt --output vocab.tiktoken --vocab-size 32000

# WordPiece — the BERT-family shape
splintr-train wordpiece corpus.txt --output vocab.txt --vocab-size 32000 \
    --special '[UNK]' --special '[CLS]' --special '[SEP]'

# Unigram — the SentencePiece shape
splintr-train unigram corpus.txt --output vocab.spm --vocab-size 32000 \
    --special '<unk>'
```

The output format follows the extension: `.json` writes a full `tokenizer.json`, anything else writes the format native to that trainer.

Load the result straight back:

```rust
use rustc_hash::FxHashMap;
use splintr::Tokenizer;

// The pattern and specials come from the `.recipe.json` written beside it.
let specials: FxHashMap<String, u32> = FxHashMap::default();
let tokenizer = Tokenizer::from_file("vocab.tiktoken", r"\s*\S+", specials)?;
```

### Common flags

| Flag                 | Default      | Meaning                                                      |
| -------------------- | ------------ | ------------------------------------------------------------ |
| `-n`, `--vocab-size` | `32000`      | Tokens to produce, specials included                         |
| `-s`, `--special`    | none         | A special token; repeat for more                             |
| `--min-frequency`    | `2`          | Ignore words rarer than this                                 |
| `--pre-tokenizer`    | `whitespace` | `whitespace`, `byte-level`, `pattern`, `none`                |
| `--pattern`          | `\s*\S+`     | The expression for `--pre-tokenizer pattern`                 |
| `--metaspace`        | off          | Mark word starts with `U+2581` (on by default for `unigram`) |
| `--seeding`          | `bytes`      | `bytes` or `chars` — BPE only                                |
| `--prune`            | off          | Drop unreachable pieces — WordPiece only                     |

---

## Choosing a trainer

|               | BPE                    | WordPiece            | Unigram                              |
| ------------- | ---------------------- | -------------------- | ------------------------------------ |
| Segmentation  | replays the merge list | greedy longest match | maximises a sum of log-probabilities |
| Native format | `.tiktoken`            | `vocab.txt`          | `.spm`                               |
| Family        | GPT, tiktoken          | BERT                 | SentencePiece, T5, Llama             |
| Speed         | fastest                | fastest              | slowest by a wide margin             |
| Compression   | good                   | good                 | **best at small vocabularies**       |

**Default to BPE.** It is the fastest to train, matches the tiktoken shape splintr's engine is built around, and a `.tiktoken` rank file is its native artifact — splintr's BPE engine ranks merges by the token a merge _produces_ rather than by the pair it joins, which is exactly what a rank file lists.

**Reach for Unigram when the vocabulary is small.** It compresses measurably better there, at a real cost in training time — see [Scale and memory](#scale-and-memory).

---

## Library use

```rust
use splintr_train::{write, BpeTrainer, Corpus, PreTok};

// 1. Cut the corpus into word counts.
let mut corpus = Corpus::with_pre_tok(PreTok::Whitespace)?;
corpus.feed_file("corpus.txt")?;
let counts = corpus.into_counts();

println!("{} distinct words, {} occurrences", counts.len(), counts.total());

// 2. Train.
let vocab = BpeTrainer::builder()
    .vocab_size(32_000)
    .min_frequency(2)
    .special_tokens(["<|endoftext|>"])
    .build()
    .train(&counts)?;

// 3. Write it somewhere splintr can load it back.
write::tiktoken_file(&vocab, "vocab.tiktoken")?;
write::recipe_json_file(&vocab, "vocab.tiktoken.recipe.json")?;
```

`Corpus::feed_reader` streams any `Read`, a line at a time, so corpus size never enters the memory cost — only the number of _distinct_ words does. Prefer it over reading a file into a `String`: the two produce identical counts, but the string costs the whole corpus in resident memory on top of the counts.

`WordCounts::memory_bytes()` reports what the counts occupy, which is the floor under a trainer's memory. Worth checking before a long run.

---

## Pre-tokenization and the recipe

This is the part that silently goes wrong, so it is worth reading.

A vocabulary encodes assumptions about where words begin and end. Load it with different assumptions and you get **different ids, with no error anywhere** — nothing in a `.tiktoken`, a `vocab.txt` or a `.spm` file states the boundaries it was trained under.

splintr-train records them. Whenever the pre-tokenizer comes from a `PreTok`, the resulting vocabulary carries a `Recipe`:

```rust
let recipe = vocab.recipe().expect("built from a PreTok");
println!("{:?} marker={:?}", recipe.pre_tok, recipe.word_marker);
```

Two things follow, and both are automatic:

- **The JSON writers state what training actually did.** `bpe_json` emits the pre-tokenizer the vocabulary was trained under rather than whatever the options asked for, and `unigram_json` declares a Metaspace pre-tokenizer _only_ when the corpus was really marked, using that very character.
- **The plain-text formats get a companion file.** The CLI writes `vocab.tiktoken.recipe.json` next to `vocab.tiktoken`, carrying the pre-tokenizer, the pattern, the word marker and the special tokens.

A hand-assembled `PreTokenizer` cannot be written back down, so a vocabulary trained with one carries no recipe rather than a guessed one.

### Metaspace is part of the vocabulary, not preprocessing

A SentencePiece-style segmenter prepends `U+2581` before matching. A vocabulary trained on unmarked words therefore cannot spell the first character of any word, and every word picks up a spurious unknown token — measured at roughly **twice** the tokens it should need.

So `--metaspace` is on by default for `unigram`. If you train a vocabulary any other way for a SentencePiece-shaped segmenter, set it yourself:

```rust
let corpus = Corpus::with_pre_tok(PreTok::Whitespace)?.with_metaspace();
```

---

## Output formats

| Writer           | File             | Loads back through                        |
| ---------------- | ---------------- | ----------------------------------------- |
| `tiktoken`       | `.tiktoken`      | `Tokenizer::from_file`                    |
| `bpe_json`       | `tokenizer.json` | `splintr::from_json_path`                 |
| `vocab_txt`      | `vocab.txt`      | `splintr::WordPieceTokenizer::new`        |
| `wordpiece_json` | `tokenizer.json` | `splintr::from_json_path`                 |
| `spm`            | `.spm`           | `splintr::core::load_spm_vocab`           |
| `unigram_json`   | `tokenizer.json` | `splintr::from_json_path`                 |
| `recipe_json`    | `*.recipe.json`  | companion to the three piece-list formats |

Every writer has a test that loads its output back through splintr's own reader, which is the only check that matters here.

---

## Options that change the vocabulary

### `min_frequency` (default 2)

Words rarer than this are ignored. The default is right at any real scale.

**Do not raise it on multilingual text.** On a corpus with Chinese, Japanese or Thai, whitespace splitting makes each sentence one long unique "word", so a frequency floor deletes those scripts from training outright — measured at **2.8× worse** compression at a floor of 2 on a ten-script corpus.

### `seeding` — BPE only

- `Bytes` (default) — one symbol per byte, and all 256 byte values are in the alphabet whether the corpus uses them or not, so **nothing is unspellable**.
- `Chars` — one symbol per character. Required when the pre-tokenizer has a `ByteLevel` stage, which has already mapped each raw byte to a printable code point; cutting those by byte would split the UTF-8 of `Ġ` down the middle.

A `tokenizer.json` keys its vocabulary by string, so a vocabulary written to that format must be trained with `Chars` over byte-level text — under `Bytes` some pieces are not valid UTF-8 and have no key.

### `prune` — WordPiece only, off by default

A greedy longest-match segmenter never emits a large share of the vocabulary it is given: measured at **14.6% / 36.6% / 45.2%** dead pieces at 2000 / 4000 / 8000 tokens, on the corpus they were trained on.

Pruning them is _not_ free. Removing them costs about **3% more tokens**, because greedy longest-match compression is a property of the vocabulary as a _set_ — removing one piece changes which pieces fire elsewhere, so no per-piece score selects a lossless subset. It ships off by default; turn it on if you value embedding rows over tokens.

### `criterion` — BPE/WordPiece, default `Frequency`

`Criterion::Likelihood` scores a merge by the exact gain in corpus log-likelihood. It is the textbook WordPiece objective, and it is **worse**: measured at 1.27× the tokens of plain frequency. Merging `m` occurrences of a pair removes exactly `m` tokens, so greedy frequency optimises token count directly. The default is `Frequency` for that reason.

### `max_token_length`

Refuses merges producing a piece longer than the given number of bytes. The pair stays in the corpus; it is simply never joined.

---

## Scale and memory

Memory is driven by the number of **distinct words**, not by corpus size — that is what makes streaming the corpus worthwhile. Distinct words grow sublinearly (Heaps' law): a 115 KB corpus deduplicated 3.3×, an 8 MB corpus 75×.

Measured on a 1.06 GB, ten-script corpus at a 128k vocabulary:

```
feed    1.06 GB streamed -> 7,849,156 distinct words     RSS 0.67 GB
bpe     128,000 pieces                                   peak 5.60 GB
```

That is roughly **5× the corpus in peak memory**, so a rule of thumb: a corpus of _N_ GB wants about _5N_ GB of RAM. The floor is structural — the corpus has to be resident as symbol ids, plus one index over the pairs — so training a corpus much beyond a few GB means sharding it rather than tuning options.

**Unigram scales differently.** Its cost is `rounds × words × word_length × max_piece_chars`, and on scripts without spaces a "word" is 26–64 bytes where English is 7.6, which makes it 50–100× slower per MB there. BPE is untouched by this. If you are training on Chinese, Japanese or Thai and Unigram is too slow, use BPE.

---

## Compared with HuggingFace `tokenizers`

**BPE selection is exactly HuggingFace's, at scale.** Trained on 9 MB of text at a 32000-piece target with both sides configured identically, the two vocabularies agree on **all 32000 pieces and all 31673 merges, in order** — the same objects, not merely comparable ones. CI pins a 2000-piece case against a committed reference fixture, so the check runs without Python or a network. Where splintr-train diverges elsewhere, it does so deliberately and on measurement.

That has a practical consequence worth stating plainly: for BPE, choosing this trainer is not a bet on an alternative implementation. You get the vocabulary HuggingFace would have produced, faster.

On a 9.6 MB training / 2.4 MB held-out English corpus at 32k:

|           | Speed           | Held-out tokens                 |
| --------- | --------------- | ------------------------------- |
| BPE       | **3.1× faster** | identical vocabulary            |
| Unigram   | 4.0× slower     | **8% better**                   |
| WordPiece | —               | trainers agree; see below       |

**Do not read a small held-out difference as a trainer result.** An earlier run of this benchmark showed BPE 0.1% apart, which cannot be the trainer given the vocabularies are byte-identical; it is on the encode side, most likely the two harnesses pre-tokenizing the held-out text differently.

**WordPiece.** The two trainers produce vocabularies that score the same under the same segmenter (verified by swapping the vocabularies over), so any difference in a head-to-head number is the _segmenter_, not the trainer — the same trap as above, caught the same way.

**splintr-train is deterministic; HuggingFace's WordPiece trainer is not.** Five runs of theirs on one corpus with one configuration gave five different vocabularies. Ours gives the same vocabulary every time, and that is tested.

---

## What is not measured

**Downstream model quality.** Every quality number here is held-out token count — that is compression, and it is a proxy. Whether a vocabulary trained this way produces a _better model_ has not been measured, and no model has yet been trained on a splintr-produced vocabulary.

If that matters for your use, treat the compression figures as what they are: evidence that the trainers agree with well-established implementations and are efficient, not evidence about downstream loss.
