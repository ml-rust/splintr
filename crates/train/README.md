# splintr-train

Train your own tokenizer vocabulary with [splintr](https://crates.io/crates/splintr) — BPE, WordPiece or Unigram — and write it in a format splintr loads straight back.

Kept out of the tokenizer crate on purpose: nothing here is needed to *use* a tokenizer, so a consumer that loads a vocabulary and encodes text never downloads a corpus reader, a merge-loop heap or a JSON writer to do it.

## Install

```bash
cargo install splintr-train      # the CLI
```

```toml
[dependencies]
splintr-train = "0.1"
```

## Use

```bash
splintr-train bpe corpus.txt --output vocab.tiktoken --vocab-size 32000
splintr-train wordpiece corpus.txt --output vocab.txt --special '[UNK]'
splintr-train unigram corpus.txt --output vocab.spm --special '<unk>'
```

```rust
use splintr_train::{write, BpeTrainer, Corpus, PreTok};

let mut corpus = Corpus::with_pre_tok(PreTok::Whitespace)?;
corpus.feed_file("corpus.txt")?;          // streamed; corpus size is not held in memory

let vocab = BpeTrainer::builder()
    .vocab_size(32_000)
    .special_tokens(["<|endoftext|>"])
    .build()
    .train(&corpus.into_counts())?;

write::tiktoken_file(&vocab, "vocab.tiktoken")?;
```

## Why train inside splintr

A vocabulary is only meaningful against the word boundaries it was trained on, and nothing in a `.tiktoken`, a `vocab.txt` or a `.spm` file states them — load one with different boundaries and you get different ids, silently.

This crate cuts the corpus with splintr's **own** `Normalizer` and `PreTokenizer`, the same types the tokenizer takes, so training and encoding cannot drift apart. Those boundaries then travel with the vocabulary as a `Recipe`: the `tokenizer.json` writers state what training actually did, and the piece-list formats get a `.recipe.json` companion.

## Correctness

The BPE trainer produces **exactly the same pieces and merge order as HuggingFace `tokenizers`** on identical input — verified at a 32000-piece target on 9 MB of text, where all 32000 pieces and all 31673 merges matched in order, and pinned in CI by a test against a committed fixture so it needs neither Python nor a network — at 3.1x the speed. Unigram compresses about 8% better than theirs on held-out text. All three trainers here are deterministic; HuggingFace's WordPiece trainer is not.

Every writer has a test that loads its output back through splintr's own reader.

## Documentation

See the [training guide](https://github.com/ml-rust/splintr/blob/main/docs/training.md) for choosing a trainer, the options that change a vocabulary, and memory at scale.

## License

MIT
