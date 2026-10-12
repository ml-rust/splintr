# Design

This page covers why splintr exists, how it tokenizes, and the sources it cites.
For use, start with the [README](../README.md).

## Why splintr exists

Tokenization runs on every LLM prompt, training corpus, RAG chunk, and token count for billing.
Python tokenizers cannot use all CPU cores, so batch preprocessing adds wall-clock latency.
The common answer is one library per format: `tiktoken`, `sentencepiece`, and `tokenizers`.
That means three dependencies, three APIs, no shared handle, and no answer for a GGUF vocabulary.

Splintr provides one handle over every format, at Rust speed.
Reference implementations serve as the correctness oracle.

## How it works

Pre-tokenization runs on [`regexr`](https://crates.io/crates/regexr), a pure-Rust regex engine with JIT and SIMD.
Special tokens match with Aho-Corasick in a single pass.
Merging uses a linked list instead of a vector, so pathological inputs stay linear.
An LRU cache stores repeated chunks, and `FxHashMap` serves rank lookups.
Batches encode in parallel with Rayon.
Single texts stay sequential, which measures faster below roughly 1 MB.

The three non-byte-level backends are full implementations, not approximations:

- SentencePiece Unigram uses Viterbi maximum-score segmentation.
- SentencePiece BPE merges by score.
- WordPiece runs greedy longest-match with the `##` continuation prefix.

Correctness rests on differential tests.
See [CONTRIBUTING.md](../CONTRIBUTING.md#correctness-against-the-reference-implementations).

## Acknowledgments

Splintr builds on concepts from [tiktoken](https://github.com/openai/tiktoken),
[SentencePiece](https://github.com/google/sentencepiece), and
[tokenizers](https://github.com/huggingface/tokenizers).
Those projects also serve as the reference implementations for splintr's output.

## Citation

Cite splintr as follows:

```bibtex
@software{splintr,
  author = {Farhan Syah},
  title = {Splintr: High-Performance Tokenizer (BPE + SentencePiece + WordPiece)},
  year = {2025},
  url = {https://github.com/ml-rust/splintr}
}
```
