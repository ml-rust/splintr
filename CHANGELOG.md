# Changelog

All notable changes to this project are documented here. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Every release is gated on the section below carrying its version: `scripts/ci/changelog_section.sh` extracts it, `release-validate.yml` refuses a tag without it, and the GitHub release body is that section verbatim.

Releases before `0.11.0` predate this file; their contents are in the git history.

## [Unreleased]

## [0.11.0] - 2026-08-03

### Added

- **GGUF vocabulary loader** — `from_gguf_vocab(GgufVocab { .. })` builds a tokenizer from the `tokenizer.ggml.*` metadata a model runtime already parsed, without splintr ever opening a GGUF container. Dispatches on `tokenizer.ggml.model` to the backend that matches the algorithm (`gpt2` → byte-level BPE, `llama` → SPM-BPE, `t5` → Unigram, `bert` → WordPiece) and rejects what it cannot honour rather than guessing.
- **BERT boundary template for GGUF vocabularies** — a `bert` vocabulary is wrapped in `[CLS] A [SEP]` built from the ids it names, so `encode` on a GGUF and on the same model's `tokenizer.json` agree instead of handing a CLS-pooling consumer bare content tokens.
- **SPM-BPE backend** (`SpmTokenizer`) for llama.cpp-style `SPM` vocabularies — SentencePiece BPE with merge-by-rank rather than Viterbi. Mistral V1/V2 now route through it.
- **`SpecialMode`** (`All` | `Ordinary` | `Allow(&FxHashSet<String>)`) on `encode_with`, with `encode_with_special` / `encode_ordinary` / `encode_allowed_special` exposed on every Python tokenizer class, so untrusted text cannot forge a control token.
- **`base_vocab_size`** (and `base_vocab_size_by_name`) — a vocabulary's size as its upstream reference defines it, without splintr's agent tokens, for sizing embedding and logit layers.
- **Multi-pass pre-tokenizer chains** — a vocabulary may declare an ordered list of split patterns; DeepSeek V3 now uses its own three-pass pre-tokenizer and Llama 3 its own pattern, instead of borrowing o200k's.
- **Pair-sequence encoding** through the special-token policy, plus a `cls_sep_ids` accessor for BERT-style wrapping.
- **`decode` as an inherent method on `AnyTokenizer`**, alongside the unified `encode` / `encode_raw` / `encode_with` / `encode_batch` surface shared by every Python tokenizer class.
- **Differential fuzzer** (`scripts/fuzz_reference.py`) diffing splintr against `tokenizers`, `transformers` or `tiktoken` on strings assembled from each vocabulary's own added and special tokens, with deterministic seeds and shrinking to a minimal reproducer; and `examples/verify_gguf.rs` for llama.cpp's own `.inp`/`.out` fixtures.
- Integration tests covering the Unigram and WordPiece backends loaded from `tokenizer.json`.

### Changed

- `from_pretrained` returns an `AnyTokenizer` for every bundled vocabulary in both Rust and Python, so a vocabulary name means the same thing on both sides regardless of which backend it needs.
- WordPiece accent stripping is its own setting (`with_strip_accents`), seeded from `lowercase` and overridable independently — what cased multilingual BERT needs.
- Bundled vocabulary files moved out of the Python package into a top-level `vocabs/`, and are shipped through an explicit `include` list in `Cargo.toml`.
- The SentencePiece decode mode is named for what it does: the metaspace decoder.
- Dependencies: pyo3 0.29 (closes RUSTSEC-2026-0176, RUSTSEC-2026-0177) and crossbeam-epoch 0.9.20 (closes RUSTSEC-2026-0204). A `cargo-deny` gate now fails CI on any advisory, disallowed license, or non-crates.io source.
- Removed an unused `wasm` feature and a stray `pcre2` dev-dependency.

### Fixed

- Added tokens honour their per-token `lstrip` / `rstrip` flags.
- The SentencePiece dummy prefix is resolved against added-token splits, and is always prepended when `add_prefix_space` is set.
- Unigram Viterbi scores accumulate in `f64`; `f32` lost segmentations on long inputs.
- A vocabulary entry that also appears in `added_tokens` is matched literally.
- Mistral SPM vocabularies load with their real scores instead of id order.
- `AnyTokenizer`'s decoder pipeline survives the Python FFI boundary.
- `encode_ordinary` is public on every backend.
- A failure to build the added-token matcher is surfaced rather than swallowed.
