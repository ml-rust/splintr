# Changelog

All notable changes to this project are documented here. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Every release is gated on the section below carrying its version: `scripts/ci/changelog_section.sh` extracts it, `release-validate.yml` refuses a tag without it, and the GitHub release body is that section verbatim.

Releases before `0.11.0` predate this file; their contents are in the git history.

## [Unreleased]

### Added

- **Reference fixtures for every bundled vocabulary that has a reference**, covering decoded text as well as token ids — encode and decode are separate pipelines, and a fixture pinning only ids left byte-level unmapping, byte fallback and the SentencePiece dummy-prefix strip unpinned. `scripts/extract_reference_cases.py` now speaks all three authoritative reference tools (`tiktoken` for the OpenAI vocabularies, `tokenizers` for the HF-published ones, `sentencepiece` for the `.spm`-backed Mistral V1/V2), gating each pairing exhaustively where the reference exposes its whole vocabulary as data. `mistral_v3` (Tekken) has no fixture: its reference is a Tekken checkpoint, and no near neighbour is an acceptable substitute.
- **`tests/decode_agreement.rs`** — streaming decode concatenated with `flush()` equals whole-sequence `decode`/`decode_lossy` for every bundled vocabulary, every backend reachable from it and every chunk size, and `reset()` leaves a decoder byte-identical to a fresh one. Needs no reference tokenizer, so it runs in CI unconditionally.
- **`scripts/verify_external_models.py`** — the pre-release sweep of splintr's `from_json` loader and bundled SentencePiece vocabularies against the published model tokenizers on a maintainer's machine, as a pass/fail table. Aborts rather than shrinking when the model directory, a target file, or a current `splintr` wheel is absent.

### Changed

- **BREAKING: one streaming decoder, built by the tokenizer.** `ByteLevelStreamingDecoder` is gone and `StreamingDecoder::new(&tokenizer)` with it; the single `StreamingDecoder` is obtained only from `Tokenizer::streaming_decoder()`, which takes ByteLevel unmapping, the `special=true` ids to drop and the metaspace ▁ substitution from the tokenizer's own configuration. Picking the decoder that did not match the vocabulary — silently producing mojibake, and silently ignoring the skip set and the metaspace pass that `decode` applies — is no longer expressible. The decoder carries no lifetime, so it can be owned and moved into a generation task, and it shares the vocabulary map rather than copying it.
- **BREAKING: Python gets that same single decoder.** `splintr.ByteLevelStreamingDecoder` and `byte_level_streaming_decoder()` are gone; `StreamingDecoder` now wraps the Rust decoder rather than re-implementing UTF-8 assembly over cloned maps, so the Python stream honours the skip set, the metaspace substitution and `<0xNN>` byte fallback that whole-sequence `decode` applies — none of which it did before. `streaming_decoder()` exists on `Tokenizer`, `SentencePieceTokenizer`, `SpmTokenizer`, `WordPieceTokenizer` and `AnyTokenizer`, not the BPE class alone, and on `AnyTokenizer` it raises `ValueError` naming the offending step when a declared `decoder` pipeline cannot be streamed instead of returning a decoder that answers with raw pieces. `add_token`/`add_tokens` keep their signatures and their lenient treatment of an unknown id (they map to the Rust `*_lossy` pair), so a stream still survives one stray id.
- **BREAKING: the streaming API is strict by default and mirrors `decode`.** `add_token`/`add_tokens` return `Result<Option<String>, TokenizeError>` and report an id in no table as `TokenizeError::InvalidTokenId`, exactly as `decode` does; the new `add_token_lossy`/`add_tokens_lossy` skip it instead, exactly as `decode_lossy` does. Concatenating every emission plus `flush()` now equals `decode` (and always equals `decode_lossy`), for raw and ByteLevel vocabularies alike.

### Fixed

- **A HuggingFace `Split` pre-tokenizer with a string pattern now splits on that string literally**, as `tokenizers` does, instead of compiling it as a regex. Splitting `"a.b c"` on `"."` with behavior `removed` yields `["a", "b c"]`; only `Regex(".")` matches every character. `PreTokStage::Split` carries the new `SplitPattern` (`Literal` | `Regex`) to keep the two forms distinct, and a literal is escaped before compiling.

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
- **Python wheels are now abi3** (`cp38-abi3`), so one wheel per platform covers every CPython from 3.8 up. Previously each platform shipped a wheel for whichever single interpreter its builder happened to use — 3.9 from the manylinux container, 3.12 elsewhere — and every other version fell back to compiling the sdist.
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
