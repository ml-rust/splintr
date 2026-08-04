# Changelog

All notable changes to this project are documented here. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Every release is gated on the section below carrying its version: `scripts/ci/changelog_section.sh` extracts it, `release-validate.yml` refuses a tag without it, and the GitHub release body is that section verbatim.

Releases before `0.11.0` predate this file; their contents are in the git history.

## [0.12.0] - 2026-08-04

This release changes what `encode` and `decode` return for several vocabularies, and it
breaks source compatibility in four places. Read these before upgrading:

- `decode` now **drops** a vocabulary's declared special tokens by default on every bundled vocabulary — Mistral V1/V2/V3's `[INST]`/`[/INST]`/`<s>`/`</s>`, Llama 3's, DeepSeek V3's and Whisper's included. `decode_with(ids, SpecialDecode::Render)` restores the old output.
- `Tokenize` gained required methods; every external implementor of the trait must add them.
- `ByteLevelStreamingDecoder` is removed from both the Rust and Python APIs, and `StreamingDecoder::new` is no longer public.
- `decode_bytes` returns a `Result`, and the streaming `add_token`/`add_tokens` pair is strict by default.

### Added

- **`SpecialDecode` — render or skip declared-special ids on decode.** `decode` has always implemented HuggingFace's `skip_special_tokens=True` with no way to ask for `skip_special_tokens=False`. `SpecialDecode::{Skip, Render}` is threaded through the new `decode_with` and `streaming_decoder_with`, with `decode`/`streaming_decoder` remaining `Skip`-mode shorthands. Python gains `decode_with_special` and `streaming_decoder_with_special` on every tokenizer class.
- **Per-id decoding** — `decode_token_bytes` / `decode_token` render one id's contribution to decoded output, for callers that need to attribute text to tokens.
- **Byte fallback for BPE tokenizers** — the public `ByteFallback` type carries a per-byte `<0xNN>` table plus the resolved unk id, attached via `Tokenizer::with_byte_fallback` and reported by `has_byte_fallback`. A piece the merge vocabulary cannot represent is emitted through the fallback ids instead of being silently dropped. `model.fuse_unk` is honoured (`ByteFallback::with_fuse_unk`, default off as in `tokenizers`), collapsing a run of unk-resolved characters into one id.
- **A streaming decoder on every backend.** `streaming_decoder()` now exists on `SentencePieceTokenizer`, `SpmTokenizer`, `WordPieceTokenizer` and `AnyTokenizer` as well as `Tokenizer`. On `AnyTokenizer` it drives the `tokenizer.json` file's own declared decoder pipeline (`ByteLevel`, `Metaspace`, `WordPiece`, and the `Sequence[Replace, ByteFallback, Fuse, Strip]` chain SentencePiece models use); shapes that are not incrementally computable — `BPEDecoder`, a trailing `Strip` after `Fuse`, a `Replace` over fused text — return the new `TokenizeError::UnstreamableDecoder` naming the step, rather than silently answering with raw pieces.
- **`add_token_lossy` / `add_tokens_lossy`** — the skip-unknown twins of the now-strict streaming pair, mirroring `decode_lossy`.
- **The pre-tokenizer pipeline is public API** — `PreTokenizer`, `PreTokStage`, `SplitBehavior` and `SplitPattern` are re-exported, so `Tokenizer::with_pre_tokenizer` is callable from outside the crate. `PreTokenizer::new` compiles each `Split` pattern and returns a `TokenizerError` instead of panicking or silently dropping a stage that fails to compile.
- **The normalizer types are public API** — `Normalizer`, `NormOp` and `Precompiled` are re-exported, so `with_normalizer` is callable from outside the crate. `NormOp` is `#[non_exhaustive]`, since it mirrors HuggingFace's normalizer spec and will grow.
- **Reference fixtures for every bundled vocabulary that has a reference**, covering decoded text as well as token ids — encode and decode are separate pipelines, and a fixture pinning only ids left byte-level unmapping, byte fallback and the SentencePiece dummy-prefix strip unpinned. `scripts/extract_reference_cases.py` now speaks all three authoritative reference tools (`tiktoken` for the OpenAI vocabularies, `tokenizers` for the HF-published ones, `sentencepiece` for the `.spm`-backed Mistral V1/V2), gating each pairing exhaustively where the reference exposes its whole vocabulary as data. `mistral_v3` (Tekken) has no fixture: its reference is a Tekken checkpoint, and no near neighbour is an acceptable substitute.
- **`tests/decode_agreement.rs`** — streaming decode concatenated with `flush()` equals whole-sequence `decode`/`decode_lossy` for every bundled vocabulary, every backend reachable from it and every chunk size, and `reset()` leaves a decoder byte-identical to a fresh one. Needs no reference tokenizer, so it runs in CI unconditionally.
- **`scripts/verify_external_models.py`** — the pre-release sweep of splintr's `from_json` loader and bundled SentencePiece vocabularies against the published model tokenizers on a maintainer's machine, as a pass/fail table. Aborts rather than shrinking when the model directory, a target file, or a current `splintr` wheel is absent.

### Changed

- **BREAKING: one streaming decoder, built by the tokenizer.** `ByteLevelStreamingDecoder` is gone and `StreamingDecoder::new(&tokenizer)` with it; the single `StreamingDecoder` is obtained only from `Tokenizer::streaming_decoder()`, which takes ByteLevel unmapping, the `special=true` ids to drop and the metaspace ▁ substitution from the tokenizer's own configuration. Picking the decoder that did not match the vocabulary — silently producing mojibake, and silently ignoring the skip set and the metaspace pass that `decode` applies — is no longer expressible. The decoder carries no lifetime, so it can be owned and moved into a generation task, and it shares the vocabulary map rather than copying it.
- **BREAKING: Python gets that same single decoder.** `splintr.ByteLevelStreamingDecoder` and `byte_level_streaming_decoder()` are gone; `StreamingDecoder` now wraps the Rust decoder rather than re-implementing UTF-8 assembly over cloned maps, so the Python stream honours the skip set, the metaspace substitution and `<0xNN>` byte fallback that whole-sequence `decode` applies — none of which it did before. `streaming_decoder()` exists on `Tokenizer`, `SentencePieceTokenizer`, `SpmTokenizer`, `WordPieceTokenizer` and `AnyTokenizer`, not the BPE class alone, and on `AnyTokenizer` it raises `ValueError` naming the offending step when a declared `decoder` pipeline cannot be streamed instead of returning a decoder that answers with raw pieces. `add_token`/`add_tokens` keep their signatures and their lenient treatment of an unknown id (they map to the Rust `*_lossy` pair), so a stream still survives one stray id.
- **BREAKING: the streaming API is strict by default and mirrors `decode`.** `add_token`/`add_tokens` return `Result<Option<String>, TokenizeError>` and report an id in no table as `TokenizeError::InvalidTokenId`, exactly as `decode` does; the new `add_token_lossy`/`add_tokens_lossy` skip it instead, exactly as `decode_lossy` does. Concatenating every emission plus `flush()` now equals `decode` (and always equals `decode_lossy`), for raw and ByteLevel vocabularies alike.
- **BREAKING: `Tokenize` gained required methods.** `decode_lossy`, `streaming_decoder`, `decode_token_bytes`, `decode_token`, `decode_with` and `streaming_decoder_with` are now part of the trait, implemented by `Tokenizer`, `AnyTokenizer`, `SentencePieceTokenizer`, `SpmTokenizer` and `WordPieceTokenizer`. Any external implementor of `Tokenize` must supply them.
- **BREAKING: `decode_bytes` reports an unknown id instead of hiding it.** It returns `Result<Vec<u8>, TokenizerError>` and errors with `InvalidTokenId` where it previously rendered the id as empty bytes; `decode_lossy` keeps the infallible skip-unknown behaviour. In Python, `decode` raises `ValueError` on an unknown id rather than returning empty bytes.
- **BREAKING: default `decode` output changed for the bundled vocabularies.** Each vocabulary's declared special ids are now dropped by default, so `from_pretrained("mistral_v2").decode(..)` and `from_json` on the same `tokenizer.json` agree — previously they disagreed with their reference tokenizers and with each other, and `[INST]`, `<s>`, `</s>`, the Whisper control tokens and the agent tokens were rendered verbatim. `decode_with(ids, SpecialDecode::Render)` reaches every declared-special id, including the ones a vocabulary's default now drops. `mistral_v3`'s arm is set by consistency with V1/V2, not by measurement: no Tekken reference is available to check it against, and the code, tests and a ready `bjson` target in `verify_external_models.py` say so.
- **Encode output changed for HuggingFace-style vocabularies with their own merge ranks.** BPE now seeds merges by character rather than by byte whenever the vocabulary carries `merges` and is not ByteLevel — byte-seeded merging can never reassemble a UTF-8 character 3 bytes or wider, so characters like Mistral/Llama-SPM's `▁` shattered into byte fallbacks instead of merging.
- **Byte fallback is resolved before merging when a merge needs it.** `tokenizers`' `BPE::merge_word` resolves `<0xNN>`/`<unk>` per character first, making those tokens ordinary word symbols its merge list may combine with their neighbours. splintr now does the same, gated on there being an unresolved run at all. No published vocabulary on the shelf exercises this order, so `mistral-7b-v0.3` and `embeddinggemma-300m` are unchanged; the behaviour is pinned against a `tokenizers` 0.22.1 measurement on a purpose-built vocabulary.
- **BPE merge selection uses a binary heap** instead of rescanning the linked list for the lowest-rank pair, dropping the merge loop from O(N×M) to O(N log N). The previous linear scan is retained in tests as a reference oracle, checked against the new implementation with proptest-generated inputs.

### Fixed

- **A HuggingFace `Split` pre-tokenizer with a string pattern now splits on that string literally**, as `tokenizers` does, instead of compiling it as a regex. Splitting `"a.b c"` on `"."` with behavior `removed` yields `["a", "b c"]`; only `Regex(".")` matches every character. `PreTokStage::Split` carries the new `SplitPattern` (`Literal` | `Regex`) to keep the two forms distinct, and a literal is escaped before compiling.
- **A `tokenizer.json` with no pre-tokenizer no longer gets one guessed.** A `pre_tokenizer` that is absent, `null`, or a declared-but-empty `Sequence` now runs the model over the whole normalized string, as HuggingFace does, instead of falling back to the GPT-2 split pattern. This changed encode output for files such as `mistral-7b-awq-int4`/`gptq-int4`, whose metaspace transform lives in the normalizer with `pre_tokenizer` left null: the prepended metaspace character was cut off the word behind it and could never merge. A `Sequence` missing its `pretokenizers` key, a `Split` with no pattern, and a node with no type are now reported as unknown and refused, matching `tokenizers`, which fails to load those shapes outright.
- **A bare `Metaspace` pre-tokenizer on a BPE model is honoured** — the Mistral/Gemma/Llama-SPM shape now builds through the metaspace-decoder constructor and applies `add_prefix_space` from the pre-tokenizer directly, instead of falling through to the plain BPE path with the prefix force-disabled.
- **`add_prefix_space` tests for the literal space** ByteLevel and Metaspace actually check, not whitespace in general, so a leading tab or newline no longer suppresses the prefix.
- **Byte fallback is resolved per character**, as HuggingFace does — a character whose bytes all have `<0xNN>` entries becomes those byte tokens, otherwise the whole character collapses to the unk id. A vocabulary declaring only some of the 256 entries is a valid file that previously failed to load with `MissingSpecial("byte_fallback")`.
- **`model.unk_token` is honoured even when `model.byte_fallback` is false**, matching HuggingFace's BPE model. Gating the whole fallback on the flag silently dropped unrepresentable pieces instead of emitting the unk.
- **`<0xNN>` byte-fallback ids decode to the byte they denote**, not to their literal vocabulary spelling, on both whole-sequence and streaming decode — a character split across several fallback tokens reassembles correctly across `add_token` calls. Fallback surfaces are parsed with the strict two-hex-digit rule everywhere, so a spelling like `<0x1>` stays literal text as `tokenizers` 0.22.1's `decoders.ByteFallback` leaves it.
- **SentencePiece decode no longer leaks `<s>`/`</s>`/`<unk>`** into the decoded text; the ids to drop are declarable through `with_special_decode_ids`, and the GGUF and pretrained loaders pass their file-declared BOS/EOS/UNK through it.
- **WordPiece resolves decode-dropped specials by id, not by spelling.** A BERT-family GGUF vocabulary spelling its specials `<s>`/`</s>`/`<unk>` leaked them into decoded text, while a plain content token spelled `[unusedN]` was silently dropped even though HuggingFace never treats it as special.
- **A GGUF `t5` vocabulary drops its declared BOS/EOS/unk ids on decode**, like the `llama` arm already did; it was the one arm that never passed them through, so they leaked into decoded text.
- **`decode_token_bytes` no longer rejects an id that is in the vocabulary but carries no surface** (a special the pipeline skips) — it returns empty bytes, and only an id past the end of the vocabulary is `InvalidTokenId`. `AnyTokenizer`'s own APIs previously disagreed about the same id.
- **Streaming decode no longer stalls on undecodable bytes.** A byte that can never start or continue a valid UTF-8 sequence is emitted as U+FFFD and scanning continues; only a trailing sequence that is still incomplete but possible is held back. The Python decoders share this buffer rather than hand-rolling their own, which had the same stall.
- **`encode_rayon` goes through the same dispatch as `encode`**, so it no longer skips the normalizer and added-token handling for non-metaspace tokenizers and produces identical ids.
- **The chunk cache is keyed by the chunk bytes**, not by a bare `u64` hash, so a hash collision can no longer return another chunk's token ids.
- **`scripts/extract_reference_cases.py` parses tiktoken vocabulary lines by their trailing separator**, so a line whose base64 payload is legitimately empty — the Whisper vocabulary's rank-50256 entry for the empty byte string — is no longer indistinguishable from a malformed line.

### Removed

- **`ByteLevelStreamingDecoder`** and the Python `byte_level_streaming_decoder()`, superseded by `streaming_decoder()` on every tokenizer class.
- **`StreamingDecoder::new`** is no longer public; a decoder comes only from the tokenizer that will feed it.

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
