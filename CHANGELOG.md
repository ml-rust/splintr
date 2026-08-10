# Changelog

All notable changes to this project are documented here. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Every release is gated on the section below carrying its version: `scripts/ci/changelog_section.sh` extracts it, `release-validate.yml` refuses a tag without it, and the GitHub release body is that section verbatim.

Releases before `0.11.0` predate this file; their contents are in the git history.

## [Unreleased]

### Fixed

- `WhitespaceSplit`/`Whitespace` in front of `Metaspace` now drops the whitespace it splits on, instead of adding a spurious `▁` per run. T5 was wrong wherever a double space or newline appeared.
- WordPiece and the `Lowercase` normalizer lowercase per character, as HuggingFace does: word-final `Σ` is `σ`, not `ς`.
- BERT's `clean_text` keeps unassigned (`Cn`) codepoints, so a word carrying one becomes `[UNK]` as in the reference.
- A `Precompiled` charsmap from a `tokenizer.json` is read as `spm_precompiled` reads it, via the new `CharsmapDialect`; the sentencepiece reading stays the default and serves the GGUF path.

### Changed

- The long-piece merge path reuses per-thread buffers instead of allocating per piece, cutting bytes allocated per encode by roughly 20x on non-Latin scripts. Token ids are unchanged.
- Merge selection splits its queue — initial pairs sorted once, only merge-created pairs heaped — and picks its strategy per script as well as per length: 30-48% fewer instructions per byte on Cyrillic, Arabic, Korean, Chinese and Japanese, English unchanged. Token ids are unchanged.
- WordPiece matches through a byte trie instead of re-hashing every candidate subword, streams borrowed words instead of allocating one per word, and caches segmented words as the BPE backend does. bert-base-uncased encodes about 6.5x faster; allocations per encode fall from millions to thousands. Token ids are unchanged.
- The Unigram vocabulary is an `FxHashMap`, as every other vocabulary in the crate already was; it was the one left on the default `SipHash`. 37-56% fewer instructions per byte on t5-base. Token ids are unchanged.
- Unigram caches the segments it has already resolved, as the BPE and WordPiece backends do — it was the one backend re-running its whole lattice for every repeat of a common word. 17-47% fewer instructions per byte on English, code and Cyrillic. Token ids are unchanged.
- A `Precompiled` charsmap resolves each ASCII byte at construction and copies runs of untouched ASCII wholesale, so grapheme segmentation only runs over text that needs clustering. Normalization falls from a fifth of a t5-base encode to a fortieth.
- The Unigram lattice's three buffers are owned by the encode rather than the sweep, which allocated them per word.
- Unigram's lattice finds each position's candidate pieces by one walk down the byte trie WordPiece already used, instead of re-hashing a growing prefix once per candidate length: a further 11-61% off instructions per byte on t5-base. Token ids are unchanged.
- The chunk cache returns a one-token chunk with a push instead of a length-driven copy, and the WordPiece trie indexes its root by byte rather than searching it: 4% fewer instructions per byte and 6% more throughput on bert-base-uncased. Token ids are unchanged.
- BERT's normalization runs as one pass instead of four, deciding ASCII by table and copying caseless ideographs through untouched. A document that is 99% ASCII no longer takes the fully general path for the other 1%: 25-42% fewer instructions per byte on English, code and Chinese. Token ids are unchanged.
- The chunk cache is 16-way set-associative, probed by a one-byte tag per slot (eight lanes per `u64`), instead of direct-mapped: two chunks sharing a home slot no longer evict each other forever. Another 12-30% off instructions per byte, English included.
- The chunk cache gives long chunks as many slots as short ones and overwrites their buffers in place, so a script whose chunks are all long stops evicting its own working set. Allocations per encode fall to single digits.
- Bundled vocabularies are no longer default features; enable `vocabs`, or the `vocab-*` families needed. The Python wheel is unaffected — `python` pulls them in.
- The case-split pre-tokenizers skip runs of CJK ideographs from their lead byte instead of decoding each character. Token ids are unchanged.

## [0.17.0] - 2026-08-10

### Changed

- The chunk cache holds more entries by default, retiring 5-9% fewer instructions per encode across the bundled vocabularies.
- Chunk cache entries store their chunk and ids inline rather than in two heap allocations, cutting allocations per encode by 98% and cache misses by 7-12%. Chunks too long to store inline — which is every CJK chunk, and every chunk at all under ByteLevel — go to a second tier rather than going uncached. Its memory is now reserved up front instead of growing.
- The encoder holds every token's bytes in one arena addressed by span, rather than a map of individually allocated keys: allocations per `from_json` load drop a further 35-45%, load retires 7-11% fewer instructions, and encode retires 1.5-3.8% fewer with 15-39% fewer cache misses.
- The decode table addresses token bytes by id in one buffer instead of a map keyed by id, retiring 3-6% fewer instructions per load, cutting allocations per `from_json` load by 15-19%, and holding 3-6 MB less.
- Merge ranks for three- and four-byte keys come from a packed side table rather than the vocabulary map, retiring ~1% fewer instructions per encode and taking 9-13% off cache misses for models with their own merge list.
- The case-split pre-tokenizers take an all-ASCII shortcut through their letter branches, and skip them outright where no letter branch can match: Kimi encodes ~31% and o200k/gpt-oss ~23% fewer instructions. Token ids are unchanged.
- `from_json` reads `model.merges` without materializing a `serde_json::Value` per entry, sizes its merge-rank tables up front, and builds its vocabulary tables once instead of rebuilding or re-copying them, retiring 20-35% fewer instructions per load.
- `Encoder` is a concrete type rather than a `HashMap` alias: `get` returns `Option<u32>` by value and `insert` takes `&[u8]`.
- `Decoder` is a `DecodeTable` rather than a map: `get` takes a `u32` by value and returns `Option<&[u8]>`.
- `Tokenizer::with_merge_ranks` takes an `Encoder` rather than a `FxHashMap<Vec<u8>, u32>`, matching what `encoder()` already returns.

### Removed

- `TokenBytes`. The encoder owns its token bytes in one arena, so there is nothing left for it to spell.

## [0.16.1] - 2026-08-09

### Added

- `encode_batch_flat` returns a batch as one flat id buffer plus row offsets instead of `list[list[int]]`, 4.7x faster on large batches.

### Changed

- Batch encoding releases the GIL.
- Batches too small to pay for a thread pool encode on the calling thread, making them ~1.4x faster.
- ByteLevel `tokenizer.json` vocabularies resolve most pre-tokens without mapping them into ByteLevel space first, ~17% fewer instructions per encode. Token ids are unchanged.
- Kimi encodes ~16% faster: its case-split letter runs skip ASCII eight bytes at a time instead of one character per call.
- Bundled vocabularies embed a packed binary form instead of base64 text, loading ~2x faster. Encode and decode are unchanged.
- The bundled vocabulary constants are now `*_VOCAB_PACKED` and take `Tokenizer::from_packed_chain` / `from_packed_byte_level_chain`. `.tiktoken` files and `from_file` are unaffected.
- Bundled vocabularies borrow their token bytes from the embedded payload instead of copying them, loading up to 3.6x faster. Encode is unchanged.
- The encoder and decoder maps are keyed by `TokenBytes` rather than `Vec<u8>`; `Tokenizer::encoder()` and `decoder()` return those maps.

## [0.15.0] - 2026-08-09

### Added

- `qwen3` (also `qwen`, `qwen2`, `qwen2.5`, `baichuan_m2`) — Qwen 2/3's 151,643-token vocabulary. Baichuan-M2 ships it unchanged, so the alias costs nothing.
- `glm4` (also `glm`, `glm-4`, `glm4.5`, `glm-4.5`) — GLM-4/4.5's 151,329-token vocabulary.
- `gpt-oss` (also `gpt_oss`, `o200k_harmony`) — OpenAI gpt-oss, o200k_base's ranks with the harmony special tokens and no vocabulary data of its own.
- `kimi_k2` (also `kimi`, `kimi_k2.5`, `kimi_linear`) and `kimi_k3` — Moonshot's Kimi. One embedded rank file serves both: K2 and K3 share every merge rank and the pre-tokenizer, and differ only in what the 256 reserved ids above them are called.
- A direct pre-tokenizer scanner for Kimi, making it 2.1x faster than the regex path.
- A `vocab-*` cargo feature per bundled vocabulary family, all on by default and all enabled in the Python wheel. Turning one off drops its embedded data, and `from_pretrained` names the missing feature.
- `special_tokens()` on `AnyTokenizer` (Rust and Python) lists every special token a tokenizer knows, for any loader.
- Agent-token constants for the three new vocabularies: `QWEN3_AGENT_TOKENS`, `GLM4_AGENT_TOKENS`, `GPT_OSS_AGENT_TOKENS`.
- Rust agent-token constant modules for all ten agent-carrying vocabularies, not just the two OpenAI ones, generated from the same table as the Python classes.
- The agent-token tables in `docs/special_tokens.md` are generated from that same table and gated in CI, listing each token's offset rather than an id column per vocabulary.

### Changed

- An agent token whose name a vocabulary already defines keeps the vocabulary's id — Qwen's `<|im_start|>` stays 151644 — instead of being overwritten.

## [0.14.4] - 2026-08-08

### Changed

- Pre-tokenization runs a direct scanner instead of the regex engine, making single-text encode 1.4-2x faster. Token ids are unchanged.
- Tokenizers loaded with `from_json` no longer allocate a string per pre-token, size their buffers up front, and skip the segment bookkeeping their split behaviour discards, making them 2x faster on single texts.
- Normalizers that leave the text unchanged no longer copy it. A vocabulary declaring `NFC`, as most HuggingFace files do, encodes up to 2.4x faster.
- Chunk-cache hits take a shared lock instead of an exclusive one, making `encode_batch` 1.2-1.3x faster. Cache eviction is now nearer insertion order than least-recently-used.

## [0.14.3] - 2026-08-08

### Fixed

- A `tokenizer.json` whose `Split` sets `"invert": true` is split again. Its matches are contiguous, so deriving the content by complementing the gaps left one span and the text was never pre-tokenized — wrong ids for any whitespace run, Phi-4, OLMo 2 and the Xenova GPT-4/GPT-4o exports among the vocabularies affected.

## [0.14.2] - 2026-08-08

### Changed

- Bumped the regexr dependency to `0.3.1`.

## [0.14.1] - 2026-08-08

### Changed

- Bumped the regexr dependency to `0.3`.

## [0.14.0] - 2026-08-06

Short-chunk encoding regressed 20–38% in `0.12.0` and is now faster than `0.11.0`; `encode_batch` scales with its cores instead of serializing on one lock.

### Fixed

- Merge selection scans below 64 symbols and uses the heap above, instead of always heaping — `0.12.0` optimized an asymptote that pre-tokenized chunks never reach.
- `encode_batch` serialized every worker on a single chunk-cache mutex; the cache is now sharded.
- `byte_level_encode` under-reserved, reallocating on every ByteLevel chunk.
- A chunk's ids were built into a fresh vector and copied; they now write through to one buffer, with the merge's node list on the stack for short pieces.
- The pre-tokenizer copied the input to seed its pipeline and again per piece; splitting stages now yield subslices.
- Two-byte merge ranks come from a direct-indexed table rather than a hash lookup — about a third of all rank lookups.

### Added

- `NO_SPLIT_PATTERN`, what a `tokenizer.json` with `pre_tokenizer: null` loads as.
- `tests/encode_cost.rs` — allocation-flat merging, sub-quadratic unsplit pieces, real batch parallelism.
- `benches/encode.rs`, a `fluxbench` suite over short chunks, cached chunks, unsplit pieces and batch scaling.

## [0.13.0] - 2026-08-05

### Added

- `pre_tokenize` on `Tokenizer` and `AnyTokenizer` — the pre-tokenizer's pieces for a text.
- `normalize` on `Tokenizer`, `SpmTokenizer`, `SentencePieceTokenizer` and `AnyTokenizer` — the text after normalization, before the model.
- Python `decode_token_bytes` and `decode_token` on every tokenizer class.
- The reference fixtures pin the pre-tokenizer split and the normalization stage alongside ids and decoded text. SentencePiece has no split, so `mistral`/`mistral_v2` pin normalization only.
- The differential fuzzer runs in CI against the bundled OpenAI vocabularies.
- A `Guarantees` section in the API guide, each entry citing the test that enforces it.

### Fixed

- The differential fuzzer compared `tiktoken`'s `encode_ordinary` against splintr's `encode` rather than its `encode_ordinary`, so one of its three modes checked the wrong pair.
- `maturin develop --features pcre2` is documented with the `python` feature it also needs.

## [0.12.0] - 2026-08-04

Breaking: `decode` drops declared special tokens by default on every bundled vocabulary; `Tokenize` gained required methods; `ByteLevelStreamingDecoder` and `StreamingDecoder::new` are gone; `decode_bytes` returns a `Result`; streaming `add_token`/`add_tokens` are strict.

### Added

- `SpecialDecode::{Skip, Render}`, threaded through `decode_with` and `streaming_decoder_with`; Python gains `decode_with_special` and `streaming_decoder_with_special`. Previously there was no equivalent of `skip_special_tokens=False`.
- `decode_token_bytes` and `decode_token` — one id's contribution to decoded output.
- Byte fallback for BPE tokenizers: the `ByteFallback` type, `Tokenizer::with_byte_fallback`, `has_byte_fallback`, and `model.fuse_unk`.
- `streaming_decoder()` on `SentencePieceTokenizer`, `SpmTokenizer`, `WordPieceTokenizer` and `AnyTokenizer`. On `AnyTokenizer` it drives the `tokenizer.json` file's declared decoder pipeline; a shape that cannot be computed incrementally returns `TokenizeError::UnstreamableDecoder` naming the step.
- `add_token_lossy` and `add_tokens_lossy`, the skip-unknown twins of the now-strict streaming pair.
- `PreTokenizer`, `PreTokStage`, `SplitBehavior` and `SplitPattern` are public, so `with_pre_tokenizer` is callable from outside the crate.
- `Normalizer`, `NormOp` and `Precompiled` are public, so `with_normalizer` is callable from outside the crate.
- Reference fixtures for every bundled vocabulary that has a reference, pinning decoded text as well as token ids.
- `tests/decode_agreement.rs` — streaming decode plus `flush()` equals whole-sequence decode for every bundled vocabulary, backend and chunk size.
- `scripts/verify_external_models.py` — pre-release sweep of `from_json` and the bundled SentencePiece vocabularies against the published model tokenizers.

### Changed

- `decode` drops each bundled vocabulary's declared special ids by default, so it agrees with the reference tokenizers and with `from_json` on the same file; `decode_with(ids, SpecialDecode::Render)` restores the previous output. `mistral_v3`'s arm follows V1/V2 by consistency — no Tekken reference was available to measure it against.
- A streaming decoder comes only from `Tokenizer::streaming_decoder()`, which takes ByteLevel unmapping, the ids to drop and the metaspace substitution from the tokenizer's own configuration. It carries no lifetime and shares the vocabulary map rather than copying it.
- Python's `StreamingDecoder` wraps the Rust decoder instead of re-implementing UTF-8 assembly over cloned maps, so it honours the skip set, the metaspace substitution and byte fallback — and is 5-30x faster.
- `add_token`/`add_tokens` return `Result<Option<String>, TokenizeError>` and report an id in no table, mirroring `decode`.
- `Tokenize` requires `decode_lossy`, `streaming_decoder`, `decode_token_bytes`, `decode_token`, `decode_with` and `streaming_decoder_with`.
- `decode_bytes` returns `Result<Vec<u8>, TokenizerError>` where it previously rendered an unknown id as empty bytes; Python's `decode` raises instead.
- BPE seeds merges by character rather than by byte for vocabularies that carry `merges` and are not ByteLevel. Byte seeding can never reassemble a UTF-8 character 3 bytes or wider, so `▁` shattered into byte fallbacks.
- Byte fallback is resolved before merging when a merge needs it, as `tokenizers` does. No published vocabulary exercises this order; it is pinned against a measurement on a purpose-built vocabulary.
- BPE merge selection uses a binary heap instead of rescanning the linked list, dropping the merge loop from O(N×M) to O(N log N).

### Fixed

- A `Split` pre-tokenizer with a string pattern splits on that string literally instead of compiling it as a regex.
- A `pre_tokenizer` that is absent, `null` or a declared-but-empty `Sequence` runs the model over the whole normalized string instead of falling back to the GPT-2 split pattern. A `Sequence` with no `pretokenizers` key, a `Split` with no pattern, and a node with no type are refused rather than guessed.
- A bare `Metaspace` pre-tokenizer on a BPE model is honoured, instead of falling through to the plain BPE path with the prefix force-disabled.
- `add_prefix_space` tests for the literal space, so a leading tab or newline no longer suppresses the prefix.
- Byte fallback resolves per character, so a vocabulary declaring only some of the 256 `<0xNN>` entries loads.
- `model.unk_token` is honoured even when `model.byte_fallback` is false.
- `<0xNN>` ids decode to the byte they denote rather than their literal spelling, on both decode paths; a character split across several of them reassembles across `add_token` calls. Surfaces are parsed with a strict two-hex-digit rule, so `<0x1>` stays literal text.
- SentencePiece decode no longer leaks `<s>`/`</s>`/`<unk>`; the ids to drop are declarable with `with_special_decode_ids`.
- WordPiece resolves decode-dropped specials by id, not by spelling, so `[unusedN]` is no longer dropped.
- A GGUF `t5` vocabulary drops its declared BOS/EOS/unk ids on decode, as the `llama` arm already did.
- `decode_token_bytes` returns empty bytes for an id that is in the vocabulary but carries no surface; only an id past the end of the vocabulary is `InvalidTokenId`.
- Streaming decode emits U+FFFD for a byte that can never be valid UTF-8 and continues, instead of stalling.
- `encode_rayon` goes through the same dispatch as `encode`, so it no longer skips the normalizer and added-token handling.
- The chunk cache is keyed by the chunk bytes, so a hash collision cannot return another chunk's token ids.
- `scripts/extract_reference_cases.py` parses tiktoken vocabulary lines by their trailing separator, so a legitimately empty base64 payload is not read as malformed.

### Removed

- `ByteLevelStreamingDecoder` and the Python `byte_level_streaming_decoder()`, superseded by `streaming_decoder()`.
- `StreamingDecoder::new` — a decoder comes only from the tokenizer that will feed it.

## [0.11.0] - 2026-08-03

### Added

- `from_gguf_vocab(GgufVocab { .. })` builds a tokenizer from the `tokenizer.ggml.*` metadata a model runtime already parsed, without splintr opening a GGUF container. It dispatches on `tokenizer.ggml.model` (`gpt2` → byte-level BPE, `llama` → SPM-BPE, `t5` → Unigram, `bert` → WordPiece) and rejects what it cannot honour rather than guessing.
- A `bert` GGUF vocabulary is wrapped in `[CLS] A [SEP]` built from the ids it names, so it agrees with the same model's `tokenizer.json` instead of handing a CLS-pooling consumer bare content tokens.
- `SpmTokenizer`, an SPM-BPE backend for llama.cpp-style `SPM` vocabularies — SentencePiece BPE with merge-by-rank rather than Viterbi. Mistral V1/V2 route through it.
- `SpecialMode` (`All` | `Ordinary` | `Allow(&FxHashSet<String>)`) on `encode_with`, with `encode_with_special` / `encode_ordinary` / `encode_allowed_special` on every Python tokenizer class, so untrusted text cannot forge a control token.
- `base_vocab_size` and `base_vocab_size_by_name` — a vocabulary's size as its upstream reference defines it, without splintr's agent tokens, for sizing embedding and logit layers.
- Multi-pass pre-tokenizer chains: DeepSeek V3 uses its own three-pass pre-tokenizer and Llama 3 its own pattern, instead of borrowing o200k's.
- Pair-sequence encoding through the special-token policy, plus a `cls_sep_ids` accessor for BERT-style wrapping.
- `decode` as an inherent method on `AnyTokenizer`, alongside the `encode` / `encode_raw` / `encode_with` / `encode_batch` surface shared by every Python tokenizer class.
- `scripts/fuzz_reference.py`, a differential fuzzer diffing splintr against `tokenizers`, `transformers` or `tiktoken` on strings assembled from each vocabulary's own added and special tokens, with deterministic seeds and shrinking; and `examples/verify_gguf.rs` for llama.cpp's own `.inp`/`.out` fixtures.
- Integration tests covering the Unigram and WordPiece backends loaded from `tokenizer.json`.

### Changed

- `from_pretrained` returns an `AnyTokenizer` for every bundled vocabulary in both Rust and Python, so a vocabulary name means the same thing on both sides regardless of backend.
- WordPiece accent stripping is its own setting (`with_strip_accents`), seeded from `lowercase` and overridable independently — what cased multilingual BERT needs.
- Bundled vocabulary files moved out of the Python package into a top-level `vocabs/`, shipped through an explicit `include` list in `Cargo.toml`.
- The SentencePiece decode mode is named for what it does: the metaspace decoder.
- Python wheels are abi3 (`cp38-abi3`), so one wheel per platform covers every CPython from 3.8 up. Previously each platform shipped a wheel for whichever single interpreter its builder used, and every other version fell back to compiling the sdist.
- pyo3 0.29 (closes RUSTSEC-2026-0176, RUSTSEC-2026-0177) and crossbeam-epoch 0.9.20 (closes RUSTSEC-2026-0204). A `cargo-deny` gate fails CI on any advisory, disallowed license, or non-crates.io source.
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
