# Splintr API Guide

This guide provides comprehensive documentation for using Splintr's Python and Rust APIs. For a quick start, see the [main README](../README.md).

## Table of Contents

- [Python API Reference](#python-api-reference)
  - [Tokenizer Class](#tokenizer-class) (BPE)
  - [Encoding Methods](#encoding-methods)
  - [Special tokens in untrusted text](#special-tokens-in-untrusted-text)
  - [Decoding Methods](#decoding-methods)
  - [Cache Management](#cache-management)
  - [Sizing against the reference vocabulary](#sizing-against-the-reference-vocabulary)
  - [SentencePiece Tokenizer Class](#sentencepiece-tokenizer-class) (Unigram)
  - [Loading any model from `tokenizer.json`](#loading-any-model-from-tokenizerjson)
- [Guarantees](#guarantees)
- [Streaming Decoder](#streaming-decoder)
  - [Basic Usage](#basic-usage)
  - [ByteLevel Vocabularies](#bytelevel-vocabularies)
  - [API Methods](#api-methods)
  - [When Streaming Is Refused](#when-streaming-is-refused)
- [Rust API Reference](#rust-api-reference)
  - [Loading a bundled vocabulary](#loading-a-bundled-vocabulary)
  - [Tokenize Trait](#tokenize-trait)
  - [Per-Token Decoding](#per-token-decoding)
  - [AnyTokenizer](#anytokenizer)
  - [SpecialMode](#specialmode)
  - [BPE Tokenizer](#bpe-tokenizer)
  - [SentencePiece Tokenizer](#sentencepiece-tokenizer)
  - [WordPiece Tokenizer](#wordpiece-tokenizer)
- [Detailed Usage Examples](#detailed-usage-examples)
  - [Basic Encoding and Decoding](#basic-encoding-and-decoding)
  - [Batch Processing](#batch-processing)
  - [Special Tokens Usage](#special-tokens-usage)
  - [Agent Tokens Usage](#agent-tokens-usage)
  - [Streaming Examples](#streaming-examples)

## Python API Reference

### Tokenizer Class

The `Tokenizer` class is the main entry point for tokenization in Python.

#### Loading

**Load a pretrained model:**

```python
from splintr import Tokenizer

# Load pretrained model (includes vocabulary and special tokens)
tokenizer = Tokenizer.from_pretrained("cl100k_base")  # OpenAI GPT-4/3.5
tokenizer = Tokenizer.from_pretrained("o200k_base")   # OpenAI GPT-4o
tokenizer = Tokenizer.from_pretrained("llama3")       # Meta Llama 3 family
tokenizer = Tokenizer.from_pretrained("deepseek_v3")  # DeepSeek V3/R1
tokenizer = Tokenizer.from_pretrained("qwen3")        # Qwen 2/3, Baichuan-M2
tokenizer = Tokenizer.from_pretrained("glm4")         # GLM-4/4.5
tokenizer = Tokenizer.from_pretrained("gpt-oss")      # OpenAI gpt-oss
tokenizer = Tokenizer.from_pretrained("kimi")         # Kimi K2/K2.5/K2.6/K2.7
tokenizer = Tokenizer.from_pretrained("kimi_k3")      # Kimi K3
tokenizer = Tokenizer.from_pretrained("mistral_v1")   # Mistral 7B v0.1/v0.2, Mixtral 8x7B
tokenizer = Tokenizer.from_pretrained("mistral_v2")   # Mistral 7B v0.3, Codestral, Mixtral 8x22B
tokenizer = Tokenizer.from_pretrained("mistral_v3")   # Mistral NeMo, Large 2, Pixtral
tokenizer = Tokenizer.from_pretrained("whisper_v3")   # OpenAI Whisper multilingual (v1/v2/v3; bare "whisper" → v2)
```

> Whisper English-only checkpoints (`*.en`) use a different base BPE and are not bundled — load those with [`from_json`](#loading-any-model-from-tokenizerjson).

`from_pretrained` returns an [`AnyTokenizer`](#anytokenizer) for **every** bundled vocabulary — the same universal handle `from_json` returns, and the same one `splintr::pretrained::from_pretrained` returns in Rust. It delegates to that one loader, so a vocabulary name means the same thing, and produces the same ids, on both sides of the binding. Query `.family` for the backend it dispatched to (`"BPE"` for the byte-level vocabularies, `"Spm"` for Mistral V1/V2).

**Load a raw `.tiktoken` file:**

```python
from splintr import Tokenizer, CL100K_BASE_PATTERN

tokenizer = Tokenizer(
    vocab_path="path/to/vocab.tiktoken",
    pattern=CL100K_BASE_PATTERN,
    special_tokens={"<|endoftext|>": 100257}  # optional
)
```

A `.tiktoken` file is `base64(token bytes) rank` per line and states nothing else — no pre-tokenizer pattern, no special tokens, no decoder chain — so `pattern` is required and `special_tokens` is yours to supply. This is the one loader that returns the concrete `Tokenizer` rather than an [`AnyTokenizer`](#anytokenizer): with no `model.type` to read there is no backend to dispatch on, and it is always byte-level BPE.

Ids are the same as any other route to the same vocabulary — loading `crates/vocab-cl100k/vocabs/cl100k_base.tiktoken` this way encodes identically to `from_pretrained("cl100k_base")`. In Rust: `Tokenizer::from_file(vocab_path, pattern, special_tokens)`, or `Tokenizer::from_bytes(&data, pattern, special_tokens)` for a vocabulary already in memory.

### Encoding Methods

Every Python tokenizer class — `Tokenizer`, `AnyTokenizer`, `SpmTokenizer`, `SentencePieceTokenizer`, `WordPieceTokenizer` — exposes the same six encoding methods, and each means the same thing on all of them:

| Method                                  | Meaning                                               | HuggingFace / tiktoken equivalent                                |
| --------------------------------------- | ----------------------------------------------------- | ---------------------------------------------------------------- |
| `encode(text)`                          | Model-ready: boundary template applied                | `tokenizer.encode(text)` (its default `add_special_tokens=True`) |
| `encode_raw(text)`                      | Content tokens only, no template                      | `tokenizer.encode(text, add_special_tokens=False)`               |
| `encode_ordinary(text)`                 | Never match a special token spelled out in `text`     | tiktoken `allowed_special=set()`                                 |
| `encode_with_special(text)`             | Match every special token spelled out in `text`       | tiktoken `allowed_special="all"`                                 |
| `encode_allowed_special(text, allowed)` | Match only the listed ones; `ValueError` on any other | tiktoken `allowed_special={...}`                                 |
| `encode_batch(texts)`                   | Batch form of `encode`, parallel across texts         | `tokenizer.encode_batch(texts)`                                  |

Two independent questions are at work here, and mixing them up is the one way to get this wrong:

1. **Boundary tokens** — the `[CLS]…[SEP]` / `<s>…</s>` wrapper the model was trained to receive. It comes from the tokenizer's `post_processor` template, never from the text. `encode` adds it; `encode_raw` does not. Every _other_ method on the list also adds it, because refusing to match a special token inside untrusted content says nothing about whether the model wants a BOS.
2. **Special tokens spelled out in the input** — a user typing `<|endoftext|>`. `encode_ordinary` / `encode_with_special` / `encode_allowed_special` decide whether those become real control-token ids. `encode` uses the tokenizer's own default (see below).

#### `encode(text: str) -> list[int]`

Encode text to token IDs, model-ready. Sequential processing, optimal for texts under ~1MB.

```python
tokens = tokenizer.encode("Hello, world!")
print(tokens)  # [9906, 11, 1917, 0]
```

Vocabularies loaded through `Tokenizer.from_pretrained` declare no boundary template, so for them `encode` and `encode_raw` return the same ids. A tokenizer loaded from a `tokenizer.json` usually does declare one:

```python
from splintr import from_json

tok = from_json("/path/to/llama-3.2-1b/tokenizer.json")
tok.encode("Hello, world!")      # [128000, 9906, 11, 1917, 0]  — 128000 is BOS
tok.encode_raw("Hello, world!")  # [9906, 11, 1917, 0]
```

#### `encode_raw(text: str) -> list[int]`

Encode text to token IDs, content only — no boundary template. Use it when you assemble the sequence yourself (a chat template, a reranker pair) and place the boundary tokens by hand. Whatever `encode` adds and this does not _is_ the template.

```python
tok = from_json("/path/to/bge-m3-tokenizer/tokenizer.json")
tok.encode("Hello, world!")      # [0, 35378, 4, 8999, 38, 2]  — [CLS] … [SEP]
tok.encode_raw("Hello, world!")  # [35378, 4, 8999, 38]
```

#### `encode_with_special(text: str) -> list[int]`

Encode text, matching every configured special token spelled out in it: the special token becomes its single control-token id rather than being split into ordinary pieces.

```python
tokenizer = Tokenizer.from_pretrained("cl100k_base")
text = "Start <|endoftext|> End"

tokenizer.encode(text)               # [3563, 220, 100257, 4060]  — 100257 is <|endoftext|>
tokenizer.encode_with_special(text)  # [3563, 220, 100257, 4060]  — the same
tokenizer.encode_ordinary(text)      # [3563, 83739, 8862, 728, 428, 91, 29, 4060]
```

Every loader — `from_pretrained`, `from_json`, the GGUF loader — turns added-token matching **on**, so `encode` and `encode_with_special` agree on everything they return; the method exists so the name means the same thing on every tokenizer class. `encode_ordinary` is how you decline the match. Name the mode explicitly whenever the text is untrusted — see below.

#### `encode_batch(texts: list[str]) -> list[list[int]]`

Encode multiple texts in parallel using Rayon — the batch form of `encode`, with the boundary template applied to each result. This is where Splintr really shines: roughly 10-12x tiktoken's throughput on the batch workloads `benchmarks/benchmark_batch.py` runs.

```python
texts = ["Hello, world!", "How are you?"]
batch_tokens = tokenizer.encode_batch(texts)
# [[9906, 11, 1917, 0], [4438, 527, 499, 30]]
```

#### `encode_rayon(text: str) -> list[int]`

Same result as `encode`, but Rayon parallelizes _within_ the single text. This is only beneficial for very large texts (>1MB); for typical use cases `encode()` is faster. A backend with no intra-text parallel path simply runs `encode`, so the ids never depend on which one you hold.

```python
# Only useful for very large texts
large_text = "..." * 1000000  # >1MB of text
tokens = tokenizer.encode_rayon(large_text)
```

#### `encode_batch_with_special(texts: list[str]) -> list[list[int]]`

The batch form of `encode_with_special`, parallel across texts.

#### `encode_batch_flat(texts)`

The flat counterpart to `encode_batch`: one contiguous buffer of ids plus the offsets that cut it into rows, as two `bytes` objects, rather than `list[list[int]]`.

```python
import numpy as np

ids, offsets = tok.encode_batch_flat(texts)
ids = np.frombuffer(ids, dtype=np.uint32)        # every token, concatenated
offsets = np.frombuffer(offsets, dtype=np.uint64)  # len(texts) + 1 entries
assert ids[offsets[i]:offsets[i + 1]].tolist() == tok.encode_batch(texts)[i]
```

`ids` is little-endian `uint32`; `offsets` is little-endian `uint64` and always starts at 0 and ends at the total token count, so `offsets[i]:offsets[i + 1]` is row `i` and an empty text is an empty (not missing) row. Both buffers are one `memcpy` each.

It exists because the object construction dominates: a Python list of ints costs ~18 ns per token, measured at 89% of `encode_batch`'s wall time on a 220k-token batch, which makes the flat form ~4.7x faster end to end. Available on `Tokenizer` and `AnyTokenizer`, so `from_pretrained`, `from_json` and the raw `.tiktoken` loader all have it. See [Best Practices](best_practices.md#getting-the-throughput).

Batch calls also release the GIL for the duration of the encode, so a threaded data loader is no longer serialized against the tokenizer.

### Special tokens in untrusted text

A tokenizer that matches special tokens will promote text that _spells_ a control token to that token's real id. `<|im_start|>` typed by a user becomes the same id the server emits when it opens a turn, and downstream nothing can tell the two apart — that is how a user message forges a system turn. Denylisting the spelling beforehand does not close it: the spelling is not the only thing that maps to the id.

So the three matching modes are explicit methods. In Rust they are the [`SpecialMode`](#specialmode) enum passed to `Tokenize::encode_with`.

```python
from splintr import Tokenizer

tokenizer = Tokenizer.from_pretrained("cl100k_base")
untrusted = "Start <|endoftext|> End"

# Match none: the literal spelling stays ordinary content.
tokenizer.encode_ordinary(untrusted)
# [3563, 83739, 8862, 728, 428, 91, 29, 4060]

# Match a named subset.
tokenizer.encode_allowed_special(untrusted, ["<|endoftext|>"])
# [3563, 220, 100257, 4060]

# Anything outside the allow-list is rejected, naming the token and its offset.
tokenizer.encode_allowed_special(untrusted, [])
# ValueError: special token "<|endoftext|>" at byte offset 6 is not in
#             the caller's allow-list
```

The model's own boundary tokens are unaffected by the mode — they come from the `post_processor` template, not from matching text against the vocabulary, so locking down matching does not silently strip the BOS the model was trained with. Use `encode_raw` when you want no boundary tokens at all.

### Decoding Methods

#### `decode(tokens: list[int]) -> str`

Decode token IDs back to text. Raises an error if the decoded bytes are not valid UTF-8.

```python
tokens = [9906, 11, 1917, 0]
text = tokenizer.decode(tokens)
print(text)  # "Hello, world!"
```

Control tokens render as nothing — HuggingFace's default `skip_special_tokens=True`, which every loader implements, so the same vocabulary decodes the same way whether it came from `from_pretrained`, from a `tokenizer.json` or from a GGUF file:

```python
tok = Tokenizer.from_pretrained("mistral_v2")
tok.decode([3])                          # ""  — [INST]
tok.decode(tok.encode_with_special("[INST]Hi[/INST]"))  # "Hi"
```

What counts as a control token is the vocabulary's own declaration, not a guess: DeepSeek marks `<｜User｜>` and `<｜Assistant｜>` as ordinary added tokens, so those still render, and Whisper's timestamp tokens (`<|0.00|>`…`<|30.00|>`) are transcript content and render too. The OpenAI vocabularies (`cl100k_base`, `o200k_base`) follow `tiktoken`, which renders `<|endoftext|>` and its siblings. Use `special_token_id(name)` when you want a marker's spelling or id rather than its decoded text.

#### `decode_bytes(tokens: list[int]) -> bytes`

Decode token IDs to raw bytes without UTF-8 validation. Needs the byte-level BPE backend (`family == "BPE"`) and a source that declares no `decoder` pipeline — reading token bytes directly is exactly what would bypass such a pipeline — and raises `ValueError` otherwise. Every bundled byte-level vocabulary qualifies; use `decode` for the rest.

```python
tokens = [9906, 11, 1917, 0]
raw_bytes = tokenizer.decode_bytes(tokens)
print(raw_bytes)  # b'Hello, world!'
```

#### `decode_lossy(tokens: list[int]) -> str`

Decode token IDs to text, replacing any invalid UTF-8 sequences with the replacement character (�). Same backend requirement as `decode_bytes`.

```python
tokens = [9906, 11, 1917, 0]
text = tokenizer.decode_lossy(tokens)
# Invalid UTF-8 sequences become �
```

#### `decode_token_bytes(id: int) -> bytes`

A single id's own contribution to the decoded stream — ByteLevel alphabet unmapped and `<0xNN>` byte fallback resolved — for callers that need to attribute output to one token rather than render a sequence: logprob display, token-level highlighting, debugging a vocabulary. Available on every tokenizer class (`Tokenizer`, `SentencePieceTokenizer`, `SpmTokenizer`, `WordPieceTokenizer`, `AnyTokenizer`).

No sequence-level post-processing runs — no leading-space strip, no first-token rule, no word separator — so **concatenating this over a sequence of ids is not the same as calling `decode` on that sequence**. Use [`streaming_decoder()`](#streaming-decoder) to render a whole stream.

An id in the vocabulary that carries no surface (a control token `decode` drops, say) returns empty bytes, not an error; `ValueError` is reserved for an id outside the vocabulary altogether.

```python
tok = Tokenizer.from_pretrained("cl100k_base")
ids = tok.encode("Hello, world!")
tok.decode_token_bytes(ids[0])  # b'Hello'

# Concatenating per-token bytes is not `decode`'s output — see above.
b"".join(tok.decode_token_bytes(i) for i in ids) == tok.decode(ids).encode()
# not guaranteed, and false on vocabularies with a separator or byte fallback
```

#### `decode_token(id: int) -> str`

`decode_token_bytes` as text. Raises far more often than `decode` does: a single `<0xNN>` byte-fallback id, or a token holding one byte of a multi-byte character, is not valid UTF-8 standing alone — that is the expected signal to stop decoding id-at-a-time and use `streaming_decoder()`, which buffers exactly those partial sequences across tokens.

```python
tok = Tokenizer.from_pretrained("cl100k_base")
ids = tok.encode("Hello, world!")
tok.decode_token(ids[0])  # "Hello"
```

### Properties

#### `vocab_size: int`

The total vocabulary size including special tokens and splintr's 54 agent tokens.

```python
print(tokenizer.vocab_size)  # 100331 for cl100k_base with agent tokens
```

For the size the upstream reference reports — what you need to size an embedding or logit layer — use [`base_vocab_size`](#sizing-against-the-reference-vocabulary).

#### `cache_len: int`

The number of entries currently in the LRU cache. The byte-level BPE backend is the only one that caches encoded chunks; on any other family this raises `ValueError`, as does `clear_cache()`.

```python
print(tokenizer.cache_len)  # Number of cached text chunks
```

### Cache Management

#### `clear_cache()`

Clear the LRU encoding cache. Useful if memory pressure is a concern.

```python
tokenizer.clear_cache()
```

### Sizing against the reference vocabulary

`base_vocab_size(name)` is a module-level function reporting a vocabulary's size _as its upstream reference defines it_ — without splintr's 54 agent tokens. That is the number to size a model's embedding or logit layer with, or to identify which vocabulary a checkpoint uses from the shape of its token-embedding tensor: both must match the checkpoint's vocabulary, not splintr's extended one. Agent tokens are appended strictly above every id the reference uses, so this is also exactly the id at which splintr's additions begin.

```python
from splintr import Tokenizer, base_vocab_size

tokenizer = Tokenizer.from_pretrained("cl100k_base")
print(tokenizer.vocab_size)             # 100331 — extended (base + 54 agent)
print(base_vocab_size("cl100k_base"))   # 100277 — what tiktoken reports
print(base_vocab_size("llama3"))        # 128256
print(base_vocab_size("mistral_v3"))    # 131072
```

It accepts the same names as `Tokenizer.from_pretrained` and raises `ValueError` for anything else. It is _not_ `vocab_size - 54`: several reference vocabularies leave gaps below their nominal size, so the difference varies per vocabulary. (Rust: `splintr::pretrained::base_vocab_size(vocab)` taking a `PretrainedVocab`, or `base_vocab_size_by_name(name)` returning `Result<u32, TokenizerError>`.)

### SentencePiece Tokenizer Class

The `SentencePieceTokenizer` class provides unigram tokenization for models using SentencePiece (e.g., loaded from GGUF files).

#### Creating

```python
from splintr import SentencePieceTokenizer

# Create from raw vocabulary data
tokenizer = SentencePieceTokenizer(
    tokens=["<unk>", "<s>", "</s>", "▁Hello", "▁world"],
    scores=[0.0, 0.0, 0.0, -1.2, -1.5],
    eos_token_id=2,
    bos_token_id=1,  # optional
)
```

#### `encode(text: str) -> list[int]`

Encode text using Viterbi maximum-score segmentation (true SentencePiece Unigram, not greedy), with byte fallback for unknown characters. Prepends BOS if configured — BOS is a boundary token, so `encode_raw` omits it.

```python
ids = tokenizer.encode("Hello world")
# [1, 3, 4]  (BOS + ▁Hello + ▁world)
```

The other five [encoding methods](#encoding-methods) (`encode_raw`, `encode_ordinary`, `encode_with_special`, `encode_allowed_special`, `encode_batch`) are present here too and mean exactly what they mean elsewhere.

#### `decode(ids: list[int]) -> str`

Decode token IDs to text. Skips BOS/EOS tokens, converts ▁ back to spaces.

```python
text = tokenizer.decode([1, 3, 4])
# "Hello world"
```

#### `decode_lossy(ids: list[int]) -> str`

Decode token IDs, silently skipping any invalid (out-of-range) IDs.

```python
text = tokenizer.decode_lossy([1, 3, 999, 4])
# "Hello world"  (999 is skipped)
```

#### Properties

- `vocab_size: int` — Total vocabulary size
- `eos_token_id: int` — End-of-sequence token ID
- `bos_token_id: int | None` — Beginning-of-sequence token ID (if configured)

#### Methods

- `is_eos(token_id: int) -> bool` — Check if a token is the EOS token

### Loading any model from `tokenizer.json`

For models not bundled with `from_pretrained`, load a HuggingFace `tokenizer.json` with `from_json`. It reads everything from the file — split regex, byte-level flag, BPE **merge order** (independent of token ids, so RoBERTa-style vocabs work), the full ordered normalizer (including SentencePiece's `Precompiled` charsmap), the `post_processor` template, the declared `decoder` pipeline, and special tokens.

It always returns an `AnyTokenizer` — the universal loaded-tokenizer handle — which dispatches internally to the backend matching the file's `model.type`. Query `.family` for which one:

```python
from splintr import from_json, from_json_bytes

tok = from_json("/path/to/bge-m3-tokenizer/tokenizer.json")   # from a path
# tok = from_json_bytes(open("tokenizer.json","rb").read())   # from bytes

tok.family                       # "Unigram"
tok.encode("Hello, world!")      # [0, 35378, 4, 8999, 38, 2]  — [CLS] … [SEP]
tok.encode_raw("Hello, world!")  # [35378, 4, 8999, 38]        — content only
tok.decode(tok.encode("Hello, world!"))   # "Hello, world!"
```

| `model.type` | `tok.family`  | Internal backend         | Example models                   |
| ------------ | ------------- | ------------------------ | -------------------------------- |
| `BPE`        | `"BPE"`       | `Tokenizer`              | GPT-2, RoBERTa, Qwen, Whisper.en |
| `Unigram`    | `"Unigram"`   | `SentencePieceTokenizer` | T5, Gemma, Albert, XLNet         |
| `WordPiece`  | `"WordPiece"` | `WordPieceTokenizer`     | BERT, DistilBERT, Electra        |

A fourth backend, `SpmTokenizer` (`family == "Spm"`), is not reachable from `tokenizer.json`: it is what the bundled Mistral V1/V2 vocabularies use and what the Rust `from_gguf_vocab` loader produces from a GGUF file's embedded vocabulary.

**Other members:** the six [encoding methods](#encoding-methods), `decode(ids)`, `vocab_size`, `family`, `eos_token_id`, `is_eos(id)`, and `special_token_id(name)` (the id of an added token by its content, e.g. `"[CLS]"`, or `None`).

`decode` runs the file's declared `decoder` chain after dropping `special=true` ids — HuggingFace's default `skip_special_tokens=True` — so files whose decoding _is_ that chain (Mistral, Llama, Gemma) come back as text rather than raw pieces.

**Strict by design.** Rather than silently approximate an unsupported config (which would produce wrong tokens with no signal), `from_json` raises:

- `UnsupportedModelType` — `model.type` is not BPE/Unigram/WordPiece
- `UnsupportedNormalizer` — an unrecognized normalizer step (dropping it would mis-normalize)
- `InvalidNormalizerRegex` — a `Replace` regex that fails to compile
- `UnsupportedPreTokenizer` — a declared pre-tokenizer with no recognized split (refusing to guess the pattern)

Output is verified id-for-id against HuggingFace `tokenizers` across all three families. (Rust: `splintr::from_json_path` / `from_json_bytes`.)

## Guarantees

What follows is a list of properties splintr's own test suite pins, not an aspirational list — each line names the test that fails if the property stops holding. A claim that seemed true but had no test enforcing it is left out entirely, or marked as such, rather than stated as fact.

- **Encode pipeline order.** Content runs through the declared normalizer, then the pre-tokenizer split (`add_prefix_space` applied at that stage, matching where HuggingFace's `ByteLevel`/`Metaspace` nodes own the flag — see `Tokenizer::prefixed` in `src/core/tokenizer/encode.rs`), then the BPE/Unigram/WordPiece model; special tokens are extracted before any of that runs. This is exercised end to end, not as an isolated ordering check: `tests/reference_parity.rs::pretrained_vocabularies_match_reference_tokenizers` asserts `encode`/`encode_raw` ids against each reference tokenizer's own ids for every bundled vocabulary, which cannot pass if a stage runs out of order or `add_prefix_space` is applied on the wrong side of the split.
- **Streaming decode agrees with whole-sequence decode, for any chunking.** Concatenating every `add_token`/`add_tokens` emission plus the final `flush()` equals `decode`/`decode_lossy` on the same ids, regardless of how the ids are grouped into chunks. Pinned for every bundled vocabulary and backend by `tests/decode_agreement.rs::streaming_decode_agrees_with_whole_sequence_decode`, and as a property test per backend: `src/core/streaming/decoder.rs::prop_chunking_matches_decode_cl100k_base` / `prop_chunking_matches_decode_deepseek_v3` / `prop_arbitrary_ids_match_decode_lossy`, `src/core/spm.rs::prop_chunking_matches_decode_mistral` / `prop_chunking_matches_decode_mistral_v2`, `src/core/sentencepiece.rs::prop_chunking_matches_decode`, and `src/core/wordpiece.rs::prop_chunking_matches_decode` / `prop_chunking_matches_decode_without_prefix`. `reset()` producing a decoder byte-identical to a fresh one is covered by the corresponding `prop_reset_matches_a_fresh_decoder` in each of those modules.
- **The UTF-8 reassembly buffer matches `String::from_utf8_lossy`.** Byte-at-a-time or arbitrarily chunked, the buffer's lossy output is exactly what `String::from_utf8_lossy` would produce on the same bytes fed whole. Pinned by `src/core/streaming/utf8.rs::prop_byte_at_a_time_matches_lossy` and `prop_arbitrary_chunking_matches_lossy`.
- **The pre-tokenizer split matches the reference tool's split — for bundled vocabularies that have a pre-tokenizer stage.** `AnyTokenizer::pre_tokenize` is asserted piece-for-piece against each reference's own split by `tests/reference_parity.rs::pretrained_vocabularies_match_reference_tokenizers`. The reference and its authority differ by vocabulary: for `cl100k_base`/`o200k_base` it is `regex.finditer` over the *installed* `tiktoken` encoding's own `_pat_str`, which proves splintr's pattern constant has not drifted from it and that splintr's regex engine executes that pattern identically to Python's `regex` module — it does **not** independently verify OpenAI's intent behind the pattern. For `llama3`/`deepseek_v3`/`whisper` it is HuggingFace `tokenizers`' `pre_tokenizer.pre_tokenize_str` over the normalizer's own output. `mistral` and `mistral_v2` have no split pinned here — SentencePiece has no pre-tokenizer stage, so those fixtures carry no `pieces` field at all.
- **Ids and decoded text match the reference tokenizers, for the bundled vocabularies, over the fixture corpus.** Also `tests/reference_parity.rs::pretrained_vocabularies_match_reference_tokenizers`, against `tests/fixtures/pretrained/*.json` captured by `scripts/extract_reference_cases.py` from `tiktoken` (OpenAI vocabularies), `tokenizers` (HF-published vocabularies) or `sentencepiece` (Mistral). This is a fixture-corpus guarantee, not a universal one: it holds over the cases committed to that corpus, not over all possible input. A broader, non-CI sweep against real model directories is `scripts/verify_external_models.py`; `scripts/fuzz_reference.py` looks for the corpus's blind spots (see [Correctness against the reference implementations](../CONTRIBUTING.md#correctness-against-the-reference-implementations)).
- **Concatenating `decode_token`/`decode_token_bytes` over a sequence is *not* `decode` — this is a deliberate non-guarantee, not an oversight.** Per-id decoding never includes the inter-token separator a surface may carry (WordPiece's `##` continuation spacing, a leading-space rule that only fires past the first token), because that separator is a fact about where in the sequence an id sits, not about the id itself. See [Per-Token Decoding](#per-token-decoding) for the exact contract of `decode_token`/`decode_token_bytes`.
- **`encode_ordinary` never promotes a special token's literal spelling in untrusted text.** Under `SpecialMode::Ordinary`, text that spells out `<|endoftext|>` (or any other configured special token) is tokenized as ordinary content and round-trips through `decode` unchanged — it can never become that token's control id. Pinned by `src/core/tokenizer/tests.rs::encode_with_all_vs_ordinary_diverge_on_a_special_token`. The model's own boundary tokens (BOS/EOS/`[CLS]`/`[SEP]`) are a separate concern: they come from the `post_processor` template under every mode, `Ordinary` included, and are unaffected by this guarantee — see [Special tokens in untrusted text](#special-tokens-in-untrusted-text).

Not currently pinned by a dedicated test: the streaming-decode identity above covers only the bundled vocabularies and backends reachable through `from_pretrained`; agreement for a `from_json`-loaded tokenizer's declared `decoder` pipeline is checked out of band by `scripts/verify_external_models.py` and `scripts/fuzz_reference.py`, not by an in-repo `cargo test`.

## Streaming Decoder

A streaming decoder is essential for real-time LLM applications where tokens arrive one at a time. It handles the critical problem of BPE tokens not aligning with UTF-8 character boundaries.

There is exactly one decoder class, `StreamingDecoder`, and two ways to get it — `streaming_decoder()` and `streaming_decoder_with_special()`, on the tokenizer whose stream you are decoding. They differ only in the mode `decode` and `decode_with_special` differ in: the first drops the control tokens, the second renders them, so a generation loop that needs to *see* `<|eot_id|>` go past has a stream that shows it. Every tokenizer class has it — `Tokenizer`, `SentencePieceTokenizer`, `SpmTokenizer`, `WordPieceTokenizer` and `AnyTokenizer` — and the decoder it hands back carries that tokenizer's own decode rules: the ByteLevel alphabet (DeepSeek V3, GPT-2), `<0xNN>` byte fallback, the `▁` metaspace substitution, and the `special=true` ids `decode` drops.

That is what makes the guarantee below hold on every vocabulary, and it is why there is nothing else to choose:

```python
"".join(chunks) + decoder.flush() == tokenizer.decode(ids)

# ...and, for the sibling factory, against the sibling decode:
"".join(chunks) + decoder.flush() == tokenizer.decode_with_special(ids)
```

### Why You Need This

BPE tokens don't align with UTF-8 character boundaries. A multi-byte Unicode character like "世" (3 bytes: `0xE4 0xB8 0x96`) might split across tokens. The streaming decoder:

1. Buffers incomplete byte sequences across token boundaries
2. Only outputs text when complete UTF-8 characters are available
3. Prevents display corruption in streaming LLM output
4. Handles edge cases automatically

### Basic Usage

```python
# Create a streaming decoder
decoder = tokenizer.streaming_decoder()

# Process tokens one at a time (typical LLM streaming scenario)
for token_id in token_stream:
    # Returns text only when complete UTF-8 characters are available
    if text := decoder.add_token(token_id):
        print(text, end="", flush=True)

# Flush any remaining buffered bytes at the end
print(decoder.flush())
```

### Real-World Example

```python
import openai
from splintr import Tokenizer

tokenizer = Tokenizer.from_pretrained("cl100k_base")
decoder = tokenizer.streaming_decoder()

# Stream tokens from OpenAI API
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        # Process each token as it arrives
        token_ids = get_token_ids(chunk)  # pseudo-code
        for token_id in token_ids:
            if text := decoder.add_token(token_id):
                print(text, end="", flush=True)

# Don't forget to flush at the end
print(decoder.flush())
```

### ByteLevel Vocabularies

A ByteLevel BPE vocabulary (DeepSeek V3, GPT-2) encodes raw bytes (0-255) as printable Unicode characters — space `0x20` becomes `Ġ` — so its tokens need one extra unmapping step before UTF-8 assembly. That step is part of the tokenizer's configuration, so the same call serves it:

```python
from splintr import Tokenizer

# DeepSeek V3 uses ByteLevel BPE encoding — same call, no second class
tokenizer = Tokenizer.from_pretrained("deepseek_v3")
decoder = tokenizer.streaming_decoder()

for token_id in token_stream:
    if text := decoder.add_token(token_id):
        print(text, end="", flush=True)

print(decoder.flush())
```

See [bytelevel_bpe.md](bytelevel_bpe.md) for details on ByteLevel encoding.

### API Methods

**Core operations:**

- `add_token(token_id: int) -> str | None`: Add a token, return complete characters or None if buffering. An id in no table at all is skipped, matching `decode_lossy` — one stray id must not abort a generation
- `add_tokens(token_ids: list[int]) -> str | None`: Add multiple tokens at once. Grouping changes only *when* text is emitted, never *what*
- `flush() -> str`: Flush buffered bytes (incomplete sequences become �)
- `reset()`: Clear the buffer and start fresh

**Properties:**

- `has_pending: bool`: Whether there are buffered bytes waiting
- `pending_bytes: int`: Number of bytes currently buffered

### When Streaming Is Refused

`AnyTokenizer.streaming_decoder()` is the one form that can raise. A `tokenizer.json` may declare a `decoder` pipeline holding a step that cannot be evaluated one chunk at a time (a `BPEDecoder`, a trailing `Strip`, a `Replace` over the fused text). Answering with the backend's own decode would render the raw pieces (`▁hello▁world`) the pipeline exists to turn into text, so it raises `ValueError` naming the step instead:

```python
tokenizer = from_json("path/to/tokenizer.json")
try:
    decoder = tokenizer.streaming_decoder()
except ValueError as e:
    # ... its `Strip` step is not incrementally computable — decode the whole sequence instead
    text = tokenizer.decode(ids)
```

Whole-sequence `decode` handles those files. Every other tokenizer class always returns a decoder.

## Rust API Reference

The Rust API provides similar functionality with strongly-typed interfaces. For complete documentation, see [docs.rs/splintr](https://docs.rs/splintr).

### Setup

Add Splintr to your `Cargo.toml`:

```toml
[dependencies]
splintr = "*"  # or pin to a specific version
```

### Loading a bundled vocabulary

`pretrained::from_pretrained(name)` is the Rust entry point, and it returns an `AnyTokenizer` for **every** bundled vocabulary — so the same code works whether the vocabulary needs the byte-level BPE backend or the SPM-BPE one (Mistral V1/V2):

```rust
use splintr::{pretrained::from_pretrained, Tokenize};

// fn from_pretrained(name: &str) -> Result<AnyTokenizer, TokenizerError>
let tokenizer = from_pretrained("cl100k_base")?;

let tokens = tokenizer.encode("Hello, world!");
let batch_tokens = tokenizer.encode_batch(&["Hello, world!", "How are you?"]);
let text = tokenizer.decode(&tokens)?; // `decode` comes from the `Tokenize` trait
```

`pretrained::from_vocab(vocab)` takes a `PretrainedVocab` enum instead of a name, with the same return type. To build a tokenizer from your own vocabulary rather than a bundled one, use `Tokenizer::new` — see [BPE Tokenizer](#bpe-tokenizer).

### Tokenize Trait

All tokenizer backends implement the `Tokenize` trait, enabling generic code:

```rust
use splintr::Tokenize;

fn count_tokens(tokenizer: &dyn Tokenize, text: &str) -> usize {
    tokenizer.encode(text).len()
}
```

**Methods:**

- `encode(&self, text: &str) -> Vec<u32>`: Encode text to token IDs
- `encode_with(&self, text: &str, mode: &SpecialMode<'_>) -> Result<Vec<u32>, PolicyError>`: Encode under an explicit [`SpecialMode`](#specialmode)
- `decode(&self, ids: &[u32]) -> Result<String, TokenizeError>`: Decode token IDs to text
- `vocab_size(&self) -> usize`: Vocabulary size

Implemented by `Tokenizer` (BPE), `SentencePieceTokenizer` (unigram), `SpmTokenizer` (SentencePiece BPE), `WordPieceTokenizer` (WordPiece), and by `AnyTokenizer` itself.

### AnyTokenizer

The universal loaded-tokenizer handle: a backend plus the `SpecialPolicy` parsed from the same source, plus the declared `decoder` pipeline. It is what `from_pretrained`, `from_json_path`/`from_json_bytes` and `from_gguf_vocab` all return.

```rust
use splintr::{from_json_path, Tokenize};

let tok = from_json_path("tokenizer.json")?;

let ids = tok.encode("Hello, world!");      // boundary template applied
let raw = tok.encode_raw("Hello, world!");  // content tokens only
let pair = tok.encode_pair("query", "document")?; // [CLS] q [SEP] d [SEP]
let text = tok.decode(&ids)?;               // via `Tokenize`
```

**Methods:**

- `encode(&self, text: &str) -> Vec<u32>`: Content tokens with the single-sequence template applied (HF `add_special_tokens=True`)
- `encode_raw(&self, text: &str) -> Vec<u32>`: Content tokens alone (HF `add_special_tokens=False`)
- `encode_with(&self, text: &str, mode: &SpecialMode<'_>) -> Result<Vec<u32>, PolicyError>`: `encode` under an explicit matching mode
- `encode_batch(&self, texts: &[&str]) -> Vec<Vec<u32>>`: Batch form of `encode`, parallel across texts
- `encode_batch_with(&self, texts: &[&str], mode: &SpecialMode<'_>) -> Result<Vec<Vec<u32>>, PolicyError>`: Batch form of `encode_with`; fails as a whole rather than dropping an offending text
- `encode_rayon(&self, text: &str) -> Vec<u32>`: `encode` parallelized _within_ one text, where the backend supports it (same ids either way)
- `encode_pair(&self, a: &str, b: &str) -> Result<Vec<u32>, PolicyError>`: The pair template; errors with `PolicyError::NoPairTemplate` rather than concatenating without the model's separator
- `decode_batch(&self, token_lists: &[Vec<u32>]) -> Result<Vec<String>, TokenizeError>`: Batch form of `decode`, running the same declared pipeline
- `set_pcre2(&mut self, use_pcre2: bool) -> Result<(), TokenizerError>` / `set_jit(&mut self, use_jit: bool)`: Reconfigure the BPE backend's regex engine in place, keeping the policy, decoder pipeline and special-id set attached; `TokenizerError::NotBpeBackend` on any other family
- `declares_decoder(&self) -> bool`: Whether the source declared a `decoder` pipeline — consult before reaching past this handle for a backend's raw byte-level decode
- `family(&self) -> &'static str`: `"BPE"` | `"Unigram"` | `"WordPiece"` | `"Spm"`
- `backend(&self) -> &Backend` / `into_backend(self) -> Backend`: Reach a backend-specific API
- `policy(&self) -> &SpecialPolicy`, `eos_token_id(&self) -> Option<u32>`, `is_eos(&self, id: u32) -> bool`, `special_token_id(&self, name: &str) -> Option<u32>`

The boundary template applies under **every** `SpecialMode`, including `Ordinary`: boundary tokens come from the template, not from matching text against the vocabulary, so the two concerns stay independent.

To fit a sequence into a fixed model length, reserve the template's own slots first: `policy().single_overhead()` is how many special tokens `encode` adds around the content (2 for `[CLS] A [SEP]`, 1 for a lone BOS, 0 for none), so truncate the content to `max_len - single_overhead()` rather than truncating the wrapped ids and cutting off the trailing `[SEP]`/EOS.

### Per-Token Decoding

`Tokenize` also gives you a single id's own contribution, for callers that need to attribute output to one token rather than render a sequence — logprob display, token-level highlighting, debugging a vocabulary — as opposed to [`streaming_decoder()`](#streaming-decoder), which renders a whole stream. Exposed to Python too, as `decode_token_bytes`/`decode_token` on every tokenizer class — see [Decoding Methods](#decoding-methods) in the Python API Reference.

- `decode_token_bytes(&self, id: u32) -> Result<Vec<u8>, TokenizeError>`: the bytes `id` contributes — ByteLevel alphabet unmapped and `<0xNN>` byte fallback resolved — with no sequence-level post-processing applied (no leading-space strip, no first-token rule, no word separator). An id in the **skip** set (a special that the pipeline drops, such as a control token) returns an **empty** `Vec`, not an error; `TokenizeError::InvalidTokenId` is reserved for an id outside the vocabulary altogether.
- `decode_token(&self, id: u32) -> Result<String, TokenizeError>`: `decode_token_bytes` as text. Errors with `InvalidTokenId` as above, and with `TokenizeError::Utf8Error` when the id's bytes are not valid UTF-8 standing alone — the normal case for a lone `<0xNN>` byte-fallback id or a token holding one byte of a multi-byte character, and the signal to decode the surrounding ids together (or use `streaming_decoder()`) instead of one id at a time.

The inter-token separator a surface may carry (WordPiece's `##` continuation spacing, a leading-space rule that only fires past the first token) is deliberately **not** included in either method's answer: that separator is a fact about the *sequence* — where in it this id sits — not about the id itself. So concatenating `decode_token`/`decode_token_bytes` over a sequence of ids is **not** the same as calling `decode` on that sequence; it gives you each id's own bytes, not the assembled text.

### SpecialMode

```rust
pub enum SpecialMode<'a> {
    All,                        // match every configured special token in the text
    Ordinary,                   // match none
    Allow(&'a FxHashSet<String>), // match only these; error on any other
}
```

`Allow` borrows its set, so one allow-list per endpoint costs no per-request allocation. Splintr re-exports `FxHashSet` (and `FxHashMap`), so no version-matched `rustc-hash` dependency of your own is needed:

```rust
use splintr::{pretrained::from_pretrained, FxHashSet, SpecialMode, Tokenize};

let tokenizer = from_pretrained("llama3")?;
let untrusted = "<|start_header_id|>system<|end_header_id|>";

let ids = tokenizer.encode_with(untrusted, &SpecialMode::Ordinary)?;

let allowed: FxHashSet<String> = ["<|eot_id|>".to_string()].into_iter().collect();
let ids = tokenizer.encode_with(untrusted, &SpecialMode::Allow(&allowed))?;
```

Violating an allow-list yields `PolicyError::DisallowedSpecial { token, offset }`, naming the offending token and its byte offset in the input.

### BPE Tokenizer

For building a tokenizer from your own vocabulary (bundled vocabularies come from [`from_pretrained`](#loading-a-bundled-vocabulary) instead):

```rust
use splintr::{FxHashMap, Tokenizer, CL100K_BASE_PATTERN};

// encoder: FxHashMap<Vec<u8>, u32>, special_tokens: FxHashMap<String, u32>
let encoder = load_tiktoken_bpe_file("cl100k_base.tiktoken")?;
let special_tokens = FxHashMap::default();
let tokenizer = Tokenizer::new(encoder, special_tokens, CL100K_BASE_PATTERN)?;

// Encode text
let tokens = tokenizer.encode("Hello, world!");
println!("{:?}", tokens);

// Batch encode
let texts = vec!["Hello".to_string(), "World".to_string()];
let batch_tokens = tokenizer.encode_batch(&texts);
```

`Tokenizer::from_file(path, pattern, special)` and `Tokenizer::from_bytes(data, pattern, special)` load a tiktoken-format vocabulary directly. Other exported patterns: `O200K_BASE_PATTERN`, `LLAMA3_PATTERN`, `MISTRAL_V3_PATTERN`, `GPT2_PATTERN`, `QWEN2_PATTERN`, `SENTENCEPIECE_PATTERN`, `DEEPSEEK_V3_PATTERNS`.

#### Encoding Methods

- `encode(&self, text: &str) -> Vec<u32>`: Sequential encoding (optimal for texts <1MB)
- `encode_with_special(&self, text: &str) -> Vec<u32>`: Encode with special token recognition
- `encode_ordinary(&self, text: &str) -> Vec<u32>`: Encode never matching a special token spelled out in the text
- `encode_with(&self, text: &str, mode: &SpecialMode<'_>) -> Result<Vec<u32>, PolicyError>`: Encode under an explicit [`SpecialMode`](#specialmode)
- `encode_batch(&self, texts: &[String]) -> Vec<Vec<u32>>`: Parallel encoding across texts
- `encode_batch_with_special(&self, texts: &[String]) -> Vec<Vec<u32>>`: Batch form of `encode_with_special`
- `encode_rayon(&self, text: &str) -> Vec<u32>`: Parallel encoding within text (for texts >1MB)

This backend has no notion of boundary tokens — no `encode_raw`, because there is no template to leave off. Those live on [`AnyTokenizer`](#anytokenizer), which pairs a backend with the `SpecialPolicy` that owns them.

#### Decoding Methods

- `decode(&self, tokens: &[u32]) -> Result<String, TokenizerError>`: Decode to UTF-8 string
- `decode_bytes(&self, tokens: &[u32]) -> Vec<u8>`: Decode to raw bytes
- `decode_lossy(&self, tokens: &[u32]) -> String`: Decode with replacement for invalid UTF-8

### SentencePiece Tokenizer

For models using SentencePiece Unigram tokenization (e.g., T5, Gemma, Albert, XLNet, loaded via `from_json`). Mistral V1/V2 are SentencePiece **BPE** (merge-by-score, not Unigram) and load through the Rust-only `SpmTokenizer` via `from_pretrained` instead — see [`src/core/spm.rs`](../src/core/spm.rs).

```rust
use splintr::SentencePieceTokenizer;

// Create from raw vocabulary data
let tokenizer = SentencePieceTokenizer::new(
    tokens,       // Vec<String> — token strings indexed by ID
    scores,       // Vec<f64> — per-token Unigram scores maximized by Viterbi (empty for uniform)
    Some(1),      // Optional BOS token ID
    2,            // EOS token ID
)?;

// Encode (prepends BOS if configured, uses ▁ word boundaries)
let ids = tokenizer.encode("Hello world");

// Decode (skips BOS/EOS, converts ▁ back to spaces)
let text = tokenizer.decode(&ids)?;

// Lossy decode (skips invalid token IDs instead of erroring)
let text = tokenizer.decode_lossy(&ids);
```

#### Methods

- `encode(&self, text: &str) -> Vec<u32>`: Viterbi maximum-score (true SentencePiece Unigram) segmentation with byte fallback
- `encode_ordinary(&self, text: &str) -> Vec<u32>`: Encode never matching an added token spelled out in the text
- `encode_with(&self, text: &str, mode: &SpecialMode<'_>) -> Result<Vec<u32>, PolicyError>`: Encode under an explicit [`SpecialMode`](#specialmode)
- `decode(&self, ids: &[u32]) -> Result<String, SentencePieceError>`: Decode to UTF-8 string
- `decode_lossy(&self, ids: &[u32]) -> String`: Decode, skipping invalid token IDs
- `vocab_size(&self) -> usize`: Vocabulary size
- `is_eos(&self, token_id: u32) -> bool`: Check if token is EOS
- `eos_token_id(&self) -> u32`: Get EOS token ID
- `bos_token_id(&self) -> Option<u32>`: Get BOS token ID

### WordPiece Tokenizer

For BERT-family models using WordPiece subword tokenization:

```rust
use splintr::{WordPieceTokenizer, Tokenize};

// Create from a flat vocabulary (index = token ID)
let vocab = vec![
    "[PAD]", "[UNK]", "[CLS]", "[SEP]",
    "hello", "world", "##ing", "##s",
].into_iter().map(String::from).collect();

let tokenizer = WordPieceTokenizer::new(
    vocab,    // Vec<String> — token strings indexed by ID
    1,        // UNK token ID
    200,      // Max word length before mapping to UNK
    true,     // Lowercase and strip accents (for uncased models)
);

// Encode (BasicTokenizer + WordPiece greedy longest-match)
let ids = tokenizer.encode("Hello world");

// Decode (reconstructs text, skips [CLS]/[SEP]/[PAD] special tokens)
let text = tokenizer.decode(&ids)?;
```

#### Methods

- `encode(&self, text: &str) -> Vec<u32>`: BasicTokenizer + WordPiece subword tokenization (via the `Tokenize` trait)
- `encode_ordinary(&self, text: &str) -> Vec<u32>`: Encode never matching an added token spelled out in the text
- `encode_with(&self, text: &str, mode: &SpecialMode<'_>) -> Result<Vec<u32>, PolicyError>`: Encode under an explicit [`SpecialMode`](#specialmode)
- `decode(&self, ids: &[u32]) -> Result<String, TokenizeError>`: Decode, joining subwords and removing `##` prefixes
- `vocab_size(&self) -> usize`: Vocabulary size
- `cls_token_id(&self) -> Option<u32>`: `[CLS]` token ID
- `sep_token_id(&self) -> Option<u32>`: `[SEP]` token ID
- `pad_token_id(&self) -> Option<u32>`: `[PAD]` token ID
- `unk_token_id(&self) -> u32`: `[UNK]` token ID

### Error Handling

The Rust API uses `Result` types for operations that can fail:

```rust
match tokenizer.decode(&tokens) {
    Ok(text) => println!("Decoded: {}", text),
    Err(e) => eprintln!("Decoding error: {}", e),
}
```

## Detailed Usage Examples

### Basic Encoding and Decoding

```python
from splintr import Tokenizer

# Load tokenizer
tokenizer = Tokenizer.from_pretrained("cl100k_base")

# Simple encoding
text = "The quick brown fox jumps over the lazy dog."
tokens = tokenizer.encode(text)
print(f"Text: {text}")
print(f"Tokens: {tokens}")
print(f"Token count: {len(tokens)}")

# Simple decoding
decoded = tokenizer.decode(tokens)
print(f"Decoded: {decoded}")
assert decoded == text

# Handle different languages
chinese = "你好世界"
tokens_cn = tokenizer.encode(chinese)
print(f"Chinese tokens: {tokens_cn}")
decoded_cn = tokenizer.decode(tokens_cn)
print(f"Decoded Chinese: {decoded_cn}")
```

### Batch Processing

```python
from splintr import Tokenizer
import time

tokenizer = Tokenizer.from_pretrained("cl100k_base")

# Prepare a batch of texts
texts = [
    "First text to encode",
    "Second text to encode",
    "Third text with different content",
    "Fourth text for batch processing",
] * 100  # 400 texts

# Measure batch encoding performance
start = time.time()
batch_tokens = tokenizer.encode_batch(texts)
elapsed = time.time() - start

print(f"Encoded {len(texts)} texts in {elapsed:.3f}s")
print(f"Throughput: {len(texts)/elapsed:.1f} texts/second")

# Process results
for i, tokens in enumerate(batch_tokens[:5]):
    print(f"Text {i}: {len(tokens)} tokens")
```

### Special Tokens Usage

```python
from splintr import Tokenizer

tokenizer = Tokenizer.from_pretrained("cl100k_base")

# Encode without special token recognition
# The special token gets split into multiple tokens
text = "Start <|endoftext|> End"
tokens_no_special = tokenizer.encode(text)
print(f"Without special tokens: {len(tokens_no_special)} tokens")

# Encode with special token recognition
# The special token becomes a single token
tokens_with_special = tokenizer.encode_with_special(text)
print(f"With special tokens: {len(tokens_with_special)} tokens")

# Verify the difference
decoded = tokenizer.decode(tokens_with_special)
print(f"Decoded: {decoded}")
```

### Agent Tokens Usage

```python
from splintr import Tokenizer, CL100K_AGENT_TOKENS, LLAMA3_AGENT_TOKENS, DEEPSEEK_V3_AGENT_TOKENS

# OpenAI models with agent tokens
tokenizer_openai = Tokenizer.from_pretrained("cl100k_base")

# Chain-of-Thought reasoning
cot_text = "<|think|>Let me break this down step by step...<|/think|>The answer is 42."
tokens = tokenizer_openai.encode_with_special(cot_text)
print(f"Thinking token ID: {CL100K_AGENT_TOKENS.THINK}")
print(f"Thinking end token ID: {CL100K_AGENT_TOKENS.THINK_END}")

# ReAct agent pattern
react_text = """<|plan|>I need to search for information
<|act|>search("climate change")
<|observe|>Found 10 results...
<|think|>Based on these results..."""

tokens = tokenizer_openai.encode_with_special(react_text)
print(f"Encoded {len(tokens)} tokens")

# Function calling
function_text = """<|function|>calculate_sum
<|result|>42
<|/result|>"""

tokens = tokenizer_openai.encode_with_special(function_text)
print(f"Function token ID: {CL100K_AGENT_TOKENS.FUNCTION}")
print(f"Result token ID: {CL100K_AGENT_TOKENS.RESULT}")

# RAG with citations
rag_text = """<|context|>This is source material...
<|cite|>According to the documentation...
<|source|>docs.example.com"""

tokens = tokenizer_openai.encode_with_special(rag_text)
print(f"Context token ID: {CL100K_AGENT_TOKENS.CONTEXT}")
print(f"Cite token ID: {CL100K_AGENT_TOKENS.CITE}")

# Llama 3 models
tokenizer_llama = Tokenizer.from_pretrained("llama3")

# Use Llama 3 native tokens
llama_text = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\nHello<|eot_id|>"
tokens = tokenizer_llama.encode_with_special(llama_text)
print(f"Llama begin_of_text: {LLAMA3_AGENT_TOKENS.BEGIN_OF_TEXT}")
print(f"Llama start_header_id: {LLAMA3_AGENT_TOKENS.START_HEADER_ID}")

# DeepSeek V3 models with native thinking tokens
tokenizer_deepseek = Tokenizer.from_pretrained("deepseek_v3")

# Use DeepSeek's native thinking tokens for R1-style reasoning
deepseek_text = "<think>Let me reason through this problem step by step...</think>The solution is X."
tokens = tokenizer_deepseek.encode_with_special(deepseek_text)
print(f"DeepSeek think token (native): {DEEPSEEK_V3_AGENT_TOKENS.THINK_NATIVE}")
print(f"DeepSeek think_end token (native): {DEEPSEEK_V3_AGENT_TOKENS.THINK_END_NATIVE}")

# DeepSeek V3 also has tool calling tokens
tool_text = """<｜tool▁calls▁begin｜>
<｜tool▁call▁begin｜>
function_name
<｜tool▁call▁end｜>
<｜tool▁calls▁end｜>"""

tokens = tokenizer_deepseek.encode_with_special(tool_text)
print(f"Encoded tool calling pattern with {len(tokens)} tokens")
```

### Streaming Examples

#### Streaming a Raw Vocabulary

```python
from splintr import Tokenizer
import time

tokenizer = Tokenizer.from_pretrained("cl100k_base")

# Simulate streaming token generation
text = "Hello, 世界! This is a test of streaming decoding with Unicode characters: 你好"
tokens = tokenizer.encode(text)

# Create streaming decoder
decoder = tokenizer.streaming_decoder()

print("Streaming output:")
for token in tokens:
    # Simulate network delay
    time.sleep(0.05)

    # Add token and print if we get complete characters
    if chunk := decoder.add_token(token):
        print(chunk, end="", flush=True)

# Flush any remaining bytes
if remaining := decoder.flush():
    print(remaining, end="", flush=True)

print("\n\nStreaming complete!")
```

#### Streaming a ByteLevel Vocabulary (DeepSeek V3)

```python
from splintr import Tokenizer
import time

tokenizer = Tokenizer.from_pretrained("deepseek_v3")

# Test text with Unicode
text = "DeepSeek V3 supports ByteLevel BPE! 中文测试"
tokens = tokenizer.encode(text)

# Same call as above: the ByteLevel unmapping comes from the tokenizer
decoder = tokenizer.streaming_decoder()

print("ByteLevel streaming output:")
for token in tokens:
    time.sleep(0.05)

    if chunk := decoder.add_token(token):
        print(chunk, end="", flush=True)

# Flush remaining
if remaining := decoder.flush():
    print(remaining, end="", flush=True)

print("\n\nByteLevel streaming complete!")

# Check pending state
print(f"Has pending bytes: {decoder.has_pending}")
print(f"Pending byte count: {decoder.pending_bytes}")
```

#### Advanced Streaming with Error Handling

```python
from splintr import Tokenizer

tokenizer = Tokenizer.from_pretrained("cl100k_base")
decoder = tokenizer.streaming_decoder()

def stream_tokens(token_generator):
    """Stream tokens with proper error handling."""
    try:
        for token_id in token_generator:
            try:
                if text := decoder.add_token(token_id):
                    yield text
            except Exception as e:
                print(f"\nError processing token {token_id}: {e}")
                # Reset decoder and continue
                decoder.reset()
                continue

        # Always flush at the end
        if remaining := decoder.flush():
            yield remaining

    except Exception as e:
        print(f"\nFatal streaming error: {e}")
        # Final flush attempt
        try:
            if remaining := decoder.flush():
                yield remaining
        except:
            pass

# Use the streaming function
text = "Test streaming with proper error handling"
tokens = tokenizer.encode(text)

for chunk in stream_tokens(iter(tokens)):
    print(chunk, end="", flush=True)

print("\nDone!")
```

## Additional Resources

- [Main README](../README.md) - Quick start and overview
- [Best Practices](best_practices.md) - Which vocabulary, which encode mode, how to size a model
- [Vocabularies](vocabularies.md) - What is bundled and what each one contains
- [Special Tokens Documentation](special_tokens.md) - Complete agent tokens reference
- [ByteLevel BPE Documentation](bytelevel_bpe.md) - ByteLevel encoding details
- [API Documentation (Rust)](https://docs.rs/splintr) - Complete Rust API reference
- [GitHub Repository](https://github.com/ml-rust/splintr) - Source code and examples
