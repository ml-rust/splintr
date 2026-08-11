# Supported Vocabularies

## Bundled Vocabularies

| Vocabulary                                             | Used By                                      | `base_vocab_size`           | Special Tokens  | Pre-tokenizer (`pretrained::patterns`) |
| ------------------------------------------------------ | -------------------------------------------- | --------------------------- | --------------- | -------------------------------------- |
| `cl100k_base`                                          | GPT-4, GPT-3.5-turbo                         | 100,277                     | 5 + 54 agent    | `CL100K_BASE_PATTERN`                  |
| `o200k_base`                                           | GPT-4o                                       | 200,019                     | 2 + 54 agent    | `O200K_BASE_PATTERN`                   |
| `llama3`                                               | Llama 3, 3.1, 3.2, 3.3 (Meta)                | 128,256                     | 11 + 54 agent   | `LLAMA3_PATTERN`                       |
| `deepseek_v3`                                          | DeepSeek V3, DeepSeek R1                     | 128,815                     | 17 + 54 agent   | `DEEPSEEK_V3_PATTERNS` (three passes)  |
| `qwen3` / `qwen` / `qwen2` / `qwen2.5` / `baichuan_m2` | Qwen 2, Qwen 3, Baichuan-M2                  | 151,669                     | 26 + 54 agent   | `QWEN2_PATTERN`                        |
| `glm4` / `glm` / `glm-4` / `glm4.5` / `glm-4.5`        | GLM-4, GLM-4.5, GLM-4.6                      | 151,365                     | 36 + 54 agent   | `LLAMA3_PATTERN`                       |
| `gpt-oss` / `gpt_oss` / `o200k_harmony`                | OpenAI gpt-oss (20B, 120B)                   | 200,019                     | 21 + 54 agent   | `O200K_BASE_PATTERN`                   |
| `kimi` / `kimi_k2` / `kimi_k2.5` / `kimi_linear`       | Kimi K2, K2.5, K2.6, K2.7, Kimi-Linear       | 163,840                     | 256 + 54 agent  | `KIMI_PATTERN`                         |
| `kimi_k3` / `kimi-k3`                                  | Kimi K3                                      | 163,840                     | 256 + 54 agent  | `KIMI_PATTERN`                         |
| `mistral_v1`                                           | Mistral 7B v0.1/v0.2, Mixtral 8x7B           | 32,000                      | 3 + 54 agent    | none — SPM-BPE, no split regex         |
| `mistral_v2`                                           | Mistral 7B v0.3, Codestral, 8x22B            | 32,768                      | 10 + 54 agent   | none — SPM-BPE, no split regex         |
| `mistral_v3`                                           | Mistral NeMo, Large 2, Pixtral               | 131,072                     | 10 + 54 agent   | `MISTRAL_V3_PATTERN`                   |
| `whisper` / `whisper_v1` / `whisper_v2` / `whisper_v3` | OpenAI Whisper multilingual (tiny..large-v3) | 51,865 (v1/v2), 51,866 (v3) | 1608 (no agent) | `GPT2_PATTERN`                         |

`pretrained::patterns(vocab)` returns `Option<&'static [&'static str]>`. It is `None` for Mistral V1/V2 — not "unknown", but "this vocabulary does not pre-tokenize with a regex": both run on the SPM-BPE backend, which segments by merging pieces and never applies a split pattern.

**Qwen and GLM already ship some of the agent tokens.** Qwen defines `<|im_start|>`/`<|im_end|>` itself; GLM defines `<|system|>`, `<|user|>`, `<|assistant|>`, `<|image|>` and `<|video|>`. All 54 still resolve — those five or two simply resolve to the _model's_ ids rather than to splintr-appended ones, so a chat template encodes to the ids the checkpoint was trained on. Splintr appends the rest, and the slot a shared name would have taken is left reserved rather than repacked, so every other agent token keeps the offset it has in every other vocabulary. **Baichuan-M2** ships Qwen's tokenizer verbatim (151,643 ids, identical), so it is an alias rather than a second copy.

**gpt-oss** is o200k_base's 199,998 ranks, id for id, under a different special-token block: where o200k_base names two of 199999-200018, gpt-oss fills the range with the harmony response format's markers (`<|start|>`, `<|channel|>`, `<|message|>`, `<|end|>`, `<|call|>`, `<|return|>`, `<|constrain|>` and OpenAI's reserved slots). It therefore embeds no vocabulary data of its own. The two also differ on decode: gpt-oss's own `tokenizer.json` declares its added tokens `special: true` so they render as nothing, while o200k_base follows `tiktoken`, which renders `<|endoftext|>`.

**Kimi is one vocabulary under two names.** Moonshot ships a byte-identical `tiktoken.model` for K2, K2.5, K2.6, K2.7, K3, Kimi-Linear and Kimi-VL, and an identical `pat_str` with them, so the merge ranks and the pre-tokenizer are shared and only one payload is embedded. What differs is the 256-slot reserved block above the ranks: id 163586 is `<|im_end|>` on K2 and `<|end_of_msg|>` on K3, and K3 drops K2's tool-call markers entirely. Encoding a K2 chat template against `kimi_k3` would therefore produce ids that checkpoint never saw, which is why they are separate names rather than aliases. Bare `kimi` resolves to K2. Both reserve all 256 slots — the ones the model does not name are `<|reserved_token_N|>`, exactly as Moonshot's own tokenizer generates them, so every id in the block decodes.

**Kimi's pattern is the only bundled one needing class intersection.** `KIMI_PATTERN` is o200k's shape plus a leading `[\p{Han}]+` branch, with Han subtracted from the letter branches (`[\p{Lu}…&&[^\p{Han}]]`) so that branch actually fires. regexr implements `&&`; note that Python's `regex` module only does so under `regex.V1`, which is a trap when building a reference by hand. Kimi has a direct scanner like the other byte-level families, which needs a `\p{Han}` table — a Unicode *script*, so the general-category tables cannot answer it. That table is derived from regexr itself rather than transcribed, and a test re-derives it over all 1.1M scalar values on every run, so an engine upgrade that moved a boundary fails loudly instead of silently changing Kimi's token ids.

**Bundled vocabularies are feature-gated.** Each family has a `vocab-*` cargo feature (`vocab-cl100k`, `vocab-o200k`, `vocab-gpt-oss`, `vocab-llama3`, `vocab-deepseek`, `vocab-qwen`, `vocab-glm`, `vocab-kimi`, `vocab-mistral`, `vocab-whisper`), all enabled by default and all enabled in the Python wheel. Turning one off drops its embedded data; `from_pretrained` then reports which feature is missing rather than rejecting the name. The full set is ~23 MB of the Python extension module's 26 MB.

**Whisper** is a speech model, so it carries no agent tokens — its special tokens are the standard Whisper set (`<|startoftranscript|>`, language tokens, `<|transcribe|>`/`<|translate|>`, 1501 timestamp tokens). Bare `whisper` resolves to v2. The **English-only** checkpoints (`*.en`) use a different base BPE and are **not bundled**; load those with `from_json` (below).

## Loading any model from `tokenizer.json`

For models not bundled above, point `splintr.from_json` at a HuggingFace `tokenizer.json`. It returns an `AnyTokenizer` — the universal loaded-tokenizer handle, which dispatches internally to the right backend for the file's `model.type` while keeping everything else the file declares: the special-token policy, the `decoder` pipeline, and the ids to skip on decode:

```python
from splintr import from_json

tok = from_json("tokenizer.json")   # BERT, T5, Gemma, Qwen, Whisper.en, ...
ids = tok.encode("Hello, world!")       # + [CLS]/[SEP]/<s> etc. (post_processor)
ids = tok.encode_raw("Hello, world!")   # content tokens only
text = tok.decode(ids)
tok.family                              # "BPE" | "Unigram" | "WordPiece"
```

`encode` applies the model's `post_processor` template (HF's default `encode`); `encode_raw` returns content tokens alone (HF's `add_special_tokens=False`). `decode` runs the file's declared `decoder` chain (`Replace`, `ByteFallback`, `Fuse`, `Strip`, `Metaspace`, `ByteLevel`, `WordPiece`, `BPEDecoder`, `Sequence`) after dropping `special=true` ids, so files whose decoding _is_ that chain — Mistral, Llama, Gemma — come back as text rather than raw pieces. Honored end-to-end: the multi-stage pre-tokenizer pipeline (`ByteLevel`, `Split` incl. `invert`, `Digits`, `Punctuation`/`Contiguous`, `Sequence`, `add_prefix_space`/`prepend_scheme`), the full ordered normalizer (`Replace`, `Strip`, `Prepend`, NFC/NFD/NFKC/NFKD, `Precompiled` charsmap, …), BPE merge order, and `added_tokens` matching. Verified id-for-id (content **and** with special tokens) against GPT-2, RoBERTa, Qwen, Whisper, T5, Albert, XLNet, BERT, DistilBERT, **Falcon, StarCoder2, DeepSeek-Coder, GPT-NeoX**.

Every family comes back as the same `AnyTokenizer` type; `family` names the backend it dispatches to internally (in Rust, `AnyTokenizer::backend()` borrows it as a `Backend` enum when you need a backend-specific API):

| `model.type`       | `tok.family`  | Internal backend         | Example models                          |
| ------------------ | ------------- | ------------------------ | --------------------------------------- |
| `BPE` (byte-level) | `"BPE"`       | `Tokenizer`              | GPT-2, Whisper, Llama 3, Qwen, DeepSeek |
| `Unigram`          | `"Unigram"`   | `SentencePieceTokenizer` | T5, Gemma, Albert, XLNet                |
| `WordPiece`        | `"WordPiece"` | `WordPieceTokenizer`     | BERT, DistilBERT, Electra               |

A fourth backend, `SpmTokenizer` (`family == "Spm"`), covers llama.cpp-style `SPM` vocabularies — SentencePiece **BPE**, merge-by-rank rather than Viterbi. It is not reachable from `tokenizer.json`: it is what the bundled Mistral V1/V2 vocabularies use, and what the GGUF loader below produces for a `llama` vocabulary.

The split regex, byte-level flag, merge order, normalizer (including SentencePiece's `Precompiled` charsmap), and special tokens are all read from the file itself. Output is verified id-for-id against HuggingFace `tokenizers` across every family — GPT-2, RoBERTa, BART, Qwen, Whisper (BPE); T5, Albert, XLNet (Unigram); BERT, DistilBERT (WordPiece). (Rust: `splintr::from_json_path` / `from_json_bytes`.)

**Strict by design.** Rather than silently approximate a config it doesn't fully support (which would emit wrong tokens with no signal), `from_json` raises — `UnsupportedModelType`, `UnsupportedNormalizer`, `InvalidNormalizerRegex`, or `UnsupportedPreTokenizer` (a declared pre-tokenizer with no recognized split, so the pattern is never guessed).

### OpenAI standard tokens

- **cl100k_base**: `<|endoftext|>`, `<|fim_prefix|>`, `<|fim_middle|>`, `<|fim_suffix|>`, `<|endofprompt|>`
- **o200k_base**: `<|endoftext|>`, `<|endofprompt|>`

### Meta Llama 3 standard tokens

- **llama3**: `<|begin_of_text|>`, `<|end_of_text|>`, `<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>`, `<|eom_id|>` (3.1+), `<|python_tag|>` (3.1+), `<|step_id|>` (3.2-Vision), `<|image|>` (3.2-Vision)

### DeepSeek V3 standard tokens

- **deepseek_v3**: `<｜begin▁of▁sentence｜>`, `<｜end▁of▁sentence｜>`, `<think>`, `</think>`, `<｜User｜>`, `<｜Assistant｜>`, `<|EOT|>`, FIM tokens (`<｜fim▁hole｜>`, `<｜fim▁begin｜>`, `<｜fim▁end｜>`), tool calling tokens (`<｜tool▁calls▁begin｜>`, `<｜tool▁call▁begin｜>`, etc.)

### Mistral standard tokens

- **mistral_v1**: `<unk>`, `<s>`, `</s>` (SentencePiece native)
- **mistral_v2**: Same as V1 + control tokens: `[INST]`, `[/INST]`, `[TOOL_CALLS]`, `[AVAILABLE_TOOLS]`, `[/AVAILABLE_TOOLS]`, `[TOOL_RESULTS]`, `[/TOOL_RESULTS]`
- **mistral_v3**: `<unk>`, `<s>`, `</s>` + control tokens (Tekken/Tiktoken-based, NOT SentencePiece)

## Loading a raw `.tiktoken` file

A `.tiktoken` file is `base64(token bytes) rank`, one per line — the format OpenAI publishes ranks in, and the format `crates/vocab-*/vocabs/*.tiktoken` uses. It carries the merge ranks and nothing else: no pre-tokenizer pattern, no special tokens, no decoder chain. So those are arguments rather than file contents:

```python
from splintr import Tokenizer, CL100K_BASE_PATTERN

tok = Tokenizer("vocab.tiktoken", CL100K_BASE_PATTERN)
tok = Tokenizer("vocab.tiktoken", CL100K_BASE_PATTERN, {"<|endoftext|>": 100257})
```

```rust
use splintr::{Tokenizer, CL100K_BASE_PATTERN};

let tokenizer = Tokenizer::from_file("vocab.tiktoken", CL100K_BASE_PATTERN, special)?;
let tokenizer = Tokenizer::from_bytes(&data, CL100K_BASE_PATTERN, special)?;
```

This is the one loader that returns the concrete `Tokenizer` rather than an `AnyTokenizer`: with no `model.type` to read there is no backend to dispatch on, and a `.tiktoken` vocabulary is always byte-level BPE. Ids are unaffected by the route — loading `crates/vocab-cl100k/vocabs/cl100k_base.tiktoken` this way encodes exactly as `from_pretrained("cl100k_base")` does.

Use it when you have ranks and no `tokenizer.json`: OpenAI's published rank files, a vocabulary you extracted yourself (`scripts/extract_byte_level_vocab.py` writes this format), or one you are iterating on before it has a HuggingFace config. If you *do* have a `tokenizer.json`, prefer `from_json` — it reads the pattern, special tokens and decoder from the file instead of asking you to restate them correctly.

## Loading a GGUF vocabulary

Splintr **never opens a GGUF container**. Parsing the header, the metadata key-value block and the tensor table is the model runtime's job, and pulling a GGUF parser into a tokenizer crate would make every consumer pay for it. What splintr owns is the tokenizer half: the caller fills a `GgufVocab` — one field per `tokenizer.ggml.*` key — and hands it to `splintr::from_gguf_vocab`, which returns the same `AnyTokenizer` every other loader does. (Rust-only; there is no Python binding for this loader.)

```rust
use splintr::{from_gguf_vocab, GgufVocab};

// Fields mirror the GGUF keys with the `tokenizer.ggml.` prefix dropped; every
// one but `tokens` is optional exactly as the key is, and `None` means "the
// file does not say" — never "false" or "zero", because the defaults differ per
// dialect and the loader is the one that knows them.
let tokenizer = from_gguf_vocab(GgufVocab {
    model: "bert".to_string(),           // absent key ⇒ "llama", as in llama.cpp
    tokens,                              // Vec<String>, indexed by token id
    token_type: Some(token_type),        // 3 == CONTROL
    cls_token_id: Some(101),
    sep_token_id: Some(102),
    ..Default::default()
})?;
```

`tokenizer.ggml.model` names the _algorithm_, and the four values in circulation are genuinely different algorithms over superficially similar data. The loader dispatches on it and rejects what it cannot honour rather than guessing:

| `tokenizer.ggml.model` | Backend                  | Algorithm                                         |
| ---------------------- | ------------------------ | ------------------------------------------------- |
| `gpt2`                 | `Tokenizer`              | byte-level BPE over the explicit `merges` list    |
| `llama`                | `SpmTokenizer`           | SentencePiece BPE — `scores` are merge ranks      |
| `t5`                   | `SentencePieceTokenizer` | Unigram, Viterbi — `scores` are log-probabilities |
| `bert`                 | `WordPieceTokenizer`     | greedy longest match with `##`                    |

Collapsing these is not a rounding error, and the failure is invisible downstream: run Unigram Viterbi over a `llama` vocabulary and its ranks maximise the wrong objective (`▁sourdough` → `▁s|ou|rd|ou|gh`); the ids stay in range, the embedding shapes stay right, and retrieval quietly degrades.

Boundary tokens live in the returned `SpecialPolicy`, not in the backend, so `add_bos_token` / `add_eos_token` are honoured in exactly one place. A `bert` vocabulary is wrapped in the `[CLS] A [SEP]` template built from the ids it names, through the same internal cls/sep policy constructor the `tokenizer.json` path uses — so `encode` on a GGUF and on the _same model's_ `tokenizer.json` agree, instead of the GGUF returning bare content tokens for a CLS-pooling consumer to misread a content token as the sentence vector. Measured on all-MiniLM-L6-v2: `"hello world"` → `[101, 7592, 2088, 102]`. A vocabulary naming neither id keeps the identity policy — inventing one would be worse than placing none.

Because the template is applied _after_ encoding, a caller enforcing a maximum length must truncate the content first: `SpecialPolicy::single_overhead()` (reachable as `tokenizer.policy().single_overhead()`) reports how many slots the single-sequence template adds, so the content budget is `max_len - single_overhead()`.

## Agent Tokens (54 per model)

Splintr extends all vocabularies with 54 specialized tokens for building agent systems:

```python
from splintr import Tokenizer, CL100K_AGENT_TOKENS

tokenizer = Tokenizer.from_pretrained("cl100k_base")
text = "<|think|>Let me reason...<|/think|>The answer is 42."
tokens = tokenizer.encode_with_special(text)
print(CL100K_AGENT_TOKENS.THINK)      # 100282
print(CL100K_AGENT_TOKENS.FUNCTION)   # 100292
```

| Category     | Example Tokens                                      | Purpose                    |
| ------------ | --------------------------------------------------- | -------------------------- |
| Conversation | `system`, `user`, `assistant`, `im_start`, `im_end` | ChatML format              |
| Thinking     | `think`                                             | Chain-of-Thought reasoning |
| ReAct        | `plan`, `step`, `act`, `observe`                    | Agent action loops         |
| Tools        | `function`, `result`, `error`                       | Function calling           |
| RAG          | `context`, `quote`, `cite`, `source`                | Citations                  |

**Agent tokens never disturb the original vocabulary.** They are appended strictly _above_ every id the reference vocabulary uses, so no original id is shifted and none can collide — ordinary text encodes to exactly the ids the reference tokenizer produces. cl100k_base's reference tops out at 100276 and its agent tokens occupy 100277–100330; llama3's tops out at 128255 with agent tokens at 128256–128353.

### Sizing against the reference vocabulary

The `base_vocab_size` column in the table above is the number to size a model's embedding or logit layer with — the checkpoint's own vocabulary, not splintr's extended one — and, because agent tokens sit strictly above everything, it is also the id at which splintr's additions begin. It is _not_ `vocab_size - 54`: several reference vocabularies leave gaps below their nominal size (llama3 is 128,256 against an extended 128,354; deepseek_v3 is 128,815 against 128,954).

For the accessor and its Rust twin see the API guide's [Sizing against the reference vocabulary](api_guide.md#sizing-against-the-reference-vocabulary); for which of the two numbers to reach for see [Best Practices](best_practices.md#sizing-a-model-against-a-vocabulary). The full agent-token list is in [special_tokens.md](special_tokens.md).

## Choosing a vocabulary for a new model

Training from scratch is the one case where the vocabulary is a decision rather than a given — a bundled vocabulary as-is (which is what the agent tokens are for), your own `tokenizer.json`, or a bundled merge table under your own special tokens. The three routes are laid out with code in [Best Practices](best_practices.md#choosing-a-vocabulary-for-a-new-model).

Splintr does not _train_ vocabularies: it is a tokenizer runtime, and producing a new merge table is HuggingFace `tokenizers`' job.
