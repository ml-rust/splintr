# Best Practices

Decisions, not signatures. Each section states a choice you have to make, the reason one answer is usually right, and where in the [API Guide](api_guide.md) the calls are documented.

- [Where should the vocabulary come from?](#where-should-the-vocabulary-come-from)
- [Choosing a vocabulary for a new model](#choosing-a-vocabulary-for-a-new-model)
- [Sizing a model against a vocabulary](#sizing-a-model-against-a-vocabulary)
- [Encoding text you did not write](#encoding-text-you-did-not-write)
- [Rendering a token stream](#rendering-a-token-stream)
- [Getting the throughput](#getting-the-throughput)

## Where should the vocabulary come from?

It comes down to one question: **does the vocabulary already exist, or are you choosing one?**

**You are matching an existing checkpoint** — inference, serving, token counting, fine-tuning. There is nothing to decide: the ids have to be the ones the model was trained on, or every embedding lookup is wrong. Use whatever that model ships, through [`from_json`](api_guide.md#loading-any-model-from-tokenizerjson). A [bundled vocabulary](vocabularies.md#bundled-vocabularies) is a convenience when the model happens to be one of the eleven — the same ranks with no file to carry — but it is a convenience, not a different answer.

Two things a bundled vocabulary does _not_ reproduce, both by design:

- **No boundary template.** `from_pretrained` states no `post_processor`, so `encode` returns content tokens and you place BOS/EOS yourself. `from_json` on the same model applies the template the file declares. If your pipeline depends on that template, load the file.
- **Ids above `base_vocab_size`.** Splintr's agent tokens live there, and a published checkpoint has no embedding rows for them. Never feed the model an id at or above that number.

**You are choosing one** — see the next section.

## Choosing a vocabulary for a new model

Training from scratch is the case where the vocabulary is a decision. Three routes, in order of how much you own.

Splintr does not **train** vocabularies. It is a tokenizer runtime: it loads merge tables and applies them. Producing a new one is HuggingFace `tokenizers`' job (`BpeTrainer`), and splintr reads the result.

### 1. A bundled vocabulary, as-is

The shortest route, and what the agent tokens exist for. Pick a vocabulary whose merge table suits your corpus, and size the embedding to the **extended** size:

```python
from splintr import Tokenizer, base_vocab_size

tok = Tokenizer.from_pretrained("qwen3")
n_embd_rows = tok.vocab_size                # 151723 — includes the agent tokens
first_agent_id = base_vocab_size("qwen3")   # 151669
```

Everything below `first_agent_id` is Qwen's vocabulary unchanged, so a corpus tokenized here is tokenized the way Qwen would. Everything from it up is yours to train: `<|system|>`, `<|think|>`, `<|plan|>`, `<|function|>`, `<|cite|>` and the rest, at offsets that mean the same thing in every bundled vocabulary.

That last property is the point. A new model needs markers no published vocabulary contains, and the usual answer is hand-editing a `tokenizer.json`, choosing ids and hoping nothing collides. Here the layout is fixed and shared, so a model trained on cl100k and one trained on Qwen agree on what `<|think|>` means. All 54 resolve on every bundled vocabulary except Whisper; on the ones that already ship some of the names — Qwen's `<|im_start|>`, GLM's `<|system|>` — those resolve to the model's own ids, which is what you want, and the arithmetic above is unaffected.

### 2. Your own `tokenizer.json`

Full control of the merge table and the id layout. Train the vocabulary with HuggingFace `tokenizers`, declare your markers in the file's `added_tokens`, and load it with [`from_json`](api_guide.md#loading-any-model-from-tokenizerjson). The file is the sole authority: you own which ids the markers get, and there are no agent tokens.

This is also the route for a vocabulary trained on a domain corpus, where a general-purpose merge table is the wrong starting point however convenient it is — a chemistry or source-code corpus is the usual case.

### 3. A bundled merge table, your own special tokens

The middle ground: keep a proven vocabulary, define your own marker set. `Tokenizer::from_bytes_chain` takes the embedded ranks and a map you build (Rust; the Python bindings expose only the named loaders):

```rust
use rustc_hash::FxHashMap;
use splintr::pretrained::{patterns, PretrainedVocab, QWEN3_VOCAB};
use splintr::Tokenizer;

let mut special = FxHashMap::default();
special.insert("<|endoftext|>".to_string(), 151643);
special.insert("<|my_marker|>".to_string(), 151669);   // above the base

let tokenizer = Tokenizer::from_bytes_chain(
    QWEN3_VOCAB,
    patterns(PretrainedVocab::Qwen3).unwrap(),
    special,
)?;

tokenizer.encode_with_special("hi <|my_marker|>");  // [6023, 220, 151669]
tokenizer.encode("hi <|my_marker|>");               // markers stay ordinary text
```

Place every addition **above** `base_vocab_size`, for the reason splintr does: an id below it is one the merge table already uses, and overriding it silently changes what ordinary text encodes to. And note which entry point matches your markers — on the concrete `Tokenizer`, `encode` is the _ordinary_ mode; see [Encoding text you did not write](#encoding-text-you-did-not-write).

## Sizing a model against a vocabulary

Two numbers, and picking the wrong one is silent until it is not:

| You are                                        | Use                     | Because                                          |
| ---------------------------------------------- | ----------------------- | ------------------------------------------------ |
| Loading a published checkpoint                 | `base_vocab_size(name)` | Its embedding has exactly that many rows         |
| Training a new model on a bundled vocabulary   | `tokenizer.vocab_size`  | You want the agent tokens to be trainable        |
| Identifying a checkpoint from its tensor shape | `base_vocab_size(name)` | The shape is the reference's size, not splintr's |

It is **not** `vocab_size - 54`. Several vocabularies leave gaps below their nominal size, so the difference varies: 54 for cl100k_base, 98 for llama3, 139 for deepseek_v3. Read it, do not derive it. Accessor semantics are in [Sizing against the reference vocabulary](api_guide.md#sizing-against-the-reference-vocabulary); the per-vocabulary numbers are in [vocabularies.md](vocabularies.md#bundled-vocabularies).

## Encoding text you did not write

A tokenizer that matches special tokens will promote text that _spells_ a control token to that token's real id — a user typing `<|im_start|>` produces the same id the server emits to open a turn, and downstream nothing can tell them apart.

The rule is simple: **the mode should match where the text came from.**

| Text                                                                 | Method                                |
| -------------------------------------------------------------------- | ------------------------------------- |
| User input, retrieved documents, tool output, anything over the wire | `encode_ordinary`                     |
| Your own chat template, where you meant the markers                  | `encode_with_special`                 |
| A template with untrusted text interpolated into it                  | `encode_allowed_special(text, [...])` |

`encode_allowed_special` is the one to reach for when the two are mixed: it promotes the markers you name and _raises_ on any other, so a forged token is an error rather than a silent substitution. Denylisting spellings before encoding does not work — the spelling is not the only thing that maps to the id.

The model's own boundary tokens are unaffected by the mode: they come from the `post_processor` template, not from matching text, so locking down matching never strips the BOS the model was trained with. Full semantics and the Rust `SpecialMode` enum are in [Special tokens in untrusted text](api_guide.md#special-tokens-in-untrusted-text).

## Rendering a token stream

Decode through [`streaming_decoder()`](api_guide.md#streaming-decoder), not by calling `decode` on each id and concatenating. BPE tokens do not align with UTF-8 boundaries, so a multi-byte character can split across two tokens; the streaming decoder buffers the incomplete sequence and emits only complete characters. One decoder per tokenizer, and `"".join(chunks) + flush()` equals `decode(ids)`.

Use the Rust-only `decode_token` / `decode_token_bytes` when you want one id's own contribution — logprob display, token-level highlighting — and not before. Concatenating those does **not** reassemble `decode`'s output: the inter-token separator is a property of where an id sits in the sequence, not of the id itself. See [Per-Token Decoding](api_guide.md#per-token-decoding).

## Getting the throughput

- **Batch whatever you can.** `encode_batch` parallelizes across texts and is where the ~10-12x over tiktoken lives. Two texts are already worth batching; a loop calling `encode` is not.
- **Do not reach for `encode_rayon` on single texts.** The sequential path wins below roughly 1 MB, which is nearly every real input. Splitting one text across threads costs more than it saves.
- **Let the cache work.** Repeated chunks are served from an LRU cache automatically. Call `clear_cache()` only if you are streaming millions of unique texts and memory matters — clearing it as routine hygiene just throws away hits.
- **Do not pay for special-token matching you do not need.** `encode_ordinary` skips the matcher entirely; `encode_with_special` runs it over every input. On text that cannot contain markers, that work is pure overhead — and per [above](#encoding-text-you-did-not-write), ordinary is the safer default anyway.
- **Build the tokenizer once.** Construction parses a vocabulary of up to 200,000 entries; hold the instance rather than rebuilding it per request. It is safe to share across threads.

## Additional Resources

- [API Guide](api_guide.md) — every method, its semantics and its guarantees
- [Vocabularies](vocabularies.md) — what is bundled and what each one contains
- [Special Tokens](special_tokens.md) — the complete agent-token reference
- [Benchmarks](benchmarks.md) — methodology and per-content-type numbers
