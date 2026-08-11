# MBPE — a text format for a BPE vocabulary and its merge order

`.mbpe` states two things about a byte-pair-encoding tokenizer: **what its tokens are**, and **in what order its merges apply**. Nothing else.

It exists because the convention in wide use cannot state the second one independently of the first, and some published tokenizers need them stated apart.

The format is deliberately vendor-neutral. `.tiktoken` spread because it was a format rather than a product, and nothing here is specific to any tokenizer implementation.

## The problem

A `.tiktoken` line is `base64(token) rank`. That single number is the token's **id** and its **merge priority** at once, so the merge list is never stored — it is reconstructed: join a pair, look the result up, and its id _is_ the priority. Every OpenAI vocabulary is built so that works.

Not every vocabulary is. Measured over 39 BPE `tokenizer.json` files from the HuggingFace hub — `injected` counts ids inside the merge-result range that no merge produces, `descents` counts merges yielding a lower id than the one before:

| model                                       |   vocab |   injected |   descents |
| ------------------------------------------- | ------: | ---------: | ---------: |
| gemma-4-12B-it                              | 262,144 | **19,552** |         30 |
| CodeLlama-7b                                |  32,016 |      2,129 |          1 |
| Yi-1.5-9B                                   |  63,992 |        425 |          0 |
| roberta-base                                |  50,265 |        255 | **24,929** |
| Llama-3.1-8B, Qwen3-8B, SmolLM3-3B, gpt2, … |         |          0 |          0 |

**29 of the 39 have zero of both**, including the newest and largest — for those a `.tiktoken` is exactly right and this format buys nothing. The other ten break for one of two reasons:

- **injection** — tokens curated into the vocabulary that BPE never produces, placed on ids _inside_ the range BPE's own results occupy. Gemma 4's 19,552 are markup and structural tokens (`<table>`, `<caption>`, `<thead>`, `<tr>`, `<td>` …). You cannot renumber around them without changing ids, so one number can no longer serve as both. Those same entries are the orphans described below.
- **id assignment** — fairseq numbered RoBERTa's ids by corpus frequency while its merges stayed in training order, so the two are essentially unrelated.

Forcing Gemma 4 into one rank per token mistokenizes **8.1%** of real documents. Its merge list is not even the same length as its vocabulary, so no per-token column can hold it.

So this is insurance, not a migration: it costs nothing when ids already follow merge order, and it is the only thing that survives the day a vocabulary is curated.

A `.spm` line is `base64(piece) score type` — a SentencePiece score and piece type, the type column being optional. It carries no merge list at all, and a BPE vocabulary has no scores to put in one. (Gemma 4 has none: the official GGUFs carry a placeholder −1000.0 for every one of its 262,144 pieces.)

So `.mbpe` is not a replacement for either. It is the case neither covers.

## The format

```
mbpe 1
vocab 262144
merges 236339

<one token line per vocabulary entry, position = id>
<one merge line per rule, in priority order>
```

Line 1 is the magic. It is needed because the name alone is ambiguous: GPT-2 shipped a _merges-only_ `vocab.bpe`, and a reader must be able to tell the two apart before parsing.

Header lines are `key value`, in any order, terminated by one blank line. Both keys are required, and **an unknown key is an error** — a reader must not skip what it does not recognize. That is the forward-compatibility contract: a v1 reader refuses a v2 file outright rather than reading the part it happens to understand, because silently ignoring a declared field is how a tokenizer ends up plausibly and quietly wrong.

The counts are exact: a file whose body does not hold `vocab` token lines followed by `merges` merge lines is rejected.

### Token lines

A token's **id is its line number**, counting from zero at the first token line. Nothing states it, because nothing needs to.

That is only sound for a vocabulary whose ids are contiguous from zero, so a writer must **refuse** to emit `.mbpe` for one that is not, rather than renumber it. A hole would shift every id after it — wrong ids, with no error, which is the one failure a tokenizer format must not permit.

A line is the token's bytes, with four escapes:

| escape | byte                               |
| ------ | ---------------------------------- |
| `\\`   | `\` (0x5C)                         |
| `\n`   | line feed (0x0A)                   |
| `\r`   | carriage return (0x0D)             |
| `\xNN` | any byte, two lowercase hex digits |

Everything else is literal. `\xNN` exists for tokens that are not well-formed UTF-8 — cl100k_base alone has 773 — and a writer uses it for exactly those bytes, leaving the rest as text.

Literal rather than base64 for size, not legibility: base64 costs a third for nothing when 262,113 of Gemma 4's 262,144 pieces need no escape at all (2.58 MB against 3.60 MB; 1.31 MB against 1.51 MB gzipped).

An empty token is an empty line, and is a real entry — several published vocabularies have one.

### Merge lines

A line is `id split`, both decimal:

- `id` — the vocabulary id of the token this merge **produces**;
- `split` — the byte length of its **left operand**, so the pair the rule joined is `token[..split]` and `token[split..]`.

Priority is the line's position: the first merge line is the highest-priority rule.

**The split is not optional.** Without it a reader knows what each merge produces but not what it consumes, and the operand set is the only thing that separates the three kinds of vocabulary entry:

- a **merge result** — BPE can produce it;
- an **atom** — never a result, but an operand of some merge, so BPE builds from it (the byte alphabet, `<0xNN>` byte-fallback pieces, `x</w>` end-of-word forms);
- an **orphan** — neither. BPE can never produce it and nothing is built from it, yet it holds an id.

A tokenizer that answers a whole word from the vocabulary before merging — the standard fast path — will emit an orphan that no correct BPE can reach. That is a wrong id rather than a slow one: Gemma 4's `<blockquote>` (id 190) against the `236820 37548 236813` that merging the same bytes gives. Gemma 4 has 6,298 orphans longer than one character. A format that cannot express the operand set makes this bug **unrepresentable-away**, which is the sharpest argument for storing the merge list at all.

Duplicate rules — several merges producing the same token — are collapsed to their first occurrence, since a token's priority is fixed by where it _first_ appears. This is what takes Gemma 4 from 514,906 rules to 236,339.

## Scope, and what it deliberately omits

`.mbpe` states what a vocabulary **is**. It does not state how a tokenizer is **configured**: no normalizer, no pre-tokenizer or regex, no added-token or special-token table, no decoder, no `end_of_word_suffix`, no `byte_fallback` flag. That is the same scope `.tiktoken` has, and for the same reason — those belong to the tokenizer that loads the vocabulary, not to the vocabulary.

That includes the **byte space** the tokens are spelled in, which is the case most likely to be argued about. GPT-2, RoBERTa, Qwen and CLIP spell their vocabularies in the byte-level alphabet, where a leading space is `Ġ` and `é` is `Ã©`; cl100k and the SentencePiece-derived vocabularies spell theirs in the bytes themselves. The file does not say which, and should not: what makes `Ġ` mean a space is that a `ByteLevel` stage exists in the pre-tokenizer, and that stage is configuration. There is no byte-level-spelled vocabulary without a byte-level pre-tokenizer — the two always arrive together — so a `space` field would restate a fact its own consumer already holds, with nothing to arbitrate a disagreement. `.tiktoken` has the identical silence and it has never cost anyone anything, because a vocabulary does not travel without something that knows what to do with it.

It does **not** supersede `.spm`, which carries SentencePiece scores and piece types that this format has no place for.

One consequence is worth stating plainly, because it is easy to get wrong. A consumer classifying atoms and orphans must apply its own `end_of_word_suffix` before doing so: under a word-final marker, one character _plus_ the marker is a seed spelling. CLIP names 139 of its 256 marked characters in no merge at all, and a reader that classified them without knowing about the marker would call them orphans and drop the id of every one-character word.

## Size

Gemma 4, `google/gemma-4-12B-it`:

|                     |      raw |     gzip |
| ------------------- | -------: | -------: |
| `tokenizer.json`    | 30.68 MB |  5.15 MB |
| `.mbpe`, vocabulary |     2.46 |     1.25 |
| `.mbpe`, merges     |     1.95 |     0.63 |
| **`.mbpe`, total**  | **4.41** | **1.88** |

Per vocabulary entry that is **6.9 gzipped bytes**, against 8.5–8.7 for `.tiktoken` and `.spm` — while carrying strictly more.

## Adoption

Nothing above requires a new tokenizer library. The vocabulary is the payload; a `tokenizer.json` is the configuration, and it is _tiny_ — strip `model.vocab` and `model.merges` from Gemma 4's and the entire pipeline definition (normalizer, pre-tokenizer, post-processor, decoder, added tokens) is **3.9 KB** of the file's 32.17 MB. Qwen3: 11.42 MB → 4.4 KB. GPT-2: 1.36 MB → 0.6 KB.

The one change that would let a provider ship the two apart is for `tokenizer.json` to be able to reference an external vocabulary:

```json
"model": { "type": "BPE", "vocab_file": "vocab.mbpe" }
```

With that, a published tokenizer is a few kilobytes of config plus a payload in whatever format states it correctly, versioned and checksummed on its own, and shareable between checkpoints that use the same vocabulary. Without it, adopting any vocabulary format means shipping the vocabulary twice.

## Verifying a conversion

A derived vocabulary that is wrong is worse than a large one. The check that matters _means_: pack both the source and the `.mbpe` into the loader's binary form and require the results to be byte-identical. splintr's converter does this before it writes anything:

```
cargo run -p splintr-vocab-pack --example mbpe_from_json -- tokenizer.json out.mbpe
```
