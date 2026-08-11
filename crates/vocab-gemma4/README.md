# splintr-vocab-gemma4

Gemma 4.

Data only — the packed bytes of one vocabulary family, with nothing that
interprets them. Use [`splintr`](https://crates.io/crates/splintr) with its
`vocab-gemma4` feature; that is what re-exports these constants and can load them.

## Provenance

Google's `tokenizer.json` from `google/gemma-4-12B-it`, shipped **byte for
byte** — SHA-256 `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`.
The vocabulary is upstream's, not splintr's, and keeps upstream's licence:
Apache-2.0, per Google's separate Gemma 4 licence at
[ai.google.dev/gemma/apache_2](https://ai.google.dev/gemma/apache_2). Gemma 1, 2
and 3 are **not** Apache — they are under the Gemma Terms of Use — and none of
them is here.

## Why this crate ships a `tokenizer.json`

Every other `splintr-vocab-*` crate ships a small derived text file: a
`.tiktoken` (`base64(token) rank`) or a `.spm` (`base64(piece) score type`).
Neither can hold Gemma 4.

A `.tiktoken` rank is both the token's id and its merge priority. That works for
OpenAI vocabularies, which are built so the two coincide. Gemma 4's do not: 465
places where a later merge yields a lower id, and 514,906 merges collapsing onto
236,339 distinct tokens — the two orders are not even the same length. Forcing
them into one column mistokenizes 8.1% of real documents.

A `.spm` needs SentencePiece scores, and Gemma 4 has none. Google publishes no
`tokenizer.model` for it, and the official Apache-2.0 GGUFs carry a placeholder
score of `-1000.0` for every one of the 262,144 pieces.

So Gemma 4 has to ship a file stating id order and merge order separately, and
its `tokenizer.json` is the only one Google publishes that does. Shipping it
verbatim rather than converting it also means this crate can be checked against
upstream with `sha256sum` and nothing else.

The build script packs it into two binaries in `OUT_DIR` — the vocabulary and
the merge order — so the json is what ships and both binaries are derived,
leaving nothing that can disagree.
