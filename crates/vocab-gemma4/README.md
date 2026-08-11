# splintr-vocab-gemma4

Gemma 4.

Data only — the packed bytes of one vocabulary family, with nothing that interprets them. Use [`splintr`](https://crates.io/crates/splintr) with its `vocab-gemma4` feature; that is what re-exports these constants and can load them.

## Provenance

Converted from Google's `tokenizer.json` for `google/gemma-4-12B-it`, SHA-256 `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`, into the plain-text `.mbpe` format. The vocabulary is upstream's, not splintr's, and keeps upstream's licence: Apache-2.0, per Google's separate Gemma 4 licence at [ai.google.dev/gemma/apache_2](https://ai.google.dev/gemma/apache_2). Gemma 1, 2 and 3 are **not** Apache — they are under the Gemma Terms of Use — and none of them is here.

**This is a modified file.** All 262,144 pieces keep their ids and spellings and all 236,339 distinct merges keep their priority order; the tokenizer configuration around them — normalizer, pre-tokenizer, added tokens, decoder — is not carried, because it is not vocabulary. Regenerate it from upstream's file with `cargo run -p splintr-vocab-pack --example mbpe_from_json -- tokenizer.json gemma4.mbpe`, which refuses to write unless both files pack to identical bytes.

## Why this crate ships an `.mbpe`

Every other `splintr-vocab-*` crate ships a small derived text file: a `.tiktoken` (`base64(token) rank`) or a `.spm` (`base64(piece) score type`). Neither can hold Gemma 4.

A `.tiktoken` rank is both the token's id and its merge priority. That works for OpenAI vocabularies, which are built so the two coincide. Gemma 4's do not: 465 places where a later merge yields a lower id, and 514,906 merges collapsing onto 236,339 distinct tokens — the two orders are not even the same length. Forcing them into one column mistokenizes 8.1% of real documents.

A `.spm` needs SentencePiece scores, and Gemma 4 has none. Google publishes no `tokenizer.model` for it, and the official Apache-2.0 GGUFs carry a placeholder score of `-1000.0` for every one of the 262,144 pieces.

So Gemma 4 has to ship a file stating id order and merge order separately. Its `tokenizer.json` is the only file Google publishes that does, and shipping that verbatim is what this crate used to do — at 30.7 MB, packing to a 5.2 MB crate. `.mbpe` states the same two things in 4.4 MB of text, packing to 1.9 MB, and the conversion is checked by requiring both files to pack to byte-identical binaries. See `docs/mbpe.md` in the splintr repository for the format.

The build script packs the text into two binaries in `OUT_DIR` — the vocabulary and the merge order — so the text is what ships and both binaries are derived, leaving nothing that can disagree.
