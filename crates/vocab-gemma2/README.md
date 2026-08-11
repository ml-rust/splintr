# splintr-vocab-gemma2

Gemma 2.

Data only — the bytes of one vocabulary family, with nothing that interprets them. Use [`splintr`](https://crates.io/crates/splintr) with its `vocab-gemma2` feature; that is what re-exports this constant and can load it.

## Provenance

Converted from Google's Gemma 2 `tokenizer.model`, MD5 `f9e2445870ec741aa6346bbd75531bb4`, id for id into the `.spm` text format (`base64(piece) score type` per line). The vocabulary is upstream's, not splintr's.

**This is a modified file** — the SentencePiece protocol buffer converted to text. Nothing was added, removed, reordered or rounded: all 256,000 pieces keep their ids, scores and piece types. See `NOTICE`.

The same vocabulary is published standalone, outside this crate, at [`fs90/gemma-2-tokenizer-spm`](https://huggingface.co/fs90/gemma-2-tokenizer-spm).

## Licence

Gemma is provided under and subject to the **Gemma Terms of Use**, found at [ai.google.dev/gemma/terms](https://ai.google.dev/gemma/terms) and reproduced in full in `LICENSE`. **Use is subject to the use restrictions in Section 3.2**, and if you redistribute this crate or anything derived from it you must pass those restrictions on, provide a copy of the agreement, and include `NOTICE`.

Gemma 4 is **not** under these terms — it is Apache-2.0, and is `splintr-vocab-gemma4`.
