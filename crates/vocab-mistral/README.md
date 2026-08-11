# splintr-vocab-mistral

Mistral V1, V2 and V3/Tekken.

Data only — the packed bytes of one vocabulary family, with nothing that interprets them. Use [`splintr`](https://crates.io/crates/splintr) with its `vocab-mistral` feature; that is what re-exports these constants and can load them.

## Provenance

Extracted from `mistralai/Mistral-7B-Instruct-v0.3` (V1/V2 SentencePiece) and `mistralai/Mistral-Nemo-Instruct-2407` (V3/Tekken) on HuggingFace. The vocabulary is upstream's, not splintr's, and keeps upstream's licence — see `LICENSE`.
