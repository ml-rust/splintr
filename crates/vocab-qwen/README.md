# splintr-vocab-qwen

Qwen 2 and 3 — and Baichuan-M2, which ships the same file.

Data only — the packed bytes of one vocabulary family, with nothing that interprets them. Use [`splintr`](https://crates.io/crates/splintr) with its `vocab-qwen` feature; that is what re-exports these constants and can load them.

## Provenance

Extracted from `Qwen/Qwen3-8B` on HuggingFace. The vocabulary is upstream's, not splintr's, and keeps upstream's licence — see `LICENSE`.
