# splintr-vocab-llama2

Llama 2 and Code Llama.

Data only — the packed bytes of one vocabulary family, with nothing that interprets them. Use [`splintr`](https://crates.io/crates/splintr) with its `vocab-llama2` feature; that is what re-exports these constants and can load them.

## Provenance

Extracted from `codellama/CodeLlama-7b-hf` on HuggingFace. The vocabulary is upstream's, not splintr's, and keeps upstream's licence — the Llama 2 Community License, in `LICENSE`.

Code Llama rather than Llama 2 because Meta publishes Llama 2's `tokenizer.model` only behind a gated repository, while Code Llama's is open and carries the same licence. Code Llama extended Llama 2's SentencePiece model in place, so its first 32,000 ids are Llama 2's pieces and Llama 2's scores exactly — checked piece-for-piece and score-for-score against Meta's file (md5 `eeec4125e9c7560836b4873b6f8e3025`), as republished verbatim by `TinyLlama/TinyLlama-1.1B-Chat-v1.0`. The Llama 2 vocabulary is that prefix, derived by the build script rather than committed a second time.
