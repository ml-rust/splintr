# splintr-vocab-whisper

OpenAI Whisper (multilingual).

Data only — the packed bytes of one vocabulary family, with nothing that
interprets them. Use [`splintr`](https://crates.io/crates/splintr) with its
`vocab-whisper` feature; that is what re-exports these constants and can load them.

## Provenance

Extracted from `openai/whisper-large-v3` on HuggingFace. The vocabulary is upstream's, not splintr's, and
keeps upstream's licence — see `LICENSE`.
