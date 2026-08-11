# splintr-vocab-cl100k

OpenAI cl100k_base — GPT-4 and GPT-3.5-turbo.

Data only — the packed bytes of one vocabulary family, with nothing that interprets them. Use [`splintr`](https://crates.io/crates/splintr) with its `vocab-cl100k` feature; that is what re-exports these constants and can load them.

## Provenance

Extracted from OpenAI, `https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken`. The vocabulary is upstream's, not splintr's, and keeps upstream's licence — see `LICENSE`.
