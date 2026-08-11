# splintr-vocab-pack

Build-time packer for splintr's bundled vocabularies.

Every `splintr-vocab-*` crate ships its vocabulary as text. The two that need decoding before use are packed here, from the crate's build script, into the binary form splintr's loader reads:

- `.tiktoken` — `base64(token) rank`, one number serving as both id and merge priority. cl100k, o200k, llama3, deepseek, qwen, glm, kimi, modernbert, whisper, and Mistral's Tekken vocabulary.
- `.mbpe` — a vocabulary whose merge order is *not* its id order, so the two are stated apart. Gemma 4. Specified in `docs/mbpe.md` in the splintr repository.

`.spm` (`base64(piece) score [type]` — the SentencePiece vocabularies: gemma2, gemma3, mistral v1/v2, llama2) does **not** come through here. Those crates compile their text in directly, because splintr's SentencePiece loader reads that text as it stands.

The `mbpe_from_json` example derives an `.mbpe` from a HuggingFace `tokenizer.json`, verifying that both pack to identical bytes before it writes — it is the reference writer for the format:

```
cargo run --example mbpe_from_json -- tokenizer.json vocab.mbpe
```

Nobody needs to depend on this crate directly.
