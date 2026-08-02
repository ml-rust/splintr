#!/usr/bin/env python3
"""
Extract the Whisper base BPE vocabulary to tiktoken format.

Whisper uses GPT-2 byte-level BPE for its base vocabulary (~50,257 tokens). The
special tokens (endoftext, startoftranscript, language, timestamp, etc.) are NOT
stored here -- they are generated programmatically per-variant in
`src/core/whisper.rs::whisper_special_tokens`. This mirrors how the bundled
DeepSeek/Mistral-V3 byte-level vocabs are stored.

The base BPE is identical across every Whisper variant (tiny .. large-v3,
English-only, multilingual), so a single `whisper.tiktoken` serves all of them.

tiktoken (byte-level) format: `base64(byte_level_string.utf8) <space> rank`
where the key is the byte-level-encoded string (e.g. "Ġa") exactly as stored in
`model.vocab`, and rank is the 0-based token id.

Usage:
    python scripts/extract_whisper_vocab.py
    python scripts/extract_whisper_vocab.py --model openai/whisper-large-v3
    python scripts/extract_whisper_vocab.py --tokenizer-json /path/to/tokenizer.json
"""

import argparse
import base64
import json
import sys
import urllib.request
from pathlib import Path


def load_tokenizer_json(model: str, tokenizer_json: str | None) -> dict:
    if tokenizer_json:
        print(f"Reading tokenizer.json from {tokenizer_json}")
        return json.loads(Path(tokenizer_json).read_text(encoding="utf-8"))

    url = f"https://huggingface.co/{model}/resolve/main/tokenizer.json"
    print(f"Downloading tokenizer.json from {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "splintr-extract"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def extract(model: str, tokenizer_json: str | None, output_path: str) -> None:
    data = load_tokenizer_json(model, tokenizer_json)

    vocab = data.get("model", {}).get("vocab")
    if not isinstance(vocab, dict):
        sys.exit("error: tokenizer.json missing model.vocab object")

    sorted_vocab = sorted(vocab.items(), key=lambda kv: kv[1])
    n = len(sorted_vocab)
    max_id = sorted_vocab[-1][1]
    print(f"Base vocab entries: {n} (max id {max_id})")

    # The base BPE must be contiguous 0..n-1 with no special tokens mixed in;
    # specials are generated at runtime starting above this range.
    if max_id != n - 1:
        print(
            f"  warning: ids are not contiguous (max id {max_id} != {n - 1}). "
            "model.vocab may contain non-base tokens."
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for token_str, token_id in sorted_vocab:
            token_b64 = base64.b64encode(token_str.encode("utf-8")).decode("ascii")
            f.write(f"{token_b64} {token_id}\n")

    print(f"Wrote {n} tokens to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Whisper base BPE vocab to tiktoken format")
    parser.add_argument("--model", default="openai/whisper-large-v3",
                        help="HF model id to pull tokenizer.json from (default: openai/whisper-large-v3)")
    parser.add_argument("--tokenizer-json", default=None,
                        help="Local tokenizer.json path (skips download)")
    parser.add_argument("--output", default="vocabs/whisper.tiktoken",
                        help="Output path for vocab file")
    args = parser.parse_args()

    extract(args.model, args.tokenizer_json, args.output)


if __name__ == "__main__":
    main()
