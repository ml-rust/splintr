#!/usr/bin/env python3
"""
Convert DeepSeek V3 tokenizer.json to tiktoken format.

The tiktoken format is: base64_encoded_token<space>rank
where rank is the token ID (0-based).

Usage:
    python scripts/convert_deepseek_vocab.py
"""

import json
import base64
from pathlib import Path


def convert_deepseek_to_tiktoken(input_path: str, output_path: str) -> None:
    """Convert HuggingFace tokenizer.json to tiktoken format."""

    print(f"Reading {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        tokenizer_data = json.load(f)

    # Extract vocabulary from model.vocab
    vocab = tokenizer_data["model"]["vocab"]
    print(f"Found {len(vocab)} tokens in vocabulary")

    # Sort by token ID to ensure correct order
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

    print(f"Writing to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for token_str, token_id in sorted_vocab:
            # Encode token string to bytes, then to base64
            token_bytes = token_str.encode("utf-8")
            token_b64 = base64.b64encode(token_bytes).decode("ascii")
            f.write(f"{token_b64} {token_id}\n")

    print(f"Done! Wrote {len(sorted_vocab)} tokens to {output_path}")

    # Print some stats
    print(f"\nVocab stats:")
    print(f"  First token: {sorted_vocab[0]}")
    print(f"  Last token: {sorted_vocab[-1]}")

    # Check for any gaps in token IDs
    expected_ids = set(range(len(sorted_vocab)))
    actual_ids = set(vocab.values())
    missing = expected_ids - actual_ids
    extra = actual_ids - expected_ids

    if missing:
        print(f"  Warning: Missing token IDs: {sorted(missing)[:10]}...")
    if extra:
        print(f"  Warning: Extra token IDs outside range: {sorted(extra)[:10]}...")


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    input_path = project_root / "research" / "deepseek_v3_tokenizer" / "tokenizer.json"
    output_path = project_root / "python" / "splintr" / "vocabs" / "deepseek_v3.tiktoken"

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    convert_deepseek_to_tiktoken(str(input_path), str(output_path))
    return 0


if __name__ == "__main__":
    exit(main())
