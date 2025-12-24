#!/usr/bin/env python3
"""
Extract DeepSeek V3 vocabulary from HuggingFace to tiktoken format.

The tiktoken format is: base64_encoded_token<space>rank
where rank is the token ID (0-based).

Usage:
    python scripts/extract_deepseek_vocab.py
    python scripts/extract_deepseek_vocab.py --model deepseek-ai/deepseek-coder-7b-instruct-v1.5 --output custom_vocab.tiktoken
"""

import argparse
import base64
import json
from transformers import AutoTokenizer


def extract_deepseek_vocab(model_name: str, output_path: str) -> None:
    """Extract DeepSeek vocabulary from HuggingFace to tiktoken format."""

    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Tokenizer class: {tokenizer.__class__.__name__}")

    # Get tokenizer.json from backend_tokenizer
    import tempfile
    import os
    backend = tokenizer.backend_tokenizer
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
        backend.save(temp_path)

    with open(temp_path, 'r') as f:
        data = json.load(f)
    os.unlink(temp_path)

    # Extract vocab
    vocab = data['model']['vocab']
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

    print(f"Total tokens: {len(sorted_vocab)}")
    print(f"\nFirst 20 tokens:")
    for token, idx in sorted_vocab[:20]:
        print(f"  {idx}: {repr(token)}")

    print(f"\nLast 20 tokens:")
    for token, idx in sorted_vocab[-20:]:
        print(f"  {idx}: {repr(token)}")

    print(f"\nWriting to {output_path}...")
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

    print(f"\nSpecial tokens:")
    print(f"  {tokenizer.special_tokens_map}")


def main():
    parser = argparse.ArgumentParser(description="Extract DeepSeek vocabulary from HuggingFace")
    parser.add_argument("--model", default="deepseek-ai/deepseek-v3",
                        help="Model name on HuggingFace Hub (default: deepseek-ai/deepseek-v3)")
    parser.add_argument("--output", default="python/splintr/vocabs/deepseek_v3.tiktoken",
                        help="Output path for vocab file")
    args = parser.parse_args()

    extract_deepseek_vocab(args.model, args.output)


if __name__ == "__main__":
    main()
