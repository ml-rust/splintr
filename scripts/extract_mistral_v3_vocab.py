#!/usr/bin/env python3
"""
Extract Mistral V3/Tekken vocabulary from HuggingFace to tiktoken format.

V3 uses Tiktoken (NOT SentencePiece) with a much larger vocabulary (~131k tokens).
"""

import argparse
import base64
import json
from transformers import AutoTokenizer


def extract_v3_vocab(model_name: str, output_path: str):
    """Extract V3 Tekken vocab in tiktoken format."""
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Tokenizer class: {tokenizer.__class__.__name__}")

    # Get tokenizer.json
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

    # Convert to tiktoken format
    output_lines = []

    for token, rank in sorted_vocab:
        # For Tekken/Tiktoken, tokens are already byte sequences
        # Just base64 encode them
        token_bytes = token.encode('utf-8')
        b64_token = base64.b64encode(token_bytes).decode('ascii')
        output_lines.append(f"{b64_token} {rank}")

    # Write output
    print(f"\nWriting to {output_path}...")
    with open(output_path, 'w') as f:
        f.write('\n'.join(output_lines))

    print(f"Wrote {len(output_lines)} tokens")

    # Print special tokens info
    print(f"\nSpecial tokens:")
    print(f"  {tokenizer.special_tokens_map}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistralai/Mistral-Nemo-Instruct-2407")
    parser.add_argument("--output", default="python/splintr/vocabs/mistral_v3_tekken.tiktoken")
    args = parser.parse_args()
    extract_v3_vocab(args.model, args.output)


if __name__ == "__main__":
    main()
