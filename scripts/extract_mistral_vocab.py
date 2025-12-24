#!/usr/bin/env python3
"""
Extract Mistral V1/V2 vocabulary from HuggingFace to tiktoken format.

Mistral uses SentencePiece with byte fallback, which we can convert to tiktoken format
(base64-encoded tokens with ranks).

V1 vocab (32,768 tokens): Mistral 7B v0.1/v0.2, Mixtral 8x7B
V2 vocab (32,768 tokens): Mistral 7B v0.3, Mixtral 8x22B, Codestral
"""

import argparse
import base64
import json
from transformers import AutoTokenizer


def extract_mistral_vocab(model_name: str, output_path: str):
    """Extract Mistral V1/V2 vocabulary from HuggingFace to tiktoken format."""
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

    # Convert to tiktoken format
    output_lines = []

    for token, rank in sorted_vocab:
        # Handle byte fallback tokens: <0x00> -> raw byte 0x00
        if token.startswith('<0x') and token.endswith('>'):
            hex_val = token[3:-1]
            token_bytes = bytes([int(hex_val, 16)])
        elif token in ['<unk>', '<s>', '</s>']:
            # Special tokens - encode as-is
            token_bytes = token.encode('utf-8')
        else:
            # Regular BPE token - encode as UTF-8
            # This includes ▁ (U+2581) for SentencePiece word boundary
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
    parser = argparse.ArgumentParser(description="Extract Mistral vocabulary from HuggingFace")
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.1",
                        help="Model name on HuggingFace Hub (default: Mistral 7B v0.1 for V1)")
    parser.add_argument("--output", default="python/splintr/vocabs/mistral.tiktoken",
                        help="Output path for vocab file")
    args = parser.parse_args()

    extract_mistral_vocab(args.model, args.output)


if __name__ == "__main__":
    main()
