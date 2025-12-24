#!/usr/bin/env python3
"""
Extract Mistral V2 vocabulary from HuggingFace to tiktoken format.

V2 includes 771 added tokens (IDs 0-770) plus the same BPE merges as V1.
The format must match V1: base64-encoded byte sequences, not token strings.
"""

import argparse
import base64
import json
from transformers import AutoTokenizer


def extract_v2_vocab(model_name: str, output_path: str):
    """Extract V2 vocab matching V1's tiktoken format."""
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Vocab size: {tokenizer.vocab_size}")

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

    # Convert to tiktoken format (same as convert_mistral_vocab.py)
    output_lines = []

    for token, rank in sorted_vocab:
        # Handle byte fallback tokens
        if token.startswith('<0x') and token.endswith('>'):
            # Byte fallback: <0x00> -> byte 0x00
            hex_val = token[3:-1]
            byte_val = bytes([int(hex_val, 16)])
            token_bytes = byte_val
        elif token in ['<unk>', '<s>', '</s>']:
            # Special tokens - encode as-is
            token_bytes = token.encode('utf-8')
        elif token.startswith('[') and token.endswith(']'):
            # Control tokens [INST], [/INST], etc. - encode as-is
            token_bytes = token.encode('utf-8')
        elif token.startswith('[control_'):
            # Control placeholder tokens - encode as-is
            token_bytes = token.encode('utf-8')
        else:
            # Regular BPE token - encode as UTF-8
            # This includes ▁ (U+2581) for SentencePiece word boundary
            token_bytes = token.encode('utf-8')

        # Base64 encode
        b64_token = base64.b64encode(token_bytes).decode('ascii')
        output_lines.append(f"{b64_token} {rank}")

    # Write output
    print(f"\nWriting to {output_path}...")
    with open(output_path, 'w') as f:
        f.write('\n'.join(output_lines))

    print(f"Wrote {len(output_lines)} tokens")
    print(f"\nToken ID ranges:")
    print(f"  IDs 0-2: <unk>, <s>, </s>")
    print(f"  IDs 3-770: Control tokens ([INST], [TOOL_CALLS], etc.)")
    print(f"  IDs 771-32767: BPE merges (same structure as V1)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--output", default="python/splintr/vocabs/mistral_v2.tiktoken")
    args = parser.parse_args()
    extract_v2_vocab(args.model, args.output)


if __name__ == "__main__":
    main()
