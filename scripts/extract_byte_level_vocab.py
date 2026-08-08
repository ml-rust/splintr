#!/usr/bin/env python3
"""Convert a HuggingFace byte-level BPE `tokenizer.json` to tiktoken format.

The tiktoken format is `base64(raw token bytes) rank`, one per line. A
HuggingFace file does not store the raw bytes: a ByteLevel tokenizer maps every
byte to a printable character first, so `" the"` is stored as `"Ġthe"`. This
script inverts that mapping, which is what makes the result loadable by
`Tokenizer::from_bytes_chain` with no byte-level stage — the same shape as
`vocabs/llama3.tiktoken`.

Only the base vocabulary is written. `added_tokens` are special tokens and
belong in `pretrained.rs`, where they are placed above the base vocabulary so no
original id moves.

Usage:
    python scripts/extract_byte_level_vocab.py <tokenizer.json> <out.tiktoken>
"""

import argparse
import base64
import json
import sys


def byte_decoder() -> dict[str, int]:
    """Invert GPT-2's byte-to-unicode map (`bytes_to_unicode` in HF)."""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    mapped = printable[:]
    n = 0
    for b in range(256):
        if b not in printable:
            printable.append(b)
            mapped.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(printable, mapped)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tokenizer_json")
    ap.add_argument("output")
    args = ap.parse_args()

    with open(args.tokenizer_json, encoding="utf-8") as f:
        data = json.load(f)

    model = data["model"]
    if model["type"] != "BPE":
        print(f"error: model.type is {model['type']!r}, expected BPE", file=sys.stderr)
        return 1

    decoder = byte_decoder()
    vocab = model["vocab"]
    rows = []
    for token, rank in sorted(vocab.items(), key=lambda kv: kv[1]):
        try:
            raw = bytes(decoder[c] for c in token)
        except KeyError as e:
            print(f"error: token {token!r} (rank {rank}) has un-mappable char {e}", file=sys.stderr)
            return 1
        rows.append((raw, rank))

    ranks = [r for _, r in rows]
    if ranks != list(range(len(ranks))):
        print(f"error: ranks are not 0..{len(ranks)} contiguous", file=sys.stderr)
        return 1
    if len({raw for raw, _ in rows}) != len(rows):
        print("error: duplicate token bytes after byte-level decoding", file=sys.stderr)
        return 1

    with open(args.output, "w", encoding="ascii") as f:
        for raw, rank in rows:
            f.write(f"{base64.b64encode(raw).decode('ascii')} {rank}\n")

    added = data.get("added_tokens", [])
    print(f"wrote {len(rows)} tokens to {args.output}")
    print(f"base vocab size: {len(rows)}")
    if added:
        print(f"added_tokens (NOT written, ids {added[0]['id']}..{added[-1]['id']}):")
        for t in added:
            print(f"  {t['id']}: {t['content']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
