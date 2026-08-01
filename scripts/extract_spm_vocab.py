#!/usr/bin/env python3
"""
Extract a SentencePiece vocabulary losslessly into splintr's ``.spm`` bundled format.

Why this exists
---------------
splintr previously stored SentencePiece vocabularies in tiktoken format
(``base64(token_bytes) rank``). That is lossy for SentencePiece in two ways:

1. **Scores are destroyed.** SentencePiece merges by *score*, not by id order.
   The 15 whitespace-run pieces (``▁``, ``▁▁``, ...) carry a ``-1e9``
   "never merge" sentinel; recovering merge order from id order inverts them.
2. **Byte-fallback spelling is destroyed.** Real SentencePiece pieces are
   spelled ``<0x41>``; storing them as the raw byte forces the loader to
   *reconstruct* the spelling by scanning for a run of 256 consecutive ids.

The ``.spm`` format written here fixes both.

Format
------
One line per token id, in ascending id order, no gaps::

    <base64 of piece encoded UTF-8> <space> <score>

* The piece is SentencePiece's own ``id_to_piece(i)``, so ``<0x41>`` keeps its
  real spelling and ``▁`` runs keep theirs.
* The score is ``get_score(i)`` written with Python's ``repr``, the shortest
  decimal that round-trips the IEEE-754 double. Mistral's scores are whole
  numbers in ``[-31740, 0]`` plus ``-1e9`` sentinels, all of which are exactly
  representable in ``f32``, so ``repr`` -> ``str::parse::<f32>`` is exact.
  ``repr`` (rather than a fixed ``%.*f``) keeps that true for any vocabulary
  whose scores are genuine log-probabilities.

Usage
-----
    python scripts/extract_spm_vocab.py \
        --model /path/to/tokenizer.model \
        --output python/splintr/vocabs/mistral.spm [--verify]
"""

import argparse
import base64
import sys

import sentencepiece


def extract(model_path: str, output_path: str) -> int:
    sp = sentencepiece.SentencePieceProcessor(model_file=model_path)
    size = sp.get_piece_size()

    lines = []
    sentinels = 0
    for i in range(size):
        piece = sp.id_to_piece(i)
        score = sp.get_score(i)
        if score == -1e9:
            sentinels += 1
        b64 = base64.b64encode(piece.encode("utf-8")).decode("ascii")
        lines.append(f"{b64} {score!r}")

    with open(output_path, "w", encoding="ascii", newline="\n") as f:
        f.write("\n".join(lines))
        f.write("\n")

    print(f"{model_path} -> {output_path}")
    print(f"  pieces: {size}")
    print(f"  -1e9 sentinels: {sentinels}")
    return size


def verify(model_path: str, output_path: str) -> None:
    """Re-read what was written and assert piece and score equality for every id."""
    sp = sentencepiece.SentencePieceProcessor(model_file=model_path)
    size = sp.get_piece_size()

    with open(output_path, "r", encoding="ascii") as f:
        lines = [line for line in f.read().split("\n") if line]

    assert len(lines) == size, f"line count {len(lines)} != piece count {size}"

    import struct

    for i, line in enumerate(lines):
        b64, _, score_str = line.rpartition(" ")
        piece = base64.b64decode(b64).decode("utf-8")
        score = float(score_str)
        assert piece == sp.id_to_piece(i), f"id {i}: piece {piece!r} != {sp.id_to_piece(i)!r}"
        assert score == sp.get_score(i), f"id {i}: score {score!r} != {sp.get_score(i)!r}"
        # The Rust side parses into f32; confirm the value survives that too.
        as_f32 = struct.unpack("<f", struct.pack("<f", score))[0]
        assert as_f32 == score, f"id {i}: score {score!r} is not exact in f32"

    print(f"  verified {size} pieces and scores round-trip exactly (including f32)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="path to a SentencePiece tokenizer.model")
    parser.add_argument("--output", required=True, help="path to the .spm file to write")
    parser.add_argument("--verify", action="store_true", help="re-read the output and check it")
    args = parser.parse_args()

    extract(args.model, args.output)
    if args.verify:
        verify(args.model, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
