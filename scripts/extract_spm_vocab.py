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

3. **Piece type is destroyed.** SentencePiece matches ``USER_DEFINED`` pieces
   verbatim *before* BPE runs; they are never merge candidates. Storing score
   alone cannot express that, and it is not recoverable afterwards: a
   ``USER_DEFINED`` piece and a ``CONTROL`` piece both score ``0.0``, but
   ``CONTROL`` is *never* matched from text. Measured with ``sentencepiece``
   0.2.0 on Gemma 2, whose 245 ``USER_DEFINED`` and 3 ``CONTROL`` pieces are
   indistinguishable by score: ``encode("<blockquote>")`` is ``[191]`` (the
   whole user-defined piece) while ``encode("<pad>")`` is
   ``[235322, 8939, 235313]`` (the control piece shattered).

The ``.spm`` format written here fixes all three.

Format
------
One line per token id, in ascending id order, no gaps::

    <base64 of piece encoded UTF-8> <space> <score> <space> <type>

* The piece is SentencePiece's own ``id_to_piece(i)``, so ``<0x41>`` keeps its
  real spelling and ``▁`` runs keep theirs.
* The score is ``get_score(i)`` written with Python's ``repr``, the shortest
  decimal that round-trips the IEEE-754 double. Mistral's scores are whole
  numbers in ``[-31740, 0]`` plus ``-1e9`` sentinels, all of which are exactly
  representable in ``f32``, so ``repr`` -> ``str::parse::<f32>`` is exact.
  ``repr`` (rather than a fixed ``%.*f``) keeps that true for any vocabulary
  whose scores are genuine log-probabilities.
* The type is SentencePiece's own ``ModelProto.SentencePiece.Type`` enum value
  (1 NORMAL, 2 UNKNOWN, 3 CONTROL, 4 USER_DEFINED, 6 BYTE), read from the
  proto rather than guessed from the spelling -- ``<blockquote>`` and ``<0x41>``
  and ``<pad>`` are all ``<...>``-shaped and all three are different types.

A two-column line (no type) is still accepted by the loader and read as
NORMAL, which is what every ``.spm`` written before the type column carried
implicitly. That is correct for those files only because none of them declares
a ``USER_DEFINED`` piece; the loader documents the same caveat.

Usage
-----
    python scripts/extract_spm_vocab.py \
        --model /path/to/tokenizer.model \
        --output crates/vocab-mistral/vocabs/mistral.spm [--verify]
"""

import argparse
import base64
import sys

import sentencepiece
from sentencepiece import sentencepiece_model_pb2

#: ``ModelProto.SentencePiece.Type.USER_DEFINED``: matched verbatim, never merged.
USER_DEFINED = 4


def extract(model_path: str, output_path: str) -> int:
    sp = sentencepiece.SentencePieceProcessor(model_file=model_path)
    size = sp.get_piece_size()

    # Types come from the proto: the processor exposes IsControl/IsByte/IsUnused
    # as separate predicates, but not USER_DEFINED, which is the one that
    # decides whether a piece is matched verbatim.
    proto = sentencepiece_model_pb2.ModelProto()
    with open(model_path, "rb") as handle:
        proto.ParseFromString(handle.read())

    lines = []
    sentinels = 0
    user_defined = 0
    for i in range(size):
        piece = sp.id_to_piece(i)
        score = sp.get_score(i)
        piece_type = int(proto.pieces[i].type)
        if score == -1e9:
            sentinels += 1
        if piece_type == USER_DEFINED:
            user_defined += 1
        b64 = base64.b64encode(piece.encode("utf-8")).decode("ascii")
        lines.append(f"{b64} {score!r} {piece_type}")

    with open(output_path, "w", encoding="ascii", newline="\n") as f:
        f.write("\n".join(lines))
        f.write("\n")

    print(f"{model_path} -> {output_path}")
    print(f"  pieces: {size}")
    print(f"  -1e9 sentinels: {sentinels}")
    print(f"  USER_DEFINED pieces: {user_defined}")
    return size


def verify(model_path: str, output_path: str) -> None:
    """Re-read what was written and assert piece, score and type equality for every id."""
    sp = sentencepiece.SentencePieceProcessor(model_file=model_path)
    size = sp.get_piece_size()
    proto = sentencepiece_model_pb2.ModelProto()
    with open(model_path, "rb") as handle:
        proto.ParseFromString(handle.read())

    with open(output_path, "r", encoding="ascii") as f:
        lines = [line for line in f.read().split("\n") if line]

    assert len(lines) == size, f"line count {len(lines)} != piece count {size}"

    import struct

    for i, line in enumerate(lines):
        b64, score_str, type_str = line.split(" ")
        piece = base64.b64decode(b64).decode("utf-8")
        score = float(score_str)
        assert piece == sp.id_to_piece(i), f"id {i}: piece {piece!r} != {sp.id_to_piece(i)!r}"
        assert score == sp.get_score(i), f"id {i}: score {score!r} != {sp.get_score(i)!r}"
        assert int(type_str) == int(proto.pieces[i].type), (
            f"id {i}: type {type_str} != {int(proto.pieces[i].type)}"
        )
        # The Rust side parses into f32; confirm the value survives that too.
        as_f32 = struct.unpack("<f", struct.pack("<f", score))[0]
        assert as_f32 == score, f"id {i}: score {score!r} is not exact in f32"

    print(f"  verified {size} pieces, scores and types round-trip exactly (including f32)")


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
