#!/usr/bin/env python3
"""Dump a GGUF file's `tokenizer.ggml.*` block into a self-contained JSON fixture.

The JSON mirrors splintr's `GgufVocab` struct exactly (`src/core/gguf/vocab.rs`),
one JSON key per struct field, plus — when llama.cpp ships them next to the
`.gguf` — the parsed `<file>.inp` test cases and `<file>.out` expected token ids.
One JSON per vocabulary, so the Rust side (`examples/verify_gguf.rs`) needs
nothing but the JSON.

A metadata key the file does not declare is simply absent from the JSON, with one
exception: `model` is a required field of `GgufVocab`, so an absent key resolves to
llama.cpp's own documented default ("llama") rather than being omitted. Every other
default is left to splintr's loader, which is the component that knows them per
dialect.

Usage:
    python3 scripts/extract_gguf_vocab.py <file.gguf|dir> [...] --out-dir DIR

With `--tiktoken-dir DIR` the same vocabulary is ALSO rendered in the tiktoken
text format splintr's byte-level loader reads (`python/splintr/vocabs/*.tiktoken`):
one `base64(token_bytes) rank` line per token, in id order, rank == token id.
That rendering exists so the byte-level BPE path can be pointed at a GGUF
`model=llama` vocabulary and scored against the same llama.cpp fixtures as the
piece-level `SpmTokenizer` (see `examples/verify_spm_vs_bpe.rs`).

Requires the `gguf` package (`pip install gguf`).
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from pathlib import Path

# A SentencePiece byte-fallback piece: `<0x00>` .. `<0xFF>`.
BYTE_PIECE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")

try:
    from gguf import GGUFReader
except ImportError:  # pragma: no cover - tooling script
    sys.exit("error: the 'gguf' package is required (pip install gguf)")

# The delimiter llama.cpp's tests/test-tokenizer-0.cpp splits `.inp` on. The
# surrounding newlines are PART of the delimiter, so a case never carries the
# trailing newline that precedes the next delimiter.
CASE_SEPARATOR = b"\n__ggml_vocab_test__\n"

# GGUF metadata key (minus the `tokenizer.ggml.` prefix) -> GgufVocab field.
# `model` and `tokens` are handled separately: they are the two required fields.
OPTIONAL_KEYS = {
    "scores": "scores",
    "merges": "merges",
    "token_type": "token_type",
    "add_space_prefix": "add_space_prefix",
    "remove_extra_whitespaces": "remove_extra_whitespaces",
    "add_bos_token": "add_bos_token",
    "add_eos_token": "add_eos_token",
    "bos_token_id": "bos_token_id",
    "eos_token_id": "eos_token_id",
    "unknown_token_id": "unknown_token_id",
    "padding_token_id": "padding_token_id",
    "cls_token_id": "cls_token_id",
    # llama.cpp writes the separator id under a misspelled key; accept both, the
    # correctly spelled one winning if a file somehow carries both.
    "seperator_token_id": "sep_token_id",
    "sep_token_id": "sep_token_id",
    "pre": "pre",
}


def read_metadata(path: Path) -> dict:
    """Read `tokenizer.ggml.*` out of a GGUF file into GgufVocab-shaped data."""
    reader = GGUFReader(str(path))
    raw = {}
    for name, field in reader.fields.items():
        if not name.startswith("tokenizer.ggml."):
            continue
        raw[name[len("tokenizer.ggml.") :]] = field.contents()

    if "tokens" not in raw:
        raise ValueError(f"{path}: no tokenizer.ggml.tokens — not a vocabulary file")

    vocab = {
        # llama.cpp treats an absent `model` key as the SentencePiece dialect.
        "model": raw.get("model", "llama"),
        "tokens": list(raw["tokens"]),
    }

    for key, field_name in OPTIONAL_KEYS.items():
        if key not in raw:
            continue
        value = raw[key]
        if value is None:
            continue
        if field_name in ("scores", "merges", "token_type"):
            value = list(value)
        elif field_name.endswith("_token_id"):
            value = int(value)
        elif field_name in (
            "add_space_prefix",
            "remove_extra_whitespaces",
            "add_bos_token",
            "add_eos_token",
        ):
            value = bool(value)
        vocab[field_name] = value

    return vocab


def read_cases(gguf_path: Path) -> list[dict]:
    """Parse the sibling `.inp`/`.out` fixtures, exactly as llama.cpp does.

    `.inp` is split on the delimiter; `.out` is one whitespace-stripped line of
    space-separated ids per case. An empty `.out` line is a legitimately empty
    expectation (the empty-string case).
    """
    inp_path = gguf_path.with_suffix(gguf_path.suffix + ".inp")
    out_path = gguf_path.with_suffix(gguf_path.suffix + ".out")
    if not inp_path.exists() or not out_path.exists():
        return []

    raw = inp_path.read_bytes()
    inputs: list[bytes] = []
    pos = 0
    while pos < len(raw):
        nxt = raw.find(CASE_SEPARATOR, pos)
        if nxt == -1:
            inputs.append(raw[pos:])
            break
        inputs.append(raw[pos:nxt])
        pos = nxt + len(CASE_SEPARATOR)

    expected = [
        [int(tok) for tok in line.strip().split()]
        for line in out_path.read_text(encoding="utf-8").splitlines()
    ]

    if len(inputs) != len(expected):
        raise ValueError(
            f"{gguf_path.name}: {len(inputs)} input cases but {len(expected)} "
            f"expected-output lines"
        )

    return [
        {"input": text.decode("utf-8"), "expected": ids}
        for text, ids in zip(inputs, expected)
    ]


def piece_to_bytes(piece: str) -> bytes:
    """Render one GGUF `model=llama` piece as the raw bytes it stands for.

    Two rules, and only two:
      * `<0xNN>` (byte fallback) is the single raw byte `0xNN`;
      * anything else is its own UTF-8 bytes, `▁` (U+2581) included verbatim.

    Raises for a piece that fits neither — notably a lone surrogate, which has no
    UTF-8 encoding. Such a piece is reported, never silently dropped.
    """
    match = BYTE_PIECE.match(piece)
    if match is not None:
        return bytes([int(match.group(1), 16)])
    try:
        return piece.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"piece {piece!r} is not encodable as UTF-8: {exc}") from exc


def write_tiktoken(vocab: dict, out_path: Path) -> str:
    """Write the tiktoken rendering and return a one-line sanity report.

    The report is the point: the conversion is only trustworthy if the line count
    matches the piece count and the duplicate byte sequences are the ones we
    expect (a `<0xNN>` byte-fallback piece colliding with a real single-character
    piece of the same bytes). Duplicates are NOT resolved here — both lines are
    written, in id order, and it is the Rust loader's documented policy that
    decides which id wins.
    """
    tokens: list[str] = vocab["tokens"]

    rendered: list[bytes] = []
    problems: list[str] = []
    for token_id, piece in enumerate(tokens):
        try:
            rendered.append(piece_to_bytes(piece))
        except ValueError as exc:
            problems.append(f"id {token_id}: {exc}")
            # Keep ids aligned with line numbers; the caller sees `problems`.
            rendered.append(piece.encode("utf-8", "surrogatepass"))

    lines = [
        f"{base64.b64encode(raw).decode('ascii')} {token_id}"
        for token_id, raw in enumerate(rendered)
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="ascii")

    seen: dict[bytes, list[int]] = {}
    for token_id, raw in enumerate(rendered):
        seen.setdefault(raw, []).append(token_id)
    dups = {raw: ids for raw, ids in seen.items() if len(ids) > 1}

    # Classify each collision: byte-fallback vs real piece is expected and
    # harmless (the loaders have a stated tie-break); anything else is not.
    unexpected = [
        (raw, ids)
        for raw, ids in dups.items()
        if not (
            len(ids) == 2
            and BYTE_PIECE.match(tokens[ids[0]])
            and not BYTE_PIECE.match(tokens[ids[1]])
        )
    ]

    report = (
        f"lines={len(lines)} pieces={len(tokens)} distinct={len(seen)} "
        f"dups={len(dups)} (byte-fallback collisions; unexpected={len(unexpected)})"
    )
    for raw, ids in unexpected[:5]:
        report += f"\n        UNEXPECTED DUP {raw!r} -> ids {ids}"
    for problem in problems:
        report += f"\n        PIECE PROBLEM {problem}"
    return report


def collect(paths: list[Path]) -> list[Path]:
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            found.extend(sorted(path.glob("*.gguf")))
        else:
            found.append(path)
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help=".gguf files or directories")
    parser.add_argument("--out-dir", type=Path, required=True, help="where to write the JSON")
    parser.add_argument(
        "--tiktoken-dir",
        type=Path,
        help="also write a `<name>.tiktoken` rendering of each vocabulary here "
        "(only meaningful for model=llama SentencePiece vocabularies)",
    )
    parser.add_argument(
        "--require-cases",
        action="store_true",
        help="skip vocabularies that ship no .inp/.out fixtures",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.tiktoken_dir is not None:
        args.tiktoken_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for gguf_path in collect(args.inputs):
        try:
            cases = read_cases(gguf_path)
            if args.require_cases and not cases:
                print(f"skip  {gguf_path.name}: no .inp/.out fixtures")
                continue
            vocab = read_metadata(gguf_path)
        except Exception as exc:  # tooling: report and keep going
            print(f"FAIL  {gguf_path.name}: {exc}", file=sys.stderr)
            failures += 1
            continue

        payload = {
            "name": gguf_path.stem,
            "source": str(gguf_path),
            "vocab": vocab,
            "cases": cases,
        }
        out_path = args.out_dir / f"{gguf_path.stem}.json"
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        print(
            f"ok    {out_path}  model={vocab['model']} "
            f"pre={vocab.get('pre', '-')} tokens={len(vocab['tokens'])} "
            f"cases={len(cases)}"
        )

        if args.tiktoken_dir is not None:
            tiktoken_path = args.tiktoken_dir / f"{gguf_path.stem}.tiktoken"
            try:
                report = write_tiktoken(vocab, tiktoken_path)
            except Exception as exc:  # tooling: report and keep going
                print(f"FAIL  {tiktoken_path}: {exc}", file=sys.stderr)
                failures += 1
                continue
            print(f"      {tiktoken_path}  {report}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
