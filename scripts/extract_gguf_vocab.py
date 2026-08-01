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

Requires the `gguf` package (`pip install gguf`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
        "--require-cases",
        action="store_true",
        help="skip vocabularies that ship no .inp/.out fixtures",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

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

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
