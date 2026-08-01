#!/usr/bin/env python3
"""Generate reference fixtures for splintr's *bundled* pretrained vocabularies.

`examples/verify_gguf.rs` diffs splintr against llama.cpp's own `.inp`/`.out`
fixtures (or, via `extract_gguf_vocab.py --reference-spm/--reference-hf`, a
SentencePiece/HuggingFace reference tokenizer) for GGUF-loaded vocabularies.
The vocabularies in `splintr::pretrained` (`cl100k_base`, `o200k_base`,
`llama3`, `deepseek_v3`, `mistral_v3`, `whisper`, ...) are compiled straight
into the crate instead, so there is no GGUF file to attach a reference to —
the only independent ground truth is whatever HuggingFace `tokenizer.json` a
model repo publishes for that same vocabulary. This script produces the
fixture `examples/verify_pretrained.rs` diffs against: `REFERENCE_CORPUS`
(shared with `extract_gguf_vocab.py`, see `reference_corpus.py`) run through
`tokenizers.Tokenizer.from_file(path).encode(text,
add_special_tokens=False).ids` -- `add_special_tokens=False` to match
`AnyTokenizer::encode_raw` on the Rust side, which is what these fixtures are
diffed against; `True` would wrap every case in BOS/EOS ids the Rust harness
never adds on its own.

Usage:
    python scripts/extract_reference_cases.py \\
        --vocab deepseek_v3 \\
        --reference-hf /path/to/tokenizer.json \\
        --out-dir /tmp/pretrained

# SANITY GATE

Pairing a bundled vocabulary with the wrong HF `tokenizer.json` would
silently produce a fixture that looks authoritative but diffs splintr
against nonsense, so before writing anything this script verifies `PATH` is
actually the tokenizer that produced `python/splintr/vocabs/<vocab>.tiktoken`
-- the exact file `splintr::pretrained::from_pretrained` embeds
(`src/core/pretrained.rs`):

  1. The `.tiktoken` file's line count must equal
     `tokenizer.get_vocab_size(with_added_tokens=False)` exactly.
  2. A sample of ~256 evenly spaced ids must resolve to matching pieces.

Point 2 needs care because `python/splintr/vocabs/*.tiktoken` is not one
format: whether a bundled vocabulary loads through
`Tokenizer::from_bytes_chain` (raw semantic bytes per line -- `cl100k_base`,
`o200k_base`, `llama3`) or `Tokenizer::from_bytes_byte_level_chain`
(GPT-2/HuggingFace ByteLevel encoding -- `deepseek_v3`, `mistral_v3`,
`whisper`) changes what a `.tiktoken` line actually contains (see
`splintr::pretrained::uses_byte_level`, `src/core/pretrained.rs`, and the
loaders in `src/core/tokenizer.rs`). Verified against the two vocabularies
whose reference tokenizers are available locally:

  * `llama3.tiktoken` line 220 decodes to the raw byte `b' '` (0x20), while
    the matching Llama-3.2 `tokenizer.json`'s `id_to_token(220)` is `'Ġ'` --
    HuggingFace's ByteLevel pre-tokenizer's printable stand-in for that same
    byte. Comparing raw file bytes to the HF piece therefore requires
    decoding the HF piece through the standard GPT-2 byte-to-unicode table
    first (`decode_byte_level` below) -- this is the `byte_level=False`
    column in `TIKTOKEN_VOCABS`, naming what the *loader* does, not what the
    HF reference looks like (every modern HF BPE `tokenizer.json` declares a
    ByteLevel pre-tokenizer regardless).
  * `deepseek_v3.tiktoken` line 300 decodes to the four raw bytes
    `b'\\xc3\\xa2\\xc4\\xa2'`, and the DeepSeek-V3 `tokenizer.json`'s
    `id_to_token(300)` is `'âĢ'` -- `'âĢ'.encode("utf-8") ==
    b'\\xc3\\xa2\\xc4\\xa2'` exactly. `extract_deepseek_vocab.py` writes this
    family's `.tiktoken` files by UTF-8-encoding the *already
    ByteLevel-encoded* piece string, not by decoding it back to the piece's
    original bytes, so comparing here means UTF-8-decoding the file bytes
    and comparing the two ByteLevel strings directly -- no GPT-2 table
    involved. This is the `byte_level=True` column.

A char in an HF piece with no entry in the GPT-2 byte-to-unicode table, or a
file line that is not valid UTF-8 where one is required, counts as a
mismatch rather than being skipped -- the gate must never pass by silently
ignoring what it cannot resolve.

Any mismatch raises and nothing is written. The output JSON's `reference`
block records the HF path and installed `tokenizers` version so a fixture
can never be mistaken for one sourced any other way.

Requires the `tokenizers` package (`pip install tokenizers`).
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

from reference_corpus import REFERENCE_CORPUS

# The repository root, so `--vocab` never needs the caller to know where
# `python/splintr/vocabs/` lives relative to their current directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
VOCABS_DIR = REPO_ROOT / "python" / "splintr" / "vocabs"

# Bundled vocabulary name (every alias `PretrainedVocab::from_name` accepts,
# `src/core/pretrained.rs`) -> (`.tiktoken` filename, byte_level).
#
# `byte_level` says which loader `splintr::pretrained::from_vocab` uses for
# this vocabulary -- `Tokenizer::from_bytes_byte_level_chain` (True) vs
# `Tokenizer::from_bytes_chain` (False) -- which is what fixes the sanity
# gate's comparison strategy (see the module docstring). It is hand-verified
# per vocabulary above, not `splintr::pretrained::uses_byte_level`, because
# that Rust accessor is a policy helper for a different purpose (query
# pattern selection) and is not necessarily kept in sync with every loader
# arm in `from_vocab` (e.g. `mistral_v3` also loads through the byte-level
# path but is absent from that accessor's match).
#
# Mistral V1/V2 are intentionally absent: they are bundled as SentencePiece
# `.spm` files (piece + score), not `.tiktoken`, so this script's sanity gate
# -- which is specifically a `.tiktoken` line comparison -- does not apply to
# them.
TIKTOKEN_VOCABS: dict[str, tuple[str, bool]] = {
    "cl100k_base": ("cl100k_base.tiktoken", False),
    "o200k_base": ("o200k_base.tiktoken", False),
    "llama3": ("llama3.tiktoken", False),
    "llama3.1": ("llama3.tiktoken", False),
    "llama3.2": ("llama3.tiktoken", False),
    "llama3.3": ("llama3.tiktoken", False),
    "deepseek_v3": ("deepseek_v3.tiktoken", True),
    "deepseek-v3": ("deepseek_v3.tiktoken", True),
    "mistral_v3": ("mistral_v3_tekken.tiktoken", True),
    "whisper": ("whisper.tiktoken", True),
    "whisper_v1": ("whisper.tiktoken", True),
    "whisper-v1": ("whisper.tiktoken", True),
    "whisper-multilingual-v1": ("whisper.tiktoken", True),
    "whisper_v2": ("whisper.tiktoken", True),
    "whisper-v2": ("whisper.tiktoken", True),
    "whisper-multilingual": ("whisper.tiktoken", True),
    "whisper_v3": ("whisper.tiktoken", True),
    "whisper-v3": ("whisper.tiktoken", True),
    "whisper-large-v3": ("whisper.tiktoken", True),
}

# Same sampling budget as `extract_gguf_vocab.py`'s SANITY_SAMPLE_SIZE, for
# the same reason: not exhaustive by design, sampling keeps the check's cost
# and failure output describable, and a genuinely mismatched pairing
# disagrees on nearly every id so a few hundred samples catch it reliably.
SANITY_SAMPLE_SIZE = 256


def bytes_to_unicode() -> dict[int, str]:
    """The standard GPT-2 byte -> printable-unicode-character mapping.

    Verbatim port of the well-known `bytes_to_unicode()` from OpenAI's GPT-2
    `encoder.py` (also what HuggingFace's ByteLevel pre-tokenizer implements):
    printable ASCII/Latin-1 map to themselves, every other byte value is
    shifted into the range starting at U+0100. Splintr's own Rust mirror of
    this table is `BYTE_TO_CHAR` in `src/core/byte_level.rs`.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


BYTE_DECODER: dict[str, int] = {v: k for k, v in bytes_to_unicode().items()}


def decode_byte_level(piece: str) -> bytes:
    """Map a GPT-2/HuggingFace ByteLevel-encoded piece back to its raw bytes.

    Raises `ValueError` (never returns a partial result) if any character in
    `piece` has no entry in the standard byte-to-unicode table -- such a
    character cannot legitimately appear in a ByteLevel piece, so treating it
    as "just drop it" would let a garbled comparison pass silently.
    """
    raw = bytearray()
    for ch in piece:
        byte = BYTE_DECODER.get(ch)
        if byte is None:
            raise ValueError(f"{ch!r} has no GPT-2 byte-level mapping")
        raw.append(byte)
    return bytes(raw)


def read_tiktoken_lines(path: Path) -> list[bytes]:
    """Parse a `.tiktoken` file into its raw per-line byte payloads, in id order.

    Format: `base64(payload) rank`, one per line, rank == line index -- the
    same format `scripts/extract_gguf_vocab.py`'s `write_tiktoken` produces
    and `Tokenizer::from_bytes`/`from_bytes_byte_level` (`src/core/vocab.rs`)
    reads.
    """
    lines: list[bytes] = []
    with path.open("r", encoding="ascii") as handle:
        for line_no, line in enumerate(handle):
            stripped = line.strip()
            if not stripped:
                continue
            b64, _, rank_text = stripped.partition(" ")
            try:
                rank = int(rank_text)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no + 1}: bad rank {rank_text!r}") from exc
            if rank != len(lines):
                raise ValueError(
                    f"{path}:{line_no + 1}: rank {rank} out of order (expected {len(lines)})"
                )
            lines.append(base64.b64decode(b64))
    return lines


def sanity_check_hf_pairing(
    vocab_name: str, tiktoken_lines: list[bytes], byte_level: bool, tokenizer
) -> None:
    """Verify `tokenizer` is the one that actually produced this bundled vocabulary.

    See the module docstring's "SANITY GATE" section for what each check
    means and why the comparison differs by `byte_level`. Raises `ValueError`
    (never returns a partial/soft result) on any mismatch.
    """
    hf_size = tokenizer.get_vocab_size(with_added_tokens=False)
    if hf_size != len(tiktoken_lines):
        raise ValueError(
            f"reference HF tokenizer has {hf_size} tokens (with_added_tokens=False) "
            f"but splintr's bundled {vocab_name!r} vocabulary has {len(tiktoken_lines)} "
            f"tokens -- refusing to pair a reference tokenizer with a mismatched vocabulary"
        )

    n = len(tiktoken_lines)
    if n <= SANITY_SAMPLE_SIZE:
        sample_ids = list(range(n))
    else:
        step = n / SANITY_SAMPLE_SIZE
        sample_ids = sorted({int(i * step) for i in range(SANITY_SAMPLE_SIZE)} | {n - 1})

    mismatches: list[str] = []
    for token_id in sample_ids:
        hf_piece = tokenizer.id_to_token(token_id)
        file_bytes = tiktoken_lines[token_id]
        if hf_piece is None:
            mismatches.append(f"    id {token_id}: HF tokenizer has no piece for this id")
            continue
        try:
            if byte_level:
                # The file stores the UTF-8 bytes of the ByteLevel string
                # itself (see the module docstring's deepseek_v3 example).
                ok = file_bytes.decode("utf-8") == hf_piece
            else:
                # The file stores the piece's raw semantic bytes; the HF
                # piece must be decoded through the GPT-2 table first (see
                # the module docstring's llama3 example).
                ok = decode_byte_level(hf_piece) == file_bytes
        except (UnicodeDecodeError, ValueError) as exc:
            mismatches.append(f"    id {token_id}: unresolvable mapping: {exc}")
            continue
        if not ok:
            mismatches.append(f"    id {token_id}: hf={hf_piece!r} file={file_bytes!r}")

    if mismatches:
        detail = "\n".join(mismatches[:10])
        raise ValueError(
            f"reference HF tokenizer and splintr's bundled {vocab_name!r} vocabulary "
            f"disagree on {len(mismatches)}/{len(sample_ids)} sampled token(s) -- "
            f"refusing to pair a reference tokenizer with a mismatched vocabulary:\n{detail}"
        )


def generate_cases(tokenizer) -> list[dict[str, object]]:
    """Run `REFERENCE_CORPUS` through the HF reference tokenizer.

    `add_special_tokens=False` matches `AnyTokenizer::encode_raw` on the Rust
    side, which is what `examples/verify_pretrained.rs` diffs these cases
    against -- `True` would wrap every case in BOS/EOS ids the raw backend
    output never carries.
    """
    return [
        {
            "input": text,
            "expected": [int(i) for i in tokenizer.encode(text, add_special_tokens=False).ids],
        }
        for text in REFERENCE_CORPUS
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--vocab",
        required=True,
        help="a splintr bundled pretrained vocabulary name accepted by "
        "PretrainedVocab::from_name, e.g. deepseek_v3, llama3, cl100k_base",
    )
    parser.add_argument(
        "--reference-hf",
        required=True,
        type=Path,
        help="path to the HF tokenizer.json believed to have produced this "
        "bundled vocabulary",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="where to write the fixture JSON")
    args = parser.parse_args()

    entry = TIKTOKEN_VOCABS.get(args.vocab)
    if entry is None:
        parser.error(
            f"--vocab {args.vocab!r} is not supported by this script. Supported: "
            f"{', '.join(sorted(TIKTOKEN_VOCABS))} (SentencePiece-backed vocabularies "
            f"-- mistral, mistral_v1, mistral_v2 -- have no .tiktoken file to sanity-"
            f"gate against and are not supported here)"
        )
    tiktoken_filename, byte_level = entry
    tiktoken_path = VOCABS_DIR / tiktoken_filename
    if not tiktoken_path.exists():
        sys.exit(f"error: {tiktoken_path} does not exist -- is the repo layout intact?")

    try:
        from tokenizers import Tokenizer
        import tokenizers
    except ImportError:
        sys.exit("error: the 'tokenizers' package is required (pip install tokenizers)")

    if not args.reference_hf.exists():
        sys.exit(f"error: {args.reference_hf} does not exist")
    tokenizer = Tokenizer.from_file(str(args.reference_hf))

    tiktoken_lines = read_tiktoken_lines(tiktoken_path)

    try:
        sanity_check_hf_pairing(args.vocab, tiktoken_lines, byte_level, tokenizer)
    except ValueError as exc:
        sys.exit(f"error: {exc}")

    cases = generate_cases(tokenizer)

    payload = {
        "vocab": args.vocab,
        "reference": {
            "source": "tokenizers",
            "path": str(args.reference_hf.resolve()),
            "tokenizers_version": tokenizers.__version__,
        },
        "cases": cases,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.vocab}.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)

    print(
        f"ok    {out_path}  vocab={args.vocab} byte_level={byte_level} "
        f"tokens={len(tiktoken_lines)} cases={len(cases)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
