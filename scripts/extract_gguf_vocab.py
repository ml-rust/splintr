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
dialect. The one binary field, `precompiled_charsmap`, is base64-encoded — the
same encoding a HuggingFace `tokenizer.json` uses for the identical bytes.

Recognised `tokenizer.ggml.*` keys that have no `GgufVocab` field (see
`UNMAPPED_KEYS`) are written to a separate top-level `unmapped_metadata` block,
never into `vocab`, so the "one JSON key per struct field" rule above stays
literally true.

Usage:
    python3 scripts/extract_gguf_vocab.py <file.gguf|dir> [...] --out-dir DIR

With `--tiktoken-dir DIR` the same vocabulary is ALSO rendered in the tiktoken
text format splintr's byte-level loader reads (`vocabs/*.tiktoken`):
one `base64(token_bytes) rank` line per token, in id order, rank == token id.
That rendering exists so the byte-level BPE path can be pointed at a GGUF
`model=llama` vocabulary and scored against the same llama.cpp fixtures as the
piece-level `SpmTokenizer` (see `examples/verify_spm_vs_bpe.rs`).

With `--reference-spm PATH` the `cases[]` are generated instead from a
SentencePiece reference model: a fixed, committed corpus of test strings (see
`REFERENCE_CORPUS` in `reference_corpus.py`) is run through
`sentencepiece.SentencePieceProcessor(
PATH).encode(text)` and the resulting ids become the expected output, in the
exact same `{"input": ..., "expected": [...]}` shape the `.inp`/`.out` path
produces — `examples/verify_gguf.rs` needs no changes to consume either kind.
Before writing anything, the SP model and the GGUF vocabulary are sanity-gated
against each other (piece counts and a sample of piece strings must match
exactly); a mismatch aborts with an error rather than silently emitting a
fixture paired with the wrong tokenizer. The JSON also records a `reference`
block naming the SP model path and installed `sentencepiece` version, so an
SPM-sourced fixture can never be mistaken for an llama.cpp-sourced one.

`--reference-hf PATH` is the same idea for vocabularies where the GGUF ids do
NOT align with the raw SentencePiece model -- notably XLM-RoBERTa-family
(`model=t5`/Unigram) vocabularies, which offset every SentencePiece id by +1
and add 2 specials (`<s>`/`</s>`) on top. There, `--reference-spm` against the
raw `.model` file is refused by the sanity gate (piece-count mismatch, by
design). `--reference-hf` instead loads `tokenizers.Tokenizer.from_file(PATH)`
(a `tokenizer.json`) and runs the same `REFERENCE_CORPUS` through
`tokenizer.encode(text, add_special_tokens=False).ids` -- `add_special_tokens`
is False to match `encode_raw` on the Rust side, which is what these fixtures
are diffed against; `True` would wrap every case in CLS/SEP ids the Rust
harness never adds. It produces the identical `cases[]` shape, is sanity-gated
the same way (vocab size, then a sample of ids compared via `id_to_token`
against the GGUF `tokens` array), and records its own `reference` block
(`source: "tokenizers"`) so it can't be confused with an SPM- or
llama.cpp-sourced fixture. `--reference-spm` and `--reference-hf` are mutually
exclusive.

Requires the `gguf` package (`pip install gguf`); `--reference-spm` further
requires the `sentencepiece` package (`pip install sentencepiece`);
`--reference-hf` requires the `tokenizers` package (`pip install tokenizers`).
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

from reference_corpus import REFERENCE_CORPUS

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
    # SentencePiece's normalization table. GGUF stores it as a UINT8 array of a
    # quarter-megabyte or so; the JSON carries it base64-encoded, the same
    # encoding a HuggingFace `tokenizer.json` uses for the identical bytes, so a
    # fixture stays a fraction of the size a number array would be.
    "precompiled_charsmap": "precompiled_charsmap",
}

# `tokenizer.ggml.*` keys that are recorded but have NO `GgufVocab` field,
# because nothing in splintr's tokenization reads them:
#
#   * `mask_token_id` — the id of `<mask>`, used by masked-LM training heads.
#     Tokenization never inserts or strips it: it is neither a boundary token
#     (that is bos/eos) nor something matched in input text (the vocabulary's
#     own CONTROL entry already covers `<mask>` as an added token, by string).
#   * `token_type_count` — the number of *segment* embeddings the model has
#     (BERT's sentence A / B), a model-architecture number. Note it is unrelated
#     to `tokenizer.ggml.token_type`, which is the per-id token-kind enum and
#     IS read.
#
# They are dumped so a fixture is a faithful record of the file and so the
# question "does splintr ignore something?" is answerable from the JSON rather
# than by re-reading the GGUF. They live outside the `vocab` block precisely
# because that block mirrors `GgufVocab` one key per field.
UNMAPPED_KEYS = ("mask_token_id", "token_type_count")


def read_metadata(path: Path) -> tuple[dict, dict]:
    """Read `tokenizer.ggml.*` out of a GGUF file.

    Returns `(vocab, unmapped)`: the GgufVocab-shaped data, and the recognised
    keys that deliberately have no struct field (see `UNMAPPED_KEYS`).
    """
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
        elif field_name == "precompiled_charsmap":
            value = base64.b64encode(bytes(value)).decode("ascii")
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

    unmapped = {
        key: int(raw[key]) for key in UNMAPPED_KEYS if raw.get(key) is not None
    }

    return vocab, unmapped


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


# The fixed test corpus for the `--reference-spm`/`--reference-hf` paths
# lives in `reference_corpus.py`, imported above and shared verbatim with
# `extract_reference_cases.py` (see that module's docstring for why and
# what it covers).

# How many (evenly spaced) ids the sanity gate below cross-checks between the
# GGUF vocabulary and the SentencePiece reference model. Not exhaustive by
# design -- an exhaustive compare is just `tokens == [sp.id_to_piece(i) ...]`,
# which is cheap enough, but sampling keeps the check's cost describable and
# its failure output short; a genuinely mismatched pairing disagrees on
# nearly every id, so a sample of a few hundred catches it just as reliably.
SANITY_SAMPLE_SIZE = 256


def sanity_check_spm_pairing(vocab: dict, sp) -> None:
    """Verify `sp` is the tokenizer that actually produced this GGUF vocabulary.

    Pairing a GGUF file with the wrong reference `.model` would silently
    produce a `cases[]` fixture that looks authoritative but diffs splintr
    against nonsense. Two checks, both required:

      1. `sp.get_piece_size()` equals the GGUF's token count exactly.
      2. A sample of ids resolve to byte-identical piece strings in both.

    Raises `ValueError` (never returns a partial/soft result) on any mismatch.
    """
    tokens: list[str] = vocab["tokens"]
    sp_size = sp.get_piece_size()
    if sp_size != len(tokens):
        raise ValueError(
            f"reference SPM has {sp_size} pieces but GGUF vocabulary has "
            f"{len(tokens)} tokens -- refusing to pair a reference tokenizer "
            f"with a mismatched vocabulary"
        )

    n = len(tokens)
    if n <= SANITY_SAMPLE_SIZE:
        sample_ids = range(n)
    else:
        step = n / SANITY_SAMPLE_SIZE
        sample_ids = {int(i * step) for i in range(SANITY_SAMPLE_SIZE)}
        sample_ids.add(n - 1)

    mismatches = []
    for token_id in sample_ids:
        sp_piece = sp.id_to_piece(token_id)
        gguf_piece = tokens[token_id]
        if sp_piece != gguf_piece:
            mismatches.append((token_id, sp_piece, gguf_piece))

    if mismatches:
        detail = "\n".join(
            f"    id {tid}: sp={sp_piece!r} gguf={gguf_piece!r}"
            for tid, sp_piece, gguf_piece in mismatches[:10]
        )
        raise ValueError(
            f"reference SPM and GGUF vocabulary disagree on "
            f"{len(mismatches)}/{len(sample_ids)} sampled piece(s) -- refusing "
            f"to pair a reference tokenizer with a mismatched vocabulary:\n{detail}"
        )


def generate_cases_from_spm(vocab: dict, spm_path: Path) -> tuple[list[dict], dict]:
    """Run `REFERENCE_CORPUS` through the SentencePiece reference model.

    Returns `(cases, reference_meta)`: `cases` is exactly the `.inp`/`.out`
    shape (`{"input": ..., "expected": [id, ...]}`), and `reference_meta` is
    the `reference` block recorded in the JSON so this fixture can never be
    mistaken for one sourced from llama.cpp's `.inp`/`.out` files.

    Raises `ValueError` if `sanity_check_spm_pairing` rejects the pairing --
    the caller must not write a fixture in that case.
    """
    try:
        import sentencepiece as spm
    except ImportError:  # pragma: no cover - tooling script
        raise ValueError(
            "the 'sentencepiece' package is required for --reference-spm "
            "(pip install sentencepiece)"
        ) from None

    sp = spm.SentencePieceProcessor()
    sp.load(str(spm_path))

    sanity_check_spm_pairing(vocab, sp)

    cases = [
        {"input": text, "expected": [int(i) for i in sp.encode(text)]}
        for text in REFERENCE_CORPUS
    ]
    reference_meta = {
        "source": "sentencepiece",
        "model_path": str(spm_path),
        "sentencepiece_version": spm.__version__,
    }
    return cases, reference_meta


# `tokenizer.ggml.token_type` enum value for a placeholder/unused slot (the
# `gguf` package's own `gguf.TokenType.UNUSED`). llama.cpp's GGUF converter
# gives these ids a synthetic name (observed: `[PAD250000]`) instead of the
# original piece text, so an HF reference tokenizer's real piece for that id
# (observed: the XLM-R `<mask>` token, offset onto an id llama.cpp otherwise
# treats as unused padding) legitimately disagrees on the *string* while still
# being the same *id* -- this is not evidence of a mismatched pairing.
GGUF_TOKEN_TYPE_UNUSED = 5


def sanity_check_hf_pairing(vocab: dict, tokenizer) -> None:
    """Verify `tokenizer` is the one that actually produced this GGUF vocabulary.

    Same two checks as `sanity_check_spm_pairing`, against the HF
    `tokenizers.Tokenizer` API instead of `sentencepiece`:

      1. `tokenizer.get_vocab_size()` equals the GGUF's token count exactly.
      2. A sample of ids resolve to byte-identical token strings in both,
         via `tokenizer.id_to_token(id)` against the GGUF `tokens` array --
         except ids the GGUF itself marks `UNUSED` (see
         `GGUF_TOKEN_TYPE_UNUSED`), where llama.cpp is known to substitute a
         synthetic placeholder string instead of preserving the original.

    Raises `ValueError` (never returns a partial/soft result) on any mismatch.
    """
    tokens: list[str] = vocab["tokens"]
    hf_size = tokenizer.get_vocab_size()
    if hf_size != len(tokens):
        raise ValueError(
            f"reference HF tokenizer has {hf_size} tokens but GGUF vocabulary "
            f"has {len(tokens)} tokens -- refusing to pair a reference "
            f"tokenizer with a mismatched vocabulary"
        )

    token_type: list[int] | None = vocab.get("token_type")

    n = len(tokens)
    if n <= SANITY_SAMPLE_SIZE:
        sample_ids = range(n)
    else:
        step = n / SANITY_SAMPLE_SIZE
        sample_ids = {int(i * step) for i in range(SANITY_SAMPLE_SIZE)}
        sample_ids.add(n - 1)

    mismatches = []
    for token_id in sample_ids:
        if token_type is not None and token_type[token_id] == GGUF_TOKEN_TYPE_UNUSED:
            continue
        hf_token = tokenizer.id_to_token(token_id)
        gguf_piece = tokens[token_id]
        if hf_token != gguf_piece:
            mismatches.append((token_id, hf_token, gguf_piece))

    if mismatches:
        detail = "\n".join(
            f"    id {tid}: hf={hf_token!r} gguf={gguf_piece!r}"
            for tid, hf_token, gguf_piece in mismatches[:10]
        )
        raise ValueError(
            f"reference HF tokenizer and GGUF vocabulary disagree on "
            f"{len(mismatches)}/{len(sample_ids)} sampled token(s) -- refusing "
            f"to pair a reference tokenizer with a mismatched vocabulary:\n{detail}"
        )


def generate_cases_from_hf(vocab: dict, hf_path: Path) -> tuple[list[dict], dict]:
    """Run `REFERENCE_CORPUS` through the HF `tokenizers` reference tokenizer.

    Returns `(cases, reference_meta)`: `cases` is exactly the `.inp`/`.out`
    shape (`{"input": ..., "expected": [id, ...]}`), and `reference_meta` is
    the `reference` block recorded in the JSON so this fixture can never be
    mistaken for one sourced from llama.cpp's `.inp`/`.out` files or from
    `--reference-spm`.

    Encodes with `add_special_tokens=False` to match `encode_raw` on the Rust
    side (no BOS/EOS/CLS/SEP wrapping) -- confirmed against the bge-m3
    tokenizer.json: `encode("hello world", add_special_tokens=False).ids ==
    [33600, 31, 8999]`, matching the GGUF model's own ids exactly, while
    `add_special_tokens=True` wraps that in `[0, ..., 2]`.

    Raises `ValueError` if `sanity_check_hf_pairing` rejects the pairing --
    the caller must not write a fixture in that case.
    """
    try:
        import tokenizers
        from tokenizers import Tokenizer
    except ImportError:  # pragma: no cover - tooling script
        raise ValueError(
            "the 'tokenizers' package is required for --reference-hf "
            "(pip install tokenizers)"
        ) from None

    tokenizer = Tokenizer.from_file(str(hf_path))

    sanity_check_hf_pairing(vocab, tokenizer)

    cases = [
        {
            "input": text,
            "expected": [
                int(i)
                for i in tokenizer.encode(text, add_special_tokens=False).ids
            ],
        }
        for text in REFERENCE_CORPUS
    ]
    reference_meta = {
        "source": "tokenizers",
        "model_path": str(hf_path),
        "tokenizers_version": tokenizers.__version__,
    }
    return cases, reference_meta


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
    parser.add_argument(
        "--reference-spm",
        type=Path,
        help="generate cases[] from this SentencePiece .model file run over "
        "REFERENCE_CORPUS, instead of sibling .inp/.out files (requires the "
        "'sentencepiece' package). Applies to every input in this invocation, "
        "so pass one .gguf file per run when its reference tokenizer differs.",
    )
    parser.add_argument(
        "--reference-hf",
        type=Path,
        help="generate cases[] from this HF tokenizer.json run over "
        "REFERENCE_CORPUS, instead of sibling .inp/.out files (requires the "
        "'tokenizers' package). For vocabularies (e.g. XLM-RoBERTa/model=t5) "
        "whose GGUF ids do not align with the raw SentencePiece model, so "
        "--reference-spm cannot be used. Mutually exclusive with "
        "--reference-spm. Applies to every input in this invocation, so pass "
        "one .gguf file per run when its reference tokenizer differs.",
    )
    args = parser.parse_args()

    if args.reference_spm is not None and args.reference_hf is not None:
        parser.error("--reference-spm and --reference-hf are mutually exclusive")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.tiktoken_dir is not None:
        args.tiktoken_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for gguf_path in collect(args.inputs):
        reference_meta = None
        try:
            vocab, unmapped = read_metadata(gguf_path)
            if args.reference_spm is not None:
                cases, reference_meta = generate_cases_from_spm(
                    vocab, args.reference_spm
                )
            elif args.reference_hf is not None:
                cases, reference_meta = generate_cases_from_hf(
                    vocab, args.reference_hf
                )
            else:
                cases = read_cases(gguf_path)
            if args.require_cases and not cases:
                print(f"skip  {gguf_path.name}: no .inp/.out fixtures")
                continue
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
        if unmapped:
            payload["unmapped_metadata"] = unmapped
        if reference_meta is not None:
            payload["reference"] = reference_meta
        out_path = args.out_dir / f"{gguf_path.stem}.json"
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        print(
            f"ok    {out_path}  model={vocab['model']} "
            f"pre={vocab.get('pre', '-')} tokens={len(vocab['tokens'])} "
            f"cases={len(cases)} charsmap="
            + (
                str(len(base64.b64decode(vocab["precompiled_charsmap"])))
                if "precompiled_charsmap" in vocab
                else "-"
            )
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
