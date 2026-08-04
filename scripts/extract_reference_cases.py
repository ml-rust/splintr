#!/usr/bin/env python3
"""Generate reference fixtures for splintr's *bundled* pretrained vocabularies.

`examples/verify_gguf.rs` diffs splintr against llama.cpp's own `.inp`/`.out`
fixtures (or, via `extract_gguf_vocab.py --reference-spm/--reference-hf`, a
SentencePiece/HuggingFace reference tokenizer) for GGUF-loaded vocabularies.
The vocabularies in `splintr::pretrained` (`cl100k_base`, `o200k_base`,
`llama3`, `deepseek_v3`, `mistral_v3`, `whisper`, ...) are compiled straight
into the crate instead, so there is no GGUF file to attach a reference to --
the independent ground truth is whichever reference *implementation* actually
defines that vocabulary. That is not one tool, so this script speaks three:

  * `--reference-tiktoken` -- the `tiktoken` package, for the two OpenAI
    vocabularies it is the reference for (`cl100k_base`, `o200k_base`). No
    model repo is involved; the encoding name alone identifies the vocabulary.
  * `--reference-hf PATH` -- the `tokenizers` package over a model repo's
    published `tokenizer.json` (`llama3`, `deepseek_v3`, `whisper`).
  * `--reference-spm PATH` -- the `sentencepiece` package over a model repo's
    `tokenizer.model`, for the vocabularies splintr bundles as `.spm`
    (`mistral`/`mistral_v1`, `mistral_v2`). These have no `.tiktoken` form at
    all, so an HF pairing cannot gate them.

Whichever reference answers, the fixture is the same shape: `REFERENCE_CORPUS`
(shared with `extract_gguf_vocab.py`, see `reference_corpus.py`) run through
the reference's *untemplated* encode -- matching `AnyTokenizer::encode_raw` on
the Rust side, which is what these fixtures are diffed against; a templated
encode would wrap every case in BOS/EOS ids the Rust harness never adds on its
own -- plus that reference's own decode of the ids it just produced.

The decode column is not decoration. Encode and decode are separate pipelines
(byte-level unmapping, byte fallback, the SentencePiece dummy-prefix strip,
`special = true` skipping), and a fixture that pins only ids leaves every one
of them unpinned -- which is where a large share of this crate's real
divergences have been. `tests/reference_parity.rs` asserts both columns.

Usage:
    python scripts/extract_reference_cases.py \\
        --vocab deepseek_v3 \\
        --reference-hf /path/to/tokenizer.json \\
        --out-dir tests/fixtures/pretrained
    python scripts/extract_reference_cases.py \\
        --vocab cl100k_base --reference-tiktoken \\
        --out-dir tests/fixtures/pretrained
    python scripts/extract_reference_cases.py \\
        --vocab mistral_v2 \\
        --reference-spm /path/to/mistral-7b-v0.3/tokenizer.model \\
        --out-dir tests/fixtures/pretrained

# SANITY GATE

Pairing a bundled vocabulary with the wrong reference would silently produce a
fixture that looks authoritative but diffs splintr against nonsense, so before
writing anything this script verifies the reference really is the one that
produced the file `splintr::pretrained::from_pretrained` embeds
(`src/core/pretrained.rs`). What "verifies" means depends on the reference:

  * `tiktoken` and `sentencepiece` expose the whole vocabulary as data, so the
    gate is **exhaustive** -- every id's bytes (tiktoken) or piece and score
    (sentencepiece) must match the bundled file, and the counts must be equal.
  * `tokenizers` is checked by count plus a ~256-id sample, as below.

## The HuggingFace sample gate

  1. The `.tiktoken` file's line count must equal
     `tokenizer.get_vocab_size(with_added_tokens=False)`, or fall short of it
     only by ids the reference itself declares as added/special tokens --
     which is the Whisper case: `whisper-tiny/tokenizer.json` reports 50258
     because HF also lists `<|endoftext|>` (id 50257) inside `model.vocab`,
     while the bundled `whisper.tiktoken` stops at 50256 and splintr generates
     the specials from 50257 up (`src/core/whisper.rs`). An id in the excess
     range that is *not* an added token is a hard mismatch.
  2. A sample of ~256 evenly spaced ids must resolve to matching pieces.

Whisper is also the case that shows why check 1 alone is not enough. The
*English-only* checkpoints (`whisper-tiny.en`) publish a 50,257-token
`tokenizer.json` -- the same count as the bundled multilingual vocabulary --
so a count-only gate would accept the pairing. It is a different base BPE
(GPT-2's, verbatim), and check 2 rejects it on 255 of 257 sampled ids. The
bundled `whisper.tiktoken` is the *multilingual* vocabulary and pairs with
`whisper-tiny/tokenizer.json`, on which it agrees at every one of its 50,257
ids. `src/core/pretrained.rs` says the same thing in prose: the English-only
checkpoints "use a different base BPE and are not bundled".

Point 2 needs care because `vocabs/*.tiktoken` is not one
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
block records which tool answered, what it was pointed at and that tool's
installed version, so a fixture can never be mistaken for one sourced any
other way.

# VOCABULARIES WITH NO REFERENCE HERE

`mistral_v3` (Tekken) is bundled but has no entry below: its reference is a
Mistral NeMo / Large 2 / Pixtral `tekken.json`, and pointing this script at
any *other* Mistral repo's `tokenizer.json` pairs the 131,072-token Tekken
vocabulary with a 32k SentencePiece one, which the gate rejects. Fetch a
Tekken checkpoint and pass its converted `tokenizer.json` via
`--reference-hf` to produce that fixture; do not substitute a near neighbour.
No other wiring is needed -- `mistral_v3` is already in `TIKTOKEN_VOCABS`
below, so the sanity gate and the fixture both run the moment that file
exists. `scripts/verify_external_models.py` carries the same gap as a
`bjson` target that reports MISSING until one appears; the two are the reason
`mistral_v3`'s `special_decode_ids` arm in `src/core/pretrained.rs` is an
inference from the Mistral family rather than a measurement.

Requires the `tokenizers`, `sentencepiece` and/or `tiktoken` package
depending on which reference is asked for.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

from reference_corpus import REFERENCE_CORPUS

# The repository root, so `--vocab` never needs the caller to know where
# `vocabs/` lives relative to their current directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
VOCABS_DIR = REPO_ROOT / "vocabs"

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
# `.spm` files (piece + score), not `.tiktoken`, and are listed in `SPM_VOCABS`
# instead -- the `.tiktoken` line comparison below cannot gate them.
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

# Bundled vocabulary name -> the `.spm` file `splintr::pretrained` embeds for
# it. These load through `SpmTokenizer` (pieces merged by score), never through
# a `.tiktoken` reader, so they are gated against `sentencepiece` directly --
# piece *and* score, exhaustively, because the `.spm` format was introduced
# precisely because the `.tiktoken` form destroyed the scores.
SPM_VOCABS: dict[str, str] = {
    "mistral": "mistral.spm",
    "mistral_v1": "mistral.spm",
    "mistral_v2": "mistral_v2.spm",
}

# Bundled vocabulary name -> the `tiktoken` encoding name that defines it.
# These two are OpenAI's own vocabularies, so `tiktoken` is not a third-party
# reimplementation of them, it *is* the reference -- and it exposes the whole
# mergeable-rank table, so the gate compares every id rather than a sample.
TIKTOKEN_PACKAGE_VOCABS: dict[str, str] = {
    "cl100k_base": "cl100k_base",
    "o200k_base": "o200k_base",
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
        for line_no, raw_line in enumerate(handle):
            # Only strip the trailing newline, not arbitrary whitespace: the
            # separator is the LAST space on the line (matching
            # `load_tiktoken_bpe`'s `rposition` in `src/core/vocab.rs`), and a
            # base64 payload can legitimately be empty -- e.g. the whisper
            # vocab's rank-50256 line `" 50256"` encodes the empty byte
            # string. `.strip()` followed by `partition(" ")` would eat that
            # leading space and make the payload indistinguishable from a
            # missing one.
            line = raw_line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            b64, sep, rank_text = line.rpartition(" ")
            if not sep:
                raise ValueError(f"{path}:{line_no + 1}: missing space separator")
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


def read_spm_lines(path: Path) -> list[tuple[str, float]]:
    """Parse a `.spm` file into its `(piece, score)` pairs, in id order.

    Format: `base64(piece as UTF-8) score`, one per line, id == line index --
    the same format `scripts/extract_spm_vocab.py` writes and
    `load_spm_vocab` (`src/core/vocab.rs`) reads. Unlike `.tiktoken` there is
    no rank column to cross-check the ordering against, so a truncated or
    reordered file is caught by the exhaustive piece comparison instead.
    """
    entries: list[tuple[str, float]] = []
    with path.open("r", encoding="ascii") as handle:
        for line_no, raw_line in enumerate(handle):
            line = raw_line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            b64, sep, score_text = line.rpartition(" ")
            if not sep:
                raise ValueError(f"{path}:{line_no + 1}: missing space separator")
            try:
                score = float(score_text)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no + 1}: bad score {score_text!r}") from exc
            entries.append((base64.b64decode(b64).decode("utf-8"), score))
    return entries


def sanity_check_spm_pairing(
    vocab_name: str, spm_lines: list[tuple[str, float]], processor
) -> None:
    """Verify `processor` is the SentencePiece model that produced this `.spm`.

    Exhaustive, not sampled: `sentencepiece` hands over the whole vocabulary
    as data, so there is no reason to settle for a sample. Both columns are
    compared -- the scores are the half the `.spm` format exists to preserve
    (see `scripts/extract_spm_vocab.py`), and a reference agreeing on pieces
    while disagreeing on scores is a genuinely different tokenizer.
    """
    ref_size = processor.get_piece_size()
    if ref_size != len(spm_lines):
        raise ValueError(
            f"reference SentencePiece model has {ref_size} pieces but splintr's "
            f"bundled {vocab_name!r} vocabulary has {len(spm_lines)} -- refusing to "
            f"pair a reference tokenizer with a mismatched vocabulary"
        )

    mismatches: list[str] = []
    for token_id, (piece, score) in enumerate(spm_lines):
        ref_piece = processor.id_to_piece(token_id)
        ref_score = processor.get_score(token_id)
        if piece != ref_piece:
            mismatches.append(f"    id {token_id}: ref={ref_piece!r} file={piece!r}")
        elif score != ref_score:
            mismatches.append(
                f"    id {token_id}: piece {piece!r} score ref={ref_score!r} file={score!r}"
            )

    if mismatches:
        detail = "\n".join(mismatches[:10])
        raise ValueError(
            f"reference SentencePiece model and splintr's bundled {vocab_name!r} "
            f"vocabulary disagree on {len(mismatches)}/{len(spm_lines)} piece(s) -- "
            f"refusing to pair a reference tokenizer with a mismatched vocabulary:\n{detail}"
        )


def sanity_check_tiktoken_pairing(
    vocab_name: str, tiktoken_lines: list[bytes], encoding
) -> None:
    """Verify `encoding` is the tiktoken encoding that produced this `.tiktoken`.

    Exhaustive for the same reason as the SentencePiece gate: the whole
    mergeable-rank table is available as data. `_mergeable_ranks` has no public
    accessor, which is why it is reached into here -- the alternative is
    re-deriving ranks from `encode_single_token`, one Python call per id.
    """
    ranks = encoding._mergeable_ranks  # noqa: SLF001 - no public accessor
    if len(ranks) != len(tiktoken_lines):
        raise ValueError(
            f"reference tiktoken encoding has {len(ranks)} mergeable ranks but "
            f"splintr's bundled {vocab_name!r} vocabulary has {len(tiktoken_lines)} "
            f"tokens -- refusing to pair a reference tokenizer with a mismatched vocabulary"
        )

    by_rank = {rank: token for token, rank in ranks.items()}
    mismatches: list[str] = []
    for token_id, file_bytes in enumerate(tiktoken_lines):
        ref_bytes = by_rank.get(token_id)
        if ref_bytes != file_bytes:
            mismatches.append(f"    id {token_id}: ref={ref_bytes!r} file={file_bytes!r}")

    if mismatches:
        detail = "\n".join(mismatches[:10])
        raise ValueError(
            f"reference tiktoken encoding and splintr's bundled {vocab_name!r} "
            f"vocabulary disagree on {len(mismatches)}/{len(tiktoken_lines)} token(s) -- "
            f"refusing to pair a reference tokenizer with a mismatched vocabulary:\n{detail}"
        )


def added_token_ids(reference_json: Path) -> set[int]:
    """The ids the reference `tokenizer.json` itself declares as added tokens.

    Read from the file rather than from the `Tokenizer` object because the
    question is what the *file* declares: HF lists an added token in
    `added_tokens` and, for some repos, a second time inside `model.vocab`,
    and it is exactly that second listing the size gate has to forgive.
    """
    with reference_json.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return {
        entry["id"]
        for entry in raw.get("added_tokens", [])
        if isinstance(entry, dict) and isinstance(entry.get("id"), int)
    }


def sanity_check_hf_pairing(
    vocab_name: str,
    tiktoken_lines: list[bytes],
    byte_level: bool,
    tokenizer,
    reference_json: Path,
) -> None:
    """Verify `tokenizer` is the one that actually produced this bundled vocabulary.

    See the module docstring's "SANITY GATE" section for what each check
    means and why the comparison differs by `byte_level`. Raises `ValueError`
    (never returns a partial/soft result) on any mismatch.
    """
    hf_size = tokenizer.get_vocab_size(with_added_tokens=False)
    if hf_size < len(tiktoken_lines):
        raise ValueError(
            f"reference HF tokenizer has {hf_size} tokens (with_added_tokens=False) "
            f"but splintr's bundled {vocab_name!r} vocabulary has {len(tiktoken_lines)} "
            f"tokens -- refusing to pair a reference tokenizer with a mismatched vocabulary"
        )
    if hf_size > len(tiktoken_lines):
        # The bundled file may legitimately stop short of the reference's model
        # vocabulary when the reference also lists its special tokens there and
        # splintr generates those separately (the Whisper case, see the module
        # docstring). Anything else in that range is a real size mismatch.
        added = added_token_ids(reference_json)
        excess = [
            token_id for token_id in range(len(tiktoken_lines), hf_size) if token_id not in added
        ]
        if excess:
            raise ValueError(
                f"reference HF tokenizer has {hf_size} tokens (with_added_tokens=False) "
                f"but splintr's bundled {vocab_name!r} vocabulary has {len(tiktoken_lines)} "
                f"tokens, and id(s) {excess[:10]} in the excess range are not declared "
                f"added tokens -- refusing to pair a reference tokenizer with a "
                f"mismatched vocabulary"
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


def build_cases(encode, decode) -> list[dict[str, object]]:
    """Run `REFERENCE_CORPUS` through one reference's encode and decode.

    `decode` is handed the ids `encode` just produced, not the original text,
    so the `decoded` column is the reference's own round trip -- which is the
    thing the Rust side has to reproduce. Where a reference's decode is lossy
    (SentencePiece drops the dummy prefix and normalizes whitespace runs) the
    lossy result is what gets recorded: the fixture states what the reference
    does, never what it "should" do.
    """
    cases: list[dict[str, object]] = []
    for text in REFERENCE_CORPUS:
        ids = [int(i) for i in encode(text)]
        cases.append({"input": text, "expected": ids, "decoded": decode(ids)})
    return cases


def hf_reference(reference_json: Path):
    """`(encode, decode)` for the `tokenizers` package over a `tokenizer.json`.

    `add_special_tokens=False` matches `AnyTokenizer::encode_raw` on the Rust
    side, which is what `examples/verify_pretrained.rs` diffs these cases
    against -- `True` would wrap every case in BOS/EOS ids the raw backend
    output never carries. `skip_special_tokens=False` on the decode side is
    stated rather than left to the default because these id sequences contain
    no specials at all: making the flag explicit says the column is the plain
    surface round trip, not a filtered one.
    """
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(reference_json))

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False).ids

    def decode(ids: list[int]) -> str:
        return tokenizer.decode(ids, skip_special_tokens=False)

    return tokenizer, encode, decode


def spm_reference(reference_model: Path):
    """`(encode, decode)` for the `sentencepiece` package over a `tokenizer.model`.

    `add_bos`/`add_eos` are left off, the SentencePiece equivalent of
    `add_special_tokens=False`: `SpmTokenizer` on the Rust side emits neither
    from `encode_raw`, the policy places them.
    """
    import sentencepiece

    processor = sentencepiece.SentencePieceProcessor(model_file=str(reference_model))

    def encode(text: str) -> list[int]:
        return processor.encode(text, out_type=int, add_bos=False, add_eos=False)

    def decode(ids: list[int]) -> str:
        return processor.decode(ids)

    return processor, encode, decode


def tiktoken_reference(encoding_name: str):
    """`(encode, decode)` for the `tiktoken` package.

    `encode_ordinary` rather than `encode`: it is the entry point that treats
    special-token spellings as ordinary text, which is what an untemplated
    encode of a prose corpus means. `decode` replaces undecodable bytes with
    U+FFFD, matching `AnyTokenizer::decode_lossy`'s tail behaviour -- no case
    in `REFERENCE_CORPUS` reaches it, since every id sequence here came from
    encoding valid UTF-8.
    """
    import tiktoken

    encoding = tiktoken.get_encoding(encoding_name)

    def encode(text: str) -> list[int]:
        return encoding.encode_ordinary(text)

    def decode(ids: list[int]) -> str:
        return encoding.decode(ids)

    return encoding, encode, decode


def run_hf(vocab: str, reference_json: Path) -> tuple[dict[str, object], list[dict[str, object]], str]:
    """Gate and generate for a `tokenizers` reference. Returns `(reference block, cases, note)`."""
    entry = TIKTOKEN_VOCABS.get(vocab)
    if entry is None:
        sys.exit(
            f"error: --reference-hf is for the .tiktoken-backed bundled vocabularies "
            f"({', '.join(sorted(TIKTOKEN_VOCABS))}); {vocab!r} is not one of them"
        )
    tiktoken_filename, byte_level = entry
    tiktoken_path = VOCABS_DIR / tiktoken_filename
    if not tiktoken_path.exists():
        sys.exit(f"error: {tiktoken_path} does not exist -- is the repo layout intact?")
    if not reference_json.exists():
        sys.exit(f"error: {reference_json} does not exist")

    try:
        import tokenizers
    except ImportError:
        sys.exit("error: the 'tokenizers' package is required (pip install tokenizers)")

    tokenizer, encode, decode = hf_reference(reference_json)
    tiktoken_lines = read_tiktoken_lines(tiktoken_path)
    try:
        sanity_check_hf_pairing(vocab, tiktoken_lines, byte_level, tokenizer, reference_json)
    except ValueError as exc:
        sys.exit(f"error: {exc}")

    block: dict[str, object] = {
        "source": "tokenizers",
        "path": str(reference_json.resolve()),
        "tokenizers_version": tokenizers.__version__,
    }
    return block, build_cases(encode, decode), f"byte_level={byte_level} tokens={len(tiktoken_lines)}"


def run_spm(vocab: str, reference_model: Path) -> tuple[dict[str, object], list[dict[str, object]], str]:
    """Gate and generate for a `sentencepiece` reference."""
    spm_filename = SPM_VOCABS.get(vocab)
    if spm_filename is None:
        sys.exit(
            f"error: --reference-spm is for the .spm-backed bundled vocabularies "
            f"({', '.join(sorted(SPM_VOCABS))}); {vocab!r} is not one of them"
        )
    spm_path = VOCABS_DIR / spm_filename
    if not spm_path.exists():
        sys.exit(f"error: {spm_path} does not exist -- is the repo layout intact?")
    if not reference_model.exists():
        sys.exit(f"error: {reference_model} does not exist")

    try:
        import sentencepiece
    except ImportError:
        sys.exit("error: the 'sentencepiece' package is required (pip install sentencepiece)")

    processor, encode, decode = spm_reference(reference_model)
    spm_lines = read_spm_lines(spm_path)
    try:
        sanity_check_spm_pairing(vocab, spm_lines, processor)
    except ValueError as exc:
        sys.exit(f"error: {exc}")

    block: dict[str, object] = {
        "source": "sentencepiece",
        "path": str(reference_model.resolve()),
        "sentencepiece_version": sentencepiece.__version__,
    }
    return block, build_cases(encode, decode), f"pieces={len(spm_lines)}"


def run_tiktoken(vocab: str, encoding_name: str) -> tuple[dict[str, object], list[dict[str, object]], str]:
    """Gate and generate for a `tiktoken` reference."""
    entry = TIKTOKEN_VOCABS.get(vocab)
    if entry is None or vocab not in TIKTOKEN_PACKAGE_VOCABS:
        sys.exit(
            f"error: --reference-tiktoken is for the OpenAI bundled vocabularies "
            f"({', '.join(sorted(TIKTOKEN_PACKAGE_VOCABS))}); {vocab!r} is not one of them"
        )
    tiktoken_filename, byte_level = entry
    tiktoken_path = VOCABS_DIR / tiktoken_filename
    if not tiktoken_path.exists():
        sys.exit(f"error: {tiktoken_path} does not exist -- is the repo layout intact?")

    try:
        import tiktoken
    except ImportError:
        sys.exit("error: the 'tiktoken' package is required (pip install tiktoken)")

    try:
        encoding, encode, decode = tiktoken_reference(encoding_name)
    except Exception as exc:  # noqa: BLE001 - unknown encoding name, or no cached vocabulary
        sys.exit(f"error: tiktoken could not load encoding {encoding_name!r}: {exc}")

    tiktoken_lines = read_tiktoken_lines(tiktoken_path)
    try:
        sanity_check_tiktoken_pairing(vocab, tiktoken_lines, encoding)
    except ValueError as exc:
        sys.exit(f"error: {exc}")

    block: dict[str, object] = {
        "source": "tiktoken",
        "encoding": encoding_name,
        "tiktoken_version": tiktoken.__version__,
    }
    return block, build_cases(encode, decode), f"byte_level={byte_level} tokens={len(tiktoken_lines)}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--vocab",
        required=True,
        help="a splintr bundled pretrained vocabulary name accepted by "
        "PretrainedVocab::from_name, e.g. deepseek_v3, llama3, cl100k_base",
    )
    # Exactly one reference, never inferred from the vocabulary name: which
    # tool is authoritative for a vocabulary is a fact about that vocabulary's
    # origin, and making the caller state it is what keeps a wrong pairing an
    # explicit mistake rather than a silent default.
    reference = parser.add_mutually_exclusive_group(required=True)
    reference.add_argument(
        "--reference-hf",
        type=Path,
        help="path to the HF tokenizer.json believed to have produced this "
        "bundled vocabulary",
    )
    reference.add_argument(
        "--reference-spm",
        type=Path,
        help="path to the SentencePiece tokenizer.model believed to have "
        "produced this bundled .spm vocabulary",
    )
    reference.add_argument(
        "--reference-tiktoken",
        nargs="?",
        const="",
        help="use the tiktoken package as the reference; the encoding name "
        "defaults to --vocab",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="where to write the fixture JSON")
    args = parser.parse_args()

    if args.reference_hf is not None:
        block, cases, note = run_hf(args.vocab, args.reference_hf)
    elif args.reference_spm is not None:
        block, cases, note = run_spm(args.vocab, args.reference_spm)
    else:
        block, cases, note = run_tiktoken(args.vocab, args.reference_tiktoken or args.vocab)

    payload = {
        "vocab": args.vocab,
        "reference": block,
        "cases": cases,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.vocab}.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)

    print(f"ok    {out_path}  vocab={args.vocab} reference={block['source']} {note} cases={len(cases)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
