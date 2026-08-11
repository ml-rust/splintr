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

# THE `pieces` AND `normalized` COLUMNS

For the same reason there are two more columns, both optional, pinning the
stage *between* the two ends: the reference's *pre-tokenizer split* of the
input, one string per piece, written as `"pieces"`, and the output of the
reference's own *normalization* stage, written as `"normalized"`. A
pre-tokenizer pattern -- or a normalizer pipeline -- that drifts from the one a
vocabulary was trained with is invisible in the id column until it happens to
move a token id. `tests/reference_parity.rs` compares `pieces` against
`AnyTokenizer::pre_tokenize` and `normalized` against `AnyTokenizer::normalize`,
and drives the former with the latter, which is what the two columns mean
together: `pieces` is always the split of `normalized`, never of `input`.

`normalized` is written only where it differs from `input`, so what "no
`normalized` key" means is "this reference's normalization stage is the
identity on this case", not "there is no such stage".

What each reference's normalization stage *is*:

  * `tokenizers` -- `normalizer.normalize_str(text)`, and nothing else: HF hangs
    `add_prefix_space` and the metaspace escaping off its pre-tokenizer nodes
    instead, so those show up in `pieces`. Every bundled `.tiktoken` vocabulary
    declares no normalizer at all, so in practice this key never appears in
    those fixtures.
  * `tiktoken` -- no normalization stage exists; the key never appears.
  * `sentencepiece` -- `SentencePieceProcessor.normalize(text)`: the space ->
    `▁` escaping *and* the `add_dummy_prefix` marker. This is the column that
    stands in for the pre-tokenizer split the format does not have (see below):
    it is the only stage between the input and the merge loop, so without it
    SentencePiece's whole front end is pinned by ids alone, which cannot say
    which stage moved them. It differs from `input` for essentially every
    non-empty case, which is the expected shape here, not a red flag.

Where the pieces come from, and what that does and does not prove:

  * `tiktoken` -- `regex.finditer(encoding._pat_str, text)`, reading the
    pattern **live off the installed encoding object** with the `regex`
    package, the same trust class (and the same `_`-prefixed attribute) as the
    exhaustive rank gate below. This proves splintr's `CL100K_BASE_PATTERN` /
    `O200K_BASE_PATTERN` constants have not drifted from the installed
    tiktoken's own pattern, and that splintr's regex engine executes that
    pattern identically to Python's `regex` module. It does **not**
    independently verify OpenAI's intent: both sides are then reading the same
    string from the same package.
  * `tokenizers` -- `pre_tokenizer.pre_tokenize_str(normalized)`, keeping the
    piece strings and dropping the offsets. These come back mapped into the
    ByteLevel alphabet (`Ġ` for a space), because every modern HF BPE pipeline
    ends in a `ByteLevel` stage; splintr's split yields raw text, so they are
    un-mapped back through the same GPT-2 table `decode_byte_level` already
    implements. Un-mapping is done here, in the generator, rather than in the
    Rust test, so the committed fixture reads as text and every fixture's
    `pieces` column means one thing.
  * `sentencepiece` -- nothing. SentencePiece has no pre-tokenizer split: the
    `▁` word-boundary marker lives in the vocabulary, and the whole normalized
    string goes to the merge loop. The `mistral`/`mistral_v2` fixtures
    therefore carry no `pieces` key at all. This is an omission on purpose, not
    a gap waiting to be filled -- and it is an omitted key rather than an empty
    list, which would falsely assert "splits into zero pieces". The stage those
    fixtures pin instead is `normalized`, above.

An individual case with an empty split (the empty-string corpus entry) omits
the key for that same reason.

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

Point 2 needs care because a bundled `.tiktoken` is not one
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
depending on which reference is asked for (`tiktoken` additionally needs
`regex`, which it already depends on).
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

from reference_corpus import REFERENCE_CORPUS

# The repository root, so `--vocab` never needs the caller to know where a
# vocabulary lives relative to their current directory.
REPO_ROOT = Path(__file__).resolve().parent.parent


def vocab_file(filename: str) -> Path:
    """Locate a bundled vocabulary by filename.

    Each family is its own published crate under `crates/vocab-*`, so there is
    no single directory to join onto any more. Globbing keeps callers naming
    the file and nothing else, which is what they knew before the split.
    """
    matches = sorted(REPO_ROOT.glob(f"crates/vocab-*/vocabs/{filename}"))
    if not matches:
        raise SystemExit(f"no bundled vocabulary named {filename}")
    return matches[0]

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
    "qwen": ("qwen3.tiktoken", False),
    "qwen2": ("qwen3.tiktoken", False),
    "qwen2.5": ("qwen3.tiktoken", False),
    "qwen3": ("qwen3.tiktoken", False),
    "baichuan_m2": ("qwen3.tiktoken", False),
    "glm": ("glm4.tiktoken", False),
    "glm4": ("glm4.tiktoken", False),
    "glm-4": ("glm4.tiktoken", False),
    "glm4.5": ("glm4.tiktoken", False),
    "glm-4.5": ("glm4.tiktoken", False),
    # gpt-oss is o200k_base's ranks under the harmony special tokens, so it is
    # gated against the o200k_base file it shares rather than one of its own.
    "kimi": ("kimi.tiktoken", False),
    "kimi_k2": ("kimi.tiktoken", False),
    "kimi_k3": ("kimi.tiktoken", False),
    "gpt-oss": ("o200k_base.tiktoken", False),
    "gpt_oss": ("o200k_base.tiktoken", False),
    "o200k_harmony": ("o200k_base.tiktoken", False),
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


def unmap_pieces(pieces: list[str], normalized: str) -> list[str]:
    """A reference's pre-tokenizer pieces as raw text, ByteLevel mapping undone.

    Which of the two spaces a piece list is in is not declared anywhere, so it
    is *measured*: a split partitions the text it was handed, so exactly one of
    the two readings reassembles into `normalized` (or into `normalized` with
    the pre-tokenizer's own `add_prefix_space` leading space). The un-mapped
    reading is tried first because for plain ASCII the two are identical and
    either answer is correct.

    Raises `ValueError` if neither reassembles -- a piece list that does not
    add back up to its input is not something to guess about.
    """
    try:
        unmapped: list[str] | None = [decode_byte_level(p).decode("utf-8") for p in pieces]
    except (ValueError, UnicodeDecodeError):
        unmapped = None

    for candidate in (unmapped, pieces):
        if candidate is not None and "".join(candidate) in (normalized, " " + normalized):
            return candidate

    raise ValueError(
        f"reference pre-tokenizer pieces {pieces[:8]!r} do not reassemble into "
        f"{normalized!r}, mapped or un-mapped -- refusing to record a `pieces` column "
        f"whose byte space cannot be established"
    )


def build_cases(encode, decode, split=None, normalize=None) -> list[dict[str, object]]:
    """Run `REFERENCE_CORPUS` through one reference's encode, decode and split.

    `decode` is handed the ids `encode` just produced, not the original text,
    so the `decoded` column is the reference's own round trip -- which is the
    thing the Rust side has to reproduce. Where a reference's decode is lossy
    (SentencePiece drops the dummy prefix and normalizes whitespace runs) the
    lossy result is what gets recorded: the fixture states what the reference
    does, never what it "should" do.

    `split`, when the reference has a pre-tokenizer at all, returns
    `(normalized-or-None, pieces)` for one input; `normalize` is the same
    `normalized` half on its own, for a reference that normalizes but does not
    split -- see the module docstring's "THE `pieces` AND `normalized` COLUMNS".
    A reference supplies at most one of the two, since the split's own
    normalized text is the one the pieces belong to. Every optional key is
    omitted rather than written empty: no `normalized` when the normalization
    stage changed nothing, no `pieces` when the split produced none.
    """
    cases: list[dict[str, object]] = []
    for text in REFERENCE_CORPUS:
        ids = [int(i) for i in encode(text)]
        case: dict[str, object] = {"input": text, "expected": ids, "decoded": decode(ids)}
        if normalize is not None:
            normalized = normalize(text)
            if normalized != text:
                case["normalized"] = normalized
        if split is not None:
            normalized, pieces = split(text)
            if normalized is not None:
                case["normalized"] = normalized
            if pieces:
                case["pieces"] = pieces
        cases.append(case)
    return cases


def hf_reference(reference_json: Path):
    """`(tokenizer, encode, decode, split)` for `tokenizers` over a `tokenizer.json`.

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

    def split(text: str) -> tuple[str | None, list[str]]:
        # Normalizer first, pre-tokenizer second -- HuggingFace's own order, and
        # the reason `normalized` is recorded: it, not `input`, is what the
        # pieces below are a split of.
        normalized = text
        if tokenizer.normalizer is not None:
            normalized = tokenizer.normalizer.normalize_str(text)
        if tokenizer.pre_tokenizer is None:
            return (normalized if normalized != text else None), []
        pieces = [piece for piece, _offsets in tokenizer.pre_tokenizer.pre_tokenize_str(normalized)]
        return (
            (normalized if normalized != text else None),
            unmap_pieces(pieces, normalized),
        )

    return tokenizer, encode, decode, split


def spm_reference(reference_model: Path):
    """`(processor, encode, decode, normalize)` for `sentencepiece` over a `tokenizer.model`.

    `add_bos`/`add_eos` are left off, the SentencePiece equivalent of
    `add_special_tokens=False`: `SpmTokenizer` on the Rust side emits neither
    from `encode_raw`, the policy places them.

    No `split` is returned, deliberately: SentencePiece has no pre-tokenizer
    stage to split with, so these fixtures carry no `pieces` column. What they
    carry instead is `normalize` -- `SentencePieceProcessor.normalize`, the
    space -> `▁` escaping plus the `add_dummy_prefix` marker, which is the only
    stage this format has between the input and the merge loop and therefore
    the one that stands in for the split it does not have. See the module
    docstring's "THE `pieces` AND `normalized` COLUMNS".
    """
    import sentencepiece

    processor = sentencepiece.SentencePieceProcessor(model_file=str(reference_model))

    def encode(text: str) -> list[int]:
        return processor.encode(text, out_type=int, add_bos=False, add_eos=False)

    def decode(ids: list[int]) -> str:
        return processor.decode(ids)

    def normalize(text: str) -> str:
        return processor.normalize(text)

    return processor, encode, decode, normalize


def tiktoken_reference(encoding_name: str):
    """`(encoding, encode, decode, split)` for the `tiktoken` package.

    `encode_ordinary` rather than `encode`: it is the entry point that treats
    special-token spellings as ordinary text, which is what an untemplated
    encode of a prose corpus means. `decode` replaces undecodable bytes with
    U+FFFD, matching `AnyTokenizer::decode_lossy`'s tail behaviour -- no case
    in `REFERENCE_CORPUS` reaches it, since every id sequence here came from
    encoding valid UTF-8.

    `split` runs the encoding's *own* pattern -- read off the object, not
    re-typed here -- through the `regex` package, which is the module tiktoken
    itself pre-tokenizes with. `finditer(...).group(0)` rather than `findall`
    so a pattern that ever grows a capturing group still yields whole matches
    instead of group tuples. tiktoken has no normalizer, so the first element
    is always `None`.
    """
    import regex
    import tiktoken

    encoding = tiktoken.get_encoding(encoding_name)

    def encode(text: str) -> list[int]:
        return encoding.encode_ordinary(text)

    def decode(ids: list[int]) -> str:
        return encoding.decode(ids)

    def split(text: str) -> tuple[str | None, list[str]]:
        pattern = encoding._pat_str  # noqa: SLF001 - no public accessor
        return None, [match.group(0) for match in regex.finditer(pattern, text)]

    return encoding, encode, decode, split


def run_hf(vocab: str, reference_json: Path) -> tuple[dict[str, object], list[dict[str, object]], str]:
    """Gate and generate for a `tokenizers` reference. Returns `(reference block, cases, note)`."""
    entry = TIKTOKEN_VOCABS.get(vocab)
    if entry is None:
        sys.exit(
            f"error: --reference-hf is for the .tiktoken-backed bundled vocabularies "
            f"({', '.join(sorted(TIKTOKEN_VOCABS))}); {vocab!r} is not one of them"
        )
    tiktoken_filename, byte_level = entry
    tiktoken_path = vocab_file(tiktoken_filename)
    if not tiktoken_path.exists():
        sys.exit(f"error: {tiktoken_path} does not exist -- is the repo layout intact?")
    if not reference_json.exists():
        sys.exit(f"error: {reference_json} does not exist")

    try:
        import tokenizers
    except ImportError:
        sys.exit("error: the 'tokenizers' package is required (pip install tokenizers)")

    tokenizer, encode, decode, split = hf_reference(reference_json)
    tiktoken_lines = read_tiktoken_lines(tiktoken_path)
    try:
        sanity_check_hf_pairing(vocab, tiktoken_lines, byte_level, tokenizer, reference_json)
    except ValueError as exc:
        sys.exit(f"error: {exc}")

    block: dict[str, object] = {
        "source": "tokenizers",
        "path": str(reference_json.resolve()),
        "tokenizers_version": tokenizers.__version__,
        "pieces": (
            "pre_tokenizer.pre_tokenize_str over the normalizer's output, un-mapped from "
            "the ByteLevel alphabet back to raw text"
        ),
    }
    try:
        cases = build_cases(encode, decode, split)
    except ValueError as exc:
        sys.exit(f"error: {exc}")
    return block, cases, f"byte_level={byte_level} tokens={len(tiktoken_lines)}"


def run_spm(vocab: str, reference_model: Path) -> tuple[dict[str, object], list[dict[str, object]], str]:
    """Gate and generate for a `sentencepiece` reference."""
    spm_filename = SPM_VOCABS.get(vocab)
    if spm_filename is None:
        sys.exit(
            f"error: --reference-spm is for the .spm-backed bundled vocabularies "
            f"({', '.join(sorted(SPM_VOCABS))}); {vocab!r} is not one of them"
        )
    spm_path = vocab_file(spm_filename)
    if not spm_path.exists():
        sys.exit(f"error: {spm_path} does not exist -- is the repo layout intact?")
    if not reference_model.exists():
        sys.exit(f"error: {reference_model} does not exist")

    try:
        import sentencepiece
    except ImportError:
        sys.exit("error: the 'sentencepiece' package is required (pip install sentencepiece)")

    processor, encode, decode, normalize = spm_reference(reference_model)
    spm_lines = read_spm_lines(spm_path)
    try:
        sanity_check_spm_pairing(vocab, spm_lines, processor)
    except ValueError as exc:
        sys.exit(f"error: {exc}")

    block: dict[str, object] = {
        "source": "sentencepiece",
        "path": str(reference_model.resolve()),
        "sentencepiece_version": sentencepiece.__version__,
        "normalized": (
            "SentencePieceProcessor.normalize -- the space -> U+2581 escaping and the "
            "add_dummy_prefix marker, the one stage this format has between the input and "
            "the merge loop, standing in for the pre-tokenizer split it does not have"
        ),
    }
    # No `split` argument, so no `pieces` column: SentencePiece has no
    # pre-tokenizer stage at all. See the module docstring.
    return block, build_cases(encode, decode, normalize=normalize), f"pieces={len(spm_lines)}"


# Kimi's pre-tokenizer, transcribed from the `pat_str` in the
# `tokenization_kimi.py` Moonshot ships beside the vocabulary. Kept here rather
# than read from a file because it is the thing under test: the fixture proves
# splintr's `KIMI_PATTERN` matches what Moonshot states, so a copy that could
# drift with splintr's would prove nothing.
KIMI_PAT_STR = "|".join(
    [
        r"[\p{Han}]+",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*"
        r"[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+"
        r"[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"\p{N}{1,3}",
        r" ?[^\s\p{L}\p{N}]+[\r\n]*",
        r"\s*[\r\n]+",
        r"\s+(?!\S)",
        r"\s+",
    ]
)


def run_tiktoken_ranks(
    vocab: str, ranks_path: Path
) -> tuple[dict[str, object], list[dict[str, object]], str]:
    """Gate and generate against a raw rank file plus a stated pattern.

    For a vocabulary published as tiktoken ranks with no `tokenizer.json` and no
    registered `tiktoken` encoding name — Kimi is the case, shipping
    `tiktoken.model` plus a `tokenization_kimi.py`. The reference is a
    `tiktoken.Encoding` built from those ranks and that pattern, which is exactly
    what Moonshot's own tokenizer constructs.
    """
    entry = TIKTOKEN_VOCABS.get(vocab)
    if entry is None:
        sys.exit(f"error: {vocab!r} is not a bundled .tiktoken vocabulary")
    bundled_filename, _byte_level = entry
    bundled_path = vocab_file(bundled_filename)
    if not bundled_path.exists():
        sys.exit(f"error: {bundled_path} does not exist -- is the repo layout intact?")
    if not ranks_path.is_file():
        sys.exit(f"error: no such rank file: {ranks_path}")

    try:
        import regex
        import tiktoken
        from tiktoken.load import load_tiktoken_bpe
    except ImportError as exc:
        sys.exit(f"error: the 'tiktoken' package is required ({exc})")

    # The gate: the bundled file must BE the reference's rank file, line for
    # line. Without this the fixture would pin splintr against a vocabulary it
    # does not ship.
    reference_lines = read_tiktoken_lines(ranks_path)
    bundled_lines = read_tiktoken_lines(bundled_path)
    if reference_lines != bundled_lines:
        first = next(
            (i for i, (a, b) in enumerate(zip(reference_lines, bundled_lines)) if a != b),
            min(len(reference_lines), len(bundled_lines)),
        )
        sys.exit(
            f"error: {bundled_path.name} is not {ranks_path.name}: "
            f"{len(bundled_lines)} vs {len(reference_lines)} lines, first differing rank {first}"
        )

    encoding = tiktoken.Encoding(
        name=vocab,
        pat_str=KIMI_PAT_STR,
        mergeable_ranks=load_tiktoken_bpe(str(ranks_path)),
        special_tokens={},
    )
    # `regex.V1`, not the default V0: `&&` is only set intersection in V1. Under
    # V0 the `&&` is read as literal `&` characters inside the class, so the
    # letter branches match the wrong thing and the split silently drops text —
    # `" a"` comes back as `[" "]`. The ids are unaffected (tiktoken pre-tokenizes
    # in Rust, which implements `&&`), so this shows up only in the `pieces`
    # column, which is exactly the drift that column exists to catch.
    compiled = regex.compile(KIMI_PAT_STR, regex.V1)

    def encode(text: str) -> list[int]:
        return encoding.encode_ordinary(text)

    def decode(ids: list[int]) -> str:
        return encoding.decode(ids)

    def split(text: str) -> tuple[None, list[str]]:
        return None, [m.group(0) for m in compiled.finditer(text)]

    block: dict[str, object] = {
        "source": "tiktoken",
        "encoding": f"{vocab} (ranks + pat_str from the model repo)",
        "tiktoken_version": tiktoken.__version__,
        "pieces": (
            "regex.finditer over the pat_str Moonshot's tokenization_kimi.py states, "
            "which is what proves splintr's KIMI_PATTERN has not drifted from it"
        ),
    }
    try:
        cases = build_cases(encode, decode, split)
    except ValueError as exc:
        sys.exit(f"error: {exc}")
    return block, cases, f"ranks={len(reference_lines)}"


def run_tiktoken(vocab: str, encoding_name: str) -> tuple[dict[str, object], list[dict[str, object]], str]:
    """Gate and generate for a `tiktoken` reference."""
    entry = TIKTOKEN_VOCABS.get(vocab)
    if entry is None or vocab not in TIKTOKEN_PACKAGE_VOCABS:
        sys.exit(
            f"error: --reference-tiktoken is for the OpenAI bundled vocabularies "
            f"({', '.join(sorted(TIKTOKEN_PACKAGE_VOCABS))}); {vocab!r} is not one of them"
        )
    tiktoken_filename, byte_level = entry
    tiktoken_path = vocab_file(tiktoken_filename)
    if not tiktoken_path.exists():
        sys.exit(f"error: {tiktoken_path} does not exist -- is the repo layout intact?")

    try:
        import tiktoken
    except ImportError:
        sys.exit("error: the 'tiktoken' package is required (pip install tiktoken)")

    try:
        encoding, encode, decode, split = tiktoken_reference(encoding_name)
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
        "pieces": (
            "regex.finditer over the installed encoding's own _pat_str -- proves splintr's "
            "pattern constant has not drifted from it and that splintr's regex engine runs "
            "that pattern like Python's `regex` module; it does not independently verify "
            "OpenAI's intent"
        ),
    }
    try:
        cases = build_cases(encode, decode, split)
    except ValueError as exc:
        sys.exit(f"error: {exc}")
    return block, cases, f"byte_level={byte_level} tokens={len(tiktoken_lines)}"


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
        "--reference-tiktoken-ranks",
        type=Path,
        help="path to a raw tiktoken rank file (e.g. Kimi's tiktoken.model) to "
        "build the reference encoding from, for vocabularies with no "
        "tokenizer.json and no registered tiktoken encoding name",
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

    if args.reference_tiktoken_ranks is not None:
        block, cases, note = run_tiktoken_ranks(args.vocab, args.reference_tiktoken_ranks)
    elif args.reference_hf is not None:
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
