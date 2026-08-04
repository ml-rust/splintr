#!/usr/bin/env python3
"""Encode + decode parity against the real model tokenizers on this machine.

WHY THIS EXISTS
---------------
`tests/reference_parity.rs` pins splintr against reference tokenizers *in CI*,
but only for the vocabularies the crate bundles, and only through fixtures
captured ahead of time -- because a published `tokenizer.json` is megabytes and
cannot be vendored. That leaves the broadest correctness evidence this project
has (splintr's own `from_json` loader run against a shelf of real,
independently-published model tokenizers) reproducible only by whoever happens
to have those models on disk.

It was previously reproducible only in a scratch file that got deleted after
each release, which meant the claim "verified against N real models" rested on
someone's word. This script is that check, committed: point it at a directory
of model repos and it prints one row per target with the exact number of
encode and decode cases that agreed.

It is deliberately NOT a CI test. It needs multi-gigabyte model directories and
three reference packages. `tests/reference_parity.rs` (committed fixtures) and
`tests/decode_agreement.rs` (internal streaming consistency) are the CI-side
proxies; this is the wider net a maintainer runs before a release.

RELATION TO THE OTHER SCRIPTS
-----------------------------
* `scripts/extract_reference_cases.py` captures a *bundled* vocabulary's
  reference ids and decoded text into a committed fixture. Run it when a
  bundled vocabulary changes.
* `scripts/fuzz_reference.py` searches for *new* divergences with random
  strings built from each vocabulary's own special tokens. Run it to find bugs.
* This script is the fixed, exhaustive-over-the-shelf regression sweep: same
  corpus for every target, one line of output each. Run it to answer "does
  splintr still agree with every real tokenizer I have?".

WHAT IS COMPARED
----------------
Two target kinds, never mixed, because the entry-point pairing differs:

`json`  -- splintr's own loader against the same file's reference:
    encode  ref.encode(t, add_special_tokens=False).ids == sp.encode_raw(t)
    decode  ref.decode(ids)                             == sp.decode(ids)
  `add_special_tokens=False` pairs with `encode_raw`, the untemplated form;
  the templated pair is `fuzz_reference.py`'s job, which exercises it against
  random inputs rather than a fixed corpus.

`spm`   -- a *bundled* splintr vocabulary against the SentencePiece
           `tokenizer.model` that defines it:
    encode  ref.encode(t, add_bos=False, add_eos=False) == sp.encode_raw(t)
    decode  ref.decode(ids)                             == sp.decode(ids)

The corpus is `scripts/reference_corpus.py`'s `REFERENCE_CORPUS`, shared with
every other reference extractor here so a divergence found by one is
addressable by the others.

FAILURE BEHAVIOUR
-----------------
Nothing here is ever skipped quietly. A missing `--models-dir` aborts with a
message naming the path. A target whose files are absent is reported `MISSING`
and makes the run exit non-zero, because a sweep that silently shrinks to the
models that happen to be present is exactly the reassuring-but-empty result
this script replaces. Use `--only` to run a deliberate subset.

The run also aborts if the installed `splintr` wheel is not this checkout's
version: a stale wheel fills in every row and prints real divergences of a
build nobody is working on, which is the most convincing wrong answer this
script can give. Rebuild with `maturin develop --release --features python`.

USAGE
-----
    # everything on the shelf
    python3 scripts/verify_external_models.py --models-dir ~/Projects/models

    # one target, e.g. while chasing a single divergence
    python3 scripts/verify_external_models.py --models-dir ~/Projects/models \\
        --only bge-m3 --verbose

Requires `splintr` (`maturin develop --release --features python`), plus
`tokenizers` and `sentencepiece`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from reference_corpus import REFERENCE_CORPUS

# The repository root, so the installed `splintr` wheel can be checked against
# the version this checkout actually is.
REPO_ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------
# The shelf.
#
# `(name, kind, path relative to --models-dir, vocab)`:
#
#   kind == "json" -- splintr's `from_json` loader is the subject and the same
#     `tokenizer.json` read by `tokenizers` is the reference; `vocab` is unused.
#   kind == "spm"  -- a bundled splintr vocabulary named by `vocab` is the
#     subject and the SentencePiece `tokenizer.model` at `path` is the
#     reference.
#
# A bundled vocabulary whose reference is a `tokenizer.json` (llama3,
# deepseek_v3, whisper) is covered on the `json` side by the model directory it
# came from *and* by a committed fixture in `tests/fixtures/pretrained/`; the
# fixture is the CI-side check, this is the live one.
# --------------------------------------------------------------------------
TARGETS: tuple[tuple[str, str, str, str], ...] = (
    # HuggingFace `tokenizer.json` files, one per tokenizer family splintr's
    # `from_json` loader has to handle: WordPiece, Unigram, byte-level BPE,
    # SentencePiece-derived BPE, and the two Whisper base vocabularies.
    ("all-MiniLM-L6-v2", "json", "all-MiniLM-L6-v2-tokenizer/tokenizer.json", ""),
    ("bge-m3", "json", "bge-m3-tokenizer/tokenizer.json", ""),
    ("embeddinggemma-300m", "json", "embeddinggemma-300m-tokenizer/tokenizer.json", ""),
    ("deepseek-v3", "json", "deepseek-v3-tokenizer/tokenizer.json", ""),
    ("llama-3.2-1b", "json", "llama-3.2-1b/tokenizer.json", ""),
    ("mistral-7b-awq-int4", "json", "mistral-7b-awq-int4/tokenizer.json", ""),
    ("mistral-7b-gptq-int4", "json", "mistral-7b-gptq-int4/tokenizer.json", ""),
    ("mistral-7b-v0.3", "json", "mistral-7b-v0.3/tokenizer.json", ""),
    ("whisper-tiny", "json", "whisper-tiny/tokenizer.json", ""),
    ("whisper-tiny.en", "json", "whisper-tiny.en/tokenizer.json", ""),
    # Bundled SentencePiece vocabularies against the `tokenizer.model` that
    # defines each. `mistral-7b-awq-int4` publishes the V1 32,000-piece model
    # and `mistral-7b-v0.3` the V2 32,768-piece one; both pairings are gated
    # piece-by-piece by `scripts/extract_reference_cases.py --reference-spm`.
    ("mistral (bundled V1)", "spm", "mistral-7b-awq-int4/tokenizer.model", "mistral"),
    ("mistral_v2 (bundled V2)", "spm", "mistral-7b-v0.3/tokenizer.model", "mistral_v2"),
)


class TargetMissing(Exception):
    """A target's files are not on this machine. Reported, never skipped."""


def compare(name: str, expected: object, actual: object) -> str | None:
    """A one-line description of a disagreement, or `None` when they agree."""
    if expected == actual:
        return None
    return f"{name}: expected {expected!r}, got {actual!r}"


def check_json(path: Path) -> tuple[int, int, list[str]]:
    """splintr's `from_json` loader against `tokenizers` reading the same file."""
    from tokenizers import Tokenizer as HfTokenizer
    import splintr

    if not path.is_file():
        raise TargetMissing(f"no such file: {path}")

    reference = HfTokenizer.from_file(str(path))
    # Padding and truncation are serving settings some repos bake into their
    # `tokenizer.json` (all-MiniLM-L6-v2 pads every sequence to 128). They are
    # not tokenization, splintr's `encode_raw` has no equivalent, and leaving
    # them on manufactures a divergence on literally every case -- so they are
    # turned off rather than being allowed to look like a bug.
    reference.no_padding()
    reference.no_truncation()
    subject = splintr.from_json(str(path))

    encode_ok = 0
    decode_ok = 0
    failures: list[str] = []
    for text in REFERENCE_CORPUS:
        ids = reference.encode(text, add_special_tokens=False).ids
        got_ids = subject.encode_raw(text)
        problem = compare(f"encode {text!r}", list(ids), list(got_ids))
        if problem is None:
            encode_ok += 1
        else:
            failures.append(problem)

        # `skip_special_tokens` defaults to True, which is the semantics
        # splintr's `decode` implements: HuggingFace drops `[UNK]`/`[CLS]`/…
        # here, and asking for False would compare against a mode splintr does
        # not claim to offer.
        expected_text = reference.decode(ids)
        got_text = subject.decode(list(ids))
        problem = compare(f"decode {text!r}", expected_text, got_text)
        if problem is None:
            decode_ok += 1
        else:
            failures.append(problem)

    return encode_ok, decode_ok, failures


def check_spm(path: Path, vocab: str) -> tuple[int, int, list[str]]:
    """A bundled splintr vocabulary against its SentencePiece `tokenizer.model`."""
    import sentencepiece
    import splintr

    if not path.is_file():
        raise TargetMissing(f"no such file: {path}")

    reference = sentencepiece.SentencePieceProcessor(model_file=str(path))
    subject = splintr.Tokenizer.from_pretrained(vocab)

    encode_ok = 0
    decode_ok = 0
    failures: list[str] = []
    for text in REFERENCE_CORPUS:
        ids = reference.encode(text, out_type=int, add_bos=False, add_eos=False)
        got_ids = subject.encode_raw(text)
        problem = compare(f"encode {text!r}", list(ids), list(got_ids))
        if problem is None:
            encode_ok += 1
        else:
            failures.append(problem)

        expected_text = reference.decode(ids)
        got_text = subject.decode(list(ids))
        problem = compare(f"decode {text!r}", expected_text, got_text)
        if problem is None:
            decode_ok += 1
        else:
            failures.append(problem)

    return encode_ok, decode_ok, failures


def require_packages() -> None:
    """Abort naming every reference package that is missing, not just the first."""
    missing: list[str] = []
    for module, install in (
        ("splintr", "maturin develop --release --features python"),
        ("tokenizers", "pip install tokenizers"),
        ("sentencepiece", "pip install sentencepiece"),
    ):
        try:
            __import__(module)
        except ImportError:
            missing.append(f"  {module}  ({install})")
    if missing:
        sys.exit(
            "error: this harness compares splintr against real reference "
            "implementations and cannot report anything meaningful without "
            "them. Missing:\n" + "\n".join(missing)
        )


def require_current_build() -> None:
    """Abort if the `splintr` that will actually be exercised is not this checkout.

    An out-of-date build is the one failure mode that produces a *convincing*
    wrong answer here: every row still fills in, and the divergences it prints
    are real divergences -- of a build nobody is working on. There is no
    override flag on purpose; the fix is one command, and a run whose subject
    is unknown is worth less than no run.

    The check reads the version off the **imported module**, not off installed
    distribution metadata: `importlib.metadata` reports whatever wheel is
    registered in site-packages, which is not necessarily what `import splintr`
    resolved to. Running from the source tree (`PYTHONPATH=python`, the workflow
    used when there is no virtualenv for maturin) legitimately imports a module
    that no distribution owns, and metadata would fail that run while happily
    passing a genuinely stale one.
    """
    import splintr

    version_file = REPO_ROOT / ".version"
    if not version_file.is_file():
        sys.exit(f"error: {version_file} does not exist -- is the repo layout intact?")
    expected = version_file.read_text(encoding="utf-8").strip()

    imported = getattr(splintr, "__version__", None)
    origin = getattr(splintr, "__file__", "<unknown>")
    if imported is None:
        sys.exit(
            f"error: the `splintr` imported from {origin} exposes no __version__, so "
            f"the build under test cannot be identified. Rebuild with:\n"
            f"  maturin develop --release --features python"
        )

    if imported != expected:
        sys.exit(
            f"error: the `splintr` imported from {origin} is version {imported}, but "
            f"this checkout is {expected}. Verifying a stale build reports divergences "
            f"that say nothing about the code being worked on. Rebuild with:\n"
            f"  maturin develop --release --features python\n"
            f"or run against the source tree with:\n"
            f"  PYTHONPATH=python python3 scripts/verify_external_models.py ..."
        )

    print(f"splintr {imported} from {origin}\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        required=True,
        help="directory holding the model repos named in TARGETS",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="NAME",
        help="run just this target (repeatable); default is every target",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print every disagreement, not just the first three per target",
    )
    args = parser.parse_args()

    if not args.models_dir.is_dir():
        sys.exit(
            f"error: --models-dir {args.models_dir} does not exist.\n"
            "This harness verifies splintr against real published tokenizers; "
            "without them there is nothing to verify, so it fails rather than "
            "reporting a pass over zero models. Point --models-dir at a "
            "directory holding the model repos listed in TARGETS (see this "
            "script's module docstring)."
        )

    selected = TARGETS
    if args.only:
        selected = tuple(t for t in TARGETS if t[0] in args.only)
        unknown = sorted(set(args.only) - {t[0] for t in TARGETS})
        if unknown:
            sys.exit(
                f"error: unknown --only target(s): {', '.join(unknown)}. "
                f"Known: {', '.join(t[0] for t in TARGETS)}"
            )

    require_packages()
    require_current_build()

    total = len(REFERENCE_CORPUS)
    print(f"models-dir: {args.models_dir}")
    print(f"corpus:     {total} cases (scripts/reference_corpus.py)\n")
    header = f"{'TARGET':<26} {'KIND':<5} {'ENCODE':>11} {'DECODE':>11}  STATUS"
    print(header)
    print("-" * len(header))

    bad = 0
    for name, kind, relative, vocab in selected:
        path = args.models_dir / relative
        try:
            if kind == "json":
                encode_ok, decode_ok, failures = check_json(path)
            else:
                encode_ok, decode_ok, failures = check_spm(path, vocab)
        except TargetMissing as exc:
            bad += 1
            print(f"{name:<26} {kind:<5} {'-':>11} {'-':>11}  MISSING ({exc})")
            continue

        status = "ok" if not failures else f"FAIL ({len(failures)})"
        if failures:
            bad += 1
        print(
            f"{name:<26} {kind:<5} {f'{encode_ok}/{total}':>11} "
            f"{f'{decode_ok}/{total}':>11}  {status}"
        )
        shown = failures if args.verbose else failures[:3]
        for failure in shown:
            print(f"    {failure}")
        if len(failures) > len(shown):
            print(f"    ... {len(failures) - len(shown)} more (use --verbose)")

    print()
    if bad:
        print(f"{bad}/{len(selected)} target(s) missing or disagreeing")
        return 1
    print(f"all {len(selected)} target(s) agree on {2 * total} encode+decode cases each")
    return 0


if __name__ == "__main__":
    sys.exit(main())
