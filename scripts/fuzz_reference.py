#!/usr/bin/env python3
"""Differential fuzzer: splintr vs. the real reference tokenizer implementation.

WHY THIS EXISTS
---------------
Every tokenization bug found in splintr recently was found the same way --
random strings assembled from fragments that include *the vocabulary's own
added/special tokens*, diffed against the reference implementation. The fixed
74-case corpus in `scripts/reference_corpus.py` contains no `<mask>`, no
`[INST]`, no `<s>`, so it was structurally blind to all of them:

  * the SentencePiece dummy prefix was applied once per inter-special gap
    instead of once per sequence (every Mistral chat prompt tokenized wrong);
  * `lstrip`/`rstrip` on added tokens was parsed and never read (224/5000
    random strings wrong on bge-m3);
  * Unigram Viterbi accumulated scores in f32, so exact ties broke the wrong
    way;
  * DeepSeek V3's `tokenizer.json` would not load at all.

None of those are reachable from a corpus of prose. All of them are reachable
in seconds from `"".join(random fragments incl. added tokens)`. Fragments are
joined with NO separator on purpose: text pressed directly against an added
token is the shape that breaks `lstrip`/`rstrip`, the dummy prefix, and the
decoder pipeline.

REFERENCES AND ENTRY-POINT PAIRING
----------------------------------
Getting the pairing wrong produces confident false failures. The mapping is
therefore explicit, one table per reference, and never inferred:

`tokenizers.Tokenizer` (a HuggingFace `tokenizer.json`) -- 4 modes:
    encode_raw       ref.encode(t, add_special_tokens=False).ids  == sp.encode_raw(t)
    encode           ref.encode(t, add_special_tokens=True).ids   == sp.encode(t)
    decode_raw       ref.decode(ids_raw)                          == sp.decode(ids_raw)
    decode           ref.decode(ids_tpl)                          == sp.decode(ids_tpl)
  `sp.encode` applies the file's `post_processor` template, which is exactly
  what `add_special_tokens=True` means; `sp.encode_raw` is the untemplated
  form. Both reference and splintr drop `special=true` ids on decode (HF's
  `skip_special_tokens=True` default), so the decoders pair directly.

`transformers.AutoTokenizer(..., use_fast=False)` (sentencepiece-backed) -- 2 modes:
    encode_raw       ref.encode(t, add_special_tokens=False)      == sp.encode_raw(t)
    decode           fast.decode(ids_raw, skip_special_tokens=False)
                                                                  == sp.decode(ids_raw)
  The slow, sentencepiece-backed tokenizer is the encode ground truth for
  Mistral: the *fast* Mistral tokenizer disagrees with sentencepiece on
  metaspace handling for some inputs, so pairing against it would report
  splintr as wrong when it is right. `encode_raw` (not `encode`) is the
  counterpart because splintr's bundled Mistral vocabularies carry no BOS
  template, while `add_special_tokens=True` on the reference prepends `<s>`.
  Decode is the other way round: the ground truth for *decoding* is the
  `decoder` pipeline declared in the same directory's `tokenizer.json`
  (`Replace`->`ByteFallback`->`Fuse`->`Strip`), which is what splintr
  implements and what `tokenizers` executes; the slow tokenizer detokenizes
  via its own `convert_tokens_to_string`, which inserts spaces around special
  tokens that were never in the input. So decode is diffed against
  `tokenizers.Tokenizer.from_file(<model dir>/tokenizer.json)` and is skipped,
  with a stated reason, when the directory has no `tokenizer.json`.

`tiktoken` (the bundled OpenAI vocabularies) -- 3 modes:
    encode_ordinary  ref.encode_ordinary(t)                       == sp.encode_ordinary(t)
    encode_special   ref.encode(t, allowed_special="all")          == sp.encode_with_special(t)
    decode           ref.decode(ids_special)                       == sp.decode(ids_special)
  tiktoken's plain `encode` raises on any special token appearing in the text,
  so it is never the right call for a fuzzer whose whole point is putting
  special tokens in the text. `encode_ordinary` (specials stay literal text)
  is what splintr's `encode_ordinary` does — not its `encode`, which applies
  the special-token policy and resolves a spelled-out special to its id;
  `allowed_special="all"` is what `encode_with_special` does. The special fragments are drawn from tiktoken's
  own special set, not splintr's -- splintr adds 54 agent tokens the reference
  has never heard of, and feeding those in would be a divergence by
  construction rather than a bug.

`splintr.from_json` returns a `splintr.AnyTokenizer`;
`splintr.Tokenizer.from_pretrained` returns an `AnyTokenizer` for
`mistral`/`mistral_v1`/`mistral_v2` and a `Tokenizer` otherwise. Both are
handled: the mode table is chosen per reference, and the splintr-side call is
resolved by name so the two handle types are interchangeable where their APIs
agree.

TRIAGE
------
Runs are deterministic: `--seed` is printed in the header, so any failure
reproduces exactly. On failure the input is shrunk -- fragments are dropped one
at a time for as long as the case keeps failing -- and the minimal reproducer
is printed with both id sequences windowed around the first differing index. A
14-fragment failing string says nothing; a 2-fragment one names the bug. Exit
status is non-zero if any case fails, so this can gate a release.

USAGE
-----
    # a HuggingFace tokenizer.json (reference auto-detected as `tokenizers`)
    python3 scripts/fuzz_reference.py ~/Projects/models/bge-m3-tokenizer/tokenizer.json \
        --cases 6250
    python3 scripts/fuzz_reference.py ~/Projects/models/deepseek-v3-tokenizer/tokenizer.json \
        --cases 2000

    # bundled vocabularies whose reference lives in a local model directory,
    # written `<bundled name>=<path>` (reference auto-detected as `transformers`)
    python3 scripts/fuzz_reference.py \
        mistral_v1=~/Projects/models/mistral-7b-awq-int4 \
        mistral_v2=~/Projects/models/mistral-7b-v0.3 \
        --cases 2014

    # bundled OpenAI vocabularies (reference auto-detected as `tiktoken`)
    python3 scripts/fuzz_reference.py cl100k_base o200k_base --cases 2000

MEASURED BASELINE (all zero failures; totals are cases x modes)
---------------------------------------------------------------
    bge-m3          `tokenizers`                     25,000/25,000
    Mistral V1 + V2 `transformers`, use_fast=False    8,056/8,056
    DeepSeek V3     `tokenizers`                       8,000/8,000

A drop below any of those totals at the same `--seed`/`--cases` is a
regression. `--seed` defaults to 0 so the baseline is reproducible without
extra flags.

Requires `tokenizers`, `transformers` + `sentencepiece`, and/or `tiktoken`
depending on which targets are named. A target whose reference package or
model path is absent is skipped with a stated reason rather than aborting the
run.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Callable, Sequence

# --------------------------------------------------------------------------
# Fragment pool.
#
# Structural fragments only -- the vocabulary's own added tokens are appended
# at runtime by `discover_special_fragments`. Every entry is a distinct
# phenomenon (a whitespace class, a script, a punctuation shape), not a
# variation on an existing one, because the interesting behaviour lives at the
# joins between them.
# --------------------------------------------------------------------------
STRUCTURAL_FRAGMENTS: tuple[str, ...] = (
    # the empty string: an added token adjacent to nothing at all
    "",
    # ASCII words / identifiers
    "hello",
    "world",
    "The",
    "quick",
    "getUserName",
    "toString",
    "def",
    "return",
    "x",
    "A",
    # whitespace runs, including non-ASCII whitespace
    " ",
    "  ",
    "   ",
    "\t",
    "\t\t",
    "\n",
    "\n\n",
    "\r\n",
    " ",  # NO-BREAK SPACE
    "　",  # IDEOGRAPHIC SPACE
    " ",  # THIN SPACE
    # CJK
    "世界",
    "こんにちは",
    "カタカナ",
    "北京市海淀区",
    "한국어",
    # Cyrillic
    "русский",
    "текст",
    "Ж",
    # emoji, including a ZWJ sequence and a variation-selector sequence
    "😀",
    "🎉🚀",
    "👨‍👩‍👧‍👦",
    "🏳️‍🌈",
    # accented / non-ASCII Latin
    "café",
    "résumé",
    "Zürich",
    "Ångström",
    "naïve",
    # punctuation
    ".",
    ",",
    "!?",
    ":",
    "'",
    '"',
    "()",
    "[]",
    "{}",
    "—",
    "<",
    ">",
    "|",
    "/",
    "\\",
    "_",
    # digit runs
    "0",
    "42",
    "2024",
    "0123456789",
    "12345678901234567890",
    "3.14159265358979",
)


def build_fragment_pool(specials: Sequence[str]) -> tuple[list[str], list[str]]:
    """Return `(structural, special)` fragment pools for this vocabulary."""
    return list(STRUCTURAL_FRAGMENTS), list(specials)


def make_fragments(
    rng: random.Random,
    structural: Sequence[str],
    specials: Sequence[str],
    max_fragments: int,
    special_rate: float = 0.35,
) -> list[str]:
    """Draw a random fragment list, biased so added tokens appear often.

    Fragments are returned unjoined so a failing case can be shrunk fragment by
    fragment; the caller joins them with no separator.
    """
    count = rng.randint(1, max_fragments)
    out: list[str] = []
    for _ in range(count):
        if specials and rng.random() < special_rate:
            out.append(rng.choice(specials))
        else:
            out.append(rng.choice(structural))
    return out


# --------------------------------------------------------------------------
# Result plumbing
# --------------------------------------------------------------------------


class SkipTarget(Exception):
    """A target cannot be fuzzed here (missing package or model path)."""


Comparison = Callable[[str], tuple[object, object]]
"""Maps one input string to `(expected, actual)`; unequal means failure."""


class Mode:
    """One named (reference call, splintr call) pairing."""

    def __init__(self, name: str, compare: Comparison) -> None:
        self.name = name
        self.compare = compare

    def failing(self, fragments: Sequence[str]) -> tuple[object, object] | None:
        """Return `(expected, actual)` if this input fails, else None."""
        text = "".join(fragments)
        try:
            expected, actual = self.compare(text)
        except Exception as exc:  # noqa: BLE001 - any raise is itself a failure
            return (f"<reference or splintr raised: {type(exc).__name__}: {exc}>", None)
        return None if expected == actual else (expected, actual)


def guarded(call: Callable[[str], object]) -> Callable[[str], object]:
    """Turn an exception into a comparable sentinel so one side raising fails."""

    def inner(text: str) -> object:
        try:
            return call(text)
        except Exception as exc:  # noqa: BLE001 - a raise is a real difference
            return f"<raised {type(exc).__name__}: {exc}>"

    return inner


# --------------------------------------------------------------------------
# Shrinking
# --------------------------------------------------------------------------


def shrink(mode: Mode, fragments: Sequence[str]) -> list[str]:
    """Drop fragments for as long as the case keeps failing.

    Repeated single-fragment deletion to a fixed point. Cheap, and it reliably
    takes a 14-fragment string down to the two fragments that actually name the
    bug.
    """
    current = list(fragments)
    changed = True
    while changed and len(current) > 1:
        changed = False
        for index in range(len(current)):
            candidate = current[:index] + current[index + 1 :]
            if not candidate:
                continue
            if mode.failing(candidate) is not None:
                current = candidate
                changed = True
                break
    return current


def first_difference(expected: object, actual: object) -> int | None:
    """Index of the first differing element/character, when both are sequences."""
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        pass
    elif isinstance(expected, str) and isinstance(actual, str):
        pass
    else:
        return None
    for index, (left, right) in enumerate(zip(expected, actual)):
        if left != right:
            return index
    shorter = min(len(expected), len(actual))
    return shorter if len(expected) != len(actual) else None


def window(value: object, index: int | None, radius: int = 8) -> str:
    """Render a bounded slice around `index` -- never the whole sequence."""
    if index is None or not isinstance(value, (list, tuple, str)):
        return repr(value)[:400]
    start = max(0, index - radius)
    stop = index + radius + 1
    prefix = "..." if start > 0 else ""
    suffix = "..." if stop < len(value) else ""
    return f"{prefix}{value[start:stop]!r}{suffix}"


def report_failure(
    target: str, mode: Mode, fragments: Sequence[str], case_index: int
) -> None:
    """Shrink and print a minimal reproducer for one failing case."""
    minimal = shrink(mode, fragments)
    outcome = mode.failing(minimal)
    if outcome is None:  # pragma: no cover - shrink preserves failure
        minimal = list(fragments)
        outcome = mode.failing(minimal)
    assert outcome is not None
    expected, actual = outcome
    index = first_difference(expected, actual)
    print(f"\nFAIL {target}/{mode.name}  (case #{case_index})")
    print(f"  fragments : {minimal!r}")
    print(f"  input     : {''.join(minimal)!r}")
    print(f"  first diff: index {index}" if index is not None else "  first diff: n/a")
    print(f"  reference : {window(expected, index)}")
    print(f"  splintr   : {window(actual, index)}")


# --------------------------------------------------------------------------
# Reference construction
# --------------------------------------------------------------------------


def load_tokenizers_target(path: Path) -> tuple[list[Mode], list[str]]:
    """A HuggingFace `tokenizer.json`, diffed against `tokenizers`."""
    try:
        from tokenizers import Tokenizer as HfTokenizer
    except ImportError as exc:
        raise SkipTarget(f"`tokenizers` not installed ({exc})") from exc
    import splintr

    if not path.is_file():
        raise SkipTarget(f"no such file: {path}")

    reference = HfTokenizer.from_file(str(path))
    subject = splintr.from_json(str(path))

    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    specials = [
        entry["content"]
        for entry in raw.get("added_tokens", [])
        if isinstance(entry, dict) and isinstance(entry.get("content"), str)
    ]

    def ref_raw(text: str) -> list[int]:
        return reference.encode(text, add_special_tokens=False).ids

    def ref_tpl(text: str) -> list[int]:
        return reference.encode(text, add_special_tokens=True).ids

    modes = [
        Mode("encode_raw", lambda t: (ref_raw(t), guarded(subject.encode_raw)(t))),
        Mode("encode", lambda t: (ref_tpl(t), guarded(subject.encode)(t))),
        Mode(
            "decode_raw",
            lambda t: _decode_pair(reference.decode, subject.decode, ref_raw(t)),
        ),
        Mode(
            "decode",
            lambda t: _decode_pair(reference.decode, subject.decode, ref_tpl(t)),
        ),
    ]
    return modes, specials


def _decode_pair(
    ref_decode: Callable[..., str],
    sp_decode: Callable[[list[int]], str],
    ids: list[int],
    **ref_kwargs: object,
) -> tuple[object, object]:
    """Round-trip one id sequence through both decoders."""
    try:
        expected: object = ref_decode(ids, **ref_kwargs)
    except Exception as exc:  # noqa: BLE001
        expected = f"<reference raised {type(exc).__name__}: {exc}>"
    try:
        actual: object = sp_decode(ids)
    except Exception as exc:  # noqa: BLE001
        actual = f"<splintr raised {type(exc).__name__}: {exc}>"
    return expected, actual


def load_transformers_target(name: str, model_dir: Path) -> tuple[list[Mode], list[str]]:
    """A bundled SentencePiece vocabulary, diffed against slow `transformers`."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SkipTarget(f"`transformers` not installed ({exc})") from exc
    try:
        import sentencepiece  # noqa: F401 - required by the slow tokenizer
    except ImportError as exc:
        raise SkipTarget(f"`sentencepiece` not installed ({exc})") from exc
    import splintr

    if not model_dir.is_dir():
        raise SkipTarget(f"no such model directory: {model_dir}")

    reference = AutoTokenizer.from_pretrained(str(model_dir), use_fast=False)
    subject = splintr.Tokenizer.from_pretrained(name)
    if not hasattr(subject, "encode_raw"):
        raise SkipTarget(
            f"bundled vocabulary {name!r} loads as {type(subject).__name__}, "
            "which has no `encode_raw`; the slow-transformers pairing needs it"
        )

    # Only added tokens the reference AND splintr both know: splintr's bundled
    # vocabularies carry 54 extra agent tokens the reference cannot produce, and
    # feeding those in would manufacture a divergence rather than find one.
    known = sorted(reference.get_added_vocab())
    specials = [tok for tok in known if subject.special_token_id(tok) is not None]

    def ref_raw(text: str) -> list[int]:
        return reference.encode(text, add_special_tokens=False)

    modes = [
        Mode("encode_raw", lambda t: (ref_raw(t), guarded(subject.encode_raw)(t))),
    ]

    # Decode ground truth is the declared `decoder` pipeline in the same
    # directory's tokenizer.json, executed by `tokenizers` -- see the module
    # docstring. Without that file there is no decode reference here.
    json_path = model_dir / "tokenizer.json"
    if json_path.is_file():
        try:
            from tokenizers import Tokenizer as HfTokenizer
        except ImportError:
            print(f"  note: decode mode skipped for {name}: `tokenizers` not installed")
        else:
            fast = HfTokenizer.from_file(str(json_path))
            modes.append(
                Mode(
                    "decode",
                    lambda t: _decode_pair(
                        fast.decode,
                        subject.decode,
                        ref_raw(t),
                        skip_special_tokens=False,
                    ),
                )
            )
    else:
        print(
            f"  note: decode mode skipped for {name}: no tokenizer.json in {model_dir}"
        )

    return modes, specials


def load_tiktoken_target(name: str) -> tuple[list[Mode], list[str]]:
    """A bundled OpenAI vocabulary, diffed against `tiktoken`."""
    try:
        import tiktoken
    except ImportError as exc:
        raise SkipTarget(f"`tiktoken` not installed ({exc})") from exc
    import splintr

    try:
        reference = tiktoken.get_encoding(name)
    except Exception as exc:  # noqa: BLE001 - unknown encoding name
        raise SkipTarget(f"tiktoken has no encoding {name!r} ({exc})") from exc
    subject = splintr.Tokenizer.from_pretrained(name)

    # tiktoken's own special set only: splintr's extra agent tokens are unknown
    # to the reference by design.
    specials = sorted(reference._special_tokens)  # noqa: SLF001 - no public accessor

    def ref_special(text: str) -> list[int]:
        return reference.encode(text, allowed_special="all")

    modes = [
        Mode(
            "encode_ordinary",
            lambda t: (
                reference.encode_ordinary(t),
                guarded(subject.encode_ordinary)(t),
            ),
        ),
        Mode(
            "encode_special",
            lambda t: (ref_special(t), guarded(subject.encode_with_special)(t)),
        ),
        Mode(
            "decode",
            lambda t: _decode_pair(reference.decode, subject.decode, ref_special(t)),
        ),
    ]
    return modes, specials


TIKTOKEN_VOCABS = frozenset({"cl100k_base", "o200k_base"})


def resolve_target(spec: str, reference: str) -> tuple[str, list[Mode], list[str]]:
    """Turn one CLI target into `(label, modes, special fragments)`.

    A target is either a path to a `tokenizer.json` (or a directory holding
    one), a bundled vocabulary name, or `<bundled name>=<reference model dir>`.
    """
    label, _, ref_path = spec.partition("=")
    path = Path(label).expanduser()

    if reference == "auto":
        if ref_path:
            reference = "transformers"
        elif path.exists():
            reference = "tokenizers"
        elif label in TIKTOKEN_VOCABS:
            reference = "tiktoken"
        else:
            raise SkipTarget(
                f"cannot auto-detect a reference for {spec!r}: it is not an "
                "existing path, not a tiktoken encoding, and names no "
                "`=<model dir>` for the transformers reference"
            )

    if reference == "tokenizers":
        if path.is_dir():
            path = path / "tokenizer.json"
        modes, specials = load_tokenizers_target(path)
        # Name the vocabulary by its directory ("bge-m3-tokenizer"), which is
        # what a reader recognises; every such file is called tokenizer.json.
        pretty = path.parent.name or path.name
        return pretty, modes, specials

    if reference == "transformers":
        if not ref_path:
            raise SkipTarget(
                f"target {spec!r} with --reference transformers needs a "
                "reference model directory, written `<vocab>=<path>`"
            )
        modes, specials = load_transformers_target(
            label, Path(ref_path).expanduser()
        )
        return label, modes, specials

    if reference == "tiktoken":
        modes, specials = load_tiktoken_target(label)
        return label, modes, specials

    raise SkipTarget(f"unknown reference {reference!r}")


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def fuzz_target(
    label: str,
    modes: Sequence[Mode],
    specials: Sequence[str],
    cases: int,
    seed: int,
    max_fragments: int,
    max_reports: int,
) -> tuple[int, int]:
    """Run every mode over `cases` inputs. Returns `(passed, total)`."""
    structural, special_pool = build_fragment_pool(specials)
    print(
        f"\n{label}: {len(modes)} mode(s), {cases} cases each, "
        f"{len(special_pool)} added-token fragment(s), "
        f"{len(structural)} structural fragment(s)"
    )
    if not special_pool:
        print("  note: no added tokens discovered -- the fuzzer is much weaker")

    passed_total = 0
    grand_total = 0
    for mode in modes:
        # Per-mode stream, re-seeded from the run seed so every mode sees the
        # same inputs and one failing seed reproduces the same case.
        rng = random.Random(seed)
        failures = 0
        reported = 0
        for case_index in range(cases):
            fragments = make_fragments(
                rng, structural, special_pool, max_fragments
            )
            if mode.failing(fragments) is not None:
                failures += 1
                if reported < max_reports:
                    report_failure(label, mode, fragments, case_index)
                    reported += 1
        passed = cases - failures
        passed_total += passed
        grand_total += cases
        marker = "" if failures == 0 else f"   <-- {failures} FAILED"
        print(f"  {label}/{mode.name}: {passed}/{cases}{marker}")
        if failures > reported:
            print(f"  ({failures - reported} further failure(s) not shown)")
    return passed_total, grand_total


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Differentially fuzz splintr against the reference tokenizer, "
            "using random strings built from the vocabulary's own added tokens."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Targets: a path to a tokenizer.json (or a directory holding one), "
            "a bundled vocabulary name (cl100k_base, o200k_base), or "
            "`<bundled name>=<reference model dir>` for the slow-transformers "
            "reference (mistral_v1, mistral_v2)."
        ),
    )
    parser.add_argument("targets", nargs="+", help="vocabularies to fuzz")
    parser.add_argument(
        "--reference",
        choices=("auto", "tokenizers", "transformers", "tiktoken"),
        default="auto",
        help="reference implementation (default: auto-detect per target)",
    )
    parser.add_argument(
        "--cases",
        type=int,
        default=2000,
        help="random cases per (vocabulary, mode) (default: 2000)",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="RNG seed, printed for reproducibility"
    )
    parser.add_argument(
        "--max-fragments",
        type=int,
        default=8,
        help="maximum fragments per generated string (default: 8)",
    )
    parser.add_argument(
        "--max-reports",
        type=int,
        default=3,
        help="shrunk reproducers printed per mode (default: 3)",
    )
    args = parser.parse_args(argv)

    if args.cases < 1:
        parser.error("--cases must be at least 1")
    if args.max_fragments < 1:
        parser.error("--max-fragments must be at least 1")

    print(
        f"splintr differential fuzz: seed={args.seed} cases={args.cases} "
        f"max_fragments={args.max_fragments} reference={args.reference}"
    )

    passed_total = 0
    grand_total = 0
    skipped: list[str] = []
    for spec in args.targets:
        try:
            label, modes, specials = resolve_target(spec, args.reference)
        except SkipTarget as exc:
            skipped.append(f"{spec}: {exc}")
            print(f"\nSKIP {spec}: {exc}")
            continue
        passed, total = fuzz_target(
            label,
            modes,
            specials,
            args.cases,
            args.seed,
            args.max_fragments,
            args.max_reports,
        )
        passed_total += passed
        grand_total += total

    print(f"\nTOTAL: {passed_total}/{grand_total} (seed={args.seed})")
    for note in skipped:
        print(f"SKIPPED {note}")
    return 0 if passed_total == grand_total else 1


if __name__ == "__main__":
    sys.exit(main())
