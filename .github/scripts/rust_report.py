"""Fold `rust-bench` JSON into a Markdown report.

    rust_report.py --title <title> [--note <line>]... < results.jsonl

Deliberately the same shape as `perf_report.py`: one row per suite
(`<vocabulary>-<family>`), one column per engine, and a `vs <engine>` ratio
column against each rival. Reading the two reports side by side is the point —
one says what a Python caller gets, this one says what the Rust kernels do — so
they must not be laid out differently.

A suite's *family* is how the vocabulary was loaded, and an engine that cannot
load that form is `—` rather than absent, exactly as in the Python report.
"""

import json
import sys
from collections import OrderedDict

# splintr first so the ratio columns read "how many times better is splintr";
# anything else appears in the order it is first seen.
PREFERRED = ["splintr"]
FAMILY_ORDER = ["ranks", "json"]
MISSING = "—"

FAMILY_NOTES = {
    "ranks": (
        "`-ranks` — every engine reads the same `.tiktoken` rank file and the same "
        "pre-tokenizer expression. `bpe-openai` is the exception: it has no loader "
        "and carries its own copy of the vocabulary compiled in, and is here "
        "because its ids match, so it tokenizes the same text against the same "
        "vocabulary."
    ),
    "json": "`-json` — every engine reads the same HuggingFace `tokenizer.json`.",
}


def read_documents(text):
    """Every JSON value in `text`, whatever whitespace separates them.

    The benchmark wraps each object across lines, so a line-oriented reader
    would split one in half.
    """
    decoder = json.JSONDecoder()
    out, pos, end = [], 0, len(text)
    while pos < end:
        while pos < end and text[pos] in " \t\r\n":
            pos += 1
        if pos >= end:
            break
        doc, pos = decoder.raw_decode(text, pos)
        out.append(doc)
    return out


def parse_args(argv):
    title, notes, i = "rust", [], 0
    while i < len(argv):
        if argv[i] == "--title":
            title, i = argv[i + 1], i + 2
        elif argv[i] == "--note":
            notes.append(argv[i + 1])
            i += 2
        else:
            i += 1
    return title, notes


def table(title, rows, columns, unit, higher_is_better, fmt, ratio_against):
    """`rows` maps a row label to {column: value or None}."""
    print(f"\n### {title}\n")
    header = "| | " + " | ".join(columns) + " |"
    for other in ratio_against:
        header += f" vs {other} |"
    print(header)
    print("|---" + "|---:" * (len(columns) + len(ratio_against)) + "|")

    for label, values in rows.items():
        present = [v for v in values.values() if v is not None]
        best = (max if higher_is_better else min)(present) if present else None

        cells = []
        for col in columns:
            v = values.get(col)
            if v is None:
                cells.append(MISSING)
                continue
            text = fmt.format(v) + f" {unit}"
            cells.append(f"**{text}**" if v == best else text)

        base = values.get(PREFERRED[0])
        for other in ratio_against:
            v = values.get(other)
            if base is None or v is None or base <= 0 or v <= 0:
                cells.append(MISSING)
            else:
                r = base / v if higher_is_better else v / base
                cells.append(f"{r:.1f}x")

        print(f"| {label} | " + " | ".join(cells) + " |")


def main():
    title, notes = parse_args(sys.argv[1:])

    # suite -> engine -> record, where suite is "<vocabulary>-<family>"
    suites, seen_engines, skipped = OrderedDict(), [], []
    vocabs, families = [], []
    for doc in read_documents(sys.stdin.read()):
        vocab = doc["vocab"]
        if vocab not in vocabs:
            vocabs.append(vocab)
        for entry in doc.get("skipped", []):
            skipped.append((vocab, entry["engine"], entry["reason"]))
        for rec in doc["results"]:
            if rec["family"] not in families:
                families.append(rec["family"])
            suites.setdefault(f"{vocab}-{rec['family']}", {})[rec["engine"]] = rec
            if rec["engine"] not in seen_engines:
                seen_engines.append(rec["engine"])

    columns = [c for c in PREFERRED if c in seen_engines]
    columns += [c for c in seen_engines if c not in columns]
    ratio_against = columns[1:]

    # Rows in a fixed order so two reports can be diffed line by line.
    order = [
        f"{v}-{f}"
        for f in ([f for f in FAMILY_ORDER if f in families] + [f for f in families if f not in FAMILY_ORDER])
        for v in vocabs
        if f"{v}-{f}" in suites
    ]
    ordered = OrderedDict((k, suites[k]) for k in order)

    print(f"# {title}\n")
    for note in notes:
        print(f"{note}\n")
    for family in families:
        if family in FAMILY_NOTES:
            print(f"{FAMILY_NOTES[family]}\n")

    disagree = [
        (suite, engine)
        for suite, engines in ordered.items()
        for engine, rec in engines.items()
        if not rec["agrees"]
    ]
    if disagree:
        print(
            "> **Engines disagreed on token ids — those rows are not comparisons:** "
            + ", ".join(f"{s}/{e}" for s, e in disagree)
            + "\n"
        )
    else:
        print("Every engine agreed on token ids for every suite it could load.\n")

    def gather(key):
        rows = OrderedDict()
        for suite, engines in ordered.items():
            rows[suite] = {name: rec[key] for name, rec in engines.items()}
        return rows

    table(
        "Single-text throughput (one thread)",
        gather("serial_mbps"),
        columns,
        "MB/s",
        higher_is_better=True,
        fmt="{:,.0f}",
        ratio_against=ratio_against,
    )
    table(
        "Batch throughput (all cores)",
        gather("par_mbps"),
        columns,
        "MB/s",
        higher_is_better=True,
        fmt="{:,.0f}",
        ratio_against=ratio_against,
    )

    borrowed = sorted(
        {
            name
            for engines in ordered.values()
            for name, rec in engines.items()
            if not rec["own_batch"]
        }
    )
    if borrowed:
        verb = "expose" if len(borrowed) > 1 else "exposes"
        print(
            f"\n{', '.join(borrowed)} {verb} no batch API, so the batch column there "
            "is rayon in the harness rather than a library call."
        )

    if skipped:
        print("\n### Could not load\n")
        print("| vocabulary | engine | reason |")
        print("|---|---|---|")
        for vocab, engine, reason in skipped:
            print(f"| {vocab} | {engine} | {reason} |")


main()
