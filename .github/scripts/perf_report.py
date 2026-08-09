"""Fold `perf_bench.py` JSON lines into a Markdown report.

    perf_report.py --title <title> [--note <line>]... < rounds.jsonl

One row per vocabulary, one column per engine, so every row is a direct
comparison; the winner is bolded and `—` marks a vocabulary an engine cannot
load. Values are medians across rounds.
"""

import json
import statistics
import sys
from collections import OrderedDict

# Every corpus each engine is timed on. Not a table axis — the report is
# vocabulary x engine throughout — but the token-count agreement check below
# compares all of them, since a disagreement on any one shape invalidates the
# vocabulary's whole row.
CORPUS_ORDER = ["english", "chinese", "code", "json", "multilingual", "long-docs"]
# Columns in a fixed order so every table reads the same way; anything else
# (a pinned baseline, say) is appended in the order it first appears.
PREFERRED = ["splintr", "HF tokenizers", "tiktoken"]
MISSING = "—"


def parse_args(argv):
    title, notes, i = "perf", [], 0
    while i < len(argv):
        if argv[i] == "--title":
            title, i = argv[i + 1], i + 2
        elif argv[i] == "--note":
            notes.append(argv[i + 1])
            i += 2
        else:
            i += 1
    return title, notes


def med(records, *path):
    values = []
    for rec in records:
        node = rec
        for key in path:
            node = node[key]
        values.append(node)
    return statistics.median(values)


def med_opt(records, *path):
    """`med`, but None when any record lacks the key.

    The flat-batch axis is absent from engines that have no buffer form, and
    from splintr releases that predate `encode_batch_flat`, so its table has to
    tell "not measured" apart from "measured as zero".
    """
    values = []
    for rec in records:
        node = rec
        for key in path:
            if not isinstance(node, dict) or key not in node:
                return None
            node = node[key]
        values.append(node)
    return statistics.median(values) if values else None


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

    # suite -> label -> [records]
    suites = OrderedDict()
    seen_labels = []
    for line in sys.stdin:
        line = line.strip()
        if not line.startswith("{"):
            continue
        rec = json.loads(line)
        suites.setdefault(rec["suite"], OrderedDict()).setdefault(rec["label"], []).append(rec)
        if rec["label"] not in seen_labels:
            seen_labels.append(rec["label"])

    columns = [c for c in PREFERRED if c in seen_labels]
    columns += [c for c in seen_labels if c not in columns]
    ratio_against = [c for c in columns[1:]]

    print(f"# {title}\n")
    for note in notes:
        print(f"{note}\n")

    # Any vocabulary where two engines disagree invalidates its whole row.
    for suite, engines in suites.items():
        counts = {
            lbl: {c: engines[lbl][0]["single"][c]["tokens"] for c in CORPUS_ORDER}
            for lbl in engines
        }
        first = next(iter(counts))
        if any(counts[lbl] != counts[first] for lbl in counts):
            print(f"> **{suite}: engines produced different token ids — its row is not a comparison.**\n")

    def gather(path_fn):
        rows = OrderedDict()
        for suite, engines in suites.items():
            rows[suite] = {lbl: path_fn(engines[lbl]) for lbl in engines}
        return rows

    table(
        "Batch throughput (1,000 texts, mixed corpus)",
        gather(lambda recs: med(recs, "batch", "1000", "mb_per_s")),
        columns,
        "MB/s",
        higher_is_better=True,
        fmt="{:,.1f}",
        ratio_against=ratio_against,
    )

    # The same work ending in a buffer instead of a list of lists. Only the
    # engines that offer one appear; the rest are absent from the table rather
    # than shown losing a race they were never entered in.
    flat_rows = OrderedDict()
    flat_columns = []
    for suite, engines in suites.items():
        row = {}
        for lbl in engines:
            value = med_opt(engines[lbl], "flat", "1000", "mb_per_s")
            if value is None:
                continue
            row[lbl] = value
            if lbl not in flat_columns:
                flat_columns.append(lbl)
        flat_rows[suite] = row
    if flat_columns:
        flat_columns = [c for c in columns if c in flat_columns]
        table(
            "Batch throughput, flat output (1,000 texts, mixed corpus)",
            flat_rows,
            flat_columns,
            "MB/s",
            higher_is_better=True,
            fmt="{:,.1f}",
            ratio_against=[c for c in flat_columns[1:]],
        )

    table(
        "Single-text latency (1,000 texts, english corpus)",
        gather(lambda recs: med(recs, "single", "english", "ms")),
        columns,
        "ms",
        higher_is_better=False,
        fmt="{:,.2f}",
        ratio_against=ratio_against,
    )

    table(
        "Vocabulary load",
        gather(lambda recs: med(recs, "load_ms")),
        columns,
        "ms",
        higher_is_better=False,
        fmt="{:,.1f}",
        ratio_against=ratio_against,
    )

    # Every table above is vocabulary x engine. Per-corpus and per-batch-size
    # breakdowns used to be printed for one arbitrarily chosen vocabulary, which
    # made them the only tables in the report that did not answer "how does this
    # vocabulary compare across engines". The measurements are still recorded in
    # `rounds.jsonl` under `single.<corpus>` and `batch.<size>` for anyone
    # chasing a specific shape.


main()
