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

CORPUS_ORDER = ["english", "chinese", "code", "json", "multilingual", "long-docs"]
BATCH_ORDER = ["100", "500", "1000"]
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

    # Corpus shape moves the single-text number a lot, so break one vocabulary
    # out rather than implying the headline holds for every kind of text.
    detail = "cl100k_base" if "cl100k_base" in suites else next(iter(suites))
    engines = suites[detail]
    rows = OrderedDict(
        (corpus, {lbl: med(engines[lbl], "single", corpus, "ms") for lbl in engines})
        for corpus in CORPUS_ORDER
    )
    table(
        f"Single-text latency by corpus ({detail})",
        rows,
        columns,
        "ms",
        higher_is_better=False,
        fmt="{:,.2f}",
        ratio_against=ratio_against,
    )

    rows = OrderedDict(
        (f"{int(size):,} texts", {lbl: med(engines[lbl], "batch", size, "mb_per_s") for lbl in engines})
        for size in BATCH_ORDER
    )
    table(
        f"Batch throughput by batch size ({detail})",
        rows,
        columns,
        "MB/s",
        higher_is_better=True,
        fmt="{:,.1f}",
        ratio_against=ratio_against,
    )


main()
