"""Fold `perf_bench.py` JSON lines into a Markdown report.

    perf_report.py --title <title> [--note <line>]... < rounds.jsonl

The first engine seen in a suite is the baseline every other column is measured
against, so run splintr first. Values are medians across rounds.
"""

import json
import statistics
import sys
from collections import OrderedDict

CORPUS_ORDER = ["english", "chinese", "code", "json", "multilingual", "long-docs"]


def parse_args(argv):
    title, notes = "perf", []
    i = 0
    while i < len(argv):
        if argv[i] == "--title":
            title = argv[i + 1]
            i += 2
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


def ratio(base, other, lower_is_better):
    """How many times better the baseline is; below 1.00x it lost."""
    if base <= 0 or other <= 0:
        return "n/a"
    return f"{(other / base if lower_is_better else base / other):.2f}x"


def main():
    title, notes = parse_args(sys.argv[1:])
    suites = OrderedDict()
    for line in sys.stdin:
        line = line.strip()
        if not line.startswith("{"):
            continue
        rec = json.loads(line)
        suites.setdefault(rec["suite"], OrderedDict()).setdefault(rec["label"], []).append(rec)

    print(f"# {title}\n")
    for note in notes:
        print(f"{note}\n")

    for suite, engines in suites.items():
        labels = list(engines)
        base = labels[0]
        others = labels[1:]

        print(f"\n## {suite}\n")

        # Token counts first: if they disagree, nothing below is a comparison.
        counts = {
            lbl: {c: engines[lbl][0]["corpora"][c]["tokens"] for c in CORPUS_ORDER}
            for lbl in labels
        }
        if any(counts[lbl] != counts[base] for lbl in others):
            print("> **Token counts differ — these engines did not produce the same output.**\n")

        print("### Steady-state encoding\n")
        header = "| Corpus | Mode | " + " | ".join(labels) + " |"
        if others:
            header += f" {base} advantage |"
        print(header)
        print("|---|---" + "|---:" * len(labels) + ("|---:|" if others else "|"))

        for corpus in CORPUS_ORDER:
            for mode, key in (("single", "single_ms"), ("batch", "batch_ms")):
                values = {
                    lbl: med(engines[lbl], "corpora", corpus, key) for lbl in labels
                }
                cells = " | ".join(f"{values[lbl]:,.3f} ms" for lbl in labels)
                row = f"| {corpus} | {mode} | {cells} |"
                if others:
                    row += f" {ratio(values[base], values[others[-1]], True)} |"
                print(row)

        print("\n### Vocabulary load\n")
        loads = {lbl: med(engines[lbl], "load_ms") for lbl in labels}
        print("| Metric | " + " | ".join(labels) + " |" + (f" {base} advantage |" if others else ""))
        print("|---" + "|---:" * len(labels) + ("|---:|" if others else "|"))
        cells = " | ".join(f"{loads[lbl]:,.3f} ms" for lbl in labels)
        row = f"| Load time | {cells} |"
        if others:
            row += f" {ratio(loads[base], loads[others[-1]], True)} |"
        print(row)

        # A one-line read of the headline number, so the table has a conclusion.
        single = {lbl: med(engines[lbl], "corpora", "english", "single_ms") for lbl in labels}
        batch = {lbl: med(engines[lbl], "corpora", "english", "batch_ms") for lbl in labels}
        for other in others:
            print(
                f"\n**{base} vs {other}** (english): "
                f"{ratio(single[base], single[other], True)} single, "
                f"{ratio(batch[base], batch[other], True)} batch, "
                f"{ratio(loads[base], loads[other], True)} load."
            )


main()
