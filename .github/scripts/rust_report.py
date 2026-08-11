"""Fold `rust-bench` JSON into a Markdown report.

    rust_report.py --title <title> [--note <line>]... < results.jsonl

The run summary has to be readable at a glance, and the matrix behind it does
not fit that description: every vocabulary in the manifest, times two families,
times every corpus, times every engine. So the summary leads with one row per
vocabulary — splintr against its best rival, aggregated over the corpora — and
the per-corpus detail goes inside a collapsed section. Nothing is dropped; the
uploaded JSONL has every cell either way.

Aggregating means summing bytes and summing time, never averaging the rates:
the corpora differ in size, and a mean of MB/s would weight a 1 MB corpus the
same as a 6 MB one.

Deliberately the same shape as `perf_report.py` — one row per suite, one column
per engine, a `vs <engine>` ratio column — because reading the two side by side
is the point: one says what a Python caller gets, this one what the Rust kernels
do.
"""

import json
import sys
from collections import OrderedDict

# splintr first so the ratio columns read "how many times better is splintr";
# anything else appears in the order it is first seen.
PREFERRED = ["splintr"]
FAMILY_ORDER = ["ranks", "json"]
# The two ways a cell can hold no number, kept apart because they say different
# things about the engine. `—` is "does not offer this shape" — no such file
# published, or the loader declined it. `x` is "took the vocabulary and then
# failed on the text", which is a bug in that engine rather than a documented
# gap, and hiding it behind the same dash would flatter it.
MISSING = "—"
BROKEN = "x"

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
    """`rows` maps a row label to {column: number, or `BROKEN`, or absent}."""
    print(f"\n#### {title}\n")
    header = "| | " + " | ".join(columns) + " |"
    for other in ratio_against:
        header += f" vs {other} |"
    print(header)
    print("|---" + "|---:" * (len(columns) + len(ratio_against)) + "|")

    def number(v):
        return v if isinstance(v, (int, float)) else None

    for label, values in rows.items():
        present = [n for n in (number(v) for v in values.values()) if n is not None]
        best = (max if higher_is_better else min)(present) if present else None

        cells = []
        for col in columns:
            v = values.get(col)
            if v == BROKEN:
                cells.append(BROKEN)
                continue
            if number(v) is None:
                cells.append(MISSING)
                continue
            text = fmt.format(v) + f" {unit}"
            cells.append(f"**{text}**" if v == best else text)

        base = number(values.get(PREFERRED[0]))
        for other in ratio_against:
            v = number(values.get(other))
            if base is None or v is None or base <= 0 or v <= 0:
                cells.append(BROKEN if values.get(other) == BROKEN else MISSING)
            else:
                r = base / v if higher_is_better else v / base
                cells.append(f"{r:.1f}x")

        print(f"| {label} | " + " | ".join(cells) + " |")


def cell(record, key):
    """One table cell: the measured rate, or `BROKEN` where there is none."""
    return BROKEN if record.get("broken") else record[key]


def aggregate(records):
    """Bytes-weighted MB/s over a group of records, plus its token total.

    Summing bytes and time rather than averaging rates: the corpora differ in
    size, and a mean would weight a 1 MB corpus like a 6 MB one.

    `None` for an engine that broke on every corpus — it has no rate to
    aggregate, and averaging over the ones it survived would report a number it
    cannot actually deliver.
    """
    records = [r for r in records if not r.get("broken")]
    if not records:
        return None
    out = {}
    for key in ("serial", "par"):
        mb = sum(r[f"{key}_mb"] for r in records)
        secs = sum(r[f"{key}_mb"] / r[f"{key}_mbps"] for r in records if r[f"{key}_mbps"] > 0)
        out[key] = mb / secs if secs > 0 else None
    out["tokens"] = sum(r["tokens"] for r in records)
    out["bytes"] = sum(r["serial_mb"] for r in records) * 1e6
    out["load_ms"] = max(r["load_ms"] for r in records)
    return out


def main():
    title, notes = parse_args(sys.argv[1:])

    # (vocab, family, corpus) -> engine -> record
    cells, seen_engines, skipped = OrderedDict(), [], []
    vocabs, families, corpora = [], [], []
    for doc in read_documents(sys.stdin.read()):
        vocab = doc["vocab"]
        if vocab not in vocabs:
            vocabs.append(vocab)
        for entry in doc.get("skipped", []):
            skipped.append((vocab, entry["engine"], entry["reason"]))
        for rec in doc["results"]:
            if rec["family"] not in families:
                families.append(rec["family"])
            if rec["corpus"] not in corpora:
                corpora.append(rec["corpus"])
            cells.setdefault((vocab, rec["family"], rec["corpus"]), {})[rec["engine"]] = rec
            if rec["engine"] not in seen_engines:
                seen_engines.append(rec["engine"])

    columns = [c for c in PREFERRED if c in seen_engines]
    columns += [c for c in seen_engines if c not in columns]
    ratio_against = columns[1:]
    ordered_families = [f for f in FAMILY_ORDER if f in families]
    ordered_families += [f for f in families if f not in FAMILY_ORDER]

    print(f"# {title}\n")
    for note in notes:
        print(f"{note}\n")

    # Parity as a count, and the failures grouped by engine. One line per
    # check was forty lines of "yes"; naming all nineteen failing cells inline
    # was one unreadable sentence. What a reader needs is which engine
    # disagrees and on what — the corpus list collapses to a count.
    disagreed = OrderedDict()
    total = 0
    for (vocab, family, corpus), engines in cells.items():
        for engine, rec in engines.items():
            if rec.get("broken"):
                continue
            total += 1
            if not rec["agrees"]:
                disagreed.setdefault(engine, OrderedDict()).setdefault(
                    f"{vocab}-{family}", []
                ).append(corpus)
    failed = sum(len(c) for e in disagreed.values() for c in e.values())

    if disagreed:
        print(
            f"**Token ids: {total - failed}/{total} agreed with the oracle.** The rest "
            "are not comparisons — that engine did not produce the same tokens:\n"
        )
        print("| engine | disagreed on | cells |")
        print("|---|---|---:|")
        for engine, where in disagreed.items():
            n = sum(len(c) for c in where.values())
            print(f"| {engine} | {', '.join(where)} | {n} |")
        print()
    else:
        print(f"Token ids: **{total}/{total}** agreed with the oracle, on every corpus.\n")

    # --- the matrix ---------------------------------------------------------
    # Every engine is a column, in every table. The point of this report is
    # splintr against the other crates, so that comparison is the report — not
    # a "closest rival" digest with the rest folded away where nobody opens it.
    #
    # Aggregated over the corpora first, then the same shape per corpus, since
    # the per-script spread is itself a finding: an engine can lead on Latin
    # text and lose by 5x on Han.
    def matrix(rows, key):
        return OrderedDict(
            (label, {name: cell(rec, key) for name, rec in engines.items()})
            for label, engines in rows.items()
        )

    aggregated, load_rows = OrderedDict(), OrderedDict()
    for family in ordered_families:
        for vocab in vocabs:
            by_engine = OrderedDict()
            for (v, f, _), engines in cells.items():
                if v != vocab or f != family:
                    continue
                for engine, rec in engines.items():
                    by_engine.setdefault(engine, []).append(rec)
            if not by_engine:
                continue
            label = f"{vocab}-{family}"
            agg = {name: aggregate(recs) for name, recs in by_engine.items()}
            aggregated[label] = agg
            load_rows[label] = {
                name: (BROKEN if a is None else a["load_ms"]) for name, a in agg.items()
            }

    def agg_table(title, key, unit, fmt, higher_is_better=True):  # noqa: FBT002
        table(
            title,
            OrderedDict(
                (label, {n: (BROKEN if a is None else a[key]) for n, a in agg.items()})
                for label, agg in aggregated.items()
            ),
            columns,
            unit,
            higher_is_better=higher_is_better,
            fmt=fmt,
            ratio_against=ratio_against,
        )

    print(f"## Over all {len(corpora)} corpora\n")
    agg_table("Single-text throughput (one thread)", "serial", "MB/s", "{:,.0f}")
    agg_table("Batch throughput (all cores)", "par", "MB/s", "{:,.0f}")
    table(
        "Load time",
        load_rows,
        columns,
        "ms",
        higher_is_better=False,
        fmt="{:,.0f}",
        ratio_against=ratio_against,
    )

    print(
        f"\nBold is the best cell in its row. A `vs` column is splintr over that "
        f"engine, so above 1.0x is splintr ahead. An empty cell is `{MISSING}` where "
        f"the engine does not offer that shape at all — no such file is published, or "
        f"its loader declined the one that is — and `{BROKEN}` where it accepted the "
        f"vocabulary and then failed on the text.\n"
    )
    print(
        "Throughput is aggregated by summing bytes and summing time, never by "
        "averaging rates: the corpora differ in size, and a mean would weight a 1 MB "
        "corpus like a 6 MB one. Load is the slowest of the runs, and is a one-off "
        "cost paid per process rather than per encode.\n"
    )

    for family in ordered_families:
        if family in FAMILY_NOTES:
            print(f"{FAMILY_NOTES[family]}\n")

    for corpus in corpora:
        print(f"\n## {corpus}\n")
        rows = OrderedDict()
        for family in ordered_families:
            for vocab in vocabs:
                engines = cells.get((vocab, family, corpus))
                if engines:
                    rows[f"{vocab}-{family}"] = engines
        table(
            "Single-text throughput (one thread)",
            matrix(rows, "serial_mbps"),
            columns, "MB/s", higher_is_better=True, fmt="{:,.0f}",
            ratio_against=ratio_against,
        )
        table(
            "Batch throughput (all cores)",
            matrix(rows, "par_mbps"),
            columns, "MB/s", higher_is_better=True, fmt="{:,.0f}",
            ratio_against=ratio_against,
        )
    print()

    borrowed = sorted(
        {
            name
            for engines in cells.values()
            for name, rec in engines.items()
            if not rec.get("broken") and not rec["own_batch"]
        }
    )
    if borrowed:
        verb = "expose" if len(borrowed) > 1 else "exposes"
        print(
            f"{', '.join(borrowed)} {verb} no batch API, so the batch column there "
            "is rayon in the harness rather than a library call.\n"
        )

    # Grouped by reason: seventeen vocabularies times a few engines is a long
    # list of the same three sentences, and the interesting fact is which
    # engines cannot read which shapes, not the repetition.
    # The `x` cells, with the error behind them. The table says an engine broke;
    # this says on what and why, which is the part someone would file a bug from.
    broke = OrderedDict()
    for (vocab, family, _), engines in cells.items():
        for engine, rec in engines.items():
            if rec.get("broken"):
                broke.setdefault((engine, rec.get("reason", "")), set()).add(f"{vocab}-{family}")
    if broke:
        print(f"### Loaded, then failed — the `{BROKEN}` cells ({len(broke)})\n")
        print("| engine | vocabularies | error |")
        print("|---|---|---|")
        for (engine, reason), vs in broke.items():
            print(f"| {engine} | {', '.join(sorted(vs))} | {reason} |")
        print()

    if skipped:
        by_reason = OrderedDict()
        for vocab, engine, reason in skipped:
            by_reason.setdefault((engine, reason), []).append(vocab)
        print(f"### Could not load ({len(skipped)})\n")
        print("| engine | vocabularies | reason |")
        print("|---|---|---|")
        for (engine, reason), vs in by_reason.items():
            print(f"| {engine} | {', '.join(vs)} | {reason} |")


main()
