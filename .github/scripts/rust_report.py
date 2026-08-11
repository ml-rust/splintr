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

    # Parity as a count. One line per check was fine for four vocabularies and
    # is forty lines for the manifest — and forty lines of "yes" is exactly the
    # noise that stops anyone reading the one line that says "no".
    checks = [
        (vocab, family, corpus, engine)
        for (vocab, family, corpus), engines in cells.items()
        for engine, rec in engines.items()
        if not rec.get("broken") and not rec["agrees"]
    ]
    total = sum(
        1
        for engines in cells.values()
        for rec in engines.values()
        if not rec.get("broken")
    )
    if checks:
        print(
            f"> **Token ids: {total - len(checks)}/{total} agreed with the oracle. "
            "These did not, and their rows are not comparisons:** "
            + ", ".join(f"{v}-{f}/{c}/{e}" for v, f, c, e in checks)
            + "\n"
        )
    else:
        print(f"Token ids: **{total}/{total}** agreed with the oracle, on every corpus.\n")

    # --- summary ------------------------------------------------------------
    # One row per vocabulary and family: what splintr does, who came closest,
    # and by how much. This is the whole report for most readers.
    print(f"## Summary — single text, one thread, over {len(corpora)} corpora\n")
    print("| vocabulary | splintr | closest rival | ratio | load | bytes/token |")
    print("|---|---:|---|---:|---:|---:|")
    for family in ordered_families:
        for vocab in vocabs:
            group = [
                (engine, rec)
                for (v, f, _), engines in cells.items()
                if v == vocab and f == family
                for engine, rec in engines.items()
            ]
            if not group:
                continue
            by_engine = OrderedDict()
            for engine, rec in group:
                by_engine.setdefault(engine, []).append(rec)
            agg = {name: aggregate(recs) for name, recs in by_engine.items()}
            mine = agg.get("splintr")
            if mine is None or mine["serial"] is None:
                continue
            rivals = [(n, a) for n, a in agg.items() if n != "splintr" and a and a["serial"]]
            best = max(rivals, key=lambda x: x[1]["serial"], default=None)
            ratio = f"{mine['serial'] / best[1]['serial']:.1f}x" if best else MISSING
            rival = f"{best[0]} {best[1]['serial']:,.0f} MB/s" if best else MISSING
            # A rival that took the file and then broke is worth naming here:
            # "no rival" and "every rival crashed" are not the same result.
            crashed = sorted(n for n, a in agg.items() if a is None)
            if crashed:
                rival = (rival + " " if best else "") + f"({BROKEN} {', '.join(crashed)})"
            load = f"{mine['load_ms']:,.0f} ms"
            if best:
                load += f" vs {best[1]['load_ms']:,.0f}"
            bpt = mine["bytes"] / mine["tokens"] if mine["tokens"] else 0
            ahead = not best or mine["serial"] >= best[1]["serial"]
            mbps = f"{mine['serial']:,.0f} MB/s"
            print(
                f"| {vocab}-{family} | {f'**{mbps}**' if ahead else mbps} | {rival} "
                f"| {ratio} | {load} | {bpt:.2f} |"
            )

    print(
        "\nRatio is splintr over the fastest other engine that could load the same "
        "file, so above 1.0x is splintr ahead and a row is bold only where it is. "
        "Load is the one-off cost of building the tokenizer. `bytes/token` is the "
        "vocabulary's compression on this text — identical across engines in a row, "
        "since parity above says they produced the same ids, and there to show what "
        "the throughput bought.\n"
    )
    print(
        f"An empty cell is `{MISSING}` where the engine does not offer that shape at "
        f"all — no such file is published, or its loader declined the one that is — "
        f"and `{BROKEN}` where it accepted the vocabulary and then failed on the "
        "text. Both are reported rather than dropped: an engine missing from a row "
        "is a fact about the engine, and the two facts are not the same one.\n"
    )

    for family in ordered_families:
        if family in FAMILY_NOTES:
            print(f"{FAMILY_NOTES[family]}\n")

    # --- detail -------------------------------------------------------------
    print("<details>")
    print(f"<summary>Every corpus and engine ({len(cells)} cells)</summary>\n")

    for corpus in corpora:
        print(f"\n### {corpus}\n")
        rows = OrderedDict()
        for family in ordered_families:
            for vocab in vocabs:
                engines = cells.get((vocab, family, corpus))
                if engines:
                    rows[f"{vocab}-{family}"] = engines
        table(
            "Single-text throughput (one thread)",
            OrderedDict((k, {n: cell(r, "serial_mbps") for n, r in v.items()}) for k, v in rows.items()),
            columns,
            "MB/s",
            higher_is_better=True,
            fmt="{:,.0f}",
            ratio_against=ratio_against,
        )
        table(
            "Batch throughput (all cores)",
            OrderedDict((k, {n: cell(r, "par_mbps") for n, r in v.items()}) for k, v in rows.items()),
            columns,
            "MB/s",
            higher_is_better=True,
            fmt="{:,.0f}",
            ratio_against=ratio_against,
        )

    print("\n</details>\n")

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
