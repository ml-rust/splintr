"""Summarise the Python suite's parity sweep as a count, not a list.

    perf_check.py <ids-dir>

Each file in `<ids-dir>` is `<engine>__<vocab>__<family>.json`, written by
`perf_bench.py --check`, or `<same>.fail` holding the error from an engine that
could not load that vocabulary at all.

Two questions are asked of them, and one line of output is spent on each unless
something fails:

  * every engine against the family's oracle — HuggingFace `tokenizers` for
    `json`, since that is the implementation the format is defined by, and
    `tiktoken` for `ranks`, since that is what a `.tiktoken` file means;
  * every engine against *itself*, single-text against batch. Three of the four
    timing tables read the batch call, and an engine whose batch disagrees with
    its own single-text path was being timed as though it had answered the same
    question as everyone else.

Printing one `- ok` line per check was fine for four vocabularies. For the whole
manifest it is forty lines of "yes", which is exactly the noise that stops
anyone noticing the one that says "no". So: a count, then only what failed.

Writes a `PARITY` heredoc to `$GITHUB_ENV` when set, and exits non-zero if any
check failed.
"""

import json
import os
import sys
from collections import OrderedDict

ORACLE = {"json": "tokenizers", "ranks": "tiktoken"}


def load(path):
    with open(path) as f:
        d = json.load(f)
    return d["ids"], d["batch_ids"]


def main(argv):
    ids_dir = argv[0]
    # (vocab, family) -> engine -> path
    cells, unloadable = OrderedDict(), []
    for name in sorted(os.listdir(ids_dir)):
        stem, _, ext = name.rpartition(".")
        engine, _, rest = stem.partition("__")
        vocab, _, family = rest.rpartition("__")
        if ext == "fail":
            with open(os.path.join(ids_dir, name)) as f:
                unloadable.append((engine, vocab, family, f.read().strip().splitlines()[-1:]))
            continue
        if ext != "json":
            continue
        cells.setdefault((vocab, family), OrderedDict())[engine] = os.path.join(ids_dir, name)

    passed, failed = 0, []
    for (vocab, family), engines in cells.items():
        oracle = ORACLE.get(family)
        reference = engines.get(oracle)
        for engine, path in engines.items():
            ids, batch_ids = load(path)
            passed += 1
            if ids != batch_ids:
                failed.append(f"{vocab}-{family}: {engine} batch disagrees with its own single-text call")
                passed -= 1
                continue
            if reference is None or engine == oracle:
                continue
            passed += 1
            if (ids, batch_ids) != load(reference):
                failed.append(f"{vocab}-{family}: {engine} != {oracle}")
                passed -= 1

    total = passed + len(failed)
    lines = []
    if failed:
        lines.append(f"**{passed}/{total} checks passed. These did not:**")
        lines += [f"- {f}" for f in failed]
    else:
        lines.append(f"**{total}/{total}** — every engine matched its family's oracle and its own batch call.")
    # An engine that cannot read a file is a fact about the engine, not a
    # failure of the run — but it must be visible, or its absent row looks like
    # an oversight.
    if unloadable:
        lines.append("")
        lines.append(f"{len(unloadable)} could not load their vocabulary at all:")
        for engine, vocab, family, why in unloadable:
            lines.append(f"- {vocab}-{family}: {engine} — {' '.join(why)[:120]}")

    report = "\n".join(lines)
    print(report)
    if env := os.environ.get("GITHUB_ENV"):
        with open(env, "a") as f:
            f.write(f"PARITY<<PARITY_EOF\n{report}\nPARITY_EOF\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
