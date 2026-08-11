"""The vocabulary manifest, read by the workflow and by the Python harness.

    perf_vocabs.py fields <field>...   one line per vocabulary, tab-separated
    perf_vocabs.py fetch <dir>         download every `tokenizer.json`

`.github/perf-vocabs.tsv` is the single list of what perf measures. Everything
that needs to know reads it from here rather than keeping its own copy, so a new
vocabulary is one line in one file.

Vocabularies, not models: timing is a property of the merge ranks and the
corpus, so two models sharing a vocabulary produce one row, not two. The
manifest's own header states the rule and why each absent family is absent.
"""

import os
import sys
import urllib.request

FIELDS = ("name", "ranks", "json", "pattern", "giga")
MANIFEST = os.path.join(os.path.dirname(__file__), os.pardir, "perf-vocabs.tsv")


def vocabs():
    """Every manifest row as a dict, with `-` read as absent (`None`)."""
    out = []
    with open(MANIFEST) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != len(FIELDS):
                raise SystemExit(f"perf-vocabs.tsv: expected {len(FIELDS)} columns, got {len(parts)}: {line!r}")
            out.append({k: (None if v == "-" else v) for k, v in zip(FIELDS, parts)})
    return out


def json_path(directory, vocab):
    """Where `fetch` puts this vocabulary's `tokenizer.json`."""
    return os.path.join(directory, f"{vocab['name']}.json")


def main(argv):
    if not argv:
        raise SystemExit(__doc__)
    if argv[0] == "fields":
        wanted = argv[1:] or list(FIELDS)
        for vocab in vocabs():
            print("\t".join(vocab[f] or "-" for f in wanted))
        return
    if argv[0] == "fetch":
        directory = argv[1]
        os.makedirs(directory, exist_ok=True)
        for vocab in vocabs():
            if not vocab["json"]:
                continue
            dest = json_path(directory, vocab)
            if os.path.exists(dest):
                continue
            # A repository that has moved or gone private must not take the
            # whole run down with it: the harnesses already report a vocabulary
            # they cannot load, and one absent file is a missing row, not a
            # failure.
            try:
                urllib.request.urlretrieve(vocab["json"], dest)
            except Exception as e:  # noqa: BLE001 — any failure is "no file"
                print(f"skip {vocab['name']}: {e}", file=sys.stderr)
                continue
            print(f"{vocab['name']}: {os.path.getsize(dest) / 1e6:.1f} MB", file=sys.stderr)
        return
    raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
