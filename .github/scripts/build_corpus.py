"""Fetch the benchmark corpora, one file per script or modality.

    build_corpus.py <out-dir> <corpus>...

The corpora are `hf-internal-testing/tokenizers-test-data`, which is what the
upstream `tokenizers` benchmark reads — a row here is therefore comparable with
a row there, which a corpus of our own choosing could never be.

Why several rather than one. A tokenizer's cost is not one number: the same
change measured -17% on English and -2.5% on Chinese, because a scanner walks
ASCII eight bytes at a time and Han one character at a time, and because the
merge loop dominates where the whole-word vocabulary hit does not. A single
English corpus reports the first number and hides the second, so a regression
confined to CJK — or a win confined to it — is invisible. Each corpus is
measured and reported separately for that reason, and never averaged into one.

Each fixture is one long text; the harness wants documents. Splitting at line
boundaries into chunks of roughly `DOC_BYTES` gives both the parallel path
something to distribute and the chunk cache a realistic hit rate — a corpus
built by repeating a seed sentence would measure the cache instead of the
tokenizer, and one giant document would leave every thread but one idle.
"""

import os
import sys
import urllib.request

BASE = "https://huggingface.co/datasets/hf-internal-testing/tokenizers-test-data/resolve/main/fixtures"
# Which subdirectory each corpus lives under. Languages are ISO 639-3 plus a
# script tag; everything else is a modality.
LANGS = {
    "amh_Ethi", "arb_Arab", "ben_Beng", "cmn_Hani", "ell_Grek", "eng_Latn",
    "heb_Hebr", "hin_Deva", "jpn_Jpan", "kat_Geor", "kor_Hang", "rus_Cyrl",
    "tam_Taml", "tha_Thai",
}
DOC_BYTES = 8 << 10


def documents(text):
    """`text` cut at line boundaries into chunks of about `DOC_BYTES`."""
    docs, current, size = [], [], 0
    for line in text.splitlines(keepends=True):
        current.append(line)
        size += len(line)
        if size >= DOC_BYTES:
            docs.append("".join(current))
            current, size = [], 0
    if current:
        docs.append("".join(current))
    return docs


def main(argv):
    out_dir, names = argv[0], argv[1:]
    os.makedirs(out_dir, exist_ok=True)
    for name in names:
        dest = os.path.join(out_dir, f"{name}.txt")
        if os.path.exists(dest):
            continue
        kind = "lang" if name in LANGS else "modalities"
        with urllib.request.urlopen(f"{BASE}/{kind}/{name}.txt") as r:
            text = r.read().decode("utf-8", "replace")
        docs = documents(text)
        with open(dest, "w") as f:
            f.write("\n\x00\n".join(docs))
        print(f"{name}: {len(docs):,} documents, {len(text) / 1e6:.1f} MB", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
