"""Extract WikiText-103 documents into a flat corpus file.

The Rust benchmark needs *real prose*, not repeated seed sentences: splintr
keeps an LRU chunk cache, so a corpus built by repetition would hand it a
near-100% hit rate and stop measuring the tokenizer at all. WikiText is used for
the same reason gigatoken's own benchmark uses OpenWebText — it is the shape of
text these tokenizers actually meet.

    build_corpus.py <parquet> <out> [max_mb]

Documents are separated by a NUL line, which cannot occur in the text itself.
"""

import sys

import pyarrow.parquet as pq

src, out = sys.argv[1], sys.argv[2]
max_bytes = int(float(sys.argv[3]) * 1e6) if len(sys.argv) > 3 else None

lines = pq.read_table(src, columns=["text"]).column("text").to_pylist()

# A WikiText article starts with a level-1 heading (" = Title = "), which is the
# only reliable document boundary in the flat line stream.
docs, current = [], []
for line in lines:
    if line.startswith(" = ") and not line.startswith(" = = ") and current:
        docs.append("".join(current))
        current = []
    current.append(line)
if current:
    docs.append("".join(current))

docs = [d for d in docs if len(d) > 200]

total, kept = 0, []
for doc in docs:
    if max_bytes and total >= max_bytes:
        break
    kept.append(doc)
    total += len(doc.encode())

with open(out, "w") as f:
    f.write("\n\x00\n".join(kept))
print(f"{len(kept):,} documents, {total / 1e6:.1f} MB", file=sys.stderr)
