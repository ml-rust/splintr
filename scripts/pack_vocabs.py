#!/usr/bin/env python3
"""Pack `vocabs/*.tiktoken` into the binary `.splv` form the crate embeds.

# Why a second format exists

A `.tiktoken` file is `base64(token bytes) rank` per line. Base64 spends four
characters on every three bytes, and the rank is written in decimal — so the
text form is about 47% larger than the information it carries. Across nine rank
files that is ~23 MiB of source, and crates.io rejects an upload over 10 MiB.
Version 0.16.0 packaged at 10.1 MiB and was refused.

Packing recovers ~1.1 MB of the *compressed* payload, which is 7x the overage,
so adding the next vocabulary does not put the release back over the line. It
also removes base64 decoding from `from_pretrained`, which is load-path work the
text form paid on every process start.

# Why the text files stay

`vocabs/*.tiktoken` is a documented public format — `Tokenizer::from_file` reads
it, `docs/vocabularies.md` names those paths, and `.github/workflows/perf.yml`
hands the same files to tiktoken-rs and gigatoken so the benchmark compares
engines on identical ranks. Converting in place would break all three. So the
text form remains the interchange format and the repo's source of truth; `.splv`
is a build artifact of it, committed because a published crate cannot generate
files it does not ship.

`tests/vocab_packed_parity.rs` is what keeps the two honest: it parses both and
fails if any vocabulary disagrees by a single rank. Regenerate with:

    python scripts/pack_vocabs.py

# Format

    magic    8 bytes   b"SPLNTRV1"
    count    u32 LE    number of entries
    entries  count x   varint(rank), varint(len), len raw token bytes

Varints are unsigned LEB128. Ranks are written absolutely rather than implied by
position: all nine bundled vocabularies happen to be contiguous, but nothing in
the tiktoken format promises that, and a positional format would silently
renumber a vocabulary with a gap instead of refusing it.
"""

import base64
import glob
import os
import sys

MAGIC = b"SPLNTRV1"


def varint(n: int) -> bytes:
    out = bytearray()
    while True:
        byte = n & 0x7F
        n >>= 7
        out.append(byte | (0x80 if n else 0))
        if not n:
            return bytes(out)


def pack(text: bytes, path: str) -> bytes:
    """Convert one `.tiktoken` file's bytes into the packed form."""
    entries = bytearray()
    count = 0
    for lineno, line in enumerate(text.split(b"\n"), 1):
        if not line:
            continue
        # `rsplit` on a literal space, not `.split()`: whisper's rank 50256 is
        # the *empty* token, so its line is " 50256" and whitespace-splitting
        # would silently drop the token and read the rank as the token.
        sep = line.rfind(b" ")
        if sep < 0:
            raise SystemExit(f"{path}:{lineno}: no space separator")
        token = base64.b64decode(line[:sep])
        rank = int(line[sep + 1 :])
        entries += varint(rank) + varint(len(token)) + token
        count += 1
    if count == 0:
        raise SystemExit(f"{path}: no entries")
    return MAGIC + count.to_bytes(4, "little") + bytes(entries)


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sources = sorted(glob.glob(os.path.join(root, "vocabs", "*.tiktoken")))
    if not sources:
        raise SystemExit("no vocabs/*.tiktoken found")

    total_text = total_packed = 0
    for src in sources:
        with open(src, "rb") as fh:
            text = fh.read()
        packed = pack(text, src)
        dst = src[: -len(".tiktoken")] + ".splv"
        with open(dst, "wb") as fh:
            fh.write(packed)
        total_text += len(text)
        total_packed += len(packed)
        name = os.path.basename(src)
        print(f"{name:34s} {len(text):>10,} -> {len(packed):>10,}")

    saved = total_text - total_packed
    print(f"\n{'total':34s} {total_text:>10,} -> {total_packed:>10,}  (-{saved:,})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
