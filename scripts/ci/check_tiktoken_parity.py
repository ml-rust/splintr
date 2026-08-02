#!/usr/bin/env python3
"""Assert the installed splintr extension still agrees with tiktoken, id for id.

    python scripts/ci/check_tiktoken_parity.py

Run by the `python` job in .github/workflows/test.yml against a freshly built
extension module. The pytest suite fixes behaviour splintr already knows about;
this checks the OpenAI vocabularies against the reference implementation, so a
change that is self-consistently wrong still fails.

Deliberately small and dependency-light — the exhaustive differential coverage
lives in scripts/fuzz_reference.py, which needs a seed, a case count, and far
more wall clock than a per-PR job should spend.
"""

from __future__ import annotations

import sys

import tiktoken

import splintr

# Text shapes whose tokenization differs between vocabularies and where the
# pre-tokenizer split actually matters: contractions, leading/trailing space
# runs, digits, CJK, emoji (multi-byte, and split across tokens), code, and the
# newline handling around a fenced block.
CORPUS = [
    "",
    " ",
    "Hello, world!",
    "  leading and trailing whitespace   ",
    "don't you'd we'll I'm they're it's",
    "1234567890 3.14159 0x1F 1_000_000",
    "日本語のテキストとハングルの한국어",
    "emoji: 🙂👨‍👩‍👧‍👦🇲🇾 and a lone surrogate-free ✓",
    "def f(x: int) -> int:\n    return x ** 2  # comment\n",
    '{"key": ["value", 1, null], "nested": {"a": true}}',
    "```python\nprint('hi')\n```\n\nParagraph after a fence.\n",
    "Mixed CASE and punctuation... plus -- dashes, (parens) [brackets] {braces}",
    "a" * 1000,
]

VOCABULARIES = ["cl100k_base", "o200k_base"]

failures = 0


def fail(message: str) -> None:
    global failures
    failures += 1
    print(f"FAIL  {message}")


for name in VOCABULARIES:
    tok = splintr.Tokenizer.from_pretrained(name)
    reference = tiktoken.get_encoding(name)

    for text in CORPUS:
        # `encode_ordinary` is the comparable mode: tiktoken's default `encode`
        # rejects special-token spellings rather than matching them, and
        # splintr's `encode` on a loaded tokenizer matches them.
        ours = tok.encode_ordinary(text)
        theirs = list(reference.encode_ordinary(text))
        excerpt = text[:40].replace("\n", "\\n")
        if ours != theirs:
            fail(f"{name}: encode mismatch on {excerpt!r}\n  splintr : {ours}\n  tiktoken: {theirs}")
            continue
        if tok.decode(ours) != text:
            fail(f"{name}: decode round-trip lost {excerpt!r}")

    # The streaming decoder must produce exactly what the batch decoder does,
    # one token at a time, without splitting a multi-byte character.
    for text in CORPUS:
        decoder = tok.streaming_decoder()
        chunks = [chunk for token in tok.encode_ordinary(text) if (chunk := decoder.add_token(token))]
        chunks.append(decoder.flush())
        if "".join(chunks) != text:
            excerpt = text[:40].replace("\n", "\\n")
            fail(f"{name}: streaming decode lost {excerpt!r}")

    print(f"ok    {name}: {len(CORPUS)} texts, encode + decode + streaming")

if failures:
    print(f"\n{failures} parity check(s) failed", file=sys.stderr)
    raise SystemExit(1)

print("\ntiktoken parity: ok")
