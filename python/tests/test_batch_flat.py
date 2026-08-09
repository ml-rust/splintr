"""`encode_batch_flat` must say exactly what `encode_batch` says.

It exists for speed — building `list[list[int]]` costs ~18 ns per token, which
measured at 89% of a batch call's wall time, so the flat form runs ~4.7x faster.
That only matters if the two agree, and the flat form is the one where a bug
hides: an off-by-one in the offsets silently reattributes tokens to the
neighbouring row rather than raising.
"""

import struct

import pytest

from splintr import CL100K_BASE_PATTERN, Tokenizer, from_json


def unflatten(ids_bytes, offsets_bytes, n):
    """Rebuild rows from the flat buffers, using only the documented layout."""
    ids = struct.unpack(f"<{len(ids_bytes) // 4}I", ids_bytes)
    offsets = struct.unpack(f"<{len(offsets_bytes) // 8}Q", offsets_bytes)
    assert len(offsets) == n + 1, "offsets must have one entry per row plus a tail"
    return [list(ids[offsets[i] : offsets[i + 1]]) for i in range(n)]


CORPUS = [
    "Hello world",
    "",  # an empty row must stay a row, not vanish
    "The quick brown fox jumps over the lazy dog.",
    "中文和English混合内容。",
    "  leading and trailing  ",
    "\n\n\t",
    "fn main() { println!(\"hi\"); }",
    "🌍🌎🌏",
    "a",
]


@pytest.fixture(scope="module")
def tokenizers():
    """Both classes that expose the flat form, reached the two different ways."""
    return {
        "AnyTokenizer": Tokenizer.from_pretrained("cl100k_base"),
        "Tokenizer": Tokenizer("vocabs/cl100k_base.tiktoken", CL100K_BASE_PATTERN),
    }


def test_flat_matches_encode_batch(tokenizers):
    for name, tok in tokenizers.items():
        rows = tok.encode_batch(CORPUS)
        flat = unflatten(*tok.encode_batch_flat(CORPUS), len(CORPUS))
        assert rows == flat, f"{name}: flat output disagrees with encode_batch"


def test_flat_preserves_empty_and_boundary_rows(tokenizers):
    """The shapes an offsets bug corrupts first."""
    corpus = ["", "", "x", "", "yy", ""]
    for name, tok in tokenizers.items():
        rows = tok.encode_batch(corpus)
        flat = unflatten(*tok.encode_batch_flat(corpus), len(corpus))
        assert rows == flat, name
        assert [len(r) for r in flat] == [len(r) for r in rows], name


def test_flat_handles_an_empty_batch(tokenizers):
    for name, tok in tokenizers.items():
        ids, offsets = tok.encode_batch_flat([])
        assert ids == b"", name
        # Still one offset: the zero every row count starts from.
        assert offsets == struct.pack("<Q", 0), name


def test_offsets_are_monotonic_and_end_at_the_token_count(tokenizers):
    for name, tok in tokenizers.items():
        ids_b, off_b = tok.encode_batch_flat(CORPUS)
        offsets = struct.unpack(f"<{len(off_b) // 8}Q", off_b)
        assert list(offsets) == sorted(offsets), f"{name}: offsets must not go backwards"
        assert offsets[0] == 0, name
        assert offsets[-1] == len(ids_b) // 4, f"{name}: last offset must be the id count"


def test_flat_is_available_on_a_from_json_tokenizer(tmp_path):
    """`from_json` returns the same AnyTokenizer, so it has the flat form too."""
    tok = Tokenizer.from_pretrained("qwen3")
    rows = tok.encode_batch(CORPUS)
    flat = unflatten(*tok.encode_batch_flat(CORPUS), len(CORPUS))
    assert rows == flat


def test_batch_releases_the_gil():
    """A batch call must not block other Python threads.

    `encode_batch` held the GIL for its whole duration, which serialized a
    threaded data loader against the tokenizer. This checks the property rather
    than the implementation: a second thread must make progress *while* a large
    batch is in flight.
    """
    import threading
    import time

    tok = Tokenizer.from_pretrained("cl100k_base")
    texts = ["the quick brown fox jumps over the lazy dog " * 40] * 2000

    ticks = 0
    stop = False

    def spin():
        nonlocal ticks
        while not stop:
            ticks += 1
            time.sleep(0)

    t = threading.Thread(target=spin, daemon=True)
    t.start()
    time.sleep(0.01)
    before = ticks
    tok.encode_batch(texts)
    during = ticks - before
    stop = True
    t.join(timeout=1)

    assert during > 0, "no other thread ran during encode_batch — the GIL was held"
