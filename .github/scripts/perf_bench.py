"""Tokenizer performance harness: splintr against the libraries it replaces.

One engine per invocation, emitting a JSON line the report script folds into
tables. Engines are only ever compared inside a *suite*, which fixes the
vocabulary, and the check mode proves they produce identical ids before any
timing is read.

    perf_bench.py <suite> <engine> <label> <spec> [--check]

  suite   name of the vocabulary under test, e.g. cl100k_base
  engine  splintr | tokenizers | tiktoken
  spec    path to a tokenizer.json, or a tiktoken/splintr encoding name
  --check emit token ids instead of timings

Throughput is MB/s over the input bytes rather than tokens/s, matching the
README: bytes are the one quantity every engine sees identically.

Workloads are the three that decide a migration: single-text latency by corpus
shape (per-request cost), batch throughput by batch size (bulk ingest), and
vocabulary load time (process start, serverless cold start).
"""

import json
import os
import statistics
import sys
import time

WARMUP = 2
ITERS = 10
LOAD_ITERS = 5
BATCH_SIZES = (100, 500, 1000)


# --- corpora ----------------------------------------------------------------
# Deterministic and shaped like real traffic rather than one synthetic string:
# pre-tokenizer cost depends heavily on script and punctuation density.

def _texts(seeds, count=1000, repeat=1):
    return [
        "".join(seeds[i % len(seeds)] for _ in range(repeat)) + f"\nrecord_id={i:06d}\n"
        for i in range(count)
    ]


ENGLISH = [
    "Tokenization is on the hot path of every LLM application: prompts, RAG chunks, and token counting.",
    "The quick brown fox jumps over the lazy dog while the sun sets behind the hills.",
    "Retrieval augmented generation splits documents into chunks before embedding them.",
]
CHINESE = [
    "那么，线性代数又是如何来解决这些问题的呢？在代数学中，研究一个集合、一类对象的方法之一，是找出若干个代表元。",
    "机器学习模型的训练过程需要大量的计算资源和高质量的标注数据。",
]
CODE = [
    "def normalize_document(blocks: list[str]) -> list[str]:\n"
    "    return [block.strip() for block in blocks if block.strip()]",
    "impl<T: Clone> Iterator for Windows<T> {\n"
    "    fn next(&mut self) -> Option<Self::Item> { self.inner.next() }\n}",
]
JSON_DOCS = [
    '{"event":"document.segment","page":12,"text":"A quick brown fox jumps over the lazy dog."}',
    '{"id":"a3f9","score":0.8231,"tags":["rag","chunk"],"meta":{"lang":"en","tokens":42}}',
]
MULTILINGUAL = [
    "Unicode coverage: café, naïve, 日本語、한국어、العربية、emoji 🚀 and combining é.",
    "Здравствуй мир, γειά σου κόσμε, שלום עולם, नमस्ते दुनिया.",
]

CORPORA = {
    "english": lambda: _texts(ENGLISH),
    "chinese": lambda: _texts(CHINESE),
    "code": lambda: _texts(CODE),
    "json": lambda: _texts(JSON_DOCS),
    "multilingual": lambda: _texts(MULTILINGUAL),
    # Few long documents rather than many short ones — the RAG-ingest shape,
    # where per-call overhead disappears and the merge loop dominates.
    "long-docs": lambda: _texts(ENGLISH + CHINESE + CODE, count=50, repeat=40),
}

# Batches in the wild are heterogeneous, so the batch axis uses one mixed corpus
# and varies only its size.
MIXED = _texts(ENGLISH + CHINESE + CODE + JSON_DOCS + MULTILINGUAL)


def megabytes(texts):
    return sum(len(t.encode("utf-8")) for t in texts) / (1024 * 1024)


# --- engines ----------------------------------------------------------------


def load_engine(engine, spec):
    """Returns (encode_one, encode_batch). Loading is timed separately."""
    if engine == "splintr":
        import splintr

        tok = (
            splintr.from_json(spec)
            if spec.endswith(".json")
            else splintr.Tokenizer.from_pretrained(spec)
        )
        return (lambda t: tok.encode_raw(t)), (lambda ts: tok.encode_batch(ts))

    if engine == "tokenizers":
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(spec)
        # `encode_batch_fast` skips the offset bookkeeping the plain call does,
        # which nothing here asks for — the same choice gigatoken's own
        # comparison script makes. Ids are then materialised as lists, because
        # that is what every other engine returns.
        batch = getattr(tok, "encode_batch_fast", tok.encode_batch)
        return (
            lambda t: tok.encode(t, add_special_tokens=False).ids,
            lambda ts: [e.ids for e in batch(ts, add_special_tokens=False)],
        )

    if engine == "gigatoken":
        import gigatoken as gt

        tok = gt.Tokenizer.from_json(open(spec).read()) if spec.endswith(".json") else gt.Tokenizer(spec)
        # `encode_batch` hands back an awkward Array, which is cheaper than the
        # Python lists every other engine builds — comparing against it would
        # charge them for object construction gigatoken skips. `encode_batch_list`
        # is the same work ending in plain lists, so the outputs match in kind.
        # `encode` returns a numpy array for the same reason; materialise it.
        # `.tolist()` rather than `list()`: the latter yields numpy scalars, not
        # the Python ints every other engine returns.
        return (lambda t: tok.encode(t).tolist()), (lambda ts: tok.encode_batch_list(ts))

    if engine == "tiktoken":
        import tiktoken

        # `get_encoding` memoises process-wide, so a second load would be a dict
        # lookup while the other engines do real work. Drop the cache so load
        # time means the same thing for everyone. Private API, hence the guard.
        try:
            from tiktoken import registry

            with registry._lock:
                registry.ENCODINGS.clear()
        except Exception:
            pass

        enc = tiktoken.get_encoding(spec)
        # `encode_ordinary` on both paths: it matches splintr's `encode_raw` in
        # treating no text as special, and pairs with `encode_ordinary_batch`.
        # `num_threads` defaults to 8 regardless of the machine, so it is set to
        # the core count — splintr and gigatoken both use every core, and leaving
        # tiktoken on 8 would be measuring the default, not the library.
        threads = os.cpu_count() or 8
        return (
            lambda t: enc.encode_ordinary(t),
            lambda ts: enc.encode_ordinary_batch(ts, num_threads=threads),
        )

    raise SystemExit(f"unknown engine {engine!r}")


# --- workloads --------------------------------------------------------------


def time_best(fn):
    for _ in range(WARMUP):
        fn()
    samples = []
    for _ in range(ITERS):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1e3)
    return statistics.median(samples)


def main():
    suite, engine, label, spec = sys.argv[1:5]
    encode_one, encode_batch = load_engine(engine, spec)

    if "--check" in sys.argv:
        sample = CORPORA["multilingual"]()[:3] + CORPORA["code"]()[:2] + CORPORA["json"]()[:2]
        print(json.dumps({"ids": [encode_one(t) for t in sample]}))
        return

    single = {}
    for name, build in CORPORA.items():
        texts = build()
        ms = time_best(lambda: [encode_one(t) for t in texts])
        single[name] = {
            "ms": ms,
            "mb_per_s": megabytes(texts) / (ms / 1e3),
            "tokens": sum(len(encode_one(t)) for t in texts),
        }

    batch = {}
    for size in BATCH_SIZES:
        texts = MIXED[:size]
        ms = time_best(lambda: encode_batch(texts))
        batch[str(size)] = {"ms": ms, "mb_per_s": megabytes(texts) / (ms / 1e3)}

    load_samples = []
    for _ in range(LOAD_ITERS):
        start = time.perf_counter()
        load_engine(engine, spec)
        load_samples.append((time.perf_counter() - start) * 1e3)

    print(
        json.dumps(
            {
                "suite": suite,
                "engine": engine,
                "label": label,
                "load_ms": statistics.median(load_samples),
                "single": single,
                "batch": batch,
            }
        ),
        flush=True,
    )


main()
