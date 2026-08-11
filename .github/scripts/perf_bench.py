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

Workloads are the ones that decide a migration: single-text latency by corpus
shape (per-request cost), batch throughput by batch size (bulk ingest), and
vocabulary load time (process start, serverless cold start).

Batch is measured on two axes, never mixed. The *list* axis has every engine
return `list[list[int]]`, which is what most callers use and the only form some
engines have. The *flat* axis has the engines that can return ids as one
contiguous buffer do so — splintr's `encode_batch_flat`, gigatoken's
`encode_batch`. Building Python ints measured at roughly 18 ns per token, which
is most of a batch call, so putting a buffer form in the same column as a list
form would report the object construction the other engine skipped as if it
were tokenizer speed.
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


# The pre-tokenizer expression each rank-file suite is defined by, keyed by
# **splintr's own name for it** — the name `.github/perf-vocabs.tsv` records
# against each vocabulary. Keyed that way rather than by vocabulary so that a
# new vocabulary reusing an existing expression (glm reuses Llama 3's) is one line
# in the manifest and nothing here.
#
# These are `splintr`'s constants, copied out verbatim — a hand-typed cl100k
# pattern here silently benchmarked a *different* pre-tokenizer for all three
# engines at once, which parity could not catch because they all agreed on the
# wrong answer, and which only showed up as splintr mysteriously losing its
# scanner. `--verify-patterns` re-checks every entry against those constants;
# the workflow runs it before timing.
#
# They cannot simply be imported: the other engines run in their own venvs,
# which do not have splintr installed.
RANK_FILE_PATTERNS = {
    'CL100K_BASE_PATTERN': "'(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s+$|\\s*[\\r\\n]|\\s+(?!\\S)|\\s",
    'O200K_BASE_PATTERN': "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    'QWEN2_PATTERN': "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    'LLAMA3_PATTERN': "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    'MISTRAL_V3_PATTERN': "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
    'GPT2_PATTERN': "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+",
    'KIMI_PATTERN': "[\\p{Han}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}&&[^\\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
}


def _vocab_of(suite):
    """The vocabulary a suite name refers to, dropping the `-json`/`-ranks` family."""
    return (suite or "").rsplit("-", 1)[0] if (suite or "").endswith(("-json", "-ranks")) else suite


def _pattern_of(suite):
    """The transcribed expression for a rank-file suite, via the manifest.

    The manifest is the one place that says which expression a vocabulary is
    defined by; looking it up here means a new rank family needs no edit to
    this file unless it also brings a new expression.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import perf_vocabs

    name = _vocab_of(suite)
    for vocab in perf_vocabs.vocabs():
        if vocab["name"] == name:
            return RANK_FILE_PATTERNS.get(vocab["pattern"])
    return None


# gigatoken's own name for each suite's pre-tokenization scheme, for the
# `ranks:` path. Never inferred from the file name: gigatoken looks the scheme up
# by name and an unrecognised one mistokenizes rather than failing, so a suite
# with no entry here is refused instead.
GIGATOKEN_SCHEMES = {
    "kimi": "kimi",
    "cl100k": "cl100k",
    "o200k": "o200k",
    "qwen3": "qwen2",
}


def _flat(tok, name):
    """An engine's buffer-returning batch call, or None if it has none.

    Looked up rather than called directly so a pinned older `splintr_baseline`
    — which predates `encode_batch_flat` — reports "no flat form" instead of
    crashing the whole round.
    """
    fn = getattr(tok, name, None)
    return (lambda ts: fn(ts)) if fn is not None else None


def load_engine(engine, spec, suite=None):
    """Returns (encode_one, encode_batch, encode_batch_flat|None).

    The third slot is the engine's *buffer* batch form: ids in one contiguous
    block the caller wraps zero-copy, instead of a `list[list[int]]`. It is a
    separate axis because only some engines have one, and mixing it into the
    list column would charge the others for object construction it skips.
    Loading is timed separately.
    """
    if engine == "splintr":
        import splintr

        if spec.startswith("ranks:"):
            pattern = _pattern_of(suite)
            if pattern is None:
                raise SystemExit(f"no pre-tokenizer pattern recorded for suite {suite!r}")
            tok = splintr.Tokenizer(spec[len("ranks:") :], pattern)
            return (
                (lambda t: tok.encode(t)),
                (lambda ts: tok.encode_batch(ts)),
                _flat(tok, "encode_batch_flat"),
            )

        tok = (
            splintr.from_json(spec)
            if spec.endswith(".json")
            else splintr.Tokenizer.from_pretrained(spec)
        )
        return (
            (lambda t: tok.encode_raw(t)),
            (lambda ts: tok.encode_batch(ts)),
            _flat(tok, "encode_batch_flat"),
        )

    if engine == "tokenizers":
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(spec)
        # `encode_batch_fast` skips the offset bookkeeping the plain call does,
        # which nothing here asks for — the same choice gigatoken's own
        # comparison script makes. Ids are then materialised as lists, because
        # that is what every other engine returns.
        batch = getattr(tok, "encode_batch_fast", tok.encode_batch)
        # No buffer form: `encode_batch_fast` hands back `Encoding` objects, and
        # reaching `.ids` on each is the list construction, not a way around it.
        return (
            lambda t: tok.encode(t, add_special_tokens=False).ids,
            lambda ts: [e.ids for e in batch(ts, add_special_tokens=False)],
            None,
        )

    if engine == "gigatoken":
        import gigatoken as gt

        # Only a `tokenizer.json`: a bare name is a HuggingFace repo id to
        # gigatoken, not a tiktoken encoding, and `from_tiktoken` wants a path to
        # a rank file. Both would compare against a different vocabulary than the
        # one the suite names, so refuse rather than quietly measure the wrong
        # thing — the OpenAI suites run gigatoken through their `.json` form.
        # A bare rank file, the same `ranks:<path>` form the tiktoken engine
        # takes. gigatoken carries named pre-tokenizer schemes of its own, so it
        # can read a vocabulary published without a `tokenizer.json` — which is
        # what lets the Kimi suite be a three-way comparison instead of leaving
        # splintr measured against one library.
        if spec.startswith("ranks:"):
            scheme = GIGATOKEN_SCHEMES.get(_vocab_of(suite))
            if scheme is None:
                raise SystemExit(
                    f"no gigatoken pre-tokenizer scheme recorded for suite {suite!r}; "
                    "naming the wrong one mistokenizes silently, so it is never guessed"
                )
            tok = gt.Tokenizer.from_tiktoken(spec[len("ranks:") :], pretokenizer=scheme)
            return (
                (lambda t: tok.encode(t).tolist()),
                (lambda ts: tok.encode_batch_list(ts)),
                _flat(tok, "encode_batch"),
            )

        if not spec.endswith(".json"):
            raise SystemExit(
                f"gigatoken needs a tokenizer.json or a ranks: path, got {spec!r}"
            )
        tok = gt.Tokenizer.from_json(open(spec).read())
        # `encode_batch` hands back an awkward Array, which is cheaper than the
        # Python lists every other engine builds — comparing against it would
        # charge them for object construction gigatoken skips. `encode_batch_list`
        # is the same work ending in plain lists, so the outputs match in kind.
        # `encode` returns a numpy array for the same reason; materialise it.
        # `.tolist()` rather than `list()`: the latter yields numpy scalars, not
        # the Python ints every other engine returns. That skipped Array *is*
        # gigatoken's buffer form, so it goes in the third slot, where splintr's
        # `encode_batch_flat` meets it on equal terms.
        return (
            (lambda t: tok.encode(t).tolist()),
            (lambda ts: tok.encode_batch_list(ts)),
            _flat(tok, "encode_batch"),
        )

    if engine == "tiktoken":
        import tiktoken

        # A raw rank file rather than a registered encoding name. Kimi is the
        # case: Moonshot publishes `tiktoken.model` plus a `tokenization_kimi.py`
        # and no `tokenizer.json`, so there is no encoding name to look up and no
        # file HuggingFace or gigatoken can read. Building the reference the way
        # Moonshot's own tokenizer does — those ranks, that `pat_str` — is the
        # only way to put tiktoken beside splintr on this vocabulary at all.
        if spec.startswith("ranks:"):
            from tiktoken.load import load_tiktoken_bpe

            path = spec[len("ranks:") :]
            pattern = _pattern_of(suite)
            if pattern is None:
                raise SystemExit(f"no pre-tokenizer pattern recorded for suite {suite!r}")
            enc = tiktoken.Encoding(
                name=suite,
                pat_str=pattern,
                mergeable_ranks=load_tiktoken_bpe(path),
                special_tokens={},
            )
            threads = os.cpu_count() or 8
            return (
                lambda t: enc.encode_ordinary(t),
                lambda ts: enc.encode_ordinary_batch(ts, num_threads=threads),
                None,  # tiktoken returns lists and nothing else
            )

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
            None,
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


def verify_patterns():
    """Check `RANK_FILE_PATTERNS` against splintr's own constants.

    Run in the splintr venv only — it is the one that has them. Exits non-zero
    on any mismatch, because a wrong pattern here does not fail parity: every
    engine is handed the same string, so they agree with each other while all
    three tokenize the wrong vocabulary.
    """
    import splintr

    # Every entry is checked, by asking the module for the constant of that
    # name. Naming them individually let the table grow entries nothing
    # verified — which is exactly the state a transcription must never be
    # allowed to reach.
    bad, checked = 0, 0
    for name, got in RANK_FILE_PATTERNS.items():
        want = getattr(splintr, name, None)
        if want is None:
            print(f"note: splintr exports no {name} to check against", file=sys.stderr)
            continue
        checked += 1
        if got != want:
            bad += 1
            print(f"MISMATCH {name}:\n  splintr: {want!r}\n  table:   {got!r}", file=sys.stderr)
    unchecked = len(RANK_FILE_PATTERNS) - checked
    if bad:
        sys.exit(f"{bad} pattern(s) differ from splintr's constants")
    if unchecked:
        sys.exit(f"{unchecked} pattern(s) have no splintr constant to check against")
    print(f"ok    {checked} rank-file patterns match splintr's constants")


def main():
    if "--verify-patterns" in sys.argv:
        verify_patterns()
        return
    suite, engine, label, spec = sys.argv[1:5]
    encode_one, encode_batch, encode_batch_flat = load_engine(engine, spec, suite)

    if "--check" in sys.argv:
        sample = CORPORA["multilingual"]()[:3] + CORPORA["code"]()[:2] + CORPORA["json"]()[:2]
        # Both paths. Three of the four tables time the batch call and nothing
        # checked it, so an engine whose batch disagreed with its own
        # single-text call — or returned a different kind of object from it —
        # would have been timed as though it answered the same question as
        # everyone else. Rows are materialised as lists so an engine handing
        # back a view or an array is compared by value.
        print(
            json.dumps(
                {
                    "ids": [encode_one(t) for t in sample],
                    "batch_ids": [list(row) for row in encode_batch(sample)],
                }
            )
        )
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
    flat = {}
    for size in BATCH_SIZES:
        texts = MIXED[:size]
        ms = time_best(lambda: encode_batch(texts))
        batch[str(size)] = {"ms": ms, "mb_per_s": megabytes(texts) / (ms / 1e3)}
        if encode_batch_flat is not None:
            ms = time_best(lambda: encode_batch_flat(texts))
            flat[str(size)] = {"ms": ms, "mb_per_s": megabytes(texts) / (ms / 1e3)}

    load_samples = []
    for _ in range(LOAD_ITERS):
        start = time.perf_counter()
        load_engine(engine, spec, suite)
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
                # Empty for engines with no buffer form, which the report reads
                # as "cannot appear in that table" rather than "scored zero".
                "flat": flat,
            }
        ),
        flush=True,
    )


main()
