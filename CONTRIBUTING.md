# Contributing

Contributions are welcome.

- **Report a bug** — open an issue with a minimal reproduction. For a tokenization difference, include the vocabulary, the input, what you got and what the reference produced.
- **Suggest a feature** — describe the use case rather than the implementation.
- **Send a pull request** — add tests for new behaviour, run the gates below, and add a `## [Unreleased]` entry in [CHANGELOG.md](CHANGELOG.md) for anything user-visible.

## Development setup

```bash
git clone https://github.com/ml-rust/splintr.git
cd splintr

cargo build --release
cargo build --release --no-default-features   # minimal: no Rayon, no regexr JIT/SIMD

# Python bindings
pip install maturin pytest
maturin develop --release --features python,pcre2
```

## Gates

The same checks CI runs. Everything here must pass before a pull request is ready.

**Run them when a unit of work is finished, not on every commit.** There is deliberately no pre-commit hook: the full suite is ~25 seconds, and a branch that lands as four or five commits would pay it four or five times to re-prove the same thing. CI is the enforcement; these are how you get the answer sooner.

The three that catch nearly everything, cheapest first:

```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo nextest run                          # `cargo test` also works, but see below
```

Run each as its own command rather than chaining them — concurrent `cargo` invocations fight over `target/` and delete each other's binaries mid-run.

The rest, before opening a pull request:

```bash
cargo nextest run --features pcre2         # the optional PCRE2 backend
cargo test --doc                           # doctests
python -m pytest python/tests              # Python bindings (after `maturin develop`)
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
cargo deny --exclude-dev check             # advisories, licenses, sources
python scripts/generate_agent_tokens.py --update-docs --check   # generated tables are current
```

`cargo nextest run` rather than `cargo test`: `.config/nextest.toml` is where the two wall-clock tests are given the machine to themselves and a retry. `cargo test` ignores that file, so under it those two race the rest of the suite for the cores they are measuring and fail on a busy machine with nothing wrong.

**One thing CI checks that is easy to break locally.** A gitignored `.cargo/config.toml` with a `[patch.crates-io]` entry pointing `regexr` at a sibling checkout makes cargo rewrite `Cargo.lock`, stripping that crate's registry `source` and `checksum` into a bare path entry — useless to anyone without your directory layout, and rejected by `--locked` builds and `cargo publish`. Anything that runs cargo does it, including rust-analyzer in the background, so it lands without you deciding to commit it. CI fails on it; to fix:

```bash
mv .cargo/config.toml .cargo/config.toml.bak
cargo fetch
mv .cargo/config.toml.bak .cargo/config.toml
git add Cargo.lock
```

CI additionally builds on Linux, macOS and Windows, compile-checks the `wasm32` targets, and covers every feature combination that ships.

## Correctness against the reference implementations

Unit tests fix behaviour splintr already knows about. Correctness against the real tokenizers is established _differentially_, on three levels — a change that touches loading, pre-tokenization or decoding should exercise all three.

### In CI, with no model files and no Python

`tests/reference_parity.rs` diffs every bundled vocabulary against committed fixtures in `tests/fixtures/pretrained/` on token ids, decoded text **and** the pre-tokenizer split. Those are separate pipelines: pinning only ids leaves byte-level unmapping, byte fallback, the SentencePiece dummy-prefix strip and the pre-tokenizer pattern itself unpinned.

Fixtures are captured by `scripts/extract_reference_cases.py`, which refuses to write one unless the reference provably _is_ the vocabulary splintr embeds:

```bash
# OpenAI vocabularies — `tiktoken`, gated on every mergeable rank
python3 scripts/extract_reference_cases.py --vocab cl100k_base --reference-tiktoken \
    --out-dir tests/fixtures/pretrained

# HF-published vocabularies — `tokenizers`, gated on vocab size and an id sample
python3 scripts/extract_reference_cases.py --vocab llama3 \
    --reference-hf path/to/tokenizer.json --out-dir tests/fixtures/pretrained

# SentencePiece vocabularies — `sentencepiece`, gated on every piece and score
python3 scripts/extract_reference_cases.py --vocab mistral_v2 \
    --reference-spm path/to/tokenizer.model --out-dir tests/fixtures/pretrained
```

`tests/decode_agreement.rs` needs no reference at all: for every bundled vocabulary, every backend and every chunk size, streaming decode plus `flush()` must equal whole-sequence `decode`, and `reset()` must leave a decoder byte-identical to a fresh one.

### Against published tokenizers, on your machine

`.github/scripts/perf_parity.py` sweeps `from_json` across a list of published `tokenizer.json` files spanning the major families, comparing splintr's ids against HuggingFace `tokenizers` over inputs that have historically broken loaders — indentation runs, leading and trailing whitespace, mixed scripts, digit runs. Loader bugs tend to be shaped like "this one vocabulary's JSON is laid out differently", which only a sweep finds.

```bash
python3 .github/scripts/perf_parity.py .cache/vocabs
```

`scripts/verify_external_models.py` covers the same ground for locally downloaded models and the bundled SentencePiece vocabularies. It never silently shrinks: a missing model directory, a missing target file, or an installed `splintr` wheel that is not this checkout each fail the run.

```bash
python3 scripts/verify_external_models.py --models-dir ~/Projects/models
```

### To find new bugs

`scripts/fuzz_reference.py` diffs splintr against `tokenizers`, `transformers` or `tiktoken` — auto-detected per target — using random strings assembled from each vocabulary's _own_ added and special tokens, joined with no separator. That is the shape prose corpora cannot reach and where the bugs live: `lstrip`/`rstrip` on added tokens, the SentencePiece dummy prefix, decoder pipelines. Runs are deterministic via `--seed`, and a failing case is shrunk fragment by fragment to a minimal reproducer.

```bash
# a HuggingFace tokenizer.json (reference auto-detected)
python3 scripts/fuzz_reference.py path/to/tokenizer.json --cases 6250

# a bundled vocabulary against a local model directory (`transformers`)
python3 scripts/fuzz_reference.py mistral_v2=path/to/model-dir --cases 2014

# bundled OpenAI vocabularies (`tiktoken`)
python3 scripts/fuzz_reference.py cl100k_base o200k_base --cases 2000

# GGUF loader against llama.cpp's own .inp/.out fixtures
cargo run --example verify_gguf -- /path/to/extracted-gguf-vocabs
```

CI fuzzes the bundled OpenAI vocabularies on every push with a fixed seed. The other targets need a local model directory no runner has, so those are maintainer-run before a release. Any failure at a seed and case count that previously passed is a regression.

## Benchmarking

The `perf` workflow is manual (`gh workflow run perf.yml`) and compares splintr against the libraries it replaces on the same vocabulary, checking that every engine produces identical ids before reading any timing. Version inputs default to whatever pip resolves that day; pin `splintr_baseline` to a previous release to check for a regression.

Absolute throughput moves with hardware and with the versions being compared against, so quote the run, not a remembered number.

## Releasing

1. Bump the version in `Cargo.toml`, `pyproject.toml`, `python/splintr/__init__.py` and `.version`, and add the matching `CHANGELOG.md` section — the release is gated on that section existing.
2. Push a `v*` tag. `Release Prepare` validates the tag against the version and changelog, runs the full suite, and builds the wheels and sdist.
3. Dispatch `Release` with that tag to publish exactly those artifacts.

`regexr` is a path dependency during local development via a gitignored `.cargo/config.toml` `[patch.crates-io]` entry. Any regexr change splintr depends on must be published **before** splintr's release, or a clean checkout cannot resolve it.
