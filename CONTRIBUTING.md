# Contributing

Contributions are welcome.

- **Report a bug**: open an issue with a minimal reproduction. For a tokenization difference, include the vocabulary, the input, your output, and the reference output.
- **Suggest a feature**: describe the use case, not the implementation.
- **Send a pull request**: add tests for new behavior, and run the gates below.
- Add a `## [Unreleased]` entry in [CHANGELOG.md](CHANGELOG.md) for each user-visible change.

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

The gates match the checks CI runs. Every gate must pass before a pull request is ready.

Run the gates when a unit of work is finished, not on every commit. The repo has no pre-commit hook on purpose. The full suite takes about 25 seconds. A branch with four or five commits pays that cost each time. CI enforces the gates. Run them locally to get the answer sooner.

The three gates that catch nearly everything, cheapest first:

```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo nextest run                          # `cargo test` also works, but see below
```

Run each command on its own. Concurrent `cargo` invocations fight over `target/` and delete each other's binaries mid-run.

Run the rest before you open a pull request:

```bash
cargo nextest run --features pcre2         # the optional PCRE2 backend
cargo test --doc                           # doctests
python -m pytest python/tests              # Python bindings (after `maturin develop`)
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
cargo deny --exclude-dev check             # advisories, licenses, sources
python scripts/generate_agent_tokens.py --update-docs --check   # generated tables are current
```

Use `cargo nextest run`, not `cargo test`. `.config/nextest.toml` gives the two wall-clock tests exclusive use of the machine and a retry. `cargo test` ignores that file. Without it, those two tests race the rest of the suite for the cores they measure. On a busy machine, they then fail with no real defect.

### Known local trap: `Cargo.lock` rewrites

CI checks for one error that is easy to cause locally. A gitignored `.cargo/config.toml` with a `[patch.crates-io]` entry that points `regexr` at a sibling checkout makes cargo rewrite `Cargo.lock`. Cargo strips the registry `source` and `checksum` for that crate and writes a bare path entry. That entry works only with your directory layout. `--locked` builds and `cargo publish` reject it.

Any command that runs cargo triggers the rewrite, including rust-analyzer in the background. The rewrite happens without a decision from you. CI fails on it. To fix it:

```bash
mv .cargo/config.toml .cargo/config.toml.bak
cargo fetch
mv .cargo/config.toml.bak .cargo/config.toml
git add Cargo.lock
```

CI also builds on Linux, macOS, and Windows. It compile-checks the `wasm32` targets and covers every feature combination that ships.

## Correctness against reference implementations

Unit tests fix behavior that splintr already knows about. Correctness against real tokenizers comes from differential tests on three levels. A change to loading, pre-tokenization, or decoding must exercise all three.

### In CI, without model files or Python

`tests/reference_parity.rs` diffs each bundled vocabulary against committed fixtures in `tests/fixtures/pretrained/`. It compares three outputs: token ids, decoded text, and the pre-tokenizer split. The three are separate pipelines. Pinning only ids leaves byte-level unmapping, byte fallback, the SentencePiece dummy-prefix strip, and the pre-tokenizer pattern unpinned.

`scripts/extract_reference_cases.py` captures the fixtures. It refuses to write a fixture unless the reference is provably the vocabulary splintr embeds:

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

`tests/decode_agreement.rs` needs no reference. For each bundled vocabulary, backend, and chunk size, streaming decode plus `flush()` must equal whole-sequence `decode`. `reset()` must leave a decoder byte-identical to a fresh one.

### Against published tokenizers, on your machine

`.github/scripts/perf_parity.py` runs `from_json` across a list of published `tokenizer.json` files from the major model families. It compares splintr's ids with HuggingFace `tokenizers`. The inputs include indentation runs, leading and trailing whitespace, mixed scripts, and digit runs. Loader bugs usually appear when one vocabulary's JSON is laid out differently. Only a sweep finds them.

```bash
python3 .github/scripts/perf_parity.py .cache/vocabs
```

`scripts/verify_external_models.py` covers locally downloaded models and the bundled SentencePiece vocabularies. It never shrinks silently. It fails the run on any of these:

- A missing model directory.
- A missing target file.
- An installed `splintr` wheel that is not this checkout.

```bash
python3 scripts/verify_external_models.py --models-dir ~/Projects/models
```

### To find new bugs

`scripts/fuzz_reference.py` diffs splintr against `tokenizers`, `transformers`, or `tiktoken`. It auto-detects the reference per target. It builds random strings from each vocabulary's own added and special tokens, joined with no separator. Prose corpora rarely produce that shape. Bugs hide there, in `lstrip` and `rstrip` on added tokens, the SentencePiece dummy prefix, and decoder pipelines. Runs are deterministic with `--seed`. A failing case shrinks fragment by fragment to a minimal reproducer.

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

CI fuzzes the bundled OpenAI vocabularies on every push with a fixed seed. The other targets need a local model directory that no runner has. Maintainers run those before a release. An error at a seed and case count that passed before is a regression.

## Benchmarking

The `perf` workflow is manual. Run it with `gh workflow run perf.yml`. It compares splintr with the libraries it replaces on the same vocabulary. It checks that every engine produces identical ids before it reads any timing. Version inputs default to the versions pip resolves that day. Pin `splintr_baseline` to a previous release to check for a regression.

Absolute throughput depends on hardware and on the versions compared. Quote a specific run, not a remembered number.

## Releasing

1. Bump the version in `Cargo.toml`, `pyproject.toml`, `python/splintr/__init__.py`, and `.version`. Add the matching `CHANGELOG.md` section. The release requires that section.
2. Push a `v*` tag. `Release Prepare` checks the tag against the version and changelog, runs the full suite, and builds the wheels and sdist.
3. Dispatch `Release` with that tag to publish exactly those artifacts.

`regexr` is a path dependency during local development. A gitignored `.cargo/config.toml` `[patch.crates-io]` entry sets it. Publish any regexr change that splintr depends on before the splintr release. Otherwise a clean checkout cannot resolve it.
