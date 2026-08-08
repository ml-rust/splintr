# Performance Benchmarks

All figures on this page were measured by the scripts in `benchmarks/` — `benchmark_batch.py` for batch throughput, `benchmark_single.py` for single-text latency — on an AMD Ryzen 9 5900X with 24 logical cores (Linux), against tiktoken 0.8.0, Hugging Face tokenizers 0.22.1, and TokenDagger 0.1.1 on CPython 3.12.

Absolute throughput moves with the hardware, the CPU architecture and the versions being compared against, so every figure here describes one run rather than a constant. Run the scripts to get your own.

The [README](../README.md#performance) carries a more recent run of the same machine against current library versions (tiktoken 0.13.0, tokenizers 0.23.1), so its absolute numbers are higher than this page's; where they disagree, the README's are the newer measurement.

These scripts compare each library on the vocabulary it ships with, which is not always the same vocabulary — the HuggingFace figures are `gpt2` while splintr, tiktoken and TokenDagger run `cl100k_base`. For a strict like-for-like measurement, the manually triggered `perf` workflow (`gh workflow run perf.yml`) pins every engine to the same vocabulary and verifies they produce identical ids before it reads any timing — its [run history](https://github.com/ml-rust/splintr/actions/workflows/perf.yml) carries the most recent report.

Every figure is the **default pure-Rust `regexr` backend** — what `pip install splintr-rs` gives you, since published wheels omit the optional `pcre2` feature. See [Regex Backends](#regex-backends) for what PCRE2 changes.

The charts below were plotted from an earlier run on different hardware and against different library versions, so their absolute axes do not match the figures quoted in the text. Where they disagree, the numbers in the text are measured; charts are kept for shape only, not values.

## Single Text Encoding

For single texts, splintr encodes **~2-2.6x faster** than tiktoken across five text types (short prose, medium prose, long prose, Python code, multilingual):

![Single Text Encoding Comparison](../images/benchmark_single.png)

**Latency by content type:**

![Latency Comparison](../images/benchmark_single_latency.png)

Consistent low latency across Python code, JSON, English prose, and Chinese text makes splintr ideal for interactive applications and real-time processing.

## Batch Encoding

Splintr parallelizes across texts, where the widest gap appears:

![Batch Speedup vs Tiktoken](../images/benchmark_batch_speedup.png)

The speedup grows with batch size, as the fixed parallelization overhead is amortized over more work.

## Design Decision: Sequential by Default

Splintr uses **sequential encoding for single texts** and **parallel encoding across batches** based on empirical benchmarking:

![Sequential vs Rayon Internal Parallelization](../images/benchmark_splintr.png)

**Key findings:**

- Sequential is faster for texts up to ~1MB (typical LLM prompts and documents)
- Rayon's parallelization overhead only pays off at ~1MB+ text sizes
- Most real-world inputs are well under 1MB
- `encode()` uses sequential processing for optimal single-text performance
- `encode_batch()` parallelizes across multiple texts for maximum throughput
- `encode_rayon()` available for the rare cases where you have >1MB single texts

This architecture ensures splintr is optimized for the most common tokenization patterns in LLM applications.

## Running Benchmarks Yourself

```bash
# Clone and install
git clone https://github.com/ml-rust/splintr.git
cd splintr
pip install -e .
pip install tiktoken tokenizers tokendagger

# The two scripts that produced the figures above
python benchmarks/benchmark_batch.py    # batch throughput table + speedup charts
python benchmarks/benchmark_single.py   # single-text throughput + latency charts

# The broader suite, written to a file
cd benchmarks
python benchmark.py --model cl100k_base --output results/my_benchmark.json
cat results/my_benchmark.md
```

`benchmark.py` covers single text encoding, batch encoding, streaming decoder performance, and special token handling across various content types. Expect different absolute numbers on different hardware and with different versions of comparison libraries — record which you ran against before quoting a figure.

## Regex Backends

Splintr uses a pure-Rust regex engine ([`regexr`](https://crates.io/crates/regexr)) by default, with optional PCRE2 support.

The two are not equally fast, and every performance figure on this page is the default one. Measured on the machine described above, PCRE2 encodes a **single text** about 1.1-1.5x faster than regexr — the gap widens on multilingual input — and is **level on batch encoding**, where Rayon rather than the regex engine sets the pace. Both produce identical token ids; the choice is a speed/dependency trade-off, not a correctness one. Quote a PCRE2 number only as a PCRE2 number: the published wheels do not carry the feature.

**Default Backend (regexr):**

- Pure Rust implementation (no C dependencies)
- JIT compilation and SIMD acceleration
- Native UTF-8 and Unicode property support

**Optional PCRE2 Backend:**

```python
from splintr import Tokenizer

# Default: regexr backend (pure Rust)
tokenizer = Tokenizer.from_pretrained("cl100k_base")

# Optional: switch to PCRE2 (requires --features pcre2)
tokenizer = Tokenizer.from_pretrained("cl100k_base").pcre2(True)
```

To enable PCRE2, build with the feature flag:

```bash
maturin develop --release --features python,pcre2
```

`python` has to be listed alongside it: `--features` replaces the default set rather than adding to it, and dropping the PyO3 feature leaves maturin with no binding to build.

**Benchmarking:**

```bash
# Compare backends (requires PCRE2 feature)
python benchmarks/benchmark_regexr_comparison.py --model cl100k_base

# Visual comparison with charts
python benchmarks/benchmark_regexr_viz.py --model cl100k_base
```
