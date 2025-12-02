#!/usr/bin/env python3
"""
Benchmark: Batch Encoding Comparison
Compares tokenizer throughput for batch encoding across different batch sizes.

Usage:
    python benchmarks/benchmark_batch.py
"""

import gc
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Sample text for batch benchmarks
SAMPLE_TEXT = """The quick brown fox jumps over the lazy dog.
Machine learning models require tokenization to process text efficiently.
Tokenizers convert text into numerical representations that models can understand."""

TOKENIZER_COLORS = {
    "splintr": "#2ecc71",          # Green (default, pure Rust)
    "splintr-pcre2": "#27ae60",    # Dark Green (optional)
    "tiktoken": "#3498db",         # Blue
    "huggingface": "#e74c3c",      # Red
    "tokendagger": "#9b59b6",      # Purple
}


@dataclass
class BenchmarkResult:
    name: str
    batch_size: int
    bytes_per_second: float
    tokens_per_second: float
    total_tokens: int
    total_bytes: int
    latency_ms: float


def benchmark_batch(
    name: str,
    encode_batch_fn,
    texts: list[str],
    batch_size: int,
    warmup: int = 3,
    iterations: int = 10,
) -> BenchmarkResult:
    """Benchmark batch encoding."""
    total_bytes = sum(len(t.encode("utf-8")) for t in texts)

    # Warmup
    for _ in range(warmup):
        encode_batch_fn(texts)

    gc.collect()

    times = []
    total_tokens = 0
    for _ in range(iterations):
        start = time.perf_counter_ns()
        results = encode_batch_fn(texts)
        end = time.perf_counter_ns()
        times.append((end - start) / 1e9)
        total_tokens = sum(len(r) for r in results)

    avg_time = statistics.mean(times)
    bytes_per_second = total_bytes / avg_time
    tokens_per_second = total_tokens / avg_time

    return BenchmarkResult(
        name=name,
        batch_size=batch_size,
        bytes_per_second=bytes_per_second,
        tokens_per_second=tokens_per_second,
        total_tokens=total_tokens,
        total_bytes=total_bytes,
        latency_ms=avg_time * 1000,
    )


def load_tokenizers():
    """Load all available tokenizers with batch functions.

    All tokenizers use their native batch encoding methods:
    - splintr: encode_batch (Rayon parallel, pure Rust regex with JIT)
    - splintr-pcre2: encode_batch (Rayon parallel, PCRE2 with JIT)
    - tiktoken: encode_ordinary_batch (native batch)
    - huggingface: encode_batch (native batch)
    - tokendagger: encode_batch (native batch)
    """
    tokenizers = {}

    # splintr - default backend (pure Rust with JIT)
    try:
        import splintr
        enc = splintr.Tokenizer.from_pretrained("cl100k_base")
        tokenizers["splintr"] = enc.encode_batch
        print("Loaded: splintr (native encode_batch)")
    except ImportError:
        print("splintr not available")

    # splintr-pcre2 - optional backend (requires --features pcre2)
    try:
        import splintr
        enc_pcre2 = splintr.Tokenizer.from_pretrained("cl100k_base").pcre2(True)
        tokenizers["splintr-pcre2"] = enc_pcre2.encode_batch
        print("Loaded: splintr-pcre2 (native encode_batch)")
    except (ImportError, ValueError) as e:
        print(f"splintr-pcre2 not available: {e}")

    # tiktoken - native batch
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        tokenizers["tiktoken"] = enc.encode_ordinary_batch
        print("Loaded: tiktoken (native encode_ordinary_batch)")
    except ImportError:
        print("tiktoken not available")

    # HuggingFace tokenizers - native batch
    try:
        from tokenizers import Tokenizer as HFTokenizer
        hf_enc = HFTokenizer.from_pretrained("gpt2")

        def hf_encode_batch(texts):
            return [e.ids for e in hf_enc.encode_batch(texts)]

        tokenizers["huggingface"] = hf_encode_batch
        print("Loaded: huggingface (native encode_batch)")
    except ImportError:
        print("huggingface not available")

    # TokenDagger - native batch
    try:
        import tokendagger
        import tiktoken
        tik_enc = tiktoken.get_encoding("cl100k_base")
        enc = tokendagger.Tokenizer(
            name="cl100k_base",
            pat_str=tik_enc._pat_str,
            mergeable_ranks=tik_enc._mergeable_ranks,
            special_tokens=tik_enc._special_tokens,
        )
        tokenizers["tokendagger"] = enc.encode_batch
        print("Loaded: tokendagger (native encode_batch)")
    except (ImportError, Exception) as e:
        print(f"tokendagger not available: {e}")

    return tokenizers


def run_benchmarks(tokenizers: dict) -> list[BenchmarkResult]:
    """Run batch benchmarks with various batch sizes."""
    results = []

    # Warmup all tokenizers
    print("\nWarming up all tokenizers...")
    warmup_texts = [SAMPLE_TEXT] * 100
    for name, encode_batch_fn in tokenizers.items():
        for _ in range(10):
            encode_batch_fn(warmup_texts)
    print("Warmup complete.")

    batch_sizes = [1, 10, 50, 100, 500, 1000]

    print("\n" + "=" * 70)
    print("BATCH ENCODING BENCHMARKS")
    print("=" * 70)

    for batch_size in batch_sizes:
        texts = [SAMPLE_TEXT] * batch_size
        total_bytes = sum(len(t.encode("utf-8")) for t in texts)

        print(f"\n--- Batch Size: {batch_size} ({total_bytes:,} bytes total) ---")
        print(f"{'Tokenizer':<15} {'MB/s':>10} {'Ktok/s':>10} {'Latency':>12}")
        print("-" * 50)

        for name, encode_batch_fn in tokenizers.items():
            result = benchmark_batch(name, encode_batch_fn, texts, batch_size)
            results.append(result)
            print(
                f"{name:<15} {result.bytes_per_second / 1e6:>10.2f} "
                f"{result.tokens_per_second / 1e3:>10.2f} "
                f"{result.latency_ms:>10.2f} ms"
            )

    return results


def generate_chart(results: list[BenchmarkResult], output_path: str):
    """Generate batch encoding comparison chart."""

    names = list(dict.fromkeys(r.name for r in results))
    batch_sizes = list(dict.fromkeys(r.batch_size for r in results))

    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(batch_sizes))
    width = 0.8 / len(names)

    for i, name in enumerate(names):
        throughputs = []
        for batch_size in batch_sizes:
            for r in results:
                if r.name == name and r.batch_size == batch_size:
                    throughputs.append(r.bytes_per_second / 1e6)
                    break

        offset = i * width - width * len(names) / 2 + width / 2
        bars = ax.bar(
            x + offset,
            throughputs,
            width,
            label=name,
            color=TOKENIZER_COLORS.get(name, "#95a5a6"),
        )

        # Add value labels
        for bar, val in zip(bars, throughputs):
            height = bar.get_height()
            ax.annotate(
                f'{val:.0f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center',
                va='bottom',
                fontsize=8,
            )

    ax.set_xlabel("Batch Size (number of texts)", fontsize=12)
    ax.set_ylabel("Throughput (MB/s)", fontsize=12)
    ax.set_title("Batch Encoding Throughput Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(bs) for bs in batch_sizes])
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to: {output_path}")
    plt.close()


def generate_speedup_chart(results: list[BenchmarkResult], output_path: str):
    """Generate speedup vs tiktoken chart."""

    if not any(r.name == "tiktoken" for r in results):
        print("tiktoken not available for speedup chart")
        return

    names = [n for n in dict.fromkeys(r.name for r in results) if n != "tiktoken"]
    batch_sizes = list(dict.fromkeys(r.batch_size for r in results))

    # Get tiktoken baseline
    tiktoken_throughput = {}
    for r in results:
        if r.name == "tiktoken":
            tiktoken_throughput[r.batch_size] = r.bytes_per_second

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(batch_sizes))
    width = 0.8 / len(names)

    for i, name in enumerate(names):
        speedups = []
        for batch_size in batch_sizes:
            for r in results:
                if r.name == name and r.batch_size == batch_size:
                    speedup = r.bytes_per_second / tiktoken_throughput[batch_size]
                    speedups.append(speedup)
                    break

        offset = i * width - width * len(names) / 2 + width / 2
        ax.bar(
            x + offset,
            speedups,
            width,
            label=name,
            color=TOKENIZER_COLORS.get(name, "#95a5a6"),
        )

    # Baseline line
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, label="tiktoken (baseline)")

    ax.set_xlabel("Batch Size", fontsize=12)
    ax.set_ylabel("Speedup vs tiktoken", fontsize=12)
    ax.set_title("Batch Encoding Speedup Relative to tiktoken", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(bs) for bs in batch_sizes])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Speedup chart saved to: {output_path}")
    plt.close()


def main():
    print("=" * 70)
    print("TOKENIZER BENCHMARK: BATCH ENCODING")
    print("=" * 70)

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    tokenizers = load_tokenizers()

    if len(tokenizers) < 2:
        print("\nNeed at least 2 tokenizers for comparison")
        return

    results = run_benchmarks(tokenizers)

    generate_chart(results, str(output_dir / "benchmark_batch.png"))
    generate_speedup_chart(results, str(output_dir / "benchmark_batch_speedup.png"))

    print("\nDone!")


if __name__ == "__main__":
    main()
