#!/usr/bin/env python3
"""
Benchmark: Single Text Encoding Comparison
Compares tokenizer throughput across different text types and sizes.

Usage:
    python benchmarks/benchmark_single.py
"""

import gc
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Sample texts for benchmarking
SAMPLE_TEXTS = {
    "short": "Hello, world! This is a test.",
    "medium": """The quick brown fox jumps over the lazy dog.
    Machine learning models require tokenization to process text efficiently.
    Tokenizers convert text into numerical representations that models can understand.""" * 10,
    "long": """Artificial intelligence and machine learning have revolutionized
    the way we process and understand natural language. Large language models (LLMs)
    like GPT-4, Claude, and others rely heavily on efficient tokenization to handle
    vast amounts of text data. The performance of tokenizers directly impacts the
    overall throughput of these systems, making optimization crucial for production
    deployments. BPE (Byte Pair Encoding) has become the de facto standard for
    modern tokenizers due to its balance of vocabulary efficiency and handling of
    out-of-vocabulary words.""" * 50,
    "code": '''
def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

class TokenizerBenchmark:
    def __init__(self, name: str):
        self.name = name
        self.results = []

    def run(self, text: str, iterations: int = 100):
        for _ in range(iterations):
            tokens = self.encode(text)
            self.results.append(len(tokens))
''' * 20,
    "multilingual": """
    English: The quick brown fox jumps over the lazy dog.
    中文: 快速的棕色狐狸跳过懒狗。
    日本語: 素早い茶色の狐が怠惰な犬を飛び越える。
    한국어: 빠른 갈색 여우가 게으른 개를 뛰어넘습니다.
    العربية: الثعلب البني السريع يقفز فوق الكلب الكسول.
    Русский: Быстрая коричневая лиса прыгает через ленивую собаку.
    """ * 20,
}

TOKENIZER_COLORS = {
    "splintr": "#2ecc71",      # Green
    "tiktoken": "#3498db",     # Blue
    "huggingface": "#e74c3c",  # Red
    "tokendagger": "#9b59b6",  # Purple
}


@dataclass
class BenchmarkResult:
    name: str
    text_type: str
    bytes_per_second: float
    tokens_per_second: float
    num_tokens: int
    num_bytes: int
    latency_ms: float
    latency_std_ms: float


def benchmark_encode(
    name: str,
    encode_fn,
    text: str,
    text_type: str,
    warmup: int = 50,
    iterations: int = 100,
) -> BenchmarkResult:
    """Benchmark a single encode function."""
    num_bytes = len(text.encode("utf-8"))

    # Warmup - use more iterations to ensure thread pools are initialized
    for _ in range(warmup):
        encode_fn(text)

    # Force garbage collection before timing
    gc.collect()

    # Benchmark
    times = []
    num_tokens = 0
    for _ in range(iterations):
        start = time.perf_counter_ns()
        tokens = encode_fn(text)
        end = time.perf_counter_ns()
        times.append((end - start) / 1e9)  # Convert to seconds
        num_tokens = len(tokens)

    avg_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0
    bytes_per_second = num_bytes / avg_time
    tokens_per_second = num_tokens / avg_time

    return BenchmarkResult(
        name=name,
        text_type=text_type,
        bytes_per_second=bytes_per_second,
        tokens_per_second=tokens_per_second,
        num_tokens=num_tokens,
        num_bytes=num_bytes,
        latency_ms=avg_time * 1000,
        latency_std_ms=std_time * 1000,
    )


def load_tokenizers():
    """Load all available tokenizers."""
    tokenizers = {}

    # splintr
    try:
        import splintr
        enc = splintr.Tokenizer.from_pretrained("cl100k_base")
        tokenizers["splintr"] = enc.encode
        print("Loaded: splintr")
    except ImportError:
        print("splintr not available")

    # tiktoken
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        tokenizers["tiktoken"] = enc.encode
        print("Loaded: tiktoken")
    except ImportError:
        print("tiktoken not available")

    # HuggingFace tokenizers
    try:
        from tokenizers import Tokenizer as HFTokenizer
        hf_enc = HFTokenizer.from_pretrained("gpt2")

        def hf_encode(text):
            return hf_enc.encode(text).ids

        tokenizers["huggingface"] = hf_encode
        print("Loaded: huggingface")
    except ImportError:
        print("huggingface not available")

    # TokenDagger
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
        tokenizers["tokendagger"] = enc.encode
        print("Loaded: tokendagger")
    except (ImportError, Exception) as e:
        print(f"tokendagger not available: {e}")

    return tokenizers


def run_benchmarks(tokenizers: dict) -> list[BenchmarkResult]:
    """Run benchmarks for all text types."""
    results = []

    # Global warmup to initialize thread pools
    print("\nWarming up all tokenizers...")
    warmup_text = "This is a warmup text to initialize thread pools and caches." * 10
    for name, encode_fn in tokenizers.items():
        for _ in range(100):
            encode_fn(warmup_text)
    print("Warmup complete.")

    print("\n" + "=" * 70)
    print("TEXT TYPE BENCHMARKS")
    print("=" * 70)

    for text_type, text in SAMPLE_TEXTS.items():
        num_bytes = len(text.encode("utf-8"))
        print(f"\n--- {text_type.upper()} ({num_bytes:,} bytes) ---")
        print(f"{'Tokenizer':<15} {'MB/s':>10} {'Ktok/s':>10} {'Latency':>12} {'Std':>10}")
        print("-" * 60)

        for name, encode_fn in tokenizers.items():
            result = benchmark_encode(name, encode_fn, text, text_type)
            results.append(result)
            print(
                f"{name:<15} {result.bytes_per_second / 1e6:>10.2f} "
                f"{result.tokens_per_second / 1e3:>10.2f} "
                f"{result.latency_ms:>10.3f} ms "
                f"{result.latency_std_ms:>8.3f} ms"
            )

    return results


def generate_chart(results: list[BenchmarkResult], output_path: str):
    """Generate text type comparison chart."""

    # Get unique tokenizers and text types
    names = list(dict.fromkeys(r.name for r in results))
    text_types = list(dict.fromkeys(r.text_type for r in results))

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(text_types))
    width = 0.8 / len(names)

    # Create bars for each tokenizer
    for i, name in enumerate(names):
        throughputs = []
        for text_type in text_types:
            for r in results:
                if r.name == name and r.text_type == text_type:
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

        # Add value labels on bars
        for bar, val in zip(bars, throughputs):
            height = bar.get_height()
            ax.annotate(
                f'{val:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center',
                va='bottom',
                fontsize=8,
            )

    # Add text size annotations below x-axis
    text_sizes = []
    for text_type in text_types:
        for r in results:
            if r.text_type == text_type:
                text_sizes.append(r.num_bytes)
                break

    ax.set_xlabel("Text Type", fontsize=12)
    ax.set_ylabel("Throughput (MB/s)", fontsize=12)
    ax.set_title("Tokenizer Throughput by Text Type", fontsize=14, fontweight="bold")
    ax.set_xticks(x)

    # Create x-tick labels with size info
    xlabels = [f"{t.capitalize()}\n({text_sizes[i]:,} bytes)" for i, t in enumerate(text_types)]
    ax.set_xticklabels(xlabels)

    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    # Add a note about short text performance
    ax.text(
        0.98, 0.02,
        "Lower is worse for short texts due to fixed overhead",
        transform=ax.transAxes,
        fontsize=9,
        ha='right',
        va='bottom',
        style='italic',
        color='gray',
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to: {output_path}")
    plt.close()


def generate_latency_chart(results: list[BenchmarkResult], output_path: str):
    """Generate latency comparison chart (good for seeing short text overhead)."""

    names = list(dict.fromkeys(r.name for r in results))
    text_types = list(dict.fromkeys(r.text_type for r in results))

    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(text_types))
    width = 0.8 / len(names)

    for i, name in enumerate(names):
        latencies = []
        errors = []
        for text_type in text_types:
            for r in results:
                if r.name == name and r.text_type == text_type:
                    latencies.append(r.latency_ms)
                    errors.append(r.latency_std_ms)
                    break

        offset = i * width - width * len(names) / 2 + width / 2
        bars = ax.bar(
            x + offset,
            latencies,
            width,
            label=name,
            color=TOKENIZER_COLORS.get(name, "#95a5a6"),
            yerr=errors,
            capsize=3,
        )

        # Add value labels
        for bar, val in zip(bars, latencies):
            height = bar.get_height()
            ax.annotate(
                f'{val:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center',
                va='bottom',
                fontsize=8,
            )

    ax.set_xlabel("Text Type", fontsize=12)
    ax.set_ylabel("Latency (ms) - Lower is Better", fontsize=12)
    ax.set_title("Tokenizer Latency by Text Type", fontsize=14, fontweight="bold")
    ax.set_xticks(x)

    text_sizes = []
    for text_type in text_types:
        for r in results:
            if r.text_type == text_type:
                text_sizes.append(r.num_bytes)
                break

    xlabels = [f"{t.capitalize()}\n({text_sizes[i]:,} bytes)" for i, t in enumerate(text_types)]
    ax.set_xticklabels(xlabels)

    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_yscale("log")  # Log scale to see small values

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Latency chart saved to: {output_path}")
    plt.close()


def analyze_short_text_overhead(tokenizers: dict):
    """Analyze why short texts are slow - measure fixed overhead."""

    print("\n" + "=" * 70)
    print("SHORT TEXT OVERHEAD ANALYSIS")
    print("=" * 70)

    # First, do a global warmup for all tokenizers
    print("\nWarming up all tokenizers...")
    warmup_text = "This is a warmup text to initialize thread pools and caches." * 10
    for name, encode_fn in tokenizers.items():
        for _ in range(100):
            encode_fn(warmup_text)
    print("Warmup complete.\n")

    # Test with progressively shorter texts
    test_texts = [
        ("1 char", "H"),
        ("5 chars", "Hello"),
        ("10 chars", "Hello worl"),
        ("29 chars (short)", "Hello, world! This is a test."),
        ("100 chars", "Hello, world! " * 7),
        ("500 chars", "Hello, world! " * 35),
    ]

    print(f"{'Text':<20} {'Size':>8} ", end="")
    for name in tokenizers.keys():
        print(f"{name:>12}", end=" ")
    print()
    print("-" * (30 + 13 * len(tokenizers)))

    for label, text in test_texts:
        num_bytes = len(text.encode("utf-8"))
        print(f"{label:<20} {num_bytes:>6} B ", end="")

        for name, encode_fn in tokenizers.items():
            # Additional per-text warmup
            for _ in range(20):
                encode_fn(text)

            # Measure
            gc.collect()
            times = []
            for _ in range(100):
                start = time.perf_counter_ns()
                encode_fn(text)
                end = time.perf_counter_ns()
                times.append((end - start) / 1e6)  # ms

            avg_ms = statistics.mean(times)
            print(f"{avg_ms:>10.4f}ms", end=" ")
        print()


def main():
    print("=" * 70)
    print("TOKENIZER BENCHMARK: TEXT TYPES")
    print("=" * 70)

    # Create output directory
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    # Load tokenizers
    tokenizers = load_tokenizers()

    if len(tokenizers) < 2:
        print("\nNeed at least 2 tokenizers for comparison")
        return

    # Run main benchmarks
    results = run_benchmarks(tokenizers)

    # Generate charts
    generate_chart(results, str(output_dir / "benchmark_single.png"))
    generate_latency_chart(results, str(output_dir / "benchmark_single_latency.png"))

    # Analyze short text overhead
    analyze_short_text_overhead(tokenizers)

    print("\nDone!")


if __name__ == "__main__":
    main()
