#!/usr/bin/env python3
"""
Benchmark: PCRE2 vs Regexr Backend Comparison with Visualization
Compares the performance of PCRE2 and Regexr regex backends in splintr.

Usage:
    python benchmarks/benchmark_regexr_viz.py
    python benchmarks/benchmark_regexr_viz.py --model o200k_base
    python benchmarks/benchmark_regexr_viz.py --iterations 20
"""

import argparse
import gc
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from splintr import Tokenizer
    HAS_SPLINTR = True
    # Test if PCRE2 is available
    try:
        test_tok = Tokenizer.from_pretrained("cl100k_base").pcre2(True)
        HAS_PCRE2 = True
        del test_tok
    except ValueError:
        HAS_PCRE2 = False
        print("Note: PCRE2 not available. Build with: maturin develop --release --features pcre2")
        print("      Exiting - this benchmark requires PCRE2 for comparison.\n")
        exit(1)
except ImportError:
    HAS_SPLINTR = False
    print("Error: splintr not installed. Run: pip install -e . or maturin develop")
    exit(1)


# Sample texts for benchmarking
SAMPLE_TEXTS = {
    "tiny": "Hello!",
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
    out-of-vocabulary words.""" * 100,
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
''' * 50,
    "multilingual": """
    English: The quick brown fox jumps over the lazy dog.
    中文: 快速的棕色狐狸跳过懒狗。
    日本語: 素早い茶色の狐が怠惰な犬を飛び越える。
    한국어: 빠른 갈색 여우가 게으른 개를 뛰어넘습니다.
    العربية: الثعلب البني السريع يقفز فوق الكلب الكسول.
    Русский: Быстрая коричневая лиса прыгает через ленивую собаку.
    """ * 30,
    "chinese": "你好世界！这是一个测试。人工智能正在改变世界。机器学习是人工智能的一个分支。" * 50,
    "contractions": "I'm, you're, he's, she's, it's, we're, they're, I'll, you'll, we'll, " * 50,
}

BACKEND_COLORS = {
    "pcre2": "#2ecc71",    # Green (current default)
    "regexr": "#e67e22",   # Orange (experimental)
}


@dataclass
class BenchmarkResult:
    backend: str
    text_type: str
    bytes_per_second: float
    tokens_per_second: float
    num_tokens: int
    num_bytes: int
    latency_ms: float
    latency_std_ms: float
    tokens_match: bool  # Does output match the other backend?


def benchmark_encode(
    backend: str,
    encode_fn,
    text: str,
    text_type: str,
    reference_tokens=None,
    warmup: int = 50,
    iterations: int = 100,
) -> BenchmarkResult:
    """Benchmark a single encode function."""
    num_bytes = len(text.encode("utf-8"))

    # Warmup
    for _ in range(warmup):
        encode_fn(text)

    # Force garbage collection before timing
    gc.collect()

    # Benchmark
    times = []
    num_tokens = 0
    tokens = None
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

    # Check if tokens match reference
    tokens_match = True
    if reference_tokens is not None:
        tokens_match = tokens == reference_tokens

    return BenchmarkResult(
        backend=backend,
        text_type=text_type,
        bytes_per_second=bytes_per_second,
        tokens_per_second=tokens_per_second,
        num_tokens=num_tokens,
        num_bytes=num_bytes,
        latency_ms=avg_time * 1000,
        latency_std_ms=std_time * 1000,
        tokens_match=tokens_match,
    )


def run_benchmarks(
    pcre2_tokenizer,
    regexr_tokenizer,
    iterations: int = 100,
) -> list[BenchmarkResult]:
    """Run benchmarks for all text types."""
    results = []

    # Global warmup to initialize thread pools
    print("\nWarming up tokenizers...")
    warmup_text = "This is a warmup text to initialize thread pools and caches." * 10
    for _ in range(100):
        pcre2_tokenizer.encode(warmup_text)
        regexr_tokenizer.encode(warmup_text)
    print("Warmup complete.")

    print("\n" + "=" * 90)
    print("PCRE2 vs REGEXR BACKEND COMPARISON")
    print("=" * 90)

    for text_type, text in SAMPLE_TEXTS.items():
        num_bytes = len(text.encode("utf-8"))
        print(f"\n--- {text_type.upper()} ({num_bytes:,} bytes) ---")
        print(f"{'Backend':<10} {'MB/s':>10} {'Ktok/s':>10} {'Latency':>12} {'Std':>10} {'Match':>8}")
        print("-" * 70)

        # Benchmark PCRE2 first (reference)
        pcre2_result = benchmark_encode(
            "pcre2",
            pcre2_tokenizer.encode,
            text,
            text_type,
            iterations=iterations,
        )
        results.append(pcre2_result)

        # Get reference tokens for comparison
        reference_tokens = pcre2_tokenizer.encode(text)

        # Benchmark Regexr
        regexr_result = benchmark_encode(
            "regexr",
            regexr_tokenizer.encode,
            text,
            text_type,
            reference_tokens=reference_tokens,
            iterations=iterations,
        )
        results.append(regexr_result)

        # Print results
        for result in [pcre2_result, regexr_result]:
            match_str = "✓" if result.tokens_match else "✗"
            print(
                f"{result.backend:<10} {result.bytes_per_second / 1e6:>10.2f} "
                f"{result.tokens_per_second / 1e3:>10.2f} "
                f"{result.latency_ms:>10.3f} ms "
                f"{result.latency_std_ms:>8.3f} ms "
                f"{match_str:>8}"
            )

        # Calculate and print speedup
        speedup = pcre2_result.latency_ms / regexr_result.latency_ms
        if speedup > 1.0:
            print(f"  → Regexr is {speedup:.2f}x FASTER")
        else:
            print(f"  → PCRE2 is {1/speedup:.2f}x FASTER")

    return results


def generate_throughput_chart(results: list[BenchmarkResult], output_path: str):
    """Generate throughput comparison chart."""

    # Get unique backends and text types
    backends = list(dict.fromkeys(r.backend for r in results))
    text_types = list(dict.fromkeys(r.text_type for r in results))

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(text_types))
    width = 0.35

    # Create bars for each backend
    for i, backend in enumerate(backends):
        throughputs = []
        for text_type in text_types:
            for r in results:
                if r.backend == backend and r.text_type == text_type:
                    throughputs.append(r.bytes_per_second / 1e6)
                    break

        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset,
            throughputs,
            width,
            label=backend.upper(),
            color=BACKEND_COLORS.get(backend, "#95a5a6"),
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
                fontsize=9,
                fontweight='bold',
            )

    # Add text size annotations
    text_sizes = []
    for text_type in text_types:
        for r in results:
            if r.text_type == text_type:
                text_sizes.append(r.num_bytes)
                break

    ax.set_xlabel("Text Type", fontsize=12, fontweight='bold')
    ax.set_ylabel("Throughput (MB/s)", fontsize=12, fontweight='bold')
    ax.set_title("PCRE2 vs Regexr: Throughput Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)

    # Create x-tick labels with size info
    xlabels = [f"{t.capitalize()}\n({text_sizes[i]:,} bytes)" for i, t in enumerate(text_types)]
    ax.set_xticklabels(xlabels, fontsize=10)

    ax.legend(loc="upper left", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Throughput chart saved to: {output_path}")
    plt.close()


def generate_speedup_chart(results: list[BenchmarkResult], output_path: str):
    """Generate speedup ratio chart (Regexr vs PCRE2)."""

    text_types = list(dict.fromkeys(r.text_type for r in results))

    # Calculate speedup ratios
    speedups = []
    for text_type in text_types:
        pcre2_time = None
        regexr_time = None
        for r in results:
            if r.text_type == text_type:
                if r.backend == "pcre2":
                    pcre2_time = r.latency_ms
                elif r.backend == "regexr":
                    regexr_time = r.latency_ms

        if pcre2_time and regexr_time:
            # Speedup > 1 means regexr is faster
            speedup = pcre2_time / regexr_time
            speedups.append(speedup)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(text_types))
    colors = ['#2ecc71' if s > 1.0 else '#e74c3c' for s in speedups]

    bars = ax.bar(x, speedups, color=colors, edgecolor='black', linewidth=1.5)

    # Add horizontal line at 1.0 (parity)
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Parity (1.0x)')

    # Add value labels on bars
    for bar, val in zip(bars, speedups):
        height = bar.get_height()
        label = f'{val:.2f}x'
        if val > 1.0:
            label += '\nRegexr\nFaster'
        else:
            label += '\nPCRE2\nFaster'

        y_pos = height + 0.05 if height > 1.0 else height - 0.05
        va_pos = 'bottom' if height > 1.0 else 'top'

        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, y_pos),
            ha='center',
            va=va_pos,
            fontsize=9,
            fontweight='bold',
        )

    # Add text size annotations
    text_sizes = []
    for text_type in text_types:
        for r in results:
            if r.text_type == text_type:
                text_sizes.append(r.num_bytes)
                break

    ax.set_xlabel("Text Type", fontsize=12, fontweight='bold')
    ax.set_ylabel("Speedup Ratio (PCRE2 time / Regexr time)", fontsize=12, fontweight='bold')
    ax.set_title("Regexr Performance Relative to PCRE2\n(>1.0 = Regexr faster, <1.0 = PCRE2 faster)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)

    xlabels = [f"{t.capitalize()}\n({text_sizes[i]:,} bytes)" for i, t in enumerate(text_types)]
    ax.set_xticklabels(xlabels, fontsize=10)

    ax.legend(loc="upper right", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # Set y-axis to show both faster and slower clearly
    y_max = max(speedups) * 1.15
    y_min = min(speedups) * 0.85
    ax.set_ylim(min(y_min, 0.8), max(y_max, 1.2))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Speedup chart saved to: {output_path}")
    plt.close()


def generate_latency_chart(results: list[BenchmarkResult], output_path: str):
    """Generate latency comparison chart."""

    backends = list(dict.fromkeys(r.backend for r in results))
    text_types = list(dict.fromkeys(r.text_type for r in results))

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(text_types))
    width = 0.35

    for i, backend in enumerate(backends):
        latencies = []
        errors = []
        for text_type in text_types:
            for r in results:
                if r.backend == backend and r.text_type == text_type:
                    latencies.append(r.latency_ms)
                    errors.append(r.latency_std_ms)
                    break

        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset,
            latencies,
            width,
            label=backend.upper(),
            color=BACKEND_COLORS.get(backend, "#95a5a6"),
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
                fontsize=9,
            )

    ax.set_xlabel("Text Type", fontsize=12, fontweight='bold')
    ax.set_ylabel("Latency (ms) - Lower is Better", fontsize=12, fontweight='bold')
    ax.set_title("PCRE2 vs Regexr: Latency Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)

    text_sizes = []
    for text_type in text_types:
        for r in results:
            if r.text_type == text_type:
                text_sizes.append(r.num_bytes)
                break

    xlabels = [f"{t.capitalize()}\n({text_sizes[i]:,} bytes)" for i, t in enumerate(text_types)]
    ax.set_xticklabels(xlabels, fontsize=10)

    ax.legend(loc="upper left", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_yscale("log")  # Log scale to see all values

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Latency chart saved to: {output_path}")
    plt.close()


def print_summary(results: list[BenchmarkResult]):
    """Print summary statistics."""
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    # Calculate average speedup
    text_types = list(dict.fromkeys(r.text_type for r in results))
    speedups = []

    for text_type in text_types:
        pcre2_time = None
        regexr_time = None
        for r in results:
            if r.text_type == text_type:
                if r.backend == "pcre2":
                    pcre2_time = r.latency_ms
                elif r.backend == "regexr":
                    regexr_time = r.latency_ms

        if pcre2_time and regexr_time:
            speedup = pcre2_time / regexr_time
            speedups.append(speedup)

    avg_speedup = statistics.mean(speedups)
    regexr_faster_count = sum(1 for s in speedups if s > 1.0)
    pcre2_faster_count = sum(1 for s in speedups if s < 1.0)

    # Check correctness
    all_match = all(r.tokens_match for r in results)

    print(f"Average speedup ratio: {avg_speedup:.3f}x")
    print(f"Regexr faster on: {regexr_faster_count}/{len(speedups)} workloads")
    print(f"PCRE2 faster on:  {pcre2_faster_count}/{len(speedups)} workloads")
    print(f"Correctness: {'✓ All outputs match' if all_match else '✗ Some outputs differ'}")

    if avg_speedup > 1.0:
        print(f"\n→ Overall: Regexr is {avg_speedup:.2f}x FASTER on average")
    else:
        print(f"\n→ Overall: PCRE2 is {1/avg_speedup:.2f}x FASTER on average")

    print("\nDetailed breakdown:")
    for text_type, speedup in zip(text_types, speedups):
        if speedup > 1.0:
            print(f"  {text_type:>15}: Regexr {speedup:.2f}x faster")
        else:
            print(f"  {text_type:>15}: PCRE2 {1/speedup:.2f}x faster")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize PCRE2 vs Regexr performance comparison"
    )
    parser.add_argument(
        "--model",
        default="cl100k_base",
        choices=["cl100k_base", "o200k_base", "llama3", "deepseek_v3"],
        help="Model to use for benchmarking (default: cl100k_base)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of iterations per benchmark (default: 100)",
    )

    args = parser.parse_args()

    if not HAS_SPLINTR:
        print("Error: splintr not installed")
        return 1

    print("=" * 90)
    print("PCRE2 vs REGEXR BACKEND COMPARISON")
    print("=" * 90)

    # Create output directory
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    # Load tokenizers
    print(f"\nLoading {args.model} tokenizers...")
    regexr_tokenizer = Tokenizer.from_pretrained(args.model)  # Default is regexr with JIT
    pcre2_tokenizer = Tokenizer.from_pretrained(args.model).pcre2(True)
    print("✓ Tokenizers loaded")
    print("  → Regexr: Default (JIT engine)")
    print("  → PCRE2: Enabled via .pcre2(True)")

    # Run benchmarks
    results = run_benchmarks(pcre2_tokenizer, regexr_tokenizer, iterations=args.iterations)

    # Generate charts
    print("\nGenerating visualizations...")
    generate_throughput_chart(results, str(output_dir / "regexr_throughput.png"))
    generate_speedup_chart(results, str(output_dir / "regexr_speedup.png"))
    generate_latency_chart(results, str(output_dir / "regexr_latency.png"))

    # Print summary
    print_summary(results)

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
