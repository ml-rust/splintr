#!/usr/bin/env python3
"""
Benchmark comparison: splintr vs tiktoken vs HuggingFace Tokenizers vs TokenDagger

Generates performance charts comparing encoding throughput across different tokenizers.

Usage:
    pip install tiktoken tokenizers matplotlib numpy
    pip install tokendagger  # optional
    python benchmarks/compare_tokenizers.py
"""

import gc
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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


@dataclass
class BenchmarkResult:
    name: str
    text_type: str
    bytes_per_second: float
    tokens_per_second: float
    num_tokens: int
    num_bytes: int
    latency_ms: float


def benchmark_encode(
    name: str,
    encode_fn: Callable[[str], list],
    text: str,
    text_type: str,
    warmup: int = 3,
    iterations: int = 10,
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
    for _ in range(iterations):
        start = time.perf_counter_ns()
        tokens = encode_fn(text)
        end = time.perf_counter_ns()
        times.append((end - start) / 1e9)  # Convert to seconds
        num_tokens = len(tokens)

    avg_time = statistics.mean(times)
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
    )


def benchmark_batch_encode(
    name: str,
    encode_batch_fn: Callable[[list[str]], list],
    texts: list[str],
    text_type: str,
    warmup: int = 2,
    iterations: int = 5,
) -> BenchmarkResult:
    """Benchmark batch encoding."""
    num_bytes = sum(len(t.encode("utf-8")) for t in texts)

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
    bytes_per_second = num_bytes / avg_time
    tokens_per_second = total_tokens / avg_time

    return BenchmarkResult(
        name=name,
        text_type=text_type,
        bytes_per_second=bytes_per_second,
        tokens_per_second=tokens_per_second,
        num_tokens=total_tokens,
        num_bytes=num_bytes,
        latency_ms=avg_time * 1000,
    )


def load_tokenizers():
    """Load all available tokenizers."""
    tokenizers = {}

    # splintr
    try:
        import splintr

        enc = splintr.Tokenizer.from_pretrained("cl100k_base")
        tokenizers["splintr"] = {
            "encode": enc.encode,
            "encode_batch": enc.encode_batch,
            "color": "#2ecc71",  # Green
        }
        print("Loaded: splintr")
    except ImportError:
        print("splintr not available - run: maturin develop --release")

    # tiktoken
    try:
        import tiktoken

        tik_enc = tiktoken.get_encoding("cl100k_base")

        def tik_encode_batch(texts):
            return tik_enc.encode_ordinary_batch(texts)

        tokenizers["tiktoken"] = {
            "encode": tik_enc.encode,
            "encode_batch": tik_encode_batch,
            "color": "#3498db",  # Blue
        }
        print("Loaded: tiktoken")
    except ImportError:
        print("tiktoken not available - run: pip install tiktoken")

    # HuggingFace tokenizers
    try:
        from tokenizers import Tokenizer as HFTokenizer

        # Use GPT-2 tokenizer (similar to cl100k but available)
        hf_enc = HFTokenizer.from_pretrained("gpt2")

        def hf_encode(text):
            return hf_enc.encode(text).ids

        def hf_encode_batch(texts):
            return [e.ids for e in hf_enc.encode_batch(texts)]

        tokenizers["huggingface"] = {
            "encode": hf_encode,
            "encode_batch": hf_encode_batch,
            "color": "#e74c3c",  # Red
        }
        print("Loaded: huggingface tokenizers")
    except ImportError:
        print("HuggingFace tokenizers not available - run: pip install tokenizers")

    # TokenDagger (if available)
    try:
        import tokendagger

        # TokenDagger requires loading vocab from tiktoken
        import tiktoken
        tik_enc = tiktoken.get_encoding("cl100k_base")
        enc = tokendagger.Tokenizer(
            name="cl100k_base",
            pat_str=tik_enc._pat_str,
            mergeable_ranks=tik_enc._mergeable_ranks,
            special_tokens=tik_enc._special_tokens,
        )
        tokenizers["tokendagger"] = {
            "encode": enc.encode,
            "encode_batch": enc.encode_batch,
            "color": "#9b59b6",  # Purple
        }
        print("Loaded: tokendagger")
    except (ImportError, Exception) as e:
        print(f"tokendagger not available: {e}")

    return tokenizers


def run_benchmarks(tokenizers: dict, text_types: list[str] = None):
    """Run all benchmarks."""
    if text_types is None:
        text_types = list(SAMPLE_TEXTS.keys())

    results = []

    print("\n" + "=" * 60)
    print("SINGLE TEXT ENCODING BENCHMARKS")
    print("=" * 60)

    for text_type in text_types:
        text = SAMPLE_TEXTS[text_type]
        num_bytes = len(text.encode("utf-8"))
        print(f"\n--- {text_type.upper()} ({num_bytes:,} bytes) ---")

        for name, tok in tokenizers.items():
            result = benchmark_encode(name, tok["encode"], text, text_type)
            results.append(result)
            print(
                f"{name:15} {result.bytes_per_second / 1e6:8.2f} MB/s  "
                f"{result.tokens_per_second / 1e3:8.2f} Ktok/s  "
                f"{result.latency_ms:8.3f} ms"
            )

    # Batch benchmarks
    print("\n" + "=" * 60)
    print("BATCH ENCODING BENCHMARKS (100 texts)")
    print("=" * 60)

    for text_type in ["medium", "long"]:
        texts = [SAMPLE_TEXTS[text_type]] * 100
        total_bytes = sum(len(t.encode("utf-8")) for t in texts)
        print(f"\n--- {text_type.upper()} x100 ({total_bytes:,} bytes total) ---")

        for name, tok in tokenizers.items():
            result = benchmark_batch_encode(
                f"{name}_batch", tok["encode_batch"], texts, f"{text_type}_batch"
            )
            results.append(result)
            print(
                f"{name:15} {result.bytes_per_second / 1e6:8.2f} MB/s  "
                f"{result.tokens_per_second / 1e3:8.2f} Ktok/s  "
                f"{result.latency_ms:8.3f} ms"
            )

    return results


def generate_chart(results: list[BenchmarkResult], tokenizers: dict, output_path: str):
    """Generate comparison bar chart."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib/numpy not available - run: pip install matplotlib numpy")
        return

    # Filter for single-text benchmarks only
    single_results = [r for r in results if "_batch" not in r.text_type]

    # Get unique tokenizers and text types
    names = list(tokenizers.keys())
    text_types = list(dict.fromkeys(r.text_type for r in single_results))

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Chart 1: Throughput by text type
    ax1 = axes[0]
    x = np.arange(len(text_types))
    width = 0.8 / len(names)

    for i, name in enumerate(names):
        throughputs = []
        for text_type in text_types:
            for r in single_results:
                if r.name == name and r.text_type == text_type:
                    throughputs.append(r.bytes_per_second / 1e6)
                    break
        bars = ax1.bar(
            x + i * width - width * len(names) / 2 + width / 2,
            throughputs,
            width,
            label=name,
            color=tokenizers[name]["color"],
        )

    ax1.set_xlabel("Text Type", fontsize=12)
    ax1.set_ylabel("Throughput (MB/s)", fontsize=12)
    ax1.set_title("Tokenizer Throughput Comparison", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([t.capitalize() for t in text_types])
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Chart 2: Batch encoding comparison
    batch_results = [r for r in results if "_batch" in r.text_type]
    if batch_results:
        ax2 = axes[1]
        batch_types = list(dict.fromkeys(r.text_type for r in batch_results))
        x2 = np.arange(len(batch_types))

        for i, name in enumerate(names):
            throughputs = []
            for text_type in batch_types:
                for r in batch_results:
                    if r.name == f"{name}_batch" and r.text_type == text_type:
                        throughputs.append(r.bytes_per_second / 1e6)
                        break
                else:
                    throughputs.append(0)
            if any(t > 0 for t in throughputs):
                ax2.bar(
                    x2 + i * width - width * len(names) / 2 + width / 2,
                    throughputs,
                    width,
                    label=name,
                    color=tokenizers[name]["color"],
                )

        ax2.set_xlabel("Batch Type", fontsize=12)
        ax2.set_ylabel("Throughput (MB/s)", fontsize=12)
        ax2.set_title(
            "Batch Encoding (100 texts)", fontsize=14, fontweight="bold"
        )
        ax2.set_xticks(x2)
        ax2.set_xticklabels([t.replace("_batch", "").capitalize() for t in batch_types])
        ax2.legend()
        ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to: {output_path}")

    # Also save as SVG for better quality
    svg_path = output_path.replace(".png", ".svg")
    plt.savefig(svg_path, format="svg", bbox_inches="tight")
    print(f"SVG saved to: {svg_path}")

    plt.close()


def generate_speedup_chart(results: list[BenchmarkResult], tokenizers: dict, output_path: str):
    """Generate speedup comparison chart (relative to tiktoken)."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    if "tiktoken" not in tokenizers:
        print("tiktoken not available for speedup comparison")
        return

    # Filter for single-text benchmarks
    single_results = [r for r in results if "_batch" not in r.text_type]
    text_types = list(dict.fromkeys(r.text_type for r in single_results))
    names = [n for n in tokenizers.keys() if n != "tiktoken"]

    # Get tiktoken baseline
    tiktoken_throughput = {}
    for r in single_results:
        if r.name == "tiktoken":
            tiktoken_throughput[r.text_type] = r.bytes_per_second

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(text_types))
    width = 0.8 / len(names)

    for i, name in enumerate(names):
        speedups = []
        for text_type in text_types:
            for r in single_results:
                if r.name == name and r.text_type == text_type:
                    if text_type in tiktoken_throughput:
                        speedup = r.bytes_per_second / tiktoken_throughput[text_type]
                    else:
                        speedup = 1.0
                    speedups.append(speedup)
                    break
        ax.bar(
            x + i * width - width * len(names) / 2 + width / 2,
            speedups,
            width,
            label=name,
            color=tokenizers[name]["color"],
        )

    # Add baseline line at 1.0
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, label="tiktoken (baseline)")

    ax.set_xlabel("Text Type", fontsize=12)
    ax.set_ylabel("Speedup vs tiktoken", fontsize=12)
    ax.set_title("Tokenizer Speedup Relative to tiktoken", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in text_types])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Speedup chart saved to: {output_path}")
    plt.close()


def save_results_json(results: list[BenchmarkResult], output_path: str):
    """Save benchmark results as JSON."""
    data = [
        {
            "name": r.name,
            "text_type": r.text_type,
            "bytes_per_second": r.bytes_per_second,
            "tokens_per_second": r.tokens_per_second,
            "num_tokens": r.num_tokens,
            "num_bytes": r.num_bytes,
            "latency_ms": r.latency_ms,
        }
        for r in results
    ]
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to: {output_path}")


def main():
    print("=" * 60)
    print("TOKENIZER BENCHMARK COMPARISON")
    print("splintr vs tiktoken vs HuggingFace vs TokenDagger")
    print("=" * 60)

    # Create output directory
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    # Load tokenizers
    tokenizers = load_tokenizers()

    if len(tokenizers) < 2:
        print("\nWarning: Less than 2 tokenizers available for comparison")
        print("Install missing packages:")
        print("  pip install tiktoken tokenizers matplotlib numpy")
        print("  pip install tokendagger  # optional")
        if "splintr" not in tokenizers:
            print("  maturin develop --release  # for splintr")

    if not tokenizers:
        print("No tokenizers available!")
        return

    # Run benchmarks
    results = run_benchmarks(tokenizers)

    # Generate outputs
    print("\n" + "=" * 60)
    print("GENERATING OUTPUTS")
    print("=" * 60)

    generate_chart(results, tokenizers, str(output_dir / "benchmark_comparison.png"))
    generate_speedup_chart(results, tokenizers, str(output_dir / "benchmark_speedup.png"))
    save_results_json(results, str(output_dir / "benchmark_results.json"))

    print("\nDone!")


if __name__ == "__main__":
    main()
