#!/usr/bin/env python3
"""
Benchmark comparison for o200k_base tokenizers: splintr vs tiktoken

o200k_base is used by GPT-4o.

Usage:
    pip install tiktoken matplotlib numpy
    python benchmarks/vocabs/benchmark_o200k.py
"""

import gc
import json
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
    Tokenizers convert text into numerical representations that models can understand."""
    * 10,
    "long": """Artificial intelligence and machine learning have revolutionized
    the way we process and understand natural language. Large language models (LLMs)
    like GPT-4, Claude, and others rely heavily on efficient tokenization to handle
    vast amounts of text data. The performance of tokenizers directly impacts the
    overall throughput of these systems, making optimization crucial for production
    deployments. BPE (Byte Pair Encoding) has become the de facto standard for
    modern tokenizers due to its balance of vocabulary efficiency and handling of
    out-of-vocabulary words."""
    * 50,
    "code": """
def fibonacci(n: int) -> int:
    \"\"\"Calculate the nth Fibonacci number.\"\"\"
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
"""
    * 20,
    "multilingual": """
    English: The quick brown fox jumps over the lazy dog.
    中文: 快速的棕色狐狸跳过懒狗。
    日本語: 素早い茶色の狐が怠惰な犬を飛び越える。
    한국어: 빠른 갈색 여우가 게으른 개를 뛰어넘습니다.
    العربية: الثعلب البني السريع يقفز فوق الكلب الكسول.
    Русский: Быстрая коричневая лиса прыгает через ленивую собаку.
    """
    * 20,
}


@dataclass
class BenchmarkResult:
    name: str
    text_type: str
    benchmark_type: str
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
    """Benchmark single text encoding."""
    num_bytes = len(text.encode("utf-8"))

    for _ in range(warmup):
        encode_fn(text)

    gc.collect()

    times = []
    num_tokens = 0
    for _ in range(iterations):
        start = time.perf_counter_ns()
        tokens = encode_fn(text)
        end = time.perf_counter_ns()
        times.append((end - start) / 1e9)
        num_tokens = len(tokens)

    avg_time = statistics.mean(times)
    bytes_per_second = num_bytes / avg_time
    tokens_per_second = num_tokens / avg_time

    return BenchmarkResult(
        name=name,
        text_type=text_type,
        benchmark_type="single_encode",
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
        benchmark_type="batch_encode",
        bytes_per_second=bytes_per_second,
        tokens_per_second=tokens_per_second,
        num_tokens=total_tokens,
        num_bytes=num_bytes,
        latency_ms=avg_time * 1000,
    )


def benchmark_decode(
    name: str,
    decode_fn: Callable[[list[int]], str],
    tokens: list[int],
    text_type: str,
    original_bytes: int,
    warmup: int = 3,
    iterations: int = 10,
) -> BenchmarkResult:
    """Benchmark single text decoding."""
    num_tokens = len(tokens)

    for _ in range(warmup):
        decode_fn(tokens)

    gc.collect()

    times = []
    num_bytes = 0
    for _ in range(iterations):
        start = time.perf_counter_ns()
        text = decode_fn(tokens)
        end = time.perf_counter_ns()
        times.append((end - start) / 1e9)
        num_bytes = len(text.encode("utf-8"))

    avg_time = statistics.mean(times)
    bytes_per_second = num_bytes / avg_time
    tokens_per_second = num_tokens / avg_time

    return BenchmarkResult(
        name=name,
        text_type=text_type,
        benchmark_type="single_decode",
        bytes_per_second=bytes_per_second,
        tokens_per_second=tokens_per_second,
        num_tokens=num_tokens,
        num_bytes=num_bytes,
        latency_ms=avg_time * 1000,
    )


def benchmark_batch_decode(
    name: str,
    decode_batch_fn: Callable[[list[list[int]]], list[str]],
    token_lists: list[list[int]],
    text_type: str,
    warmup: int = 2,
    iterations: int = 5,
) -> BenchmarkResult:
    """Benchmark batch decoding."""
    total_tokens = sum(len(t) for t in token_lists)

    for _ in range(warmup):
        decode_batch_fn(token_lists)

    gc.collect()

    times = []
    total_bytes = 0
    for _ in range(iterations):
        start = time.perf_counter_ns()
        results = decode_batch_fn(token_lists)
        end = time.perf_counter_ns()
        times.append((end - start) / 1e9)
        total_bytes = sum(len(r.encode("utf-8")) for r in results)

    avg_time = statistics.mean(times)
    bytes_per_second = total_bytes / avg_time
    tokens_per_second = total_tokens / avg_time

    return BenchmarkResult(
        name=name,
        text_type=text_type,
        benchmark_type="batch_decode",
        bytes_per_second=bytes_per_second,
        tokens_per_second=total_tokens / avg_time,
        num_tokens=total_tokens,
        num_bytes=total_bytes,
        latency_ms=avg_time * 1000,
    )


def load_tokenizers() -> dict:
    """Load all available o200k_base tokenizers."""
    tokenizers = {}

    # splintr
    try:
        import splintr

        enc = splintr.Tokenizer.from_pretrained("o200k_base")

        tokenizers["splintr"] = {
            "encode": enc.encode,
            "encode_batch": enc.encode_batch,
            "decode": enc.decode,
            "decode_batch": enc.decode_batch,
            "color": "#2ecc71",  # Green
        }
        print("Loaded: splintr (o200k_base)")
    except ImportError:
        print("splintr not available - run: maturin develop --release")

    # tiktoken
    try:
        import tiktoken

        tik_enc = tiktoken.get_encoding("o200k_base")

        def tik_encode_batch(texts):
            return tik_enc.encode_ordinary_batch(texts)

        tokenizers["tiktoken"] = {
            "encode": tik_enc.encode,
            "encode_batch": tik_encode_batch,
            "decode": tik_enc.decode,
            "decode_batch": tik_enc.decode_batch,
            "color": "#3498db",  # Blue
        }
        print("Loaded: tiktoken (o200k_base)")
    except ImportError:
        print("tiktoken not available - run: pip install tiktoken")

    return tokenizers


def run_benchmarks(tokenizers: dict, text_types: list[str] | None = None) -> list[BenchmarkResult]:
    """Run all benchmarks."""
    if text_types is None:
        text_types = list(SAMPLE_TEXTS.keys())

    results = []

    # Single text encoding benchmarks
    print("\n" + "=" * 70)
    print("SINGLE TEXT ENCODING BENCHMARKS")
    print("=" * 70)

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

    # Batch encoding benchmarks
    print("\n" + "=" * 70)
    print("BATCH ENCODING BENCHMARKS (100 texts)")
    print("=" * 70)

    for text_type in ["medium", "long"]:
        texts = [SAMPLE_TEXTS[text_type]] * 100
        total_bytes = sum(len(t.encode("utf-8")) for t in texts)
        print(f"\n--- {text_type.upper()} x100 ({total_bytes:,} bytes total) ---")

        for name, tok in tokenizers.items():
            result = benchmark_batch_encode(
                name, tok["encode_batch"], texts, f"{text_type}_batch"
            )
            results.append(result)
            print(
                f"{name:15} {result.bytes_per_second / 1e6:8.2f} MB/s  "
                f"{result.tokens_per_second / 1e3:8.2f} Ktok/s  "
                f"{result.latency_ms:8.3f} ms"
            )

    # Single text decoding benchmarks
    print("\n" + "=" * 70)
    print("SINGLE TEXT DECODING BENCHMARKS")
    print("=" * 70)

    for text_type in text_types:
        text = SAMPLE_TEXTS[text_type]
        num_bytes = len(text.encode("utf-8"))

        for name, tok in tokenizers.items():
            tokens = tok["encode"](text)
            print(f"\n--- {text_type.upper()} ({len(tokens):,} tokens) ---") if name == list(tokenizers.keys())[0] else None
            result = benchmark_decode(name, tok["decode"], tokens, text_type, num_bytes)
            results.append(result)
            print(
                f"{name:15} {result.bytes_per_second / 1e6:8.2f} MB/s  "
                f"{result.tokens_per_second / 1e3:8.2f} Ktok/s  "
                f"{result.latency_ms:8.3f} ms"
            )

    # Batch decoding benchmarks
    print("\n" + "=" * 70)
    print("BATCH DECODING BENCHMARKS (100 texts)")
    print("=" * 70)

    for text_type in ["medium", "long"]:
        texts = [SAMPLE_TEXTS[text_type]] * 100
        print(f"\n--- {text_type.upper()} x100 ---")

        for name, tok in tokenizers.items():
            token_lists = tok["encode_batch"](texts)
            result = benchmark_batch_decode(
                name, tok["decode_batch"], token_lists, f"{text_type}_batch"
            )
            results.append(result)
            print(
                f"{name:15} {result.bytes_per_second / 1e6:8.2f} MB/s  "
                f"{result.tokens_per_second / 1e3:8.2f} Ktok/s  "
                f"{result.latency_ms:8.3f} ms"
            )

    return results


def generate_chart(
    results: list[BenchmarkResult],
    tokenizers: dict,
    benchmark_type: str,
    title: str,
    output_path: str,
):
    """Generate a bar chart for a specific benchmark type."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib/numpy not available - run: pip install matplotlib numpy")
        return

    filtered = [r for r in results if r.benchmark_type == benchmark_type]
    if not filtered:
        return

    names = list(tokenizers.keys())
    text_types = list(dict.fromkeys(r.text_type for r in filtered))

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(text_types))
    width = 0.8 / len(names)

    for i, name in enumerate(names):
        throughputs = []
        for text_type in text_types:
            for r in filtered:
                if r.name == name and r.text_type == text_type:
                    throughputs.append(r.bytes_per_second / 1e6)
                    break
            else:
                throughputs.append(0)

        ax.bar(
            x + i * width - width * len(names) / 2 + width / 2,
            throughputs,
            width,
            label=name,
            color=tokenizers[name]["color"],
        )

    ax.set_xlabel("Text Type", fontsize=12)
    ax.set_ylabel("Throughput (MB/s)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_batch", " (batch)").capitalize() for t in text_types])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Chart saved to: {output_path}")
    plt.close()


def save_results_json(results: list[BenchmarkResult], output_path: str):
    """Save benchmark results as JSON."""
    data = [
        {
            "name": r.name,
            "text_type": r.text_type,
            "benchmark_type": r.benchmark_type,
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
    print("=" * 70)
    print("O200K_BASE TOKENIZER BENCHMARK COMPARISON")
    print("splintr vs tiktoken (GPT-4o)")
    print("=" * 70)

    # Create output directory
    output_dir = Path(__file__).parent.parent / "results" / "o200k"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tokenizers
    tokenizers = load_tokenizers()

    if len(tokenizers) < 2:
        print("\nWarning: Less than 2 tokenizers available for comparison")
        print("Install missing packages:")
        print("  pip install tiktoken matplotlib numpy")
        if "splintr" not in tokenizers:
            print("  maturin develop --release  # for splintr")

    if not tokenizers:
        print("No tokenizers available!")
        return

    # Run benchmarks
    results = run_benchmarks(tokenizers)

    # Generate outputs
    print("\n" + "=" * 70)
    print("GENERATING OUTPUTS")
    print("=" * 70)

    generate_chart(
        results,
        tokenizers,
        "single_encode",
        "o200k_base Single Text Encoding Throughput",
        str(output_dir / "single_encode.png"),
    )
    generate_chart(
        results,
        tokenizers,
        "batch_encode",
        "o200k_base Batch Encoding Throughput (100 texts)",
        str(output_dir / "batch_encode.png"),
    )
    generate_chart(
        results,
        tokenizers,
        "single_decode",
        "o200k_base Single Text Decoding Throughput",
        str(output_dir / "single_decode.png"),
    )
    generate_chart(
        results,
        tokenizers,
        "batch_decode",
        "o200k_base Batch Decoding Throughput (100 texts)",
        str(output_dir / "batch_decode.png"),
    )
    save_results_json(results, str(output_dir / "results.json"))

    print("\nDone!")


if __name__ == "__main__":
    main()
