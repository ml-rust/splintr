#!/usr/bin/env python3
"""
Benchmark script for splintr tokenizer.

Compares splintr against tiktoken across various workloads:
- Single text encoding (various sizes)
- Batch encoding (parallel vs sequential)
- Different content types (English, multilingual, code, etc.)
- Streaming decoder performance
- Cache hit scenarios
- Multiple model support (cl100k_base, o200k_base)

Usage:
    python benchmarks/benchmark.py
    python benchmarks/benchmark.py --iterations 20
    python benchmarks/benchmark.py --compare  # Compare with tiktoken
    python benchmarks/benchmark.py --name my_test  # Custom test name
    python benchmarks/benchmark.py --model o200k_base  # Use GPT-4o model
"""

import argparse
import json
import os
import platform
import time
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

try:
    from splintr import Tokenizer as SplintrTokenizer
    HAS_SPLINTR = True
    # Test if PCRE2 is available
    try:
        test_tok = SplintrTokenizer.from_pretrained("cl100k_base").pcre2(True)
        HAS_PCRE2 = True
        del test_tok
    except ValueError:
        HAS_PCRE2 = False
except ImportError:
    HAS_SPLINTR = False
    HAS_PCRE2 = False
    print("Warning: splintr not installed. Run: pip install -e . or maturin develop")

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False


@dataclass
class BenchmarkResult:
    name: str
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    throughput_mb_s: float
    iterations: int
    data_size_bytes: int = 0
    data_size_chars: int = 0


@dataclass
class SystemInfo:
    platform: str
    python_version: str
    cpu_count: int
    timestamp: str


def get_system_info() -> SystemInfo:
    """Collect system information."""
    return SystemInfo(
        platform=platform.platform(),
        python_version=platform.python_version(),
        cpu_count=os.cpu_count() or 1,
        timestamp=datetime.now().isoformat(),
    )


def benchmark(
    func: Callable,
    iterations: int = 10,
    warmup: int = 2,
    data_size_bytes: int = 0,
    data_size_chars: int = 0,
    name: str = "",
) -> BenchmarkResult:
    """Run a benchmark with warmup and statistics."""
    # Warmup
    for _ in range(warmup):
        func()

    # Timed runs
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms

    mean_ms = statistics.mean(times)
    std_ms = statistics.stdev(times) if len(times) > 1 else 0
    min_ms = min(times)
    max_ms = max(times)

    # Calculate throughput in MB/s
    if data_size_bytes > 0 and mean_ms > 0:
        throughput = (data_size_bytes / 1024 / 1024) / (mean_ms / 1000)
    else:
        throughput = 0

    return BenchmarkResult(
        name=name,
        mean_ms=mean_ms,
        std_ms=std_ms,
        min_ms=min_ms,
        max_ms=max_ms,
        throughput_mb_s=throughput,
        iterations=iterations,
        data_size_bytes=data_size_bytes,
        data_size_chars=data_size_chars,
    )


def generate_test_data():
    """Generate various test datasets."""
    return {
        "short_english": "Hello, world! This is a test.",
        "medium_english": "The quick brown fox jumps over the lazy dog. " * 100,
        "long_english": "The quick brown fox jumps over the lazy dog. " * 10000,
        "chinese": "你好世界！这是一个测试。人工智能正在改变世界。" * 500,
        "mixed_multilingual": (
            "Hello! 你好！مرحبا！Bonjour! Hola! Привет! " * 500
        ),
        "code_python": '''
def fibonacci(n):
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

class DataProcessor:
    def __init__(self, data):
        self.data = data

    def process(self):
        return [x * 2 for x in self.data if x > 0]
''' * 200,
        "code_json": '{"name": "test", "value": 123, "nested": {"key": "value"}}' * 500,
        "numbers": "1234567890 " * 5000,
        "special_chars": "!@#$%^&*()_+-=[]{}|;':\",./<>? " * 1000,
        "whitespace_heavy": "   word   " * 5000,
        "emojis": "🎉🎊🎈🎁🎀🎄🎃🎇🎆✨ " * 500,
    }


def format_result(result: BenchmarkResult, baseline: BenchmarkResult = None) -> list[str]:
    """Format a benchmark result for display."""
    lines = [
        f"  Mean: {result.mean_ms:>8.2f} ms (±{result.std_ms:.2f})",
        f"  Min:  {result.min_ms:>8.2f} ms",
        f"  Max:  {result.max_ms:>8.2f} ms",
    ]
    if result.throughput_mb_s > 0:
        lines.append(f"  Throughput: {result.throughput_mb_s:>6.2f} MB/s")
    if baseline:
        speedup = baseline.mean_ms / result.mean_ms
        lines.append(f"  Speedup vs baseline: {speedup:.2f}x")
    return lines


def print_result(result: BenchmarkResult, baseline: BenchmarkResult = None):
    """Print a benchmark result with optional comparison."""
    for line in format_result(result, baseline):
        print(line)


def run_single_text_benchmarks(
    splintr_enc,
    tiktoken_enc,
    test_data: dict,
    iterations: int,
    compare: bool,
) -> dict:
    """Run single text encoding benchmarks."""
    results = {"single_text": {}}

    print("\n" + "=" * 70)
    print("SINGLE TEXT ENCODING BENCHMARKS")
    print("=" * 70)

    for name, text in test_data.items():
        data_size = len(text.encode("utf-8"))
        print(f"\n{name} ({len(text):,} chars, {data_size:,} bytes):")
        print("-" * 50)

        results["single_text"][name] = {"chars": len(text), "bytes": data_size}

        # Splintr
        print("  Splintr:")
        result = benchmark(
            lambda t=text: splintr_enc.encode(t),
            iterations=iterations,
            data_size_bytes=data_size,
            data_size_chars=len(text),
            name=f"splintr_{name}",
        )
        print_result(result)
        results["single_text"][name]["splintr"] = asdict(result)
        splintr_result = result

        # Tiktoken (if comparing)
        if compare and tiktoken_enc:
            print("  tiktoken:")
            result = benchmark(
                lambda t=text: tiktoken_enc.encode(t),
                iterations=iterations,
                data_size_bytes=data_size,
                data_size_chars=len(text),
                name=f"tiktoken_{name}",
            )
            print_result(result)
            results["single_text"][name]["tiktoken"] = asdict(result)

            speedup = result.mean_ms / splintr_result.mean_ms
            results["single_text"][name]["speedup"] = speedup
            print(f"  >>> Splintr is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")

    return results


def run_batch_benchmarks(
    splintr_enc,
    tiktoken_enc,
    iterations: int,
    compare: bool,
) -> dict:
    """Run batch encoding benchmarks."""
    results = {"batch": {}}

    print("\n" + "=" * 70)
    print("BATCH ENCODING BENCHMARKS")
    print("=" * 70)

    batch_configs = [
        (10, 1000),    # 10 texts, 1000 chars each
        (100, 1000),   # 100 texts, 1000 chars each
        (1000, 100),   # 1000 texts, 100 chars each
        (100, 10000),  # 100 texts, 10000 chars each
    ]

    base_text = "The quick brown fox jumps over the lazy dog. "

    for num_texts, chars_per_text in batch_configs:
        text = (base_text * (chars_per_text // len(base_text) + 1))[:chars_per_text]
        texts = [text] * num_texts
        total_chars = num_texts * chars_per_text
        data_size = sum(len(t.encode("utf-8")) for t in texts)

        config_name = f"{num_texts}x{chars_per_text}"
        results["batch"][config_name] = {
            "num_texts": num_texts,
            "chars_per_text": chars_per_text,
            "total_chars": total_chars,
            "total_bytes": data_size,
        }

        print(f"\n{num_texts} texts × {chars_per_text} chars ({total_chars:,} total chars):")
        print("-" * 50)

        # Splintr batch (parallel)
        print("  splintr.encode_batch (parallel):")
        result = benchmark(
            lambda: splintr_enc.encode_batch(texts),
            iterations=iterations,
            data_size_bytes=data_size,
            data_size_chars=total_chars,
            name=f"splintr_batch_{config_name}",
        )
        print_result(result)
        results["batch"][config_name]["splintr_parallel"] = asdict(result)
        splintr_batch = result

        # Splintr sequential (for comparison)
        print("  Splintr sequential:")
        result = benchmark(
            lambda: [splintr_enc.encode(t) for t in texts],
            iterations=iterations,
            data_size_bytes=data_size,
            data_size_chars=total_chars,
            name=f"splintr_seq_{config_name}",
        )
        print_result(result)
        results["batch"][config_name]["splintr_sequential"] = asdict(result)
        parallel_speedup = result.mean_ms / splintr_batch.mean_ms
        results["batch"][config_name]["parallel_speedup"] = parallel_speedup
        print(f"  >>> Parallel speedup: {parallel_speedup:.2f}x")

        # Tiktoken (if comparing)
        if compare and tiktoken_enc:
            print("  tiktoken sequential:")
            result = benchmark(
                lambda: [tiktoken_enc.encode(t) for t in texts],
                iterations=iterations,
                data_size_bytes=data_size,
                data_size_chars=total_chars,
                name=f"tiktoken_seq_{config_name}",
            )
            print_result(result)
            results["batch"][config_name]["tiktoken_sequential"] = asdict(result)
            speedup = result.mean_ms / splintr_batch.mean_ms
            results["batch"][config_name]["vs_tiktoken_speedup"] = speedup
            print(f"  >>> Splintr batch is {speedup:.2f}x faster than tiktoken sequential")

    return results


def run_streaming_decoder_benchmark(
    splintr_enc,
    iterations: int,
) -> dict:
    """Benchmark streaming decoder performance."""
    results = {"streaming": {}}

    print("\n" + "=" * 70)
    print("STREAMING DECODER BENCHMARKS")
    print("=" * 70)

    # Various test cases for streaming
    test_cases = {
        "ascii_simple": "Hello, world! This is a test of the streaming decoder.",
        "multilingual": "Hello! 你好！مرحبا！Bonjour! Привет! 🎉",
        "long_mixed": ("The quick brown fox " * 50) + "你好世界" + (" jumps over" * 50),
    }

    for name, text in test_cases.items():
        tokens = splintr_enc.encode(text)
        data_size = len(text.encode("utf-8"))

        print(f"\n{name} ({len(tokens)} tokens, {len(text)} chars):")
        print("-" * 50)

        results["streaming"][name] = {
            "num_tokens": len(tokens),
            "chars": len(text),
            "bytes": data_size,
        }

        # Streaming decode
        def stream_decode():
            decoder = splintr_enc.streaming_decoder()
            output = []
            for token_id in tokens:
                chunk = decoder.add_token(token_id)
                if chunk:
                    output.append(chunk)
            output.append(decoder.flush())
            return "".join(output)

        print("  streaming_decoder:")
        result = benchmark(
            stream_decode,
            iterations=iterations,
            data_size_bytes=data_size,
            name=f"streaming_{name}",
        )
        print_result(result)
        results["streaming"][name]["streaming_decode"] = asdict(result)

        # Compare with regular decode
        print("  regular decode:")
        result = benchmark(
            lambda: splintr_enc.decode(tokens),
            iterations=iterations,
            data_size_bytes=data_size,
            name=f"regular_{name}",
        )
        print_result(result)
        results["streaming"][name]["regular_decode"] = asdict(result)

    return results


def run_cache_benchmark(
    splintr_enc,
    iterations: int,
) -> dict:
    """Benchmark cache effectiveness."""
    results = {"cache": {}}

    print("\n" + "=" * 70)
    print("CACHE BENCHMARKS")
    print("=" * 70)

    # Test with repeated text (should benefit from cache)
    repeated_text = "Hello, world! " * 100
    unique_texts = [f"Unique text number {i} with some content." for i in range(100)]
    data_size = len(repeated_text.encode("utf-8"))

    print("\nRepeated text (cache friendly):")
    print("-" * 50)

    # Clear cache and encode
    splintr_enc.clear_cache()

    print("  First encode (cold cache):")
    result_cold = benchmark(
        lambda: splintr_enc.encode(repeated_text),
        iterations=1,
        warmup=0,
        data_size_bytes=data_size,
        name="cold_cache",
    )
    print_result(result_cold)
    results["cache"]["cold_cache"] = asdict(result_cold)

    print("  Subsequent encodes (warm cache):")
    result_warm = benchmark(
        lambda: splintr_enc.encode(repeated_text),
        iterations=iterations,
        data_size_bytes=data_size,
        name="warm_cache",
    )
    print_result(result_warm)
    results["cache"]["warm_cache"] = asdict(result_warm)

    print(f"\n  Cache entries: {splintr_enc.cache_len}")

    print("\nBatch with repeated texts (high cache hit rate):")
    print("-" * 50)
    splintr_enc.clear_cache()
    repeated_batch = ["The quick brown fox jumps."] * 1000
    batch_size = sum(len(t.encode("utf-8")) for t in repeated_batch)

    print("  encode_batch (repeated):")
    result = benchmark(
        lambda: splintr_enc.encode_batch(repeated_batch),
        iterations=iterations,
        data_size_bytes=batch_size,
        name="batch_repeated",
    )
    print_result(result)
    results["cache"]["batch_repeated"] = asdict(result)
    print(f"  Cache entries after: {splintr_enc.cache_len}")

    return results


def run_special_tokens_benchmark(
    splintr_enc,
    tiktoken_enc,
    iterations: int,
    compare: bool,
) -> dict:
    """Benchmark special token handling performance."""
    results = {"special_tokens": {}}

    print("\n" + "=" * 70)
    print("SPECIAL TOKEN BENCHMARKS")
    print("=" * 70)

    # Text with many special tokens
    text_with_special = "Hello<|endoftext|>World<|endoftext|>Test<|endoftext|>" * 100
    data_size = len(text_with_special.encode("utf-8"))

    print(f"\nText with special tokens ({len(text_with_special)} chars):")
    print("-" * 50)

    results["special_tokens"]["chars"] = len(text_with_special)
    results["special_tokens"]["bytes"] = data_size

    print("  Splintr encode_with_special:")
    result = benchmark(
        lambda: splintr_enc.encode_with_special(text_with_special),
        iterations=iterations,
        data_size_bytes=data_size,
        name="splintr_special",
    )
    print_result(result)
    results["special_tokens"]["splintr"] = asdict(result)
    splintr_result = result

    if compare and tiktoken_enc:
        print("  tiktoken encode (with special):")
        result = benchmark(
            lambda: tiktoken_enc.encode(text_with_special, allowed_special="all"),
            iterations=iterations,
            data_size_bytes=data_size,
            name="tiktoken_special",
        )
        print_result(result)
        results["special_tokens"]["tiktoken"] = asdict(result)

        speedup = result.mean_ms / splintr_result.mean_ms
        results["special_tokens"]["speedup"] = speedup
        print(f"  >>> Splintr is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")

    return results


def run_correctness_check(splintr_enc, tiktoken_enc) -> dict:
    """Verify Splintr produces identical output to tiktoken."""
    results = {"correctness": {"tests": [], "all_passed": False}}

    print("\n" + "=" * 70)
    print("CORRECTNESS CHECK (vs tiktoken)")
    print("=" * 70)

    test_cases = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "你好世界",
        "مرحبا بالعالم",
        "🎉🎊🎈",
        "def hello():\n    print('Hello')",
        "   spaces   ",
        "CamelCaseIdentifier",
        "123456789",
        "Mixed123Text456Here",
    ]

    all_pass = True
    for text in test_cases:
        splintr_tokens = splintr_enc.encode(text)
        tiktoken_tokens = tiktoken_enc.encode(text)

        match = splintr_tokens == tiktoken_tokens
        all_pass = all_pass and match

        status = "PASS" if match else "FAIL"
        display = text[:40] + "..." if len(text) > 40 else text
        display = display.replace("\n", "\\n")
        print(f"  [{status}] \"{display}\"")

        results["correctness"]["tests"].append({
            "text": text,
            "passed": match,
            "splintr_tokens": splintr_tokens if not match else None,
            "tiktoken_tokens": tiktoken_tokens if not match else None,
        })

        if not match:
            print(f"    Splintr:  {splintr_tokens}")
            print(f"    tiktoken: {tiktoken_tokens}")

    print("-" * 50)
    print(f"All tests passed: {all_pass}")
    results["correctness"]["all_passed"] = all_pass
    return results


def save_results(
    results: dict,
    test_name: str,
    results_dir: Path,
):
    """Save benchmark results to JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{test_name}_{timestamp}.json"
    filepath = results_dir / filename

    # Add metadata
    results["metadata"] = {
        "test_name": test_name,
        "timestamp": datetime.now().isoformat(),
        "system": asdict(get_system_info()),
    }

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {filepath}")
    return filepath


def generate_summary(results: dict) -> str:
    """Generate a markdown summary of results."""
    lines = ["# Benchmark Results\n"]

    # Metadata
    if "metadata" in results:
        meta = results["metadata"]
        lines.append(f"**Test:** {meta.get('test_name', 'N/A')}")
        lines.append(f"**Date:** {meta.get('timestamp', 'N/A')}")
        if "system" in meta:
            sys_info = meta["system"]
            lines.append(f"**Platform:** {sys_info.get('platform', 'N/A')}")
            lines.append(f"**CPU Cores:** {sys_info.get('cpu_count', 'N/A')}")
        lines.append("")

    # Single text results
    if "single_text" in results:
        lines.append("## Single Text Encoding\n")
        lines.append("| Content | Size | splintr (ms) | tiktoken (ms) | Speedup |")
        lines.append("|---------|------|--------------|---------------|---------|")

        for name, data in results["single_text"].items():
            splintr_ms = data.get("splintr", {}).get("mean_ms", 0)
            tiktoken_ms = data.get("tiktoken", {}).get("mean_ms", 0)
            speedup = data.get("speedup", 0)
            size = f"{data.get('chars', 0):,} chars"

            tiktoken_str = f"{tiktoken_ms:.2f}" if tiktoken_ms else "N/A"
            speedup_str = f"{speedup:.2f}x" if speedup else "N/A"

            lines.append(f"| {name} | {size} | {splintr_ms:.2f} | {tiktoken_str} | {speedup_str} |")
        lines.append("")

    # Batch results
    if "batch" in results:
        lines.append("## Batch Encoding\n")
        lines.append("| Config | Splintr parallel (ms) | Splintr seq (ms) | Tiktoken (ms) | Parallel Speedup | vs Tiktoken |")
        lines.append("|--------|----------------------|------------------|---------------|------------------|-------------|")

        for config, data in results["batch"].items():
            sp_par = data.get("splintr_parallel", {}).get("mean_ms", 0)
            sp_seq = data.get("splintr_sequential", {}).get("mean_ms", 0)
            tk_seq = data.get("tiktoken_sequential", {}).get("mean_ms", 0)
            par_speedup = data.get("parallel_speedup", 0)
            tk_speedup = data.get("vs_tiktoken_speedup", 0)

            tk_str = f"{tk_seq:.2f}" if tk_seq else "N/A"
            tk_speedup_str = f"{tk_speedup:.2f}x" if tk_speedup else "N/A"

            lines.append(f"| {config} | {sp_par:.2f} | {sp_seq:.2f} | {tk_str} | {par_speedup:.2f}x | {tk_speedup_str} |")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Benchmark splintr tokenizer")
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=10,
        help="Number of iterations per benchmark (default: 10)"
    )
    parser.add_argument(
        "--compare", "-c",
        action="store_true",
        help="Compare with tiktoken"
    )
    parser.add_argument(
        "--correctness-only",
        action="store_true",
        help="Only run correctness check"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="benchmark",
        help="Test name for results file (default: benchmark)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cl100k_base",
        choices=["cl100k_base", "o200k_base"],
        help="Model to benchmark (default: cl100k_base)"
    )
    parser.add_argument(
        "--skip-streaming",
        action="store_true",
        help="Skip streaming decoder benchmarks"
    )
    parser.add_argument(
        "--skip-cache",
        action="store_true",
        help="Skip cache benchmarks"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="regexr",
        choices=["regexr", "pcre2"],
        help="Regex backend to use: regexr (default, pure Rust) or pcre2 (requires feature flag)"
    )
    args = parser.parse_args()

    if not HAS_SPLINTR:
        print("Error: splintr not installed")
        return 1

    # Setup results directory
    script_dir = Path(__file__).parent
    results_dir = script_dir / "results"
    results_dir.mkdir(exist_ok=True)

    all_results = {}

    print("=" * 70)
    print("SPLINTR TOKENIZER BENCHMARK")
    print("=" * 70)

    # Load tokenizers
    backend_str = "PCRE2" if args.backend == "pcre2" else "Regexr"
    print(f"\nLoading tokenizers (model: {args.model}, backend: {backend_str})...")

    if args.backend == "pcre2":
        if not HAS_PCRE2:
            print("Error: PCRE2 backend requested but not available.")
            print("       Build with: maturin develop --release --features pcre2")
            return 1
        splintr_enc = SplintrTokenizer.from_pretrained(args.model).pcre2(True)
    else:  # regexr (default)
        splintr_enc = SplintrTokenizer.from_pretrained(args.model)

    print(f"  Splintr ({backend_str}): {splintr_enc}")

    tiktoken_enc = None
    if args.compare or args.correctness_only:
        if HAS_TIKTOKEN:
            tiktoken_enc = tiktoken.get_encoding(args.model)
            print(f"  tiktoken: {args.model} (vocab={tiktoken_enc.n_vocab})")
        else:
            print("  tiktoken: not installed (pip install tiktoken)")
            if args.correctness_only:
                return 1

    # Correctness check
    if tiktoken_enc:
        correctness_results = run_correctness_check(splintr_enc, tiktoken_enc)
        all_results.update(correctness_results)
        if not correctness_results["correctness"]["all_passed"]:
            print("\nWarning: Correctness check failed!")
            if args.correctness_only:
                return 1

    if args.correctness_only:
        return 0

    # Generate test data
    test_data = generate_test_data()

    # Run benchmarks
    single_results = run_single_text_benchmarks(
        splintr_enc,
        tiktoken_enc,
        test_data,
        args.iterations,
        args.compare,
    )
    all_results.update(single_results)

    batch_results = run_batch_benchmarks(
        splintr_enc,
        tiktoken_enc,
        args.iterations,
        args.compare,
    )
    all_results.update(batch_results)

    # Streaming decoder benchmarks
    if not args.skip_streaming:
        streaming_results = run_streaming_decoder_benchmark(
            splintr_enc,
            args.iterations,
        )
        all_results.update(streaming_results)

    # Cache benchmarks
    if not args.skip_cache:
        cache_results = run_cache_benchmark(
            splintr_enc,
            args.iterations,
        )
        all_results.update(cache_results)

    # Special tokens benchmarks
    special_results = run_special_tokens_benchmark(
        splintr_enc,
        tiktoken_enc,
        args.iterations,
        args.compare,
    )
    all_results.update(special_results)

    # Store model info
    all_results["model"] = args.model

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)

    # Save results
    if not args.no_save:
        filepath = save_results(all_results, args.name, results_dir)

        # Also save markdown summary
        summary = generate_summary(all_results)
        summary_path = filepath.with_suffix(".md")
        with open(summary_path, "w") as f:
            f.write(summary)
        print(f"Summary saved to: {summary_path}")

    return 0


if __name__ == "__main__":
    exit(main())
