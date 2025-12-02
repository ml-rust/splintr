#!/usr/bin/env python3
"""
Benchmark comparing PCRE2 vs Regexr regex backends in splintr.

Tests both implementations with identical workloads to measure:
- Throughput (MB/s)
- Latency (mean, min, max, std)
- Correctness (ensure identical token outputs)
- Performance ratio (regexr vs pcre2)

Usage:
    python benchmarks/benchmark_regexr_comparison.py
    python benchmarks/benchmark_regexr_comparison.py --iterations 20
    python benchmarks/benchmark_regexr_comparison.py --model o200k_base
    python benchmarks/benchmark_regexr_comparison.py --workload long  # Only long texts
"""

import argparse
import json
import os
import platform
import statistics
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

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
        print("      Benchmark will compare regexr only.\n")
except ImportError:
    HAS_SPLINTR = False
    print("Error: splintr not installed. Run: pip install -e . or maturin develop")
    exit(1)


@dataclass
class ComparisonResult:
    """Results from comparing PCRE2 vs Regexr on a single workload."""
    workload_name: str
    data_size_bytes: int
    data_size_chars: int

    # PCRE2 results
    pcre2_mean_ms: float
    pcre2_std_ms: float
    pcre2_min_ms: float
    pcre2_max_ms: float
    pcre2_throughput_mb_s: float

    # Regexr results
    regexr_mean_ms: float
    regexr_std_ms: float
    regexr_min_ms: float
    regexr_max_ms: float
    regexr_throughput_mb_s: float

    # Comparison metrics
    speedup_ratio: float  # pcre2_time / regexr_time (>1 means regexr is faster)
    tokens_match: bool  # Do both produce identical tokens?

    iterations: int


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


def benchmark_single(func, iterations: int = 10, warmup: int = 2) -> Tuple[float, float, float, float]:
    """
    Benchmark a function and return (mean_ms, std_ms, min_ms, max_ms).
    """
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

    return mean_ms, std_ms, min_ms, max_ms


def generate_test_workloads() -> Dict[str, str]:
    """Generate test workloads of various sizes and content types."""
    return {
        # Size-based tests
        "tiny": "Hello, world!",
        "short": "The quick brown fox jumps over the lazy dog. " * 10,
        "medium": "The quick brown fox jumps over the lazy dog. " * 1000,
        "long": "The quick brown fox jumps over the lazy dog. " * 10000,
        "very_long": "The quick brown fox jumps over the lazy dog. " * 50000,

        # Content-type tests
        "multilingual": "Hello! 你好！مرحبا！Bonjour! Hola! Привет! " * 1000,
        "chinese": "你好世界！这是一个测试。人工智能正在改变世界。机器学习是人工智能的一个分支。" * 1000,
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
''' * 500,
        "code_json": '{"name": "test", "value": 123, "nested": {"key": "value", "array": [1, 2, 3]}}' * 1000,
        "numbers": "1234567890 9876543210 " * 5000,
        "special_chars": "!@#$%^&*()_+-=[]{}|;':\",./<>? " * 2000,
        "emojis": "🎉🎊🎈🎁🎀🎄🎃🎇🎆✨🌟💫⭐️🌈🦄🐉🔥💧🌊 " * 500,
        "whitespace_heavy": "   word   another   more   " * 3000,

        # Pattern-specific tests
        "contractions": "I'm, you're, he's, she's, it's, we're, they're, I'll, you'll, we'll, " * 1000,
        "punctuation_heavy": "Hello... World!!! What? Really??? Yes!!! No??? Maybe... Perhaps!!! " * 1000,
    }


def compare_backends(
    pcre2_tokenizer,  # Optional[Tokenizer]
    regexr_tokenizer: Tokenizer,
    workload_name: str,
    text: str,
    iterations: int = 10,
) -> ComparisonResult:
    """
    Compare PCRE2 and Regexr backends on a single workload.
    """
    data_size_bytes = len(text.encode('utf-8'))
    data_size_chars = len(text)

    # Regexr tokens (always available)
    regexr_tokens = regexr_tokenizer.encode(text)

    # PCRE2 comparison (if available)
    if pcre2_tokenizer is not None:
        pcre2_tokens = pcre2_tokenizer.encode(text)
        tokens_match = pcre2_tokens == regexr_tokens

        if not tokens_match:
            print(f"WARNING: Token mismatch for '{workload_name}'!")
            print(f"  PCRE2 tokens:  {len(pcre2_tokens)} tokens")
            print(f"  Regexr tokens: {len(regexr_tokens)} tokens")
            # Show first few tokens for debugging
            print(f"  First 10 PCRE2:  {pcre2_tokens[:10]}")
            print(f"  First 10 Regexr: {regexr_tokens[:10]}")

        # Benchmark PCRE2
        pcre2_mean, pcre2_std, pcre2_min, pcre2_max = benchmark_single(
            lambda: pcre2_tokenizer.encode(text),
            iterations=iterations,
        )
    else:
        # PCRE2 not available - set placeholder values
        tokens_match = True  # Can't compare, assume correct
        pcre2_mean = 0.0
        pcre2_std = 0.0
        pcre2_min = 0.0
        pcre2_max = 0.0

    # Benchmark Regexr (always available)
    regexr_mean, regexr_std, regexr_min, regexr_max = benchmark_single(
        lambda: regexr_tokenizer.encode(text),
        iterations=iterations,
    )

    # Calculate throughput
    if pcre2_tokenizer is not None and pcre2_mean > 0:
        pcre2_throughput = (data_size_bytes / 1024 / 1024) / (pcre2_mean / 1000)
    else:
        pcre2_throughput = 0.0

    if regexr_mean > 0:
        regexr_throughput = (data_size_bytes / 1024 / 1024) / (regexr_mean / 1000)
    else:
        regexr_throughput = 0.0

    # Calculate speedup ratio (pcre2 / regexr)
    # > 1 means regexr is faster
    # < 1 means pcre2 is faster
    if pcre2_tokenizer is not None and regexr_mean > 0 and pcre2_mean > 0:
        speedup_ratio = pcre2_mean / regexr_mean
    else:
        speedup_ratio = 1.0  # No comparison possible

    return ComparisonResult(
        workload_name=workload_name,
        data_size_bytes=data_size_bytes,
        data_size_chars=data_size_chars,
        pcre2_mean_ms=pcre2_mean,
        pcre2_std_ms=pcre2_std,
        pcre2_min_ms=pcre2_min,
        pcre2_max_ms=pcre2_max,
        pcre2_throughput_mb_s=pcre2_throughput,
        regexr_mean_ms=regexr_mean,
        regexr_std_ms=regexr_std,
        regexr_min_ms=regexr_min,
        regexr_max_ms=regexr_max,
        regexr_throughput_mb_s=regexr_throughput,
        speedup_ratio=speedup_ratio,
        tokens_match=tokens_match,
        iterations=iterations,
    )


def print_comparison_table(results: List[ComparisonResult], has_pcre2: bool = True):
    """Print a formatted comparison table."""
    print("\n" + "="*120)
    if has_pcre2:
        print("PCRE2 vs Regexr Performance Comparison")
    else:
        print("Regexr Performance Benchmark (PCRE2 not available)")
    print("="*120)

    if has_pcre2:
        # Header with PCRE2 comparison
        print(f"{'Workload':<20} {'Size':>10} {'PCRE2 (ms)':>15} {'Regexr (ms)':>15} {'PCRE2 MB/s':>12} {'Regexr MB/s':>12} {'Speedup':>10} {'Match':>8}")
        print("-"*120)

        # Data rows
        for r in results:
            size_kb = r.data_size_bytes / 1024
            size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb/1024:.1f} MB"

            # Color code speedup ratio
            if r.speedup_ratio > 1.1:
                speedup_str = f"{r.speedup_ratio:.2f}x ✓"  # Regexr faster
            elif r.speedup_ratio < 0.9:
                speedup_str = f"{r.speedup_ratio:.2f}x ✗"  # PCRE2 faster
            else:
                speedup_str = f"{r.speedup_ratio:.2f}x ~"  # Similar

            match_str = "✓" if r.tokens_match else "✗ FAIL"

            print(f"{r.workload_name:<20} {size_str:>10} "
                  f"{r.pcre2_mean_ms:>13.2f} ± {r.pcre2_std_ms:.2f} "
                  f"{r.regexr_mean_ms:>11.2f} ± {r.regexr_std_ms:.2f} "
                  f"{r.pcre2_throughput_mb_s:>12.1f} "
                  f"{r.regexr_throughput_mb_s:>12.1f} "
                  f"{speedup_str:>10} "
                  f"{match_str:>8}")

        print("="*120)

        # Summary statistics
        print("\nSummary:")
        avg_speedup = statistics.mean([r.speedup_ratio for r in results])
        all_match = all(r.tokens_match for r in results)
        regexr_faster_count = sum(1 for r in results if r.speedup_ratio > 1.0)
        pcre2_faster_count = sum(1 for r in results if r.speedup_ratio < 1.0)

        print(f"  Average speedup ratio: {avg_speedup:.2f}x")
        print(f"  Regexr faster: {regexr_faster_count}/{len(results)} workloads")
        print(f"  PCRE2 faster:  {pcre2_faster_count}/{len(results)} workloads")
        print(f"  Correctness: {'✓ All outputs match' if all_match else '✗ Some outputs differ'}")

        if avg_speedup > 1.0:
            print(f"\n  → Regexr is {avg_speedup:.1f}x faster on average")
        else:
            print(f"\n  → PCRE2 is {1/avg_speedup:.1f}x faster on average")
    else:
        # Regexr-only output
        print(f"{'Workload':<20} {'Size':>10} {'Regexr (ms)':>18} {'Regexr MB/s':>12}")
        print("-"*70)

        for r in results:
            size_kb = r.data_size_bytes / 1024
            size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb/1024:.1f} MB"

            print(f"{r.workload_name:<20} {size_str:>10} "
                  f"{r.regexr_mean_ms:>14.2f} ± {r.regexr_std_ms:.2f} "
                  f"{r.regexr_throughput_mb_s:>12.1f}")

        print("="*70)

        # Summary for regexr-only
        print("\nSummary:")
        avg_throughput = statistics.mean([r.regexr_throughput_mb_s for r in results if r.regexr_throughput_mb_s > 0])
        print(f"  Average throughput: {avg_throughput:.1f} MB/s")
        print(f"  Note: Build with --features pcre2 to enable comparison")


def save_results(results: List[ComparisonResult], system_info: SystemInfo, output_file: str):
    """Save results to JSON file."""
    data = {
        "system_info": asdict(system_info),
        "results": [asdict(r) for r in results],
    }

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n✓ Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare PCRE2 vs Regexr performance in splintr tokenizer"
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
        default=10,
        help="Number of iterations per benchmark (default: 10)",
    )
    parser.add_argument(
        "--workload",
        type=str,
        default=None,
        help="Run only a specific workload (e.g., 'long', 'multilingual')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results/regexr_comparison.json",
        help="Output file for results (default: benchmark_results/regexr_comparison.json)",
    )

    args = parser.parse_args()

    if not HAS_SPLINTR:
        print("Error: splintr not installed")
        return 1

    print(f"Loading {args.model} tokenizers...")
    regexr_tokenizer = Tokenizer.from_pretrained(args.model)  # Default is regexr with JIT
    print("  → Regexr using JIT engine (default)")

    if HAS_PCRE2:
        pcre2_tokenizer = Tokenizer.from_pretrained(args.model).pcre2(True)
        print("  → PCRE2 backend enabled")
    else:
        pcre2_tokenizer = None
        print("  → PCRE2 not available (benchmark regexr only)")

    print(f"Generating test workloads...")
    workloads = generate_test_workloads()

    # Filter to specific workload if requested
    if args.workload:
        if args.workload not in workloads:
            print(f"Error: Unknown workload '{args.workload}'")
            print(f"Available: {', '.join(workloads.keys())}")
            return 1
        workloads = {args.workload: workloads[args.workload]}

    print(f"Running {len(workloads)} workloads with {args.iterations} iterations each...")
    print()

    results = []
    system_info = get_system_info()

    for i, (name, text) in enumerate(workloads.items(), 1):
        print(f"[{i}/{len(workloads)}] Benchmarking: {name:<20} ({len(text.encode('utf-8'))/1024:.1f} KB)...", end=" ")

        result = compare_backends(
            pcre2_tokenizer,
            regexr_tokenizer,
            name,
            text,
            iterations=args.iterations,
        )

        results.append(result)

        # Print quick summary
        if HAS_PCRE2:
            if result.speedup_ratio > 1.0:
                print(f"Regexr {result.speedup_ratio:.2f}x faster")
            else:
                print(f"PCRE2 {1/result.speedup_ratio:.2f}x faster")
        else:
            print(f"{result.regexr_throughput_mb_s:.1f} MB/s")

    # Print detailed comparison table
    print_comparison_table(results, has_pcre2=HAS_PCRE2)

    # Save results
    save_results(results, system_info, args.output)

    return 0


if __name__ == "__main__":
    exit(main())
