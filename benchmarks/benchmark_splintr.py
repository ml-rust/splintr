#!/usr/bin/env python3
"""
Benchmark: Splintr Sequential vs Rayon (Parallel) Single Text Encoding
Finds the crossover point where Rayon parallelization becomes beneficial.

Usage:
    python benchmarks/benchmark_splintr.py
"""

import gc
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class BenchmarkResult:
    size_bytes: int
    sequential_ms: float
    rayon_ms: float
    sequential_throughput: float  # MB/s
    rayon_throughput: float  # MB/s
    speedup: float  # rayon / sequential (>1 means rayon is faster)


def create_test_text(target_size: int) -> str:
    """Create test text of approximately target size."""
    base = "The quick brown fox jumps over the lazy dog. "
    repeat = max(1, target_size // len(base))
    text = base * repeat
    return text[:target_size] if len(text) > target_size else text


def benchmark_size(size: int, warmup: int = 20, iterations: int = 50) -> BenchmarkResult:
    """Benchmark both sequential and rayon encoding for a given size."""
    import splintr

    text = create_test_text(size)
    actual_size = len(text.encode("utf-8"))

    enc = splintr.Tokenizer.from_pretrained("cl100k_base")

    # Warmup both
    for _ in range(warmup):
        enc.encode(text)
        enc.encode_rayon(text)

    gc.collect()

    # Benchmark sequential (encode)
    seq_times = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        enc.encode(text)
        end = time.perf_counter_ns()
        seq_times.append((end - start) / 1e6)  # ms

    # Benchmark rayon (encode_rayon)
    rayon_times = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        enc.encode_rayon(text)
        end = time.perf_counter_ns()
        rayon_times.append((end - start) / 1e6)  # ms

    seq_avg = statistics.mean(seq_times)
    rayon_avg = statistics.mean(rayon_times)

    seq_throughput = actual_size / seq_avg / 1000  # MB/s
    rayon_throughput = actual_size / rayon_avg / 1000  # MB/s

    speedup = seq_avg / rayon_avg  # >1 means rayon is faster

    return BenchmarkResult(
        size_bytes=actual_size,
        sequential_ms=seq_avg,
        rayon_ms=rayon_avg,
        sequential_throughput=seq_throughput,
        rayon_throughput=rayon_throughput,
        speedup=speedup,
    )


def check_rayon_available():
    """Check if encode_rayon is available."""
    try:
        import splintr
        enc = splintr.Tokenizer.from_pretrained("cl100k_base")
        if not hasattr(enc, 'encode_rayon'):
            print("ERROR: encode_rayon method not found!")
            print("You need to add encode_rayon to the Tokenizer class.")
            print("\nAdd this to src/core/tokenizer.rs:")
            print("""
    /// Encode text using Rayon parallel processing.
    /// Use this for very large texts (>100KB) where parallelization helps.
    pub fn encode_rayon(&self, text: &str) -> Vec<u32> {
        // ... parallel implementation
    }
""")
            return False
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False


def run_benchmarks() -> list[BenchmarkResult]:
    """Run benchmarks across various sizes."""
    results = []

    # Test sizes from 100 bytes to 1MB
    sizes = [
        100, 200, 500,
        1000, 2000, 5000,
        10000, 20000, 50000,
        100000, 200000, 500000,
        1000000,
    ]

    print("\n" + "=" * 80)
    print("SEQUENTIAL vs RAYON BENCHMARK")
    print("=" * 80)
    print(f"\n{'Size':>12} {'Sequential':>12} {'Rayon':>12} {'Seq MB/s':>10} {'Rayon MB/s':>10} {'Speedup':>10}")
    print("-" * 80)

    for size in sizes:
        result = benchmark_size(size)
        results.append(result)

        speedup_str = f"{result.speedup:.2f}x"
        if result.speedup > 1.1:
            speedup_str += " (rayon wins)"
        elif result.speedup < 0.9:
            speedup_str += " (seq wins)"

        print(
            f"{result.size_bytes:>10} B "
            f"{result.sequential_ms:>10.3f} ms "
            f"{result.rayon_ms:>10.3f} ms "
            f"{result.sequential_throughput:>10.1f} "
            f"{result.rayon_throughput:>10.1f} "
            f"{speedup_str:>14}"
        )

    return results


def find_crossover(results: list[BenchmarkResult]) -> int | None:
    """Find the size where Rayon becomes consistently faster."""
    for i, r in enumerate(results):
        # Check if rayon is faster for this and subsequent sizes
        if r.speedup > 1.0:
            # Verify it stays faster for larger sizes
            remaining = results[i:]
            if all(rr.speedup >= 0.95 for rr in remaining):  # Allow 5% margin
                return r.size_bytes
    return None


def generate_chart(results: list[BenchmarkResult], output_path: str):
    """Generate comparison chart."""
    sizes = [r.size_bytes for r in results]
    seq_throughput = [r.sequential_throughput for r in results]
    rayon_throughput = [r.rayon_throughput for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Chart 1: Throughput comparison
    ax1.plot(sizes, seq_throughput, 'o-', label='Sequential', color='#2ecc71', linewidth=2, markersize=6)
    ax1.plot(sizes, rayon_throughput, 's-', label='Rayon (parallel)', color='#e74c3c', linewidth=2, markersize=6)

    ax1.set_xscale('log')
    ax1.set_xlabel('Text Size (bytes)', fontsize=12)
    ax1.set_ylabel('Throughput (MB/s)', fontsize=12)
    ax1.set_title('Sequential vs Rayon Throughput', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add crossover annotation
    crossover = find_crossover(results)
    if crossover:
        ax1.axvline(x=crossover, color='gray', linestyle='--', alpha=0.7)
        ax1.annotate(
            f'Crossover\n~{crossover/1000:.0f}KB',
            xy=(crossover, max(seq_throughput) * 0.8),
            fontsize=10,
            ha='center',
        )

    # Chart 2: Speedup ratio
    speedups = [r.speedup for r in results]

    colors = ['#2ecc71' if s < 1 else '#e74c3c' for s in speedups]
    ax2.bar(range(len(sizes)), speedups, color=colors)
    ax2.axhline(y=1.0, color='black', linestyle='-', linewidth=1)

    ax2.set_xticks(range(len(sizes)))
    ax2.set_xticklabels([f'{s/1000:.0f}K' if s >= 1000 else str(s) for s in sizes], rotation=45, ha='right')
    ax2.set_xlabel('Text Size', fontsize=12)
    ax2.set_ylabel('Speedup (Rayon / Sequential)', fontsize=12)
    ax2.set_title('Rayon Speedup Ratio (>1 = Rayon faster)', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Sequential wins'),
        Patch(facecolor='#e74c3c', label='Rayon wins'),
    ]
    ax2.legend(handles=legend_elements, loc='upper left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nChart saved to: {output_path}")
    plt.close()


def main():
    print("=" * 80)
    print("SPLINTR: SEQUENTIAL vs RAYON SINGLE TEXT ENCODING")
    print("=" * 80)

    if not check_rayon_available():
        return

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    # Warmup
    print("\nWarming up...")
    import splintr
    enc = splintr.Tokenizer.from_pretrained("cl100k_base")
    warmup_text = "warmup " * 1000
    for _ in range(50):
        enc.encode(warmup_text)
        enc.encode_rayon(warmup_text)
    print("Warmup complete.")

    results = run_benchmarks()

    crossover = find_crossover(results)
    print("\n" + "=" * 80)
    if crossover:
        print(f"CROSSOVER POINT: ~{crossover:,} bytes ({crossover/1024:.1f} KB)")
        print(f"Recommendation: Use Rayon for texts > {crossover:,} bytes")
    else:
        print("No clear crossover found - Sequential is generally faster")
    print("=" * 80)

    generate_chart(results, str(output_dir / "benchmark_splintr.png"))

    print("\nDone!")


if __name__ == "__main__":
    main()
