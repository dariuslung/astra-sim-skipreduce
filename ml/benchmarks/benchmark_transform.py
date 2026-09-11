"""
Benchmark execution latency of DCT and Walsh-Hadamard transforms on the local GPU.
Measures forward transform, retention filtering, and inverse transform across realistic tensor sizes.
"""

import argparse
import json
import os
import sys
import time

# Ensure repo root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from ml.transforms.dct import compress_dct
from ml.transforms.hadamard import compress_hadamard


def benchmark_kernel(func, tensor, warmup=25, iters=100):
    # Warmup
    for _ in range(warmup):
        _ = func(tensor)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        _ = func(tensor)
    end.record()
    torch.cuda.synchronize()

    # elapsed_time returns milliseconds; convert to microseconds
    latency_us = (start.elapsed_time(end) / iters) * 1000.0
    return latency_us


def main():
    parser = argparse.ArgumentParser(description="Benchmark Transform Kernels on GPU")
    parser.add_argument("--retention", type=float, default=0.15, help="Retention ratio r")
    parser.add_argument("--output", type=str, default="ml/logs/transform_benchmark.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting benchmark.")
        return

    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(0)
    print(f"Benchmarking on: {device_name}")

    # Sizes in elements (float32: 4 bytes per element)
    # 64K elem = 256 KB, 256K elem = 1 MB, 1M elem = 4 MB, 4M elem = 16 MB, 16M elem = 64 MB
    sizes = [
        ("256 KB", 65536),
        ("1 MB", 262144),
        ("4 MB", 1048576),
        ("16 MB", 4194304),
        ("64 MB", 16777216)
    ]

    results = {
        "device": device_name,
        "retention_ratio": args.retention,
        "benchmarks": []
    }

    print("\n" + "=" * 70)
    print(f"{'Size':<10} | {'Bytes (MB)':<12} | {'DCT (us)':<15} | {'Hadamard (us)':<15} | {'Speedup'}")
    print("-" * 70)

    for label, numel in sizes:
        x = torch.randn(numel, device=device, dtype=torch.float32)
        bytes_mb = (numel * 4) / (1024 * 1024)

        dct_lat = benchmark_kernel(lambda t: compress_dct(t, args.retention), x)
        hadamard_lat = benchmark_kernel(lambda t: compress_hadamard(t, args.retention, mode="topk"), x)
        speedup = dct_lat / hadamard_lat if hadamard_lat > 0 else 1.0

        print(f"{label:<10} | {bytes_mb:<12.2f} | {dct_lat:<15.1f} | {hadamard_lat:<15.1f} | {speedup:.2f}x")

        results["benchmarks"].append({
            "label": label,
            "bytes_mb": bytes_mb,
            "numel": numel,
            "dct_latency_us": round(dct_lat, 2),
            "hadamard_latency_us": round(hadamard_lat, 2),
            "speedup": round(speedup, 2)
        })

    print("=" * 70)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved benchmark results to {args.output}")


if __name__ == "__main__":
    main()
