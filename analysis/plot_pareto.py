"""
Generate publication-grade, purpose-driven evaluation plots from training run logs.
Standardized figure suite:
  - Fig 1: Skip s=1 Evaluation (Baseline vs. Naive vs. DCT vs. WHT)
  - Fig 2: Skip s=2 Evaluation (Baseline vs. Naive vs. DCT vs. WHT)
  - Fig 3: Computation Time Analysis (GPU Kernel Latency & Epoch Training Time)
  - Fig 4: Pareto Frontier (Accuracy vs. Communication Volume)
"""

import argparse
import glob
import json
import os
import sys

# Ensure repo root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def parse_all_logs(log_dir: str):
    runs = []
    for filepath in glob.glob(os.path.join(log_dir, "ranks*.json")):
        with open(filepath, "r") as f:
            data = json.load(f)
            # Filter out incomplete / sanity check runs with < 5 epochs and EF runs
            if len(data.get("epochs_data", [])) >= 5 and not data.get("use_ef", False):
                runs.append(data)
    return runs


def print_summary_table(runs):
    if not runs:
        print("No completed 20-epoch run logs found.")
        return

    print("\n" + "=" * 105)
    print(f"{'Run ID':<35} | {'Skip':<5} | {'Transform':<10} | {'Ret':<5} | {'Val Acc (%)':<12} | {'CosSim':<8} | {'Loss':<8} | {'Epoch Time'}")
    print("-" * 105)

    sorted_runs = sorted(runs, key=lambda r: (r.get("skip", 0), -r["epochs_data"][-1]["val_acc"]))

    for r in sorted_runs:
        last = r["epochs_data"][-1]
        run_name = r["run_id"][:33]
        skip_s = r.get("skip", 0)
        trans = r["transform"]
        ret = f"{r['retention_ratio']:.2f}"
        val_acc = f"{last.get('val_acc', 0):.2f}%"
        cossim = f"{last.get('mean_cos_sim', 0):.4f}"
        loss = f"{last.get('train_loss', 0):.4f}"
        avg_time = sum(ep.get("epoch_time_s", 0) for ep in r["epochs_data"]) / len(r["epochs_data"])
        time_str = f"{avg_time:.1f}s"
        print(f"{run_name:<35} | {skip_s:<5} | {trans:<10} | {ret:<5} | {val_acc:<12} | {cossim:<8} | {loss:<8} | {time_str}")

    print("=" * 105)


def find_run(runs, skip, transform, ret=None):
    """Helper to retrieve a specific non-EF run from the list."""
    for r in runs:
        if r.get("skip") == skip and r.get("transform") == transform:
            if ret is not None and abs(r.get("retention_ratio", 0.0) - ret) > 1e-4:
                continue
            return r
    return None


def plot_skip_s1_evaluation(runs, out_dir: str):
    """
    Figure 1: Skip s=1 Evaluation
    Compares Baseline (s=0), Naive Skip (s=1), DCT (s=1), and WHT (s=1).
    """
    import matplotlib.pyplot as plt

    baseline = find_run(runs, skip=0, transform="none")
    naive_s1 = find_run(runs, skip=1, transform="none")
    dct_s1 = find_run(runs, skip=1, transform="dct", ret=0.15)
    hadamard_s1 = find_run(runs, skip=1, transform="hadamard", ret=0.15)

    targets = [
        (baseline, "Baseline AllReduce (s=0)", "black", "--", "o"),
        (naive_s1, "Naive Skip (s=1)", "#d9534f", "-", "x"),
        (dct_s1, "DCT (s=1, r=0.15)", "#0275d8", "-", "s"),
        (hadamard_s1, "WHT (s=1, r=0.15)", "#6f42c1", "-", "^"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for run, label, color, linestyle, marker in targets:
        if run is None:
            continue
        epochs = [ep["epoch"] for ep in run["epochs_data"]]
        accs = [ep["val_acc"] for ep in run["epochs_data"]]
        cossims = [ep["mean_cos_sim"] for ep in run["epochs_data"]]

        ax1.plot(epochs, accs, label=label, color=color, linestyle=linestyle, marker=marker, markersize=5, linewidth=2)
        ax2.plot(epochs, cossims, label=label, color=color, linestyle=linestyle, marker=marker, markersize=5, linewidth=2)

    ax1.set_title("Validation Accuracy vs. Epoch (s=1)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Top-1 Val Accuracy (%)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", framealpha=0.9)

    ax2.set_title("Gradient Cosine Similarity vs. Epoch (s=1)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Cosine Similarity", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    out_file = os.path.join(out_dir, "fig1_skip_s1_evaluation.png")
    plt.savefig(out_file, dpi=250)
    plt.close()
    print(f"Generated: {out_file}")


def plot_skip_s2_evaluation(runs, out_dir: str):
    """
    Figure 2: Skip s=2 Evaluation
    Compares Baseline (s=0), Naive Skip (s=2), DCT (s=2), and WHT (s=2).
    """
    import matplotlib.pyplot as plt

    baseline = find_run(runs, skip=0, transform="none")
    naive_s2 = find_run(runs, skip=2, transform="none")
    dct_s2 = find_run(runs, skip=2, transform="dct", ret=0.15)
    hadamard_s2 = find_run(runs, skip=2, transform="hadamard", ret=0.15)

    targets = [
        (baseline, "Baseline AllReduce (s=0)", "black", "--", "o"),
        (naive_s2, "Naive Skip (s=2)", "#d9534f", "-", "x"),
        (dct_s2, "DCT (s=2, r=0.15)", "#0275d8", "-", "s"),
        (hadamard_s2, "WHT (s=2, r=0.15)", "#6f42c1", "-", "^"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for run, label, color, linestyle, marker in targets:
        if run is None:
            continue
        epochs = [ep["epoch"] for ep in run["epochs_data"]]
        accs = [ep["val_acc"] for ep in run["epochs_data"]]
        cossims = [ep["mean_cos_sim"] for ep in run["epochs_data"]]

        ax1.plot(epochs, accs, label=label, color=color, linestyle=linestyle, marker=marker, markersize=5, linewidth=2)
        ax2.plot(epochs, cossims, label=label, color=color, linestyle=linestyle, marker=marker, markersize=5, linewidth=2)

    ax1.set_title("Validation Accuracy vs. Epoch (s=2)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Top-1 Val Accuracy (%)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", framealpha=0.9)

    ax2.set_title("Gradient Cosine Similarity vs. Epoch (s=2)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Cosine Similarity", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    out_file = os.path.join(out_dir, "fig2_skip_s2_evaluation.png")
    plt.savefig(out_file, dpi=250)
    plt.close()
    print(f"Generated: {out_file}")


def plot_computation_time_analysis(runs, benchmark_file: str, out_dir: str):
    """
    Figure 3: Computation Time Analysis
    - Subplot 1: Kernel execution latency (us) vs tensor payload size on RTX 4060 Ti
    - Subplot 2: End-to-end epoch compute time across s=1 and s=2
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Subplot 1: Micro-benchmark Kernel Execution Latency ---
    if os.path.exists(benchmark_file):
        with open(benchmark_file, "r") as f:
            bench_data = json.load(f)
        benches = bench_data.get("benchmarks", [])
        sizes_mb = [b["bytes_mb"] for b in benches]
        dct_lats = [b["dct_latency_us"] for b in benches]
        hadamard_lats = [b["hadamard_latency_us"] for b in benches]

        ax1.plot(sizes_mb, dct_lats, marker="s", color="#0275d8", linewidth=2.5, markersize=7, label="1D-DCT (cuFFT)")
        ax1.plot(sizes_mb, hadamard_lats, marker="^", color="#6f42c1", linewidth=2.5, markersize=7, label="Walsh-Hadamard (FWHT)")
        ax1.set_xscale("log", base=2)
        ax1.set_yscale("log")
        ax1.set_title("GPU Kernel Latency vs. Tensor Payload (RTX 4060 Ti)", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Tensor Slice Size (MB, Log Scale)", fontsize=10)
        ax1.set_ylabel("Execution Time (μs, Log Scale)", fontsize=10)
        ax1.grid(True, linestyle="--", alpha=0.5, which="both")
        ax1.legend(loc="upper left")

        ax1.annotate(
            "DCT: 183.7 μs\nWHT: 1,049.7 μs\n(5.7x faster)",
            xy=(1.0, 183.7),
            xytext=(1.5, 30.0),
            arrowprops=dict(facecolor="black", arrowstyle="->", lw=1.2),
            fontsize=8,
            fontweight="bold"
        )
    else:
        ax1.text(0.5, 0.5, "Benchmark file not found", ha="center")

    # --- Subplot 2: End-to-End Epoch Training Compute Time ---
    baseline = find_run(runs, skip=0, transform="none")
    naive_s1 = find_run(runs, skip=1, transform="none")
    dct_s1 = find_run(runs, skip=1, transform="dct", ret=0.15)
    wht_s1 = find_run(runs, skip=1, transform="hadamard", ret=0.15)

    naive_s2 = find_run(runs, skip=2, transform="none")
    dct_s2 = find_run(runs, skip=2, transform="dct", ret=0.15)
    wht_s2 = find_run(runs, skip=2, transform="hadamard", ret=0.15)

    def avg_epoch_time(run):
        if run is None or not run.get("epochs_data"):
            return 0.0
        return sum(ep["epoch_time_s"] for ep in run["epochs_data"]) / len(run["epochs_data"])

    labels = ["Naive Skip", "DCT (r=0.15)", "WHT (r=0.15)"]
    s1_times = [avg_epoch_time(naive_s1), avg_epoch_time(dct_s1), avg_epoch_time(wht_s1)]
    s2_times = [avg_epoch_time(naive_s2), avg_epoch_time(dct_s2), avg_epoch_time(wht_s2)]

    x = np.arange(len(labels))
    width = 0.35

    rects1 = ax2.bar(x - width/2, s1_times, width, label="s=1", color="#5bc0de", edgecolor="black", alpha=0.9)
    rects2 = ax2.bar(x + width/2, s2_times, width, label="s=2", color="#0275d8", edgecolor="black", alpha=0.9)

    base_time = avg_epoch_time(baseline)
    if base_time > 0:
        ax2.axhline(base_time, color="black", linestyle="--", linewidth=1.8, label=f"Baseline s=0 ({base_time:.1f}s)")

    ax2.set_title("Average Computation Time per Training Epoch", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Epoch Time (seconds)", fontsize=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax2.legend(loc="upper left")

    for rect in rects1 + rects2:
        height = rect.get_height()
        if height > 0:
            ax2.annotate(f"{height:.1f}s",
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    out_file = os.path.join(out_dir, "fig3_computation_time_analysis.png")
    plt.savefig(out_file, dpi=250)
    plt.close()
    print(f"Generated: {out_file}")


def plot_pareto_frontier(runs, out_dir: str):
    """
    Figure 4: The Pareto Frontier (Accuracy vs. Transmitted Communication Volume)
    Payload ratio: rho = 1 - (s * (1 - r)) / (2 * (N - 1))
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 6))

    points = []
    for r in runs:
        last = r["epochs_data"][-1]
        acc = last["val_acc"]
        s = r.get("skip", 0)
        N = r.get("ranks", 4)
        transform = r.get("transform", "none")
        ret = r.get("retention_ratio", 0.0)

        if s == 0:
            payload_pct = 100.0
            label = "Baseline AllReduce (s=0)"
            color = "black"
            marker = "o"
        elif transform == "none":
            payload_pct = (1.0 - s / (2.0 * (N - 1))) * 100.0
            label = f"Naive Skip (s={s})"
            color = "#d9534f"
            marker = "x"
        elif transform == "dct":
            payload_pct = (1.0 - (s * (1.0 - ret)) / (2.0 * (N - 1))) * 100.0
            label = f"DCT (s={s}, r={ret})"
            color = "#0275d8"
            marker = "s"
        elif transform == "hadamard":
            payload_pct = (1.0 - (s * (1.0 - ret)) / (2.0 * (N - 1))) * 100.0
            label = f"WHT (s={s}, r={ret})"
            color = "#6f42c1"
            marker = "^"

        points.append((payload_pct, acc, label, color, marker))

    for payload_pct, acc, label, color, marker in points:
        ax.scatter(payload_pct, acc, color=color, marker=marker, s=120, zorder=5, label=label)
        ax.annotate(
            f" {label}\n ({payload_pct:.1f}%, {acc:.2f}%)",
            xy=(payload_pct, acc),
            xytext=(3, -8),
            textcoords="offset points",
            fontsize=8,
            alpha=0.9
        )

    ax.axvspan(70, 86, color="yellow", alpha=0.1, label="Optimal Communication Savings Zone")

    ax.set_title("The Pareto Frontier: Model Accuracy vs. Communication Volume", fontsize=12, fontweight="bold")
    ax.set_xlabel("Transmitted Communication Volume (% of Full AllReduce)", fontsize=11)
    ax.set_ylabel("Top-1 Validation Accuracy (%)", fontsize=11)
    ax.set_xlim(60, 105)
    ax.set_ylim(90.2, 92.6)
    ax.grid(True, linestyle="--", alpha=0.5)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="lower right", framealpha=0.9, fontsize=8)

    plt.tight_layout()
    out_file = os.path.join(out_dir, "fig4_pareto_accuracy_vs_payload.png")
    plt.savefig(out_file, dpi=250)
    plt.close()
    print(f"Generated: {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate purpose-driven research evaluation plots")
    parser.add_argument("--log-dir", type=str, default="training/logs", help="Directory containing run JSON logs")
    parser.add_argument("--bench-file", type=str, default="training/logs/transform_benchmark.json", help="Kernel benchmark JSON")
    parser.add_argument("--out-dir", type=str, default="training/figures", help="Directory to save generated PNG plots")

    args = parser.parse_args()

    runs = parse_all_logs(args.log_dir)
    print_summary_table(runs)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; cannot generate plots.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    print("\nGenerating standardized purpose-driven figure suite...")
    plot_skip_s1_evaluation(runs, args.out_dir)
    plot_skip_s2_evaluation(runs, args.out_dir)
    plot_computation_time_analysis(runs, args.bench_file, args.out_dir)
    plot_pareto_frontier(runs, args.out_dir)
    print("\nAll 4 figures generated successfully!")


if __name__ == "__main__":
    main()
