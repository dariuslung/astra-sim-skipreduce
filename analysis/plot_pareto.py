"""
Generate publication-grade, purpose-driven evaluation plots from training run logs.
Separates figures by research hypothesis:
  - Fig 1: Heavy Skip (s=2) Recovery (Baseline vs. Naive vs. DCT vs. WHT)
  - Fig 2: Mild Skip (s=1) & Error Feedback Ablation
  - Fig 3: Degradation vs. Recovery Across Skip Degrees (s=0, 1, 2)
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
            # Filter out incomplete / sanity check runs with < 5 epochs
            if len(data.get("epochs_data", [])) >= 5:
                runs.append(data)
    return runs


def print_summary_table(runs):
    if not runs:
        print("No completed 20-epoch run logs found.")
        return

    print("\n" + "=" * 105)
    print(f"{'Run ID':<35} | {'Skip':<5} | {'Transform':<10} | {'Ret':<5} | {'EF':<5} | {'Val Acc (%)':<12} | {'CosSim':<8} | {'Loss':<8}")
    print("-" * 105)

    sorted_runs = sorted(runs, key=lambda r: (r.get("skip", 0), -r["epochs_data"][-1]["val_acc"]))

    for r in sorted_runs:
        last = r["epochs_data"][-1]
        run_name = r["run_id"][:33]
        skip_s = r.get("skip", 0)
        trans = r["transform"]
        ret = f"{r['retention_ratio']:.2f}"
        ef = "Yes" if r.get("use_ef") else "No"
        val_acc = f"{last.get('val_acc', 0):.2f}%"
        cossim = f"{last.get('mean_cos_sim', 0):.4f}"
        loss = f"{last.get('train_loss', 0):.4f}"
        print(f"{run_name:<35} | {skip_s:<5} | {trans:<10} | {ret:<5} | {ef:<5} | {val_acc:<12} | {cossim:<8} | {loss:<8}")

    print("=" * 105)


def find_run(runs, skip, transform, ret=None, use_ef=None):
    """Helper to retrieve a specific run from the list."""
    for r in runs:
        if r.get("skip") == skip and r.get("transform") == transform:
            if ret is not None and abs(r.get("retention_ratio", 0.0) - ret) > 1e-4:
                continue
            if use_ef is not None and r.get("use_ef") != use_ef:
                continue
            return r
    return None


def plot_heavy_skip_recovery(runs, out_dir: str):
    """
    Figure 1: Heavy Skip (s=2, 50% dropped) Recovery
    Compares Baseline (s=0), Naive Skip (s=2), DCT (s=2), and Hadamard (s=2).
    """
    import matplotlib.pyplot as plt

    baseline = find_run(runs, skip=0, transform="none")
    naive_s2 = find_run(runs, skip=2, transform="none")
    dct_s2 = find_run(runs, skip=2, transform="dct", ret=0.15, use_ef=False)
    hadamard_s2 = find_run(runs, skip=2, transform="hadamard", ret=0.15, use_ef=False)

    targets = [
        (baseline, "Baseline (s=0, Full AllReduce)", "black", "--", "o"),
        (naive_s2, "Naive Skip (s=2, 50% dropped)", "#d9534f", "-", "x"),
        (dct_s2, "DCT Recovery (s=2, r=0.15)", "#0275d8", "-", "s"),
        (hadamard_s2, "Walsh-Hadamard (s=2, r=0.15)", "#6f42c1", "-", "^"),
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

    # Subplot 1: Validation Accuracy
    ax1.set_title("Validation Accuracy under Heavy Skip (s=2, 50% dropped)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Top-1 Val Accuracy (%)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", framealpha=0.9)

    # Subplot 2: Gradient Cosine Similarity
    ax2.set_title("Gradient Direction Alignment (Cosine Similarity vs. AllReduce)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Cosine Similarity", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    out_file = os.path.join(out_dir, "fig1_heavy_skip_s2_recovery.png")
    plt.savefig(out_file, dpi=250)
    plt.close()
    print(f"Generated: {out_file}")


def plot_mild_skip_and_ef(runs, out_dir: str):
    """
    Figure 2: Mild Skip (s=1, 25% dropped) and Error Feedback Ablation
    Compares Baseline, Naive Skip (s=1), DCT (No EF), and DCT (+ EF).
    """
    import matplotlib.pyplot as plt

    baseline = find_run(runs, skip=0, transform="none")
    naive_s1 = find_run(runs, skip=1, transform="none")
    dct_noef = find_run(runs, skip=1, transform="dct", ret=0.15, use_ef=False)
    dct_ef = find_run(runs, skip=1, transform="dct", ret=0.15, use_ef=True)

    targets = [
        (baseline, "Baseline (s=0, Full AllReduce)", "black", "--", "o"),
        (naive_s1, "Naive Skip (s=1, 25% dropped)", "#d9534f", "-", "x"),
        (dct_noef, "DCT (s=1, r=0.15, No EF)", "#0275d8", "-", "s"),
        (dct_ef, "DCT + EF (s=1, r=0.15, With EF)", "#5cb85c", "-", "D"),
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

    ax1.set_title("Mild Skip (s=1) & Error Feedback Accuracy", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Top-1 Val Accuracy (%)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", framealpha=0.9)

    ax2.set_title("Gradient Cosine Similarity Drift vs. EF Stabilization", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Cosine Similarity", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    out_file = os.path.join(out_dir, "fig2_mild_skip_s1_ef_ablation.png")
    plt.savefig(out_file, dpi=250)
    plt.close()
    print(f"Generated: {out_file}")


def plot_skip_scaling_degradation(runs, out_dir: str):
    """
    Figure 3: Degradation vs. Recovery Across Skip Degrees (s = 0, 1, 2)
    Demonstrates how naive skip degrades as s increases, and how DCT bounds the degradation.
    """
    import matplotlib.pyplot as plt

    baseline = find_run(runs, skip=0, transform="none")
    naive_s1 = find_run(runs, skip=1, transform="none")
    naive_s2 = find_run(runs, skip=2, transform="none")
    dct_s1 = find_run(runs, skip=1, transform="dct", ret=0.15, use_ef=False)
    dct_s2 = find_run(runs, skip=2, transform="dct", ret=0.15, use_ef=False)

    skips = [0, 1, 2]
    naive_accs = [
        baseline["epochs_data"][-1]["val_acc"] if baseline else 0,
        naive_s1["epochs_data"][-1]["val_acc"] if naive_s1 else 0,
        naive_s2["epochs_data"][-1]["val_acc"] if naive_s2 else 0,
    ]
    dct_accs = [
        baseline["epochs_data"][-1]["val_acc"] if baseline else 0,
        dct_s1["epochs_data"][-1]["val_acc"] if dct_s1 else 0,
        dct_s2["epochs_data"][-1]["val_acc"] if dct_s2 else 0,
    ]

    naive_cossim = [
        1.0,
        naive_s1["epochs_data"][-1]["mean_cos_sim"] if naive_s1 else 0,
        naive_s2["epochs_data"][-1]["mean_cos_sim"] if naive_s2 else 0,
    ]
    dct_cossim = [
        1.0,
        dct_s1["epochs_data"][-1]["mean_cos_sim"] if dct_s1 else 0,
        dct_s2["epochs_data"][-1]["mean_cos_sim"] if dct_s2 else 0,
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    ax1.plot(skips, naive_accs, marker="o", color="#d9534f", linewidth=2.5, markersize=8, label="Naive SkipReduce (r=0)")
    ax1.plot(skips, dct_accs, marker="s", color="#0275d8", linewidth=2.5, markersize=8, label="DCT SkipReduce (r=0.15)")
    ax1.set_title("Accuracy Degradation vs. Skip Steps (s)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Skipped Steps (s) on N=4", fontsize=10)
    ax1.set_ylabel("Final Validation Accuracy (%)", fontsize=10)
    ax1.set_xticks([0, 1, 2])
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower left")

    # Annotate the gap recovery at s=2
    if naive_accs[2] > 0 and dct_accs[2] > 0:
        gain = dct_accs[2] - naive_accs[2]
        ax1.annotate(
            f"+{gain:.2f}% Recovery",
            xy=(2, dct_accs[2]),
            xytext=(1.6, dct_accs[2] - 0.4),
            arrowprops=dict(facecolor="black", arrowstyle="->", lw=1.5),
            fontweight="bold",
            fontsize=9
        )

    ax2.plot(skips, naive_cossim, marker="o", color="#d9534f", linewidth=2.5, markersize=8, label="Naive SkipReduce (r=0)")
    ax2.plot(skips, dct_cossim, marker="s", color="#0275d8", linewidth=2.5, markersize=8, label="DCT SkipReduce (r=0.15)")
    ax2.set_title("Gradient Cosine Similarity vs. Skip Steps (s)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Skipped Steps (s) on N=4", fontsize=10)
    ax2.set_ylabel("Final Gradient Cosine Similarity", fontsize=10)
    ax2.set_xticks([0, 1, 2])
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower left")

    plt.tight_layout()
    out_file = os.path.join(out_dir, "fig3_skip_scaling_ablation.png")
    plt.savefig(out_file, dpi=250)
    plt.close()
    print(f"Generated: {out_file}")


def plot_pareto_frontier(runs, out_dir: str):
    """
    Figure 4: The Pareto Frontier (Accuracy vs. Transmitted Communication Volume)
    Payload ratio: rho = 1 - (s * (1 - r)) / (2 * (N - 1))
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))

    points = []
    # (run_obj, label, color, marker)
    for r in runs:
        last = r["epochs_data"][-1]
        acc = last["val_acc"]
        s = r.get("skip", 0)
        N = r.get("ranks", 4)
        transform = r.get("transform", "none")
        ret = r.get("retention_ratio", 0.0)
        use_ef = r.get("use_ef", False)

        # Theoretical network communication payload ratio vs. full AllReduce
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
            label = f"DCT (s={s}, r={ret})" + (" +EF" if use_ef else "")
            color = "#0275d8" if not use_ef else "#5cb85c"
            marker = "s" if not use_ef else "D"
        elif transform == "hadamard":
            payload_pct = (1.0 - (s * (1.0 - ret)) / (2.0 * (N - 1))) * 100.0
            label = f"WHT (s={s}, r={ret})"
            color = "#6f42c1"
            marker = "^"

        points.append((payload_pct, acc, label, color, marker))

    # Plot points
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

    # Highlight Pareto region
    ax.axvspan(70, 86, color="yellow", alpha=0.1, label="Optimal Communication Savings Zone")

    ax.set_title("The Pareto Frontier: Model Accuracy vs. Network Payload", fontsize=12, fontweight="bold")
    ax.set_xlabel("Transmitted Communication Volume (% of Full AllReduce)", fontsize=11)
    ax.set_ylabel("Top-1 Validation Accuracy (%)", fontsize=11)
    ax.set_xlim(60, 105)
    ax.set_ylim(90.2, 92.6)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Clean deduplicated legend
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
    parser.add_argument("--log-dir", type=str, default="ml/logs", help="Directory containing run JSON logs")
    parser.add_argument("--out-dir", type=str, default="ml/logs", help="Directory to save generated PNG plots")
    args = parser.parse_args()

    runs = parse_all_logs(args.log_dir)
    print_summary_table(runs)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; cannot generate plots.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    print("\nGenerating purpose-driven figure suite...")
    plot_heavy_skip_recovery(runs, args.out_dir)
    plot_mild_skip_and_ef(runs, args.out_dir)
    plot_skip_scaling_degradation(runs, args.out_dir)
    plot_pareto_frontier(runs, args.out_dir)
    print("\nAll 4 figures generated successfully!")


if __name__ == "__main__":
    main()
