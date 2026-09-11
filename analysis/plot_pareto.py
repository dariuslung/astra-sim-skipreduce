"""
Generate comparison plots and summary tables from training run logs.
Compares convergence curves, final accuracy, and gradient cosine similarity.
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
            runs.append(data)
    return runs


def print_summary_table(runs):
    if not runs:
        print("No run logs found.")
        return

    print("\n" + "=" * 95)
    print(f"{'Run ID':<35} | {'Transform':<10} | {'Ret':<6} | {'EF':<5} | {'Val Acc (%)':<12} | {'CosSim':<8} | {'Loss':<8}")
    print("-" * 95)

    # Sort runs by validation accuracy descending
    sorted_runs = sorted(runs, key=lambda r: r["epochs_data"][-1]["val_acc"] if r["epochs_data"] else 0, reverse=True)

    for r in sorted_runs:
        last = r["epochs_data"][-1] if r["epochs_data"] else {}
        run_name = r["run_id"][:33]
        trans = r["transform"]
        ret = f"{r['retention_ratio']:.2f}"
        ef = "Yes" if r.get("use_ef") else "No"
        val_acc = f"{last.get('val_acc', 0):.2f}%"
        cossim = f"{last.get('mean_cos_sim', 0):.4f}"
        loss = f"{last.get('train_loss', 0):.4f}"
        print(f"{run_name:<35} | {trans:<10} | {ret:<6} | {ef:<5} | {val_acc:<12} | {cossim:<8} | {loss:<8}")

    print("=" * 95)


def plot_curves(runs, output_path: str = "ml/logs/convergence_comparison.png"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot generation.")
        return

    if not runs:
        return

    plt.figure(figsize=(12, 5))

    # Subplot 1: Validation Accuracy
    plt.subplot(1, 2, 1)
    for r in runs:
        epochs = [ep["epoch"] for ep in r["epochs_data"]]
        accs = [ep["val_acc"] for ep in r["epochs_data"]]
        label = f"{r['transform']} (r={r['retention_ratio']})" + (" +EF" if r.get("use_ef") else "")
        plt.plot(epochs, accs, marker="o", label=label)
    plt.title("CIFAR-10 Validation Accuracy (%)")
    plt.xlabel("Epoch")
    plt.ylabel("Top-1 Accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # Subplot 2: Gradient Cosine Similarity
    plt.subplot(1, 2, 2)
    for r in runs:
        epochs = [ep["epoch"] for ep in r["epochs_data"]]
        cossims = [ep["mean_cos_sim"] for ep in r["epochs_data"]]
        label = f"{r['transform']} (r={r['retention_ratio']})" + (" +EF" if r.get("use_ef") else "")
        plt.plot(epochs, cossims, marker="s", label=label)
    plt.title("Mean Gradient Cosine Similarity vs. AllReduce")
    plt.xlabel("Epoch")
    plt.ylabel("Cosine Similarity")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Saved convergence plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate experiment comparison tables and plots")
    parser.add_argument("--log-dir", type=str, default="ml/logs", help="Directory containing run JSON logs")
    parser.add_argument("--plot", action="store_true", help="Generate PNG plots")
    parser.add_argument("--output-plot", type=str, default="ml/logs/convergence_comparison.png")
    args = parser.parse_args()

    runs = parse_all_logs(args.log_dir)
    print_summary_table(runs)

    if args.plot:
        plot_curves(runs, args.output_plot)


if __name__ == "__main__":
    main()
