"""
Plotting script for compute time savings across layer skipping configurations.
Generates stacked breakdown and savings efficiency figures.
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt


def setup_style():
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.size": 11,
        "font.family": "sans-serif",
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "lines.linewidth": 2.0,
        "grid.alpha": 0.3,
    })


def get_model_title(model_name: str) -> str:
    if model_name == "resnet50":
        return "ResNet-50 (CV Bottleneck CNN)"
    elif model_name == "vit":
        return "Vision Transformer (ViT-Tiny)"
    elif model_name == "gpt":
        return "Causal Transformer (GPT-Tiny)"
    return model_name.upper()


def plot_compute_breakdown(data: dict, model_name: str, output_dir: str):
    """Stacked bar chart showing Forward, Backward, and Optimizer times per configuration."""
    configs = data["configurations"]
    names = [c["name"] for c in configs]
    fwd = [c["fwd_ms"] for c in configs]
    bwd = [c["bwd_ms"] for c in configs]
    opt = [c["opt_ms"] for c in configs]

    x = np.arange(len(names))
    width = 0.55

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

    color_fwd = "#1f77b4"
    color_bwd = "#d62728"
    color_opt = "#ff7f0e"

    bars_fwd = ax.bar(x, fwd, width, label="Forward Pass", color=color_fwd, alpha=0.85)
    bars_bwd = ax.bar(x, bwd, width, bottom=fwd, label="Backward Pass", color=color_bwd, alpha=0.85)
    bottom_opt = np.array(fwd) + np.array(bwd)
    bars_opt = ax.bar(x, opt, width, bottom=bottom_opt, label="Optimizer Step", color=color_opt, alpha=0.85)

    # Annotate total latency on top of bars
    for i in range(len(names)):
        total_val = bottom_opt[i] + opt[i]
        bwd_saved = configs[i]["bwd_saved_pct"]
        if bwd_saved > 0:
            label_text = f"{total_val:.2f} ms\n(-{bwd_saved:.1f}% bwd)"
        else:
            label_text = f"{total_val:.2f} ms"
        ax.text(x[i], total_val + 0.1, label_text, ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_ylabel("Execution Time per Iteration (ms)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylim(0, max(bottom_opt + opt) * 1.25)

    title_model = get_model_title(model_name)
    plt.title(f"GPU Compute Time Breakdown by Layer Skipping: {title_model}\n(RTX 4060 Ti Measured Latency)",
              fontsize=13, fontweight="bold", pad=12)

    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"fig_compute_breakdown_{model_name}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_savings_efficiency(data: dict, model_name: str, output_dir: str):
    """Dual-bar chart showing % Parameters Skipped vs % Backward Time Saved."""
    configs = [c for c in data["configurations"] if c["id"] != "baseline"]
    names = [c["name"] for c in configs]
    params_pct = [c["pct_params_skipped"] for c in configs]
    bwd_saved_pct = [c["bwd_saved_pct"] for c in configs]

    y = np.arange(len(names))
    height = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    color_params = "#7f7f7f"
    color_saved = "#2ca02c"

    rects1 = ax.barh(y - height/2, params_pct, height, label="% Model Parameters Skipped", color=color_params, alpha=0.85)
    rects2 = ax.barh(y + height/2, bwd_saved_pct, height, label="% Backward Compute Time Saved", color=color_saved, alpha=0.85)

    for i in range(len(names)):
        ax.text(params_pct[i] + 1, y[i] - height/2, f"{params_pct[i]:.1f}%", va="center", fontsize=8.5, color="#333333")
        ax.text(bwd_saved_pct[i] + 1, y[i] + height/2, f"{bwd_saved_pct[i]:.1f}%", va="center", fontsize=8.5, fontweight="bold", color=color_saved)

    ax.set_xlabel("Percentage (%)", fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, max(max(params_pct), max(bwd_saved_pct)) * 1.25)
    ax.invert_yaxis()  # Top-down order

    title_model = get_model_title(model_name)
    plt.title(f"Layer Skipping Compute Savings Efficiency: {title_model}\n(Parameters Skipped vs. Backward Latency Reduced)",
              fontsize=13, fontweight="bold", pad=12)

    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"fig_compute_savings_summary_{model_name}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot compute savings benchmarks.")
    parser.add_argument("--json", type=str, required=True, help="Path to compute savings JSON")
    parser.add_argument("--output-dir", type=str, default="training/figures/compute_savings", help="Output directory for plots")

    args = parser.parse_args()

    setup_style()
    with open(args.json, "r") as f:
        data = json.load(f)

    model_name = data["model"].lower()
    os.makedirs(args.output_dir, exist_ok=True)

    plot_compute_breakdown(data, model_name, args.output_dir)
    plot_savings_efficiency(data, model_name, args.output_dir)
    print(f"All figures generated successfully for {model_name}!")


if __name__ == "__main__":
    main()
