"""
Publication-Quality Visualization Suite for Protocol A:
Layer Recoverability & Sensitivity Ablation (Hypothesis 5).

Generates 3 publication figures saved into training/figures/convergence/:
1. fig08a_layer_recoverability_accuracy.png: Bar chart of Final Val Accuracy & Delta Acc across conditions.
2. fig08b_layer_recoverability_convergence.png: Multi-line convergence curves (Val Accuracy and Train Loss).
3. fig08c_layer_sensitivity_normalized.png: Parameter count vs accuracy drop evaluating H0 vs H1.
"""

import argparse
import json
import os
import sys

# Ensure repo root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import matplotlib.pyplot as plt
import numpy as np


def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "grid.color": "#E0E0E0",
        "grid.linestyle": "--",
        "grid.alpha": 0.6,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.2,
        "lines.markersize": 6,
    })

setup_style()


COLOR_MAP = {
    "baseline": "#1a365d",            # Dark Navy / Baseline
    "conv3x3_spatial": "#ff7f0e",     # Orange (matches 3x3 Conv in all previous tests)
    "conv1x1_expand": "#2ca02c",      # Green (matches 1x1 Expand in all previous tests)
    "conv1x1_reduce": "#1f77b4",      # Blue (matches 1x1 Reduce in all previous tests)
    "conv1x1_downsample": "#17becf",  # Cyan (matches 1x1 Downsample in all previous tests)
    "classifier_head": "#d62728",     # Red (matches Classifier Head in all previous tests)
    "skip_all": "#7f7f7f",            # Gray (All Layers)
}

LABEL_MAP = {
    "baseline": "Baseline (0% Skip)",
    "conv3x3_spatial": "3x3 Conv (Spatial)",
    "conv1x1_expand": "1x1 Conv (Expand)",
    "conv1x1_reduce": "1x1 Conv (Reduce)",
    "conv1x1_downsample": "1x1 Conv (Shortcut)",
    "classifier_head": "Classifier Head (fc)",
    "skip_all": "Skip All Layers",
}


def plot_accuracy_summary(data: dict, output_dir: str):
    """Figure 1: Final Validation Accuracy & Delta Acc across conditions."""
    conditions = data["conditions"]
    baseline = conditions.get("baseline")
    baseline_acc = baseline["final_val_acc"] if baseline else 0.0

    keys = [k for k in COLOR_MAP.keys() if k in conditions]
    names = [LABEL_MAP.get(k, k) for k in keys]
    accs = [conditions[k]["final_val_acc"] for k in keys]
    colors = [COLOR_MAP.get(k, "#333333") for k in keys]
    deltas = [round(conditions[k]["final_val_acc"] - baseline_acc, 2) for k in keys]

    fig, ax1 = plt.subplots(figsize=(11, 6))

    bars = ax1.bar(range(len(keys)), accs, color=colors, width=0.55, alpha=0.85)

    # Reference line for baseline
    ax1.axhline(baseline_acc, color="#2b5c8f", linestyle="--", linewidth=1.5, alpha=0.8,
                label=f"Baseline Reference ({baseline_acc:.2f}%)", zorder=2)

    # Annotate bars with accuracy and delta
    for i, (bar, delta, acc) in enumerate(zip(bars, deltas, accs)):
        h = bar.get_height()
        delta_str = f"{delta:+.2f}%" if keys[i] != "baseline" else "Ref (0.0%)"
        ax1.annotate(
            f"{acc:.1f}%\n({delta_str})",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=9.5, fontweight="bold",
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85),
            zorder=4
        )

    ax1.set_xticks(range(len(keys)))
    ax1.set_xticklabels(names, rotation=25, ha="right", fontweight="medium")
    ax1.set_ylabel("Final Validation Accuracy (%)", fontweight="bold")
    ax1.set_ylim(0, 115)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", framealpha=0.9)

    plt.suptitle(
        "Evaluating Hypothesis 5: Heterogeneous Layer Recoverability Under 50% Gradient Skipping\n"
        "(CIFAR-10 ResNet-50)",
        fontsize=13, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    out_path = os.path.join(output_dir, "fig08a_layer_recoverability_accuracy.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_convergence_trajectories(data: dict, output_dir: str):
    """Figure 2: Multi-line Validation Accuracy and Training Loss trajectories across epochs."""
    conditions = data["conditions"]
    keys = [k for k in COLOR_MAP.keys() if k in conditions]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for k in keys:
        c_data = conditions[k]
        history = c_data["epochs_history"]
        epochs = [e["epoch"] for e in history]
        val_accs = [e["val_acc"] for e in history]
        train_losses = [e["train_loss"] for e in history]

        color = COLOR_MAP.get(k, "#333333")
        label = LABEL_MAP.get(k, k)
        linestyle = "-" if k == "baseline" else ("-." if k == "skip_all" else "-")
        marker = "o" if k in ["baseline", "skip_all", "classifier_head"] else "s"

        ax1.plot(epochs, val_accs, label=label, color=color, linestyle=linestyle, marker=marker, markersize=5, alpha=0.9)
        ax2.plot(epochs, train_losses, label=label, color=color, linestyle=linestyle, marker=marker, markersize=5, alpha=0.9)

    ax1.set_title("Validation Accuracy Trajectory", fontweight="bold", fontsize=12)
    ax1.set_xlabel("Epoch", fontweight="bold")
    ax1.set_ylabel("Top-1 Validation Accuracy (%)", fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", framealpha=0.9, fontsize=9)

    ax2.set_title("Training Loss Trajectory", fontweight="bold", fontsize=12)
    ax2.set_xlabel("Epoch", fontweight="bold")
    ax2.set_ylabel("Cross Entropy Loss", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=9)

    plt.suptitle(
        "Evaluating Hypothesis 5: Convergence Trajectories Across Layer Skipping Conditions\n"
        "(Epoch-by-Epoch Progress Under 50% Update Skipping)",
        fontsize=13, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    out_path = os.path.join(output_dir, "fig08b_layer_recoverability_convergence.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_sensitivity_normalized(data: dict, output_dir: str):
    """Figure 3: Parameter Volume vs Accuracy Drop testing H0 (linear volume) vs H1 (intrinsic sensitivity)."""
    conditions = data["conditions"]
    baseline = conditions.get("baseline")
    baseline_acc = baseline["final_val_acc"] if baseline else 0.0

    # Probe conditions excluding baseline and skip_all for layer-intrinsic analysis
    probe_keys = [k for k in conditions.keys() if k not in ["baseline", "skip_all"]]
    if not probe_keys:
        return

    m_params = []
    acc_drops = []
    colors = []
    labels = []

    for k in probe_keys:
        c_data = conditions[k]
        mp = c_data["skipped_params"] / 1e6
        drop = baseline_acc - c_data["final_val_acc"]
        m_params.append(mp)
        acc_drops.append(drop)
        colors.append(COLOR_MAP.get(k, "#333333"))
        labels.append(LABEL_MAP.get(k, k).replace("Skip ", ""))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: Scatter of Parameter Volume vs Accuracy Drop
    for mp, drop, col, lbl in zip(m_params, acc_drops, colors, labels):
        ax1.scatter(mp, drop, color=col, s=120, alpha=0.9, zorder=5)
        xytext = (6, -12) if "Downsample" in lbl else (6, 4)
        ax1.annotate(
            lbl,
            xy=(mp, drop),
            xytext=xytext,
            textcoords="offset points",
            fontsize=9.5, fontweight="medium"
        )

    # Add H0 Linear Null Hypothesis reference line
    spatial_mp = conditions.get("conv3x3_spatial", {}).get("skipped_params", 11.3e6) / 1e6
    spatial_drop = baseline_acc - conditions.get("conv3x3_spatial", {}).get("final_val_acc", baseline_acc)
    slope = spatial_drop / spatial_mp if spatial_mp > 0 else 0.45
    x_h0 = np.linspace(0, 12.5, 50)
    ax1.plot(x_h0, slope * x_h0, "k--", alpha=0.6, linewidth=1.5, label=f"H0: Linear Parameter Scaling ({slope:.2f} pp/M)")

    ax1.set_title("Empirical Accuracy Drop vs Parameter Volume", fontweight="bold", fontsize=12)
    ax1.set_xlabel("Skipped Parameters (Millions)", fontweight="bold")
    ax1.set_ylabel("Accuracy Drop relative to Baseline (pp)", fontweight="bold")
    ax1.set_xlim(-0.5, 13.5)
    ax1.set_ylim(-1.5, 6.0)
    ax1.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=9)

    # Panel B: Sensitivity Index for Convolutional Feature Hierarchy
    conv_keys = ["conv3x3_spatial", "conv1x1_expand", "conv1x1_reduce", "conv1x1_downsample"]
    conv_keys = [k for k in conv_keys if k in conditions]
    conv_labels = [LABEL_MAP.get(k, k).replace("Skip ", "") for k in conv_keys]
    conv_colors = [COLOR_MAP.get(k, "#333333") for k in conv_keys]
    conv_sens = [
        round((baseline_acc - conditions[k]["final_val_acc"]) / (conditions[k]["skipped_params"] / 1e6), 3)
        for k in conv_keys
    ]

    bars = ax2.bar(range(len(conv_labels)), conv_sens, color=conv_colors, width=0.55, alpha=0.85)

    for bar, s_idx in zip(bars, conv_sens):
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        offset = 4 if h >= 0 else -12
        ax2.annotate(
            f"{s_idx:+.3f} pp/M",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center", va=va,
            fontsize=9.5, fontweight="bold"
        )

    ax2.set_xticks(range(len(conv_labels)))
    ax2.set_xticklabels(conv_labels, rotation=25, ha="right", fontweight="medium")
    ax2.set_title("Convolutional Layer Sensitivity (ΔAcc Drop / MParam)", fontweight="bold", fontsize=12)
    ax2.set_ylabel("Sensitivity Index (pp / MParam)", fontweight="bold")
    ax2.set_ylim(-0.12, max(conv_sens) * 1.30)
    ax2.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    if "classifier_head" in conditions:
        head_acc = conditions["classifier_head"]["final_val_acc"]
        head_delta = head_acc - baseline_acc
        ax2.text(
            0.97, 0.95,
            f"Classifier Head (0.02M params):\nΔAcc = {head_delta:+.2f}% (High Resilience)",
            transform=ax2.transAxes,
            ha="right", va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff9e6", edgecolor="#e6ab02", alpha=0.9)
        )

    plt.suptitle(
        "Evaluating Hypothesis 5: Intrinsic Layer Sensitivity Normalized by Parameter Volume\n"
        "(Disproportionate Impact Indicates Layer-Specific Sensitivity Beyond Raw Parameter Count)",
        fontsize=13, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    out_path = os.path.join(output_dir, "fig08c_layer_sensitivity_normalized.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Protocol A Layer Recoverability Results")
    parser.add_argument("--log-path", type=str, default="training/logs/layer_recoverability_ablation.json")
    parser.add_argument("--output-dir", type=str, default="training/figures/exp08_layer_recoverability")
    args = parser.parse_args()

    setup_style()

    if not os.path.exists(args.log_path):
        print(f"Error: Log file not found at {args.log_path}")
        sys.exit(1)

    with open(args.log_path, "r") as f:
        data = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    plot_accuracy_summary(data, args.output_dir)
    plot_convergence_trajectories(data, args.output_dir)
    plot_sensitivity_normalized(data, args.output_dir)
    print("\nAll figures generated successfully.")


if __name__ == "__main__":
    main()
