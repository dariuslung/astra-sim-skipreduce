"""
Visualization suite for Intra-Epoch Gradient Stability in ResNet-50.
Plots:
1. fig_intra_epoch_metric_variance.png (Anchor T0 Hoyer vs. Intra-Epoch Spread & CV%)
2. fig_intra_epoch_mask_decay.png       (Top-10% Mask IoU Decay Across Intra-Epoch Steps)
3. fig_intra_epoch_cosine_drift.png     (Gradient Vector Cosine Similarity Decay with T0)
"""

import argparse
import json
import os
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
    })


def plot_metric_variance(data: dict, output_dir: str):
    epochs_data = data["epochs_data"]
    epochs = [d["epoch"] for d in epochs_data]

    t0_hoyers = [d["checkpoints"][0]["global_hoyer"] for d in epochs_data]
    min_hoyers = [min(c["global_hoyer"] for c in d["checkpoints"]) for d in epochs_data]
    max_hoyers = [max(c["global_hoyer"] for c in d["checkpoints"]) for d in epochs_data]
    cv_hoyers = [d["intra_epoch_hoyer_cv_pct"] for d in epochs_data]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7.5), dpi=300, sharex=True)

    # Top Panel: Anchor Hoyer with Min-Max Intra-Epoch Envelope
    color_hoyer = "#2ca02c"
    color_band = "#98df8a"

    ax1.plot(epochs, t0_hoyers, color=color_hoyer, marker="o", linewidth=2.5, label="Initial Anchor (Batches 1-5)")
    ax1.fill_between(epochs, min_hoyers, max_hoyers, color=color_band, alpha=0.4, label="Intra-Epoch Range (Min to Max Checkpoint)")
    ax1.plot(epochs, min_hoyers, color=color_hoyer, linestyle=":", alpha=0.7)
    ax1.plot(epochs, max_hoyers, color=color_hoyer, linestyle=":", alpha=0.7)

    ax1.set_ylabel("Global Hoyer Sparsity [0-1]", fontweight="bold")
    ax1.set_title("ResNet-50 Intra-Epoch Sparsity Stability Across 20 Epochs\n(Evaluating Whether Initial Anchor Proxy Holds Across 390 Batches)",
                  fontsize=13, fontweight="bold", pad=10)
    ax1.grid(True)
    ax1.legend(loc="upper right", frameon=True)

    # Bottom Panel: Coefficient of Variation (CV %)
    color_cv = "#1f77b4"
    ax2.bar(epochs, cv_hoyers, color=color_cv, alpha=0.8, width=0.6, label="Intra-Epoch Hoyer CV (%)")
    ax2.axhline(5.0, color="#d62728", linestyle="--", linewidth=1.8, label="5% Invariance Threshold")

    ax2.set_xlabel("Training Epoch", fontweight="bold")
    ax2.set_ylabel("Coefficient of Variation (CV %)", fontweight="bold")
    ax2.set_xticks(epochs)
    ax2.grid(True)
    ax2.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig_intra_epoch_metric_variance.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_mask_decay(data: dict, output_dir: str):
    epochs_data = data["epochs_data"]
    positions = [0, 25, 50, 75, 100]  # percentage of epoch

    # Select representative epochs
    target_epochs = [1, 2, 5, 10, 15, 20]
    epoch_colors = {
        1: "#7f7f7f",
        2: "#d62728",
        5: "#ff7f0e",
        10: "#2ca02c",
        15: "#1f77b4",
        20: "#9467bd",
    }

    fig, (ax_stem, ax_stage3) = plt.subplots(1, 2, figsize=(12, 5.5), dpi=300)

    # Panel 1: Stem Conv (Early representation)
    # Panel 2: Stage 3 Conv (Computational bottleneck)
    target_layers = [
        ("conv1.weight", ax_stem, "Stem (conv1.weight)"),
        ("layer3.2.conv2.weight", ax_stage3, "Stage 3 (layer3.2.conv2.weight)"),
    ]

    for layer_name, ax, title_suffix in target_layers:
        for ep in target_epochs:
            if ep <= len(epochs_data):
                ep_rec = epochs_data[ep - 1]
                decay = [1.0]  # T0 relative to T0 is 1.0
                for ckpt in ep_rec["checkpoints"][1:]:
                    val = ckpt["mask_iou_from_anchor"].get(layer_name, np.nan)
                    decay.append(val)

                ax.plot(positions, decay, marker="o", linewidth=2.0, color=epoch_colors[ep], label=f"Epoch {ep}")

        # Add random chance baseline
        ax.axhline(0.0526, color="#000000", linestyle=":", linewidth=1.5, label="Random Chance (5.26%)")

        ax.set_xlabel("Intra-Epoch Step Progression (%)", fontweight="bold")
        ax.set_ylabel("Top-10% Mask IoU (Relative to Anchor T0)", fontweight="bold")
        ax.set_title(f"Mask Persistence Decay: {title_suffix}", fontsize=12, fontweight="bold")
        ax.set_xticks(positions)
        ax.set_xticklabels(["0% (T0)", "25%", "50%", "75%", "100% (T4)"])
        ax.set_ylim(0, 1.05)
        ax.grid(True)
        ax.legend(loc="upper right", frameon=True, fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig_intra_epoch_mask_decay.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_cosine_drift(data: dict, output_dir: str):
    epochs_data = data["epochs_data"]
    positions = [0, 25, 50, 75, 100]

    target_epochs = [1, 2, 5, 10, 15, 20]
    epoch_colors = {
        1: "#7f7f7f",
        2: "#d62728",
        5: "#ff7f0e",
        10: "#2ca02c",
        15: "#1f77b4",
        20: "#9467bd",
    }

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    for ep in target_epochs:
        if ep <= len(epochs_data):
            ep_rec = epochs_data[ep - 1]
            cos_decay = [1.0]
            for ckpt in ep_rec["checkpoints"][1:]:
                # Average cosine similarity across all tracked layers
                vals = list(ckpt["cosine_sim_from_anchor"].values())
                mean_cos = float(np.mean(vals)) if vals else np.nan
                cos_decay.append(mean_cos)

            ax.plot(positions, cos_decay, marker="s", linewidth=2.2, color=epoch_colors[ep], label=f"Epoch {ep}")

    ax.axhline(0.0, color="#000000", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_xlabel("Intra-Epoch Step Progression (%)", fontweight="bold")
    ax.set_ylabel("Cosine Similarity with Anchor Gradient g(T0)", fontweight="bold")
    ax.set_title("Intra-Epoch Gradient Direction Drift Across 20 Epochs\n(Mean Cosine Similarity Across All Representative Layers)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(positions)
    ax.set_xticklabels(["0% (Start)", "25%", "50%", "75%", "100% (End)"])
    ax.set_ylim(-0.1, 1.05)
    ax.grid(True)
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig_intra_epoch_cosine_drift.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_layertype_trajectories(data: dict, output_dir: str):
    epochs_data = data["epochs_data"]
    target_epochs = [2, 5, 10, 20]
    positions = [0, 25, 50, 75, 100]

    type_labels = {
        "conv3x3_spatial": "3x3 Conv (Spatial)",
        "conv1x1_reduce": "1x1 Conv (Reduce)",
        "conv1x1_expand": "1x1 Conv (Expand)",
        "conv1x1_downsample": "1x1 Conv (Downsample)",
        "classifier_head": "Classifier Head (fc)",
    }
    type_colors = {
        "conv3x3_spatial": "#ff7f0e",
        "conv1x1_reduce": "#1f77b4",
        "conv1x1_expand": "#2ca02c",
        "conv1x1_downsample": "#17becf",
        "classifier_head": "#d62728",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300, sharey=True)
    axes = axes.flatten()

    for i, ep in enumerate(target_epochs):
        ax = axes[i]
        ep_rec = epochs_data[ep - 1]
        ckpts = ep_rec["checkpoints"]

        for ltype, label in type_labels.items():
            vals = [c["by_type_hoyer"].get(ltype, np.nan) for c in ckpts]
            ax.plot(positions, vals, marker="o", linewidth=2.0, color=type_colors[ltype], label=label)

        ax.set_title(f"Epoch {ep} (Val Acc: {ep_rec['val_acc']}%)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Intra-Epoch Step Progression (%)", fontweight="bold")
        ax.set_ylabel("Hoyer Sparsity Index [0-1]", fontweight="bold")
        ax.set_xticks(positions)
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.set_ylim(0.3, 1.0)
        ax.grid(True)
        if i == 0:
            ax.legend(loc="upper right", frameon=True, fontsize=8.5)

    plt.suptitle("ResNet-50 Layer-Type Sparsity Trajectories Across Intra-Epoch Checkpoints (T0 -> T4)\n(Evaluating Hypothesis 1: Individual Layer-Type Invariance Across Intra-Epoch Steps)",
                 fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out_path = os.path.join(output_dir, "fig_intra_epoch_layertype_trajectories.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_layertype_ratios(data: dict, output_dir: str):
    epochs_data = data["epochs_data"]
    target_epochs = [5, 10, 15, 20]
    positions = [0, 25, 50, 75, 100]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    epoch_colors = {
        5: "#ff7f0e",
        10: "#2ca02c",
        15: "#1f77b4",
        20: "#9467bd",
    }

    for ep in target_epochs:
        ckpts = epochs_data[ep - 1]["checkpoints"]
        ratio_exp = [c["by_type_hoyer"]["conv3x3_spatial"] / c["by_type_hoyer"]["conv1x1_expand"] for c in ckpts]
        ratio_red = [c["by_type_hoyer"]["conv3x3_spatial"] / c["by_type_hoyer"]["conv1x1_reduce"] for c in ckpts]

        cv_exp = (np.std(ratio_exp) / np.mean(ratio_exp)) * 100.0
        cv_red = (np.std(ratio_red) / np.mean(ratio_red)) * 100.0

        ax1.plot(positions, ratio_exp, marker="o", linewidth=2.2, color=epoch_colors[ep],
                 label=f"Epoch {ep} (CV: {cv_exp:.2f}%)")
        ax2.plot(positions, ratio_red, marker="s", linewidth=2.2, color=epoch_colors[ep],
                 label=f"Epoch {ep} (CV: {cv_red:.2f}%)")

    ax1.set_title("Sparsity Ratio: 3x3 Spatial / 1x1 Expand Conv", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Intra-Epoch Step Progression (%)", fontweight="bold")
    ax1.set_ylabel("Hoyer Ratio", fontweight="bold")
    ax1.set_xticks(positions)
    ax1.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax1.set_ylim(0.95, 1.25)
    ax1.grid(True)
    ax1.legend(loc="upper right", frameon=True, fontsize=9)

    ax2.set_title("Sparsity Ratio: 3x3 Spatial / 1x1 Reduce Conv", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Intra-Epoch Step Progression (%)", fontweight="bold")
    ax2.set_ylabel("Hoyer Ratio", fontweight="bold")
    ax2.set_xticks(positions)
    ax2.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax2.set_ylim(0.95, 1.25)
    ax2.grid(True)
    ax2.legend(loc="upper right", frameon=True, fontsize=9)

    plt.suptitle("Relative Layer-Type Sparsity Ratios Across Intra-Epoch Checkpoints (T0 -> T4)\n(Evaluating Hypothesis 1: Relative Proportions Between Layer Types Across Intra-Epoch Steps)",
                 fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    out_path = os.path.join(output_dir, "fig_intra_epoch_layertype_ratios.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Intra-Epoch Gradient Stability Figures")
    parser.add_argument("--json", type=str, default="training/logs/resnet50_intra_epoch_stability.json", help="Path to JSON log")
    parser.add_argument("--output-dir", type=str, default="training/figures/intra_epoch_stability", help="Output directory")
    args = parser.parse_args()

    setup_style()
    with open(args.json, "r") as f:
        data = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    plot_metric_variance(data, args.output_dir)
    plot_mask_decay(data, args.output_dir)
    plot_cosine_drift(data, args.output_dir)
    plot_layertype_trajectories(data, args.output_dir)
    plot_layertype_ratios(data, args.output_dir)
    print("All intra-epoch stability figures successfully plotted!")


if __name__ == "__main__":
    main()

