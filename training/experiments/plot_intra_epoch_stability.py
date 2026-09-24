"""
Visualization suite for Intra-Epoch Gradient Stability in ResNet-50.
Plots:
1. fig07a_intra_epoch_metric_variance.png (Anchor T0 Hoyer vs. Intra-Epoch Spread & CV%)
2. fig07b_intra_epoch_layertype_cv.png     (Intra-Epoch Sparsity CV by Layer Type & Stage)
3. fig07c_intra_epoch_mask_decay.png       (Top-10% Mask IoU Decay Across Intra-Epoch Steps)
4. fig07d_intra_epoch_cosine_drift.png     (Gradient Vector Cosine Similarity Decay with T0)
5. fig07e_intra_epoch_layertype_trajectories.png (Layer-Type Sparsity Trajectories Across Checkpoints)
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

    # Hoyer metrics
    t0_hoyers = [d["checkpoints"][0]["global_hoyer"] for d in epochs_data]
    min_hoyers = [min(c["global_hoyer"] for c in d["checkpoints"]) for d in epochs_data]
    max_hoyers = [max(c["global_hoyer"] for c in d["checkpoints"]) for d in epochs_data]
    cv_hoyers = [d["intra_epoch_hoyer_cv_pct"] for d in epochs_data]

    # Energy-10 metrics
    t0_energies = [d["checkpoints"][0]["global_energy10"] for d in epochs_data]
    min_energies = [min(c["global_energy10"] for c in d["checkpoints"]) for d in epochs_data]
    max_energies = [max(c["global_energy10"] for c in d["checkpoints"]) for d in epochs_data]
    cv_energies = []
    for d in epochs_data:
        ckpts_e10 = [c["global_energy10"] for c in d["checkpoints"]]
        m = float(np.mean(ckpts_e10))
        s = float(np.std(ckpts_e10))
        cv = (s / m * 100.0) if m > 0 else 0.0
        cv_energies.append(round(cv, 2))

    fig, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, figsize=(15, 8.0), dpi=300, sharex=True)

    # --- Top-Left Panel: Anchor Hoyer with Min-Max Intra-Epoch Envelope ---
    color_hoyer = "#2ca02c"
    color_band_h = "#98df8a"

    ax1.plot(epochs, t0_hoyers, color=color_hoyer, marker="o", linewidth=2.5, label=r"Initial Anchor $T_0$ (Batches 1-5)")
    ax1.fill_between(epochs, min_hoyers, max_hoyers, color=color_band_h, alpha=0.4, label=r"Intra-Epoch Range ($T_0 \to T_4$)")
    ax1.plot(epochs, min_hoyers, color=color_hoyer, linestyle=":", alpha=0.7)
    ax1.plot(epochs, max_hoyers, color=color_hoyer, linestyle=":", alpha=0.7)

    ax1.set_ylabel("Global Hoyer Sparsity [0-1]", fontweight="bold")
    ax1.set_title(r"Hoyer Sparsity: Anchor $T_0$ vs. Intra-Epoch Spread", fontsize=11, fontweight="bold")
    ax1.grid(True)
    ax1.legend(loc="upper right", frameon=True)

    # --- Bottom-Left Panel: Hoyer Coefficient of Variation (CV %) ---
    color_cv_h = "#1f77b4"
    ax2.bar(epochs, cv_hoyers, color=color_cv_h, edgecolor="#333333", linewidth=0.8, alpha=0.85, width=0.6,
            label="Intra-Epoch Hoyer CV (%)")
    ax2.axhline(5.0, color="#d62728", linestyle="--", linewidth=1.8, label="5% Invariance Threshold")

    ax2.set_xlabel("Training Epoch", fontweight="bold")
    ax2.set_ylabel("Hoyer CV (%)", fontweight="bold")
    ax2.set_xticks(epochs)
    ax2.set_title("Intra-Epoch Hoyer CV (%) Across Epochs", fontsize=11, fontweight="bold")
    ax2.grid(True)
    ax2.legend(loc="upper right", frameon=True)

    # --- Top-Right Panel: Anchor E10 with Min-Max Intra-Epoch Envelope ---
    color_e10 = "#9467bd"
    color_band_e = "#c5b0d5"

    ax3.plot(epochs, t0_energies, color=color_e10, marker="s", linewidth=2.5, label=r"Initial Anchor $T_0$ (Batches 1-5)")
    ax3.fill_between(epochs, min_energies, max_energies, color=color_band_e, alpha=0.4, label=r"Intra-Epoch Range ($T_0 \to T_4$)")
    ax3.plot(epochs, min_energies, color=color_e10, linestyle=":", alpha=0.7)
    ax3.plot(epochs, max_energies, color=color_e10, linestyle=":", alpha=0.7)

    ax3.set_ylabel("Top-10% Energy Concentration ($E_{10}$ %)", fontweight="bold")
    ax3.set_title("Top-10% Energy: Anchor $T_0$ vs. Intra-Epoch Spread", fontsize=11, fontweight="bold")
    ax3.grid(True)
    ax3.legend(loc="upper right", frameon=True)

    # --- Bottom-Right Panel: Energy-10 Coefficient of Variation (CV %) ---
    color_cv_e = "#ff7f0e"
    ax4.bar(epochs, cv_energies, color=color_cv_e, edgecolor="#333333", linewidth=0.8, alpha=0.85, width=0.6,
            label="Intra-Epoch $E_{10}$ CV (%)")
    ax4.axhline(5.0, color="#d62728", linestyle="--", linewidth=1.8, label="5% Invariance Threshold")

    ax4.set_xlabel("Training Epoch", fontweight="bold")
    ax4.set_ylabel("Energy-10 CV (%)", fontweight="bold")
    ax4.set_xticks(epochs)
    ax4.set_title("Intra-Epoch Top-10% Energy ($E_{10}$) CV (%) Across Epochs", fontsize=11, fontweight="bold")
    ax4.grid(True)
    ax4.legend(loc="upper right", frameon=True)

    plt.suptitle("ResNet-50 Intra-Epoch Sparsity & Energy Stability Across 20 Epochs\n(Evaluating Whether Initial Anchor Proxy Holds Across 390 Batches for Hoyer & $E_{10}$)",
                 fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out_path = os.path.join(output_dir, "fig07a_intra_epoch_metric_variance.png")
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
    out_path = os.path.join(output_dir, "fig07c_intra_epoch_mask_decay.png")
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
    out_path = os.path.join(output_dir, "fig07d_intra_epoch_cosine_drift.png")
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
    out_path = os.path.join(output_dir, "fig07e_intra_epoch_layertype_trajectories.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_layertype_cv(data: dict, output_dir: str):
    """
    Plots the intra-epoch Coefficient of Variation (CV %) for every individual layer type
    and architectural stage across all 20 epochs.
    """
    epochs_data = data["epochs_data"]
    epochs = [d["epoch"] for d in epochs_data if d["epoch"] >= 2]

    type_labels = {
        "conv3x3_spatial": "3x3 Spatial Conv",
        "conv1x1_reduce": "1x1 Reduce Conv",
        "conv1x1_expand": "1x1 Expand Conv",
        "conv1x1_downsample": "1x1 Shortcut Conv",
        "classifier_head": "Classifier Head (fc)",
    }
    type_colors = {
        "conv3x3_spatial": "#ff7f0e",
        "conv1x1_reduce": "#1f77b4",
        "conv1x1_expand": "#2ca02c",
        "conv1x1_downsample": "#17becf",
        "classifier_head": "#d62728",
    }
    type_markers = {
        "conv3x3_spatial": "o",
        "conv1x1_reduce": "s",
        "conv1x1_expand": "^",
        "conv1x1_downsample": "v",
        "classifier_head": "D",
    }

    stage_labels = {
        "stem": "Stem (Conv1)",
        "stage1": "Stage 1 (32x32)",
        "stage2": "Stage 2 (16x16)",
        "stage3": "Stage 3 (8x8)",
        "stage4": "Stage 4 (4x4)",
        "head": "Classifier Head",
    }
    stage_colors = {
        "stem": "#8c564b",
        "stage1": "#1f77b4",
        "stage2": "#ff7f0e",
        "stage3": "#2ca02c",
        "stage4": "#d62728",
        "head": "#9467bd",
    }
    stage_markers = {
        "stem": "o",
        "stage1": "s",
        "stage2": "^",
        "stage3": "v",
        "stage4": "<",
        "head": "D",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Panel 1: By Layer Type
    for lt, label in type_labels.items():
        cvs = []
        for d in epochs_data:
            if d["epoch"] >= 2:
                ckpts = d["checkpoints"]
                vals = [c["by_type_hoyer"][lt] for c in ckpts]
                cv = (np.std(vals) / np.mean(vals)) * 100.0 if np.mean(vals) > 0 else 0.0
                cvs.append(cv)
        ax1.plot(epochs, cvs, marker=type_markers[lt], linewidth=2.0, markersize=6,
                 color=type_colors[lt], label=label)

    ax1.axhline(5.0, color="#d62728", linestyle="--", linewidth=1.5, alpha=0.8, label="5% Invariance Bound")
    ax1.set_title("Intra-Epoch Sparsity CV by Layer Type", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Training Epoch", fontweight="bold")
    ax1.set_ylabel("Intra-Epoch Hoyer CV (%)", fontweight="bold")
    ax1.set_xticks(epochs)
    ax1.set_ylim(0, 9.5)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True, fontsize=9)
    ax1.text(0.03, 0.92, "Epoch 1 (Random Init) CV ≈ 20–31%\nomitted for scale",
             transform=ax1.transAxes, fontsize=8.5,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="#cccccc", alpha=0.85))

    # Panel 2: By Architectural Stage
    for st, label in stage_labels.items():
        cvs = []
        for d in epochs_data:
            if d["epoch"] >= 2:
                ckpts = d["checkpoints"]
                vals = [c["by_stage_hoyer"][st] for c in ckpts]
                cv = (np.std(vals) / np.mean(vals)) * 100.0 if np.mean(vals) > 0 else 0.0
                cvs.append(cv)
        ax2.plot(epochs, cvs, marker=stage_markers[st], linewidth=2.0, markersize=6,
                 color=stage_colors[st], label=label)

    ax2.axhline(5.0, color="#d62728", linestyle="--", linewidth=1.5, alpha=0.8, label="5% Invariance Bound")
    ax2.set_title("Intra-Epoch Sparsity CV by Architectural Stage", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Training Epoch", fontweight="bold")
    ax2.set_ylabel("Intra-Epoch Hoyer CV (%)", fontweight="bold")
    ax2.set_xticks(epochs)
    ax2.set_ylim(0, 9.5)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True, fontsize=9)
    ax2.text(0.03, 0.92, "Epoch 1 (Random Init) CV ≈ 20–31%\nomitted for scale",
             transform=ax2.transAxes, fontsize=8.5,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="#cccccc", alpha=0.85))

    plt.suptitle("ResNet-50 Layer-Wise Intra-Epoch Sparsity Stability Across 20 Epochs\n(Evaluating Hypothesis 1: Intra-Epoch Invariance (CV < 5%) Across Individual Layers)",
                 fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    out_path = os.path.join(output_dir, "fig07b_intra_epoch_layertype_cv.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Intra-Epoch Gradient Stability Figures")
    parser.add_argument("--json", type=str, default="training/logs/resnet50_intra_epoch_stability.json", help="Path to JSON log")
    parser.add_argument("--output-dir", type=str, default="training/figures/exp07_intra_epoch_stability", help="Output directory")
    args = parser.parse_args()

    setup_style()
    with open(args.json, "r") as f:
        data = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    plot_metric_variance(data, args.output_dir)
    plot_mask_decay(data, args.output_dir)
    plot_cosine_drift(data, args.output_dir)
    plot_layertype_trajectories(data, args.output_dir)
    plot_layertype_cv(data, args.output_dir)
    print("All intra-epoch stability figures successfully plotted!")


if __name__ == "__main__":
    main()

