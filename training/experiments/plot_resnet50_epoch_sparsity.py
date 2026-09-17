"""
Visualization suite for ResNet-50 Multi-Epoch Convergence and Gradient Sparsity.
Plots:
1. fig_resnet50_convergence_vs_sparsity.png (Loss & Val Acc vs. Hoyer & Energy10)
2. fig_resnet50_stages_across_epochs.png    (Hoyer Sparsity by Stage across Epochs)
3. fig_resnet50_layertypes_across_epochs.png (Hoyer Sparsity by Layer Type across Epochs)
4. fig_resnet50_mask_iou_across_epochs.png   (Top-10% Mask Persistence across Epochs)
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


def plot_convergence_vs_sparsity(data: dict, output_dir: str):
    epochs_data = data["epochs_data"]
    epochs = [d["epoch"] for d in epochs_data]
    train_loss = [d["train_loss"] for d in epochs_data]
    val_acc = [d["val_acc"] for d in epochs_data]
    hoyers = [d["global_hoyer"] for d in epochs_data]
    energy10 = [d["global_energy10"] for d in epochs_data]

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 7.5), dpi=300, sharex=True)

    # --- TOP PANEL: Convergence (Loss & Accuracy) ---
    color_loss = "#d62728"
    color_acc = "#1f77b4"

    ax_top.plot(epochs, train_loss, color=color_loss, marker="o", linewidth=2.2, label="Train Loss (Cross-Entropy)")
    ax_top.set_ylabel("Training Loss", color=color_loss, fontweight="bold")
    ax_top.tick_params(axis="y", labelcolor=color_loss)
    ax_top.grid(True)

    ax_top_r = ax_top.twinx()
    ax_top_r.plot(epochs, val_acc, color=color_acc, marker="s", linestyle="--", linewidth=2.2, label="Validation Accuracy (%)")
    ax_top_r.set_ylabel("Validation Accuracy (%)", color=color_acc, fontweight="bold")
    ax_top_r.tick_params(axis="y", labelcolor=color_acc)
    ax_top_r.set_ylim(0, 100)

    # Combined legend for top
    lines_top = [ax_top.get_lines()[0], ax_top_r.get_lines()[0]]
    labels_top = [l.get_label() for l in lines_top]
    ax_top.legend(lines_top, labels_top, loc="center right", frameon=True)
    ax_top.set_title("ResNet-50 Convergence vs. Gradient Sparsity Over 20 Epochs\n(Evaluating Hypothesis 3 Under Convergence)",
                     fontsize=13, fontweight="bold", pad=10)

    # --- BOTTOM PANEL: Sparsity Evolution ---
    color_hoyer = "#2ca02c"
    color_e10 = "#9467bd"

    ax_bot.plot(epochs, hoyers, color=color_hoyer, marker="^", linewidth=2.5, label="Global Hoyer Sparsity [0-1]")
    ax_bot.set_xlabel("Training Epoch", fontweight="bold")
    ax_bot.set_ylabel("Hoyer Sparsity Index", color=color_hoyer, fontweight="bold")
    ax_bot.tick_params(axis="y", labelcolor=color_hoyer)
    ax_bot.set_ylim(min(hoyers) * 0.9, min(1.0, max(hoyers) * 1.1))
    ax_bot.grid(True)

    ax_bot_r = ax_bot.twinx()
    ax_bot_r.plot(epochs, energy10, color=color_e10, marker="D", linestyle="-.", linewidth=2.2, label="Top-10% Energy Concentration (%)")
    ax_bot_r.set_ylabel("Top-10% Gradient Energy (%)", color=color_e10, fontweight="bold")
    ax_bot_r.tick_params(axis="y", labelcolor=color_e10)
    ax_bot_r.set_ylim(min(energy10) * 0.9, 100)

    lines_bot = [ax_bot.get_lines()[0], ax_bot_r.get_lines()[0]]
    labels_bot = [l.get_label() for l in lines_bot]
    ax_bot.legend(lines_bot, labels_bot, loc="center right", frameon=True)

    ax_bot.set_xticks(epochs)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig_resnet50_convergence_vs_sparsity.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_stages_across_epochs(data: dict, output_dir: str):
    epochs_data = data["epochs_data"]
    epochs = [d["epoch"] for d in epochs_data]

    stages = ["stem", "stage1", "stage2", "stage3", "stage4", "head"]
    stage_labels = {
        "stem": "Stem (Conv1)",
        "stage1": "Stage 1 (32x32)",
        "stage2": "Stage 2 (16x16)",
        "stage3": "Stage 3 (8x8)",
        "stage4": "Stage 4 (4x4)",
        "head": "Classifier FC",
    }
    stage_colors = {
        "stem": "#8c564b",
        "stage1": "#1f77b4",
        "stage2": "#ff7f0e",
        "stage3": "#2ca02c",
        "stage4": "#d62728",
        "head": "#9467bd",
    }

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    for st in stages:
        vals = [d["by_stage"].get(st, {}).get("hoyer", np.nan) for d in epochs_data]
        if not any(np.isnan(vals)):
            ax.plot(epochs, vals, marker="o", linewidth=2.2, label=stage_labels.get(st, st),
                    color=stage_colors.get(st, "#333333"))

    ax.set_xlabel("Training Epoch", fontweight="bold")
    ax.set_ylabel("Hoyer Sparsity Index [0-1]", fontweight="bold")
    ax.set_title("ResNet-50 Stage-Wise Sparsity Evolution Across 20 Epochs\n(Stem through Deepest Stage 4)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(epochs)
    ax.grid(True)
    ax.legend(loc="best", frameon=True)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig_resnet50_stages_across_epochs.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_layertypes_across_epochs(data: dict, output_dir: str):
    epochs_data = data["epochs_data"]
    epochs = [d["epoch"] for d in epochs_data]

    types = ["conv1x1_reduce", "conv3x3_spatial", "conv1x1_expand", "conv1x1_downsample", "classifier_head"]
    type_labels = {
        "conv1x1_reduce": "1x1 Conv (Reduce)",
        "conv3x3_spatial": "3x3 Conv (Spatial)",
        "conv1x1_expand": "1x1 Conv (Expand)",
        "conv1x1_downsample": "1x1 Conv (Shortcut)",
        "classifier_head": "Classifier FC",
    }
    type_colors = {
        "conv1x1_reduce": "#1f77b4",
        "conv3x3_spatial": "#ff7f0e",
        "conv1x1_expand": "#2ca02c",
        "conv1x1_downsample": "#17becf",
        "classifier_head": "#d62728",
    }

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    for tp in types:
        vals = [d["by_layer_type"].get(tp, {}).get("hoyer", np.nan) for d in epochs_data]
        if not all(np.isnan(vals)):
            ax.plot(epochs, vals, marker="s", linewidth=2.2, label=type_labels.get(tp, tp),
                    color=type_colors.get(tp, "#333333"))

    ax.set_xlabel("Training Epoch", fontweight="bold")
    ax.set_ylabel("Hoyer Sparsity Index [0-1]", fontweight="bold")
    ax.set_title("ResNet-50 Layer-Type Sparsity Evolution Across 20 Epochs\n(1x1 Convolutions vs. 3x3 Spatial Convolutions)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(epochs)
    ax.grid(True)
    ax.legend(loc="best", frameon=True)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig_resnet50_layertypes_across_epochs.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_mask_iou_across_epochs(data: dict, output_dir: str):
    iou_history = data.get("epoch_mask_iou_history", {})
    if not iou_history:
        return

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    for layer_name, ious in iou_history.items():
        if ious:
            epoch_transitions = list(range(2, 2 + len(ious)))
            clean_name = layer_name.replace(".weight", "")
            ax.plot(epoch_transitions, ious, marker="^", linewidth=2.0, label=clean_name)

    ax.set_xlabel("Epoch Transition (t -> t+1)", fontweight="bold")
    ax.set_ylabel("Top-10% Coordinate Mask IoU", fontweight="bold")
    ax.set_title("Temporal Mask Persistence Across Epochs (Predictability)\n(Top-10% Coordinate Overlap Between Consecutive Epochs)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0, 1.0)
    ax.grid(True)
    ax.legend(loc="best", frameon=True)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig_resnet50_mask_iou_across_epochs.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot ResNet-50 multi-epoch sparsity figures.")
    parser.add_argument("--json", type=str, default="training/logs/resnet50_convergence_sparsity.json", help="Path to profile JSON")
    parser.add_argument("--output-dir", type=str, default="training/figures/convergence", help="Output directory for figures")
    args = parser.parse_args()

    setup_style()
    with open(args.json, "r") as f:
        data = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    plot_convergence_vs_sparsity(data, args.output_dir)
    plot_stages_across_epochs(data, args.output_dir)
    plot_layertypes_across_epochs(data, args.output_dir)
    plot_mask_iou_across_epochs(data, args.output_dir)
    print("All multi-epoch ResNet-50 figures plotted successfully!")


if __name__ == "__main__":
    main()
