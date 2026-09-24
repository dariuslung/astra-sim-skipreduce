"""
Plotting module to evaluate Hypotheses 1, 2, and 3 for gradient sparsity.
Generates:
1. fig01_layer_type_sparsity_<model>.png   (Hypothesis 1: Layer Type Heterogeneity)
2. fig02_depth_vs_density_<model>.png     (Hypothesis 2: Depth vs Gradient Density)
3. fig03_iteration_sparsity_<model>.png   (Hypothesis 3: Sparsity Evolution over Iterations)
4. fig04_temporal_mask_iou_<model>.png    (Predictability: Mask IoU Persistence)
"""

import argparse
import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


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


def get_title_model(model_name: str) -> str:
    if model_name == "resnet50":
        return "ResNet-50"
    elif model_name == "vit":
        return "Vision Transformer (ViT)"
    elif model_name == "gpt":
        return "Causal Transformer (GPT-Tiny)"
    return model_name.upper()


def plot_layer_type_sparsity(data: dict, model_name: str, output_dir: str):
    """
    Figure 1: Evaluates Hypothesis 1 (Different layer types have different sparsity patterns).
    """
    trajectory = data["trajectory"]
    last_record = trajectory[-1]
    by_type = last_record["by_layer_type"]

    # Gather mean across last 50 steps for stability
    stable_steps = trajectory[-min(50, len(trajectory)):]
    type_stats = {}
    for ltype in by_type.keys():
        h_vals = [s["by_layer_type"][ltype]["mean_hoyer"] for s in stable_steps if ltype in s["by_layer_type"]]
        e_vals = [s["by_layer_type"][ltype]["mean_energy10"] for s in stable_steps if ltype in s["by_layer_type"]]
        type_stats[ltype] = {
            "mean_hoyer": np.mean(h_vals),
            "std_hoyer": np.std(h_vals),
            "mean_energy": np.mean(e_vals),
            "std_energy": np.std(e_vals),
        }

    # Sort layer types logically
    ltypes = list(type_stats.keys())
    x = np.arange(len(ltypes))
    width = 0.35

    hoyer_means = [type_stats[lt]["mean_hoyer"] for lt in ltypes]
    hoyer_errs = [type_stats[lt]["std_hoyer"] for lt in ltypes]
    energy_means = [type_stats[lt]["mean_energy"] for lt in ltypes]
    energy_errs = [type_stats[lt]["std_energy"] for lt in ltypes]

    fig, ax1 = plt.subplots(figsize=(9, 5), dpi=300)

    color1 = "#1f77b4"
    color2 = "#ff7f0e"

    rects1 = ax1.bar(x - width/2, hoyer_means, width, yerr=hoyer_errs, capsize=4,
                     label="Hoyer Sparsity Index [0-1]", color=color1, alpha=0.85)
    ax1.set_ylabel("Hoyer Sparsity Index", color=color1, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, energy_means, width, yerr=energy_errs, capsize=4,
                     label="Top-10% Energy Concentration (%)", color=color2, alpha=0.85)
    ax2.set_ylabel("Top-10% Energy (%)", color=color2, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, 105)

    # Format labels
    clean_labels = [lt.replace("_", " ").title() for lt in ltypes]
    ax1.set_xticks(x)
    ax1.set_xticklabels(clean_labels, rotation=25, ha="right")

    title_model = get_title_model(model_name)
    plt.title(f"Gradient Sparsity by Layer Type: {title_model}\n(Hypothesis 1: Layer-Type Heterogeneity)",
              fontsize=13, fontweight="bold", pad=12)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"fig01_layer_type_sparsity_{model_name}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_depth_vs_density(data: dict, model_name: str, output_dir: str):
    """
    Figure 2: Evaluates Hypothesis 2 controlled for layer type.
    Compares identical layer types across network depth to evaluate whether
    deeper layers exhibit denser or sparser gradients.
    """
    from collections import defaultdict
    rec_last = data["trajectory"][-1]
    fls = rec_last.get("full_layer_stats", [])
    if not fls:
        return

    by_type = defaultdict(list)
    for it in fls:
        by_type[it["layer_type"]].append(it)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    if model_name == "resnet50":
        target_types = ["conv3x3_spatial", "conv1x1_reduce", "conv1x1_expand"]
        stages = ["stage1", "stage2", "stage3", "stage4"]
        colors = {"conv3x3_spatial": "#1f77b4", "conv1x1_reduce": "#ff7f0e", "conv1x1_expand": "#2ca02c"}
        markers = {"conv3x3_spatial": "o", "conv1x1_reduce": "s", "conv1x1_expand": "^"}

        for lt in target_types:
            items = by_type[lt]
            stage_vals = []
            for st in stages:
                vals = [it["density"] for it in items if it["stage"] == st]
                stage_vals.append(np.mean(vals) if vals else np.nan)

            clean_name = lt.replace("_", " ").title()
            ax.plot(stages, stage_vals, marker=markers[lt], color=colors[lt], linewidth=2.2, markersize=8, label=clean_name)

        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels([s.title() for s in stages])
        ax.set_xlabel("Depth Stage (Shallow -> Deep)", fontweight="bold")
        title_model = "ResNet-50"
    else:  # ViT or GPT
        target_types = ["attn_qkv", "attn_proj", "ffn_up", "ffn_down"]
        blocks = [f"block_{i}" for i in range(6)]
        colors = {"attn_qkv": "#d62728", "attn_proj": "#9467bd", "ffn_up": "#2ca02c", "ffn_down": "#1f77b4"}
        markers = {"attn_qkv": "o", "attn_proj": "v", "ffn_up": "s", "ffn_down": "D"}

        for lt in target_types:
            items = by_type[lt]
            block_vals = []
            for blk in blocks:
                vals = [it["density"] for it in items if it["stage"] == blk]
                block_vals.append(np.mean(vals) if vals else np.nan)

            clean_name = lt.replace("_", " ").title()
            ax.plot(blocks, block_vals, marker=markers[lt], color=colors[lt], linewidth=2.2, markersize=7, label=clean_name)

        ax.set_xticks(range(len(blocks)))
        ax.set_xticklabels([b.replace("_", " ").title() for b in blocks])
        ax.set_xlabel("Transformer Block Index (0 = Shallow -> 5 = Deep)", fontweight="bold")
        title_model = get_title_model(model_name)

    ax.set_ylabel("Gradient Density (1 - Hoyer Sparsity)", fontweight="bold")
    ax.set_ylim(0.2, 0.85)
    plt.title(f"Gradient Density vs. Depth (Controlled by Layer Type): {title_model}\n(Hypothesis 2: Evaluated Across Identical Submodules)",
              fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"fig02_depth_vs_density_{model_name}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")



def plot_iteration_sparsity_evolution(data: dict, model_name: str, output_dir: str):
    """
    Figure 3: Evaluates Hypothesis 3 (Gradient sparsity increases with iterations as loss stabilizes).
    """
    trajectory = data["trajectory"]
    steps = [r["step"] for r in trajectory]
    losses = [r["loss"] for r in trajectory]
    hoyers = [r["global_hoyer"] for r in trajectory]
    energies = [r["global_energy10"] for r in trajectory]

    fig, ax1 = plt.subplots(figsize=(9, 5), dpi=300)

    color_loss = "#7f7f7f"
    color_hoyer = "#2ca02c"
    color_energy = "#1f77b4"

    # Moving average helper for smoother curves
    def moving_avg(arr, window=10):
        if len(arr) < window:
            return arr
        return np.convolve(arr, np.ones(window)/window, mode='valid')

    w = 10
    smooth_steps = steps[w-1:] if len(steps) >= w else steps
    smooth_hoyers = moving_avg(hoyers, w)
    smooth_energies = moving_avg(energies, w)
    smooth_losses = moving_avg(losses, w)

    ax1.plot(steps, hoyers, color=color_hoyer, alpha=0.25)
    line1, = ax1.plot(smooth_steps, smooth_hoyers, color=color_hoyer, linewidth=2.5,
                      label="Mean Hoyer Sparsity (10-step MA)")
    ax1.set_xlabel("Training Iteration (Step)", fontweight="bold")
    ax1.set_ylabel("Global Hoyer Sparsity Index [0-1]", color=color_hoyer, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color_hoyer)
    ax1.set_ylim(min(hoyers) * 0.9, min(1.0, max(hoyers) * 1.1))

    ax2 = ax1.twinx()
    ax2.plot(steps, losses, color=color_loss, alpha=0.25)
    line2, = ax2.plot(smooth_steps, smooth_losses, color=color_loss, linestyle="--", linewidth=2.0,
                      label="Training Loss (Cross-Entropy)")
    ax2.set_ylabel("Training Loss", color=color_loss, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color_loss)

    title_model = get_title_model(model_name)
    plt.title(f"Gradient Sparsity Evolution Over Training: {title_model}\n(Hypothesis 3: Sparsity Increases with Iterations)",
              fontsize=13, fontweight="bold", pad=12)

    ax1.legend([line1, line2], [line1.get_label(), line2.get_label()], loc="center right", frameon=True)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"fig03_iteration_sparsity_{model_name}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_temporal_mask_iou(data: dict, model_name: str, output_dir: str):
    """
    Figure 4: Evaluates Top-10% Mask IoU across temporal step lags (Predictability).
    """
    iou_data = data.get("temporal_mask_iou", {})
    if not iou_data:
        return

    layers = list(iou_data.keys())
    lags = ["lag_1", "lag_5", "lag_10"]
    x = np.arange(len(layers))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

    colors = ["#2ca02c", "#1f77b4", "#9467bd"]
    for i, lag in enumerate(lags):
        vals = [iou_data[l].get(lag, 0.0) or 0.0 for l in layers]
        lag_label = f"Lag Δt = {lag.split('_')[1]} step" + ("s" if lag != "lag_1" else "")
        ax.bar(x + (i - 1) * width, vals, width, label=lag_label, color=colors[i], alpha=0.85)

    ax.set_ylabel("Top-10% Mask Jaccard IoU", fontweight="bold")
    ax.set_xlabel("Tracked Model Layer", fontweight="bold")
    ax.set_xticks(x)
    clean_layer_names = [l.replace(".weight", "") for l in layers]
    ax.set_xticklabels(clean_layer_names, rotation=25, ha="right")
    ax.set_ylim(0, 1.0)

    title_model = get_title_model(model_name)
    plt.title(f"Temporal Sparsity Mask Persistence (Predictability): {title_model}\n(Top-10% Coordinate IoU Across Lags Δt)",
              fontsize=13, fontweight="bold", pad=12)

    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"fig04_temporal_mask_iou_{model_name}.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate gradient sparsity figures.")
    parser.add_argument("--json", type=str, required=True, help="Path to profile JSON log")
    parser.add_argument("--output-dir", type=str, default="training/figures", help="Output base directory for plots")

    args = parser.parse_args()

    setup_style()
    with open(args.json, "r") as f:
        data = json.load(f)

    model_name = data["model"].lower()

    # Route each experiment to its respective directory
    if os.path.basename(os.path.normpath(args.output_dir)) == "figures":
        exp01_dir = os.path.join(args.output_dir, "exp01_layer_type_sparsity")
        exp02_dir = os.path.join(args.output_dir, "exp02_depth_vs_density")
        exp03_dir = os.path.join(args.output_dir, "exp03_iteration_sparsity")
        exp04_dir = os.path.join(args.output_dir, "exp04_temporal_mask_iou")
    else:
        exp01_dir = exp02_dir = exp03_dir = exp04_dir = args.output_dir

    plot_layer_type_sparsity(data, model_name, exp01_dir)
    plot_depth_vs_density(data, model_name, exp02_dir)
    plot_iteration_sparsity_evolution(data, model_name, exp03_dir)
    plot_temporal_mask_iou(data, model_name, exp04_dir)
    print(f"All figures generated successfully for {model_name}!")


if __name__ == "__main__":
    main()
