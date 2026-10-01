"""
Publication-Quality Visualization Suite for EXP-10:
Layer-Adaptive Compressive Sensing Rate-Distortion Profiling.

Generates:
1. fig10a_layer_rate_distortion.png:
   Dual-panel Cosine Similarity vs. Retention Ratio r for DCT and Hadamard across all layer types.
2. fig10b_layer_rel_l2_error.png:
   Dual-panel Relative L2 Reconstruction Error vs. Retention Ratio r.
"""

import argparse
import json
import os
import sys

# Ensure repo root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10.5,
        "axes.labelsize": 10.5,
        "axes.titlesize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 8.8,
        "figure.titlesize": 13,
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
    "conv3x3_spatial": "#ff7f0e",     # Orange
    "conv1x1_expand": "#2ca02c",      # Green
    "conv1x1_reduce": "#1f77b4",      # Blue
    "conv1x1_downsample": "#17becf",  # Cyan
    "classifier_head": "#9467bd",     # Purple
}

MARKER_MAP = {
    "conv3x3_spatial": "o",
    "conv1x1_expand": "s",
    "conv1x1_reduce": "^",
    "conv1x1_downsample": "d",
    "classifier_head": "v",
}

LABEL_MAP = {
    "conv3x3_spatial": "3x3 Spatial Convolutions [conv3x3]",
    "conv1x1_expand": "1x1 Expand Convolutions [conv1x1_exp]",
    "conv1x1_reduce": "1x1 Reduce Convolutions [conv1x1_red]",
    "conv1x1_downsample": "1x1 Shortcut Connections [conv1x1_down]",
    "classifier_head": "Linear Classifier Head [fc]",
}


def plot_fig10a_rate_distortion(data: dict, output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.5), dpi=300, sharey=True)

    r_vals = data["retention_ratios"]
    layer_types = ["conv3x3_spatial", "conv1x1_expand", "conv1x1_reduce", "conv1x1_downsample", "classifier_head"]
    transforms = ["dct", "hadamard"]

    for ax_idx, trans in enumerate(transforms):
        ax = axes[ax_idx]
        trans_name = "Discrete Cosine Transform (DCT)" if trans == "dct" else "Fast Walsh-Hadamard Transform (Hadamard)"

        for lt in layer_types:
            cos_means = []
            for r in r_vals:
                d_r = data["by_transform"][trans][str(r)]["by_layer_type"]
                cos_means.append(d_r.get(lt, {}).get("mean_cos_sim", 0.0))

            ax.plot(
                r_vals,
                cos_means,
                marker=MARKER_MAP[lt],
                color=COLOR_MAP[lt],
                label=LABEL_MAP[lt],
                linewidth=2.2,
                alpha=0.9
            )

        # Baseline reference line at pure zeroing (r=0)
        ax.axhline(0.7144, color="#888888", linestyle=":", linewidth=1.2, label="Pure Zeroing Baseline (r=0.0, Cos Sim ≈ 0.71)")
        # Target threshold for high-fidelity training (>0.85)
        ax.axhline(0.85, color="#2ca02c", linestyle="--", linewidth=1.2, alpha=0.7, label="High-Fidelity Threshold (Cos Sim ≥ 0.85)")

        ax.set_title(f"Panel {'A' if ax_idx == 0 else 'B'}: {trans_name}\n"
                     f"Reconstruction Fidelity Across CS Budgets ($N=4, s=2$)",
                     fontweight="bold", fontsize=11.5)
        ax.set_xlabel("Retention Ratio $r$ (Compressive Sensing Budget)", fontweight="bold")
        if ax_idx == 0:
            ax.set_ylabel("Cosine Similarity with True All-Reduce Gradient", fontweight="bold")
        ax.set_xticks(r_vals)
        ax.set_xticklabels([f"{r:.2f}" for r in r_vals], rotation=45)
        ax.set_ylim(0.65, 1.00)
        ax.grid(True)
        ax.legend(loc="lower right", framealpha=0.92, fontsize=8.2)

    plt.suptitle(
        "Evaluating Hypothesis 7: Layer-Adaptive Compressive Sensing Rate-Distortion Profiling\n"
        "(Measuring Gradient Reconstruction Fidelity as a Function of Measurement Budget under 50% Ring Skipping)",
        fontsize=12.5, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    out_path = os.path.join(output_dir, "fig10a_layer_rate_distortion.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_fig10b_rel_l2_error(data: dict, output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.5), dpi=300, sharey=True)

    r_vals = data["retention_ratios"]
    layer_types = ["conv3x3_spatial", "conv1x1_expand", "conv1x1_reduce", "conv1x1_downsample", "classifier_head"]
    transforms = ["dct", "hadamard"]

    for ax_idx, trans in enumerate(transforms):
        ax = axes[ax_idx]
        trans_name = "Discrete Cosine Transform (DCT)" if trans == "dct" else "Fast Walsh-Hadamard Transform (Hadamard)"

        for lt in layer_types:
            l2_means = []
            for r in r_vals:
                d_r = data["by_transform"][trans][str(r)]["by_layer_type"]
                l2_means.append(d_r.get(lt, {}).get("mean_rel_l2_error", 0.0))

            ax.plot(
                r_vals,
                l2_means,
                marker=MARKER_MAP[lt],
                color=COLOR_MAP[lt],
                label=LABEL_MAP[lt],
                linewidth=2.2,
                alpha=0.9
            )

        ax.axhline(0.7004, color="#888888", linestyle=":", linewidth=1.2, label="Pure Zeroing Error (r=0.0, Error ≈ 0.70)")
        ax.axhline(0.45, color="#2ca02c", linestyle="--", linewidth=1.2, alpha=0.7, label="Low-Distortion Target (Error ≤ 0.45)")

        ax.set_title(f"Panel {'A' if ax_idx == 0 else 'B'}: {trans_name}\n"
                     f"Relative $L_2$ Reconstruction Error Across CS Budgets ($N=4, s=2$)",
                     fontweight="bold", fontsize=11.5)
        ax.set_xlabel("Retention Ratio $r$ (Compressive Sensing Budget)", fontweight="bold")
        if ax_idx == 0:
            ax.set_ylabel("Relative $L_2$ Error ($\\|\\hat{g} - g_{true}\\|_2 / \\|g_{true}\\|_2$)", fontweight="bold")
        ax.set_xticks(r_vals)
        ax.set_xticklabels([f"{r:.2f}" for r in r_vals], rotation=45)
        ax.set_ylim(0.20, 0.75)
        ax.grid(True)
        ax.legend(loc="upper right", framealpha=0.92, fontsize=8.2)

    plt.suptitle(
        "Evaluating Hypothesis 7: Relative Reconstruction Error Across Compressive Sensing Budgets\n"
        "(Comparing Distortion Decay Between Spatial and Pointwise Convolutional Submodules)",
        fontsize=12.5, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    out_path = os.path.join(output_dir, "fig10b_layer_rel_l2_error.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_fig10c_isolated_accuracy(ablation_data: dict, recover_data: dict, output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2), dpi=300)

    layer_keys = [
        "conv3x3_spatial",
        "conv1x1_downsample",
        "conv1x1_reduce",
        "conv1x1_expand",
    ]
    layer_names = [
        "3x3 Spatial\n[conv3x3]",
        "1x1 Shortcut\n[conv1x1_down]",
        "1x1 Reduce\n[conv1x1_red]",
        "1x1 Expand\n[conv1x1_exp]",
    ]

    baseline_acc = ablation_data["conditions"]["baseline"]["final_val_acc"]

    zeroing_accs = [recover_data["conditions"][k]["final_val_acc"] for k in layer_keys]
    cs_accs = [ablation_data["conditions"][f"isolate_{k}"]["final_val_acc"] for k in layer_keys]

    zeroing_deltas = [recover_data["conditions"][k]["delta_val_acc"] for k in layer_keys]
    cs_deltas = [ablation_data["conditions"][f"isolate_{k}"]["delta_val_acc"] for k in layer_keys]

    x = np.arange(len(layer_keys))
    width = 0.35

    # Panel A: Final Top-1 Accuracy
    ax1 = axes[0]
    rects1 = ax1.bar(
        x - width / 2, zeroing_accs, width,
        label="Pure Zeroing (EXP-08, 50% updates)",
        color="#bcbddc", edgecolor="#54278f", linewidth=1.2, alpha=0.9
    )
    rects2 = ax1.bar(
        x + width / 2, cs_accs, width,
        label="Compressive Sensing (EXP-10, r=0.15 DCT, 100% updates)",
        color="#3182bd", edgecolor="#08519c", linewidth=1.2, alpha=0.9
    )

    ax1.axhline(baseline_acc, color="#d95f02", linestyle="--", linewidth=1.8, label=f"Standard Baseline ({baseline_acc:.2f}%)")

    ax1.set_ylabel("Final Validation Accuracy (%)", fontweight="bold")
    ax1.set_title(
        "Panel A: Final Validation Accuracy Across Isolated Conditions\n"
        "(Comparing 50% Zeroing vs. 50% Skip with r=0.15 CS)",
        fontweight="bold", fontsize=11
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(layer_names, fontweight="bold")
    ax1.set_ylim(80.0, 91.5)
    ax1.grid(True, axis="y")
    ax1.legend(loc="lower left", framealpha=0.92, fontsize=8.8)

    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(
            f"{h:.2f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
            fontsize=8.5, fontweight="bold"
        )
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(
            f"{h:.2f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
            fontsize=8.5, fontweight="bold", color="#08519c"
        )

    # Highlight spatial recovery
    ax1.annotate(
        "+3.15 pp Recovery\n(83.07% → 86.22%)",
        xy=(x[0] + width / 2, cs_accs[0]), xytext=(x[0] + 0.15, 87.2),
        arrowprops=dict(facecolor="#2ca02c", shrink=0.08, width=1.5, headwidth=6),
        ha="center", fontsize=8.2, fontweight="bold", color="#1b7837",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#e5f5e0", edgecolor="#31a354", alpha=0.9)
    )

    # Panel B: Delta vs. Baseline
    ax2 = axes[1]
    rects3 = ax2.bar(
        x - width / 2, zeroing_deltas, width,
        label="Pure Zeroing Drop (EXP-08)",
        color="#bcbddc", edgecolor="#54278f", linewidth=1.2, alpha=0.9
    )
    rects4 = ax2.bar(
        x + width / 2, cs_deltas, width,
        label="Compressive Sensing Drop (EXP-10, r=0.15)",
        color="#6baed6", edgecolor="#08519c", linewidth=1.2, alpha=0.9
    )

    ax2.axhline(0.0, color="#333333", linestyle="-", linewidth=1.0)
    ax2.set_ylabel(r"Validation Accuracy $\Delta$ vs. Baseline (pp)", fontweight="bold")
    ax2.set_title(
        "Panel B: Accuracy Delta Comparison Relative to Baseline\n"
        "(Evaluating Recovery and Distortion Penalty)",
        fontweight="bold", fontsize=11
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(layer_names, fontweight="bold")
    ax2.set_ylim(-6.8, 2.0)
    ax2.grid(True, axis="y")
    ax2.legend(loc="lower left", framealpha=0.92, fontsize=8.8)

    for rect in rects3:
        h = rect.get_height()
        if h >= 0:
            va = 'bottom'
            y_offset = 6
        else:
            va = 'top'
            y_offset = -5
        ax2.annotate(
            f"{h:+.2f} pp", xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, y_offset), textcoords="offset points", ha='center', va=va,
            fontsize=8.5, fontweight="bold"
        )
    for rect in rects4:
        h = rect.get_height()
        if h >= 0:
            va = 'bottom'
            y_offset = 6
        else:
            va = 'top'
            y_offset = -5
        ax2.annotate(
            f"{h:+.2f} pp", xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, y_offset), textcoords="offset points", ha='center', va=va,
            fontsize=8.5, fontweight="bold", color="#08519c"
        )

    plt.suptitle(
        "Evaluating Hypothesis 7: Isolated Layer Compressive Sensing Ablation Probe\n"
        "(Comparing Pure Zeroing Starvation vs. Layer-Adaptive CS Reconstructed Updates under 50% Ring Skipping)",
        fontsize=12.5, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.91])

    out_path = os.path.join(output_dir, "fig10c_layer_cs_accuracy.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_fig10d_isolated_convergence(ablation_data: dict, output_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2), dpi=300)

    cond_styles = {
        "baseline": {
            "color": "#333333", "linestyle": "--",
            "label": "Baseline ResNet-50 (88.19%)",
            "marker": "None", "linewidth": 2.2
        },
        "isolate_conv3x3_spatial": {
            "color": "#ff7f0e", "linestyle": "-",
            "label": "Isolated 3x3 Spatial CS (86.22%, Δ -1.97 pp)",
            "marker": "o", "linewidth": 2.0
        },
        "isolate_conv1x1_downsample": {
            "color": "#17becf", "linestyle": "-",
            "label": "Isolated 1x1 Shortcut CS (88.23%, Δ +0.04 pp)",
            "marker": "d", "linewidth": 2.0
        },
        "isolate_conv1x1_reduce": {
            "color": "#1f77b4", "linestyle": "-",
            "label": "Isolated 1x1 Reduce CS (86.55%, Δ -1.64 pp)",
            "marker": "^", "linewidth": 2.0
        },
        "isolate_conv1x1_expand": {
            "color": "#2ca02c", "linestyle": "-",
            "label": "Isolated 1x1 Expand CS (84.04%, Δ -4.15 pp)",
            "marker": "s", "linewidth": 2.0
        },
    }

    # Panel A: Validation Accuracy across Epochs
    ax1 = axes[0]
    for cond_key, style in cond_styles.items():
        cond_data = ablation_data["conditions"].get(cond_key)
        if not cond_data:
            continue
        history = cond_data.get("epochs_history", [])
        epochs = [h["epoch"] for h in history]
        val_accs = [h["val_acc"] for h in history]
        ax1.plot(
            epochs, val_accs, label=style["label"], color=style["color"],
            linestyle=style["linestyle"], marker=style["marker"],
            linewidth=style["linewidth"], markersize=5, alpha=0.9
        )

    ax1.set_title(
        "Panel A: Top-1 Validation Accuracy Trajectory\n"
        "(Multi-Epoch Convergence Under 50% Ring Skipping with r=0.15 CS)",
        fontweight="bold", fontsize=11
    )
    ax1.set_xlabel("Epoch", fontweight="bold")
    ax1.set_ylabel("Validation Accuracy (%)", fontweight="bold")
    ax1.set_xticks(range(2, 21, 2))
    ax1.set_ylim(10.0, 92.0)
    ax1.grid(True)
    ax1.legend(loc="lower right", framealpha=0.92, fontsize=8.8)

    # Panel B: Validation Loss across Epochs
    ax2 = axes[1]
    for cond_key, style in cond_styles.items():
        cond_data = ablation_data["conditions"].get(cond_key)
        if not cond_data:
            continue
        history = cond_data.get("epochs_history", [])
        epochs = [h["epoch"] for h in history]
        val_losses = [h["val_loss"] for h in history]
        ax2.plot(
            epochs, val_losses, label=style["label"], color=style["color"],
            linestyle=style["linestyle"], marker=style["marker"],
            linewidth=style["linewidth"], markersize=5, alpha=0.9
        )

    ax2.set_title(
        "Panel B: Validation Loss Trajectory\n"
        "(Evaluating Optimization Stability and Gradient Noise Dynamics)",
        fontweight="bold", fontsize=11
    )
    ax2.set_xlabel("Epoch", fontweight="bold")
    ax2.set_ylabel("Validation Cross-Entropy Loss", fontweight="bold")
    ax2.set_xticks(range(2, 21, 2))
    ax2.set_ylim(0.2, 2.9)
    ax2.grid(True)
    ax2.legend(loc="upper right", framealpha=0.92, fontsize=8.8)

    plt.suptitle(
        "Evaluating Hypothesis 7: Isolated Layer Compressive Sensing Multi-Epoch Convergence Dynamics\n"
        "(ResNet-50 on CIFAR-10 Across 20 Epochs with N=4 Ranks, 50% Ring Skip, and 15% DCT Reconstruction)",
        fontsize=12.5, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.91])

    out_path = os.path.join(output_dir, "fig10d_layer_cs_convergence.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot EXP-10 Layer Compressive Sensing Rate-Distortion Suite")
    parser.add_argument("--json", type=str, default="training/logs/layer_cs_rate_distortion.json")
    parser.add_argument("--ablation-json", type=str, default="training/logs/layer_cs_isolated_ablation.json")
    parser.add_argument("--recoverability-json", type=str, default="training/logs/layer_recoverability_ablation.json")
    parser.add_argument("--output-dir", type=str, default="training/figures/exp10_layer_cs_budget")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.exists(args.json):
        with open(args.json, "r") as f:
            data = json.load(f)
        plot_fig10a_rate_distortion(data, args.output_dir)
        plot_fig10b_rel_l2_error(data, args.output_dir)

    if os.path.exists(args.ablation_json) and os.path.exists(args.recoverability_json):
        with open(args.ablation_json, "r") as f:
            ablation_data = json.load(f)
        with open(args.recoverability_json, "r") as f:
            recover_data = json.load(f)
        plot_fig10c_isolated_accuracy(ablation_data, recover_data, args.output_dir)
        plot_fig10d_isolated_convergence(ablation_data, args.output_dir)


if __name__ == "__main__":
    main()

