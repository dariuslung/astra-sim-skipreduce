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


def main():
    parser = argparse.ArgumentParser(description="Plot EXP-10 Layer Compressive Sensing Rate-Distortion Suite")
    parser.add_argument("--json", type=str, default="training/logs/layer_cs_rate_distortion.json")
    parser.add_argument("--output-dir", type=str, default="training/figures/exp10_layer_cs_budget")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.json, "r") as f:
        data = json.load(f)

    plot_fig10a_rate_distortion(data, args.output_dir)
    plot_fig10b_rel_l2_error(data, args.output_dir)


if __name__ == "__main__":
    main()
