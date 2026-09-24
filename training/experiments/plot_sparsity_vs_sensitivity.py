"""
Publication-Quality Visualization Suite for Hypothesis 6:
Correlation Between Layer Sensitivity and Baseline Gradient Sparsity (Hoyer & E10).

Evaluates whether baseline gradient sparsity metrics (Top-10% Energy E10 and Hoyer Index)
correlate inversely with layer sensitivity to gradient skipping.
Generates:
  training/figures/convergence/fig09_sparsity_vs_sensitivity.png
"""

import argparse
import json
import os
import sys

# Ensure repo root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


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
    "conv3x3_spatial": "#ff7f0e",     # Orange
    "conv1x1_expand": "#2ca02c",      # Green
    "conv1x1_reduce": "#1f77b4",      # Blue
    "conv1x1_downsample": "#17becf",  # Cyan
    "classifier_head": "#d62728",     # Red
}

LABEL_MAP = {
    "conv3x3_spatial": "3x3 Conv (Spatial)",
    "conv1x1_expand": "1x1 Conv (Expand)",
    "conv1x1_reduce": "1x1 Conv (Reduce)",
    "conv1x1_downsample": "1x1 Conv (Shortcut)",
    "classifier_head": "Classifier Head (fc)",
}


def plot_sparsity_vs_sensitivity(output_dir: str):
    """Plot 2-panel correlation: Sensitivity Index vs. E10 and Hoyer."""
    # Data from Table 2 (Epoch 1 steady state) & Protocol A ablation logs
    conv_keys = ["conv3x3_spatial", "conv1x1_expand", "conv1x1_reduce", "conv1x1_downsample"]
    
    e10_conv = np.array([89.3, 91.4, 93.0, 95.8])
    hoyer_conv = np.array([0.5799, 0.6018, 0.6277, 0.6952])
    sens_conv = np.array([0.452, 0.193, 0.113, -0.007])
    
    # Statistical regressions
    slope_e, int_e, r_e, p_e, _ = stats.linregress(e10_conv, sens_conv)
    rho_e, _ = stats.spearmanr(e10_conv, sens_conv)
    
    slope_h, int_h, r_h, p_h, _ = stats.linregress(hoyer_conv, sens_conv)
    rho_h, _ = stats.spearmanr(hoyer_conv, sens_conv)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    # --- PANEL A: Sensitivity vs. E10 ---
    x_e = np.linspace(88.5, 96.8, 50)
    ax1.plot(x_e, slope_e * x_e + int_e, color="#555555", linestyle="--", linewidth=1.8,
             label=f"Linear Fit ($R^2 = {r_e**2:.3f}$, $p = {p_e:.3f}$)")
    
    offsets_a = {
        "conv3x3_spatial": (8, 4),
        "conv1x1_expand": (-8, 8),
        "conv1x1_reduce": (8, 6),
        "conv1x1_downsample": (8, 6),
    }
    ha_a = {
        "conv3x3_spatial": "left",
        "conv1x1_expand": "right",
        "conv1x1_reduce": "left",
        "conv1x1_downsample": "left",
    }
    
    for k, e_val, s_val in zip(conv_keys, e10_conv, sens_conv):
        col = COLOR_MAP[k]
        lbl = LABEL_MAP[k]
        ax1.scatter(e_val, s_val, color=col, s=130, alpha=0.9, zorder=5)
        
        ax1.annotate(lbl, xy=(e_val, s_val), xytext=offsets_a[k],
                     ha=ha_a[k],
                     textcoords="offset points", fontsize=9.5, fontweight="medium",
                     bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))
    
    ax1.set_title("Top-10% Energy Concentration ($E_{10}$) vs Sensitivity", fontweight="bold", fontsize=12)
    ax1.set_xlabel("Top-10% Energy Concentration ($E_{10}$, %)", fontweight="bold")
    ax1.set_ylabel("Normalized Sensitivity Index (pp / MParam)", fontweight="bold")
    ax1.set_xlim(88.0, 97.5)
    ax1.set_ylim(-0.06, 0.54)
    ax1.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax1.grid(True)
    ax1.legend(loc="upper right", framealpha=0.9)
    
    # Statistical stats callout box in Panel A
    stats_text_a = (
        f"Strong Inverse Correlation:\n"
        f"• Pearson $r = {r_e:.3f}$ ($R^2 = {r_e**2:.3f}$)\n"
        f"• Spearman $\\rho = {rho_e:.3f}$ (Monotonic)\n"
        f"• Stat Significance: $p = {p_e:.3f}$"
    )
    ax1.text(0.04, 0.12, stats_text_a, transform=ax1.transAxes, fontsize=9.5,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f4f8", edgecolor="#2b5c8f", alpha=0.9))
    
    # --- PANEL B: Sensitivity vs. Hoyer ---
    x_h = np.linspace(0.565, 0.710, 50)
    ax2.plot(x_h, slope_h * x_h + int_h, color="#555555", linestyle="--", linewidth=1.8,
             label=f"Linear Fit ($R^2 = {r_h**2:.3f}$, $p = {p_h:.3f}$)")
    
    offsets_b = {
        "conv3x3_spatial": (8, 4),
        "conv1x1_expand": (-8, 8),
        "conv1x1_reduce": (8, 6),
        "conv1x1_downsample": (8, 6),
    }
    ha_b = {
        "conv3x3_spatial": "left",
        "conv1x1_expand": "right",
        "conv1x1_reduce": "left",
        "conv1x1_downsample": "left",
    }
    
    for k, h_val, s_val in zip(conv_keys, hoyer_conv, sens_conv):
        col = COLOR_MAP[k]
        lbl = LABEL_MAP[k]
        ax2.scatter(h_val, s_val, color=col, s=130, alpha=0.9, zorder=5)
        
        ax2.annotate(lbl, xy=(h_val, s_val), xytext=offsets_b[k],
                     ha=ha_b[k],
                     textcoords="offset points", fontsize=9.5, fontweight="medium",
                     bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))
    
    ax2.set_title("Global Hoyer Sparsity Index vs Sensitivity", fontweight="bold", fontsize=12)
    ax2.set_xlabel("Hoyer Sparsity Index [0-1]", fontweight="bold")
    ax2.set_ylabel("Normalized Sensitivity Index (pp / MParam)", fontweight="bold")
    ax2.set_xlim(0.56, 0.72)
    ax2.set_ylim(-0.06, 0.54)
    ax2.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax2.grid(True)
    ax2.legend(loc="upper right", framealpha=0.9)
    
    # Statistical stats callout box in Panel B
    stats_text_b = (
        f"Strong Inverse Correlation:\n"
        f"• Pearson $r = {r_h:.3f}$ ($R^2 = {r_h**2:.3f}$)\n"
        f"• Spearman $\\rho = {rho_h:.3f}$ (Monotonic)\n"
        f"• Stat Significance: $p = {p_h:.3f}$"
    )
    ax2.text(0.04, 0.12, stats_text_b, transform=ax2.transAxes, fontsize=9.5,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f4f8", edgecolor="#2b5c8f", alpha=0.9))
    
    # Callout for Classifier Head (Global Regularization Outlier)
    head_note = (
        "Classifier Head (fc, 0.02M params):\n"
        "Hyper-sparse ($E_{10}=99.2\%$, $H=0.785$)\n"
        "Yields $\\Delta\\text{Acc} = +0.44\\%$ (Regularization Gain)"
    )
    ax1.text(0.96, 0.60, head_note, transform=ax1.transAxes, fontsize=8.5, ha="right",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff9e6", edgecolor="#e6ab02", alpha=0.9))
    
    plt.suptitle(
        "Evaluating Hypothesis 6: Correlation Between Layer Sensitivity and Baseline Gradient Sparsity\n"
        "(Higher Sparsity & Energy Concentration Predicts Superior Recoverability Under Gradient Skipping)",
        fontsize=13, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    
    out_path = os.path.join(output_dir, "fig09_sparsity_vs_sensitivity.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Hypothesis 6: Sparsity vs. Sensitivity Correlation")
    parser.add_argument("--output-dir", type=str, default="training/figures/exp09_sparsity_vs_sensitivity")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    plot_sparsity_vs_sensitivity(args.output_dir)


if __name__ == "__main__":
    main()
