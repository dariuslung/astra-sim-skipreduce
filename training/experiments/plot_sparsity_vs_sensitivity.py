"""
Publication-Quality Visualization Suite for Hypothesis 6:
Multi-Epoch Correlation Between Layer Sensitivity and Gradient Sparsity.

Generates 3 separated, focused figures in training/figures/exp09_sparsity_vs_sensitivity/:
1. fig09a_sparsity_vs_sensitivity_epochs.png:
   6-panel multi-epoch scatter plot tracking the transition across training
   (Epochs 1, 2, 3, 5, 10, and 20).
2. fig09b_correlation_trajectory.png:
   Standalone 20-epoch correlation trajectory (Pearson r and Spearman rho)
   with shaded representation discovery vs. converged separation regimes.
3. fig09c_active_skipped_e10.png:
   Standalone comparison of Baseline Unskipped vs. Active Skipped E10
   (20-epoch mean bar comparison + multi-epoch trajectory across all 20 epochs).
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
from scipy import stats


def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10.5,
        "axes.labelsize": 10.5,
        "axes.titlesize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9,
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
}

LABEL_MAP = {
    "conv3x3_spatial": "3x3 Convolutions (Spatial Details) [conv3x3_spatial]",
    "conv1x1_expand": "1x1 Convolutions (Channel Upscaling) [conv1x1_expand]",
    "conv1x1_reduce": "1x1 Convolutions (Channel Compression) [conv1x1_reduce]",
    "conv1x1_downsample": "Shortcut Connections (Residual Path) [conv1x1_downsample]",
}

SHORT_LABEL_MAP = {
    "conv3x3_spatial": "3x3 Spatial [conv3x3]",
    "conv1x1_expand": "1x1 Expand [conv1x1_exp]",
    "conv1x1_reduce": "1x1 Reduce [conv1x1_red]",
    "conv1x1_downsample": "1x1 Shortcut [conv1x1_down]",
}


def load_data(conv_path: str, abl_path: str, fallback_abl_path: str):
    with open(conv_path, "r") as f:
        conv_data = json.load(f)

    abl_data = {}
    if os.path.exists(abl_path):
        try:
            with open(abl_path, "r") as f:
                abl_data = json.load(f)
        except Exception:
            abl_data = {}

    if not abl_data and os.path.exists(fallback_abl_path):
        with open(fallback_abl_path, "r") as f:
            abl_data = json.load(f)

    return conv_data, abl_data


def get_epoch_accuracy_drop(abl_data: dict, key: str, epoch: int) -> float:
    """
    Returns the contemporaneous validation accuracy drop (in percentage points)
    at the given epoch: base_acc(epoch) - cond_acc(epoch).
    Positive indicates performance degradation, negative indicates gain (regularization).
    """
    cond_history = abl_data.get("conditions", {}).get(key, {}).get("epochs_history", [])
    base_history = abl_data.get("conditions", {}).get("baseline", {}).get("epochs_history", [])
    if cond_history and base_history and len(cond_history) >= epoch and len(base_history) >= epoch:
        base_acc = base_history[epoch - 1]["val_acc"]
        cond_acc = cond_history[epoch - 1]["val_acc"]
        return round(base_acc - cond_acc, 2)
    # Fallback to final delta if history missing
    cond = abl_data.get("conditions", {}).get(key, {})
    delta = cond.get("delta_val_acc", 0.0)
    return round(-delta, 2)


def get_layer_accuracy_drop(abl_data: dict, key: str) -> float:
    """
    Returns the final validation accuracy drop (in percentage points) when skipping layer key.
    A positive number indicates performance degradation (e.g., +5.12 pp drop).
    """
    cond = abl_data.get("conditions", {}).get(key, {})
    delta = cond.get("delta_val_acc", 0.0)
    return round(-delta, 2)


def plot_fig09a_multi_epoch_scatters(conv_data: dict, abl_data: dict, output_dir: str):
    """
    Figure 09a: 6-Panel Multi-Epoch Contemporaneous Gradient Sparsity vs. Accuracy Drop.
    Displays scatter plots, regression lines, and correlation statistics for 6 key milestone epochs.
    Both the x-axis (E10) and y-axis (Accuracy Drop) are strictly contemporaneous to each epoch.
    """
    conv_keys = ["conv3x3_spatial", "conv1x1_expand", "conv1x1_reduce", "conv1x1_downsample"]

    milestone_epochs = [
        {"epoch": 1, "title": "Epoch 1: Early Representation Learning", "regime": "Noisy Phase: Pointwise Regularization Leads"},
        {"epoch": 2, "title": "Epoch 2: Filter Formation", "regime": "Transition: Spatial Convs Lag Immediately"},
        {"epoch": 3, "title": "Epoch 3: Peak Regularization", "regime": "Pointwise Skipping Gains up to +5.7 pp"},
        {"epoch": 5, "title": "Epoch 5: Mid-Training Transition", "regime": "Pointwise Skipping Advantage Normalizing"},
        {"epoch": 10, "title": "Epoch 10: Late Training Emergence", "regime": "Spatial vs. Pointwise Separation"},
        {"epoch": 20, "title": "Epoch 20: Final Convergence", "regime": "Spatial Damage (+5.12 pp), Pointwise Parity"},
    ]

    fig, axes = plt.subplots(2, 3, figsize=(17.0, 11.2), dpi=300)
    axes = axes.flatten()

    for idx, m in enumerate(milestone_epochs):
        ax = axes[idx]
        ep_num = m["epoch"]

        ep_dict = conv_data["epochs_data"][ep_num - 1]["by_layer_type"]
        e10_vals = np.array([ep_dict[k]["energy10"] for k in conv_keys])
        acc_drops = np.array([get_epoch_accuracy_drop(abl_data, k, ep_num) for k in conv_keys])

        slope, intercept, r_val, p_val, _ = stats.linregress(e10_vals, acc_drops)
        rho_val, _ = stats.spearmanr(e10_vals, acc_drops)

        x_margin = max(2.5, (max(e10_vals) - min(e10_vals)) * 0.40)
        x_min = min(e10_vals) - x_margin
        x_max = max(e10_vals) + x_margin
        x_line = np.linspace(x_min, x_max, 50)
        ax.plot(x_line, slope * x_line + intercept, color="#555555", linestyle="--", linewidth=1.6,
                label=f"Linear Fit ($R^2 = {r_val**2:.3f}$, $p = {p_val:.3f}$)")

        # Draw baseline parity line (y = 0)
        ax.axhline(0, color="#888888", linestyle=":", linewidth=1.3, alpha=0.8, label="Baseline Parity (0 pp Drop)")

        for k, e_val, s_val in zip(conv_keys, e10_vals, acc_drops):
            col = COLOR_MAP[k]
            lbl = SHORT_LABEL_MAP[k]
            ax.scatter(e_val, s_val, color=col, s=130, alpha=0.9, zorder=5)

            # Per-epoch label positioning to eliminate collisions and axis touching
            if ep_num == 1:
                if k == "conv3x3_spatial":
                    x_off, y_off, ha = 8, 8, "left"
                elif k == "conv1x1_expand":
                    x_off, y_off, ha = 8, 6, "left"
                elif k == "conv1x1_reduce":
                    x_off, y_off, ha = 8, -12, "left"
                else:  # downsample
                    x_off, y_off, ha = -8, 8, "right"
            elif ep_num == 2:
                if k == "conv3x3_spatial":
                    x_off, y_off, ha = -8, 8, "right"
                elif k == "conv1x1_expand":
                    x_off, y_off, ha = -8, 8, "right"
                elif k == "conv1x1_reduce":
                    x_off, y_off, ha = 8, -12, "left"
                else:  # downsample
                    x_off, y_off, ha = 8, 8, "left"
            elif ep_num == 3:
                if k == "conv3x3_spatial":
                    x_off, y_off, ha = -8, 8, "right"
                elif k == "conv1x1_expand":
                    x_off, y_off, ha = -8, -12, "right"
                elif k == "conv1x1_reduce":
                    x_off, y_off, ha = 8, 8, "left"
                else:  # downsample
                    x_off, y_off, ha = -8, 8, "right"
            elif ep_num == 5:
                if k == "conv3x3_spatial":
                    x_off, y_off, ha = -8, 8, "right"
                elif k == "conv1x1_expand":
                    x_off, y_off, ha = 8, 6, "left"
                elif k == "conv1x1_reduce":
                    x_off, y_off, ha = 8, 6, "left"
                else:  # downsample
                    x_off, y_off, ha = -8, -12, "right"
            elif ep_num == 10:
                if k == "conv3x3_spatial":
                    x_off, y_off, ha = 8, -12, "left"
                elif k == "conv1x1_expand":
                    x_off, y_off, ha = 8, 8, "left"
                elif k == "conv1x1_reduce":
                    x_off, y_off, ha = 8, -12, "left"
                else:  # downsample
                    x_off, y_off, ha = 8, 8, "left"
            else:  # Epoch 20
                if k == "conv3x3_spatial":
                    x_off, y_off, ha = -8, 8, "right"
                elif k == "conv1x1_expand":
                    x_off, y_off, ha = 8, 8, "left"
                elif k == "conv1x1_reduce":
                    x_off, y_off, ha = 8, -12, "left"
                else:  # downsample
                    x_off, y_off, ha = 8, 8, "left"

            ax.annotate(lbl, xy=(e_val, s_val), xytext=(x_off, y_off),
                        ha=ha, textcoords="offset points", fontsize=8.5, fontweight="medium",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.85))

        ax.set_title(f"Panel {chr(65+idx)}: {m['title']}\n$r = {r_val:+.3f}$, $\\rho = {rho_val:+.3f}$ ({m['regime']})",
                     fontweight="bold", fontsize=10.2)
        ax.set_xlabel(f"Epoch {ep_num} Energy in Top 10% Coordinates [%] [E10]", fontweight="bold")
        ax.set_ylabel(f"Epoch {ep_num} Accuracy Drop (pp) [ΔVal Acc]", fontweight="bold")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-7.5, 12.5)
        ax.grid(True)
        ax.legend(loc="upper right" if slope < 0 else "upper left", framealpha=0.9, fontsize=8.2)

    plt.suptitle(
        "Evaluating Hypothesis 6: Contemporaneous Gradient Sparsity vs. Accuracy Drop Across Epochs\n"
        "(Each Epoch Directly Evaluates That Epoch's Gradient Energy [E10] Against Actual Accuracy Drop [ΔVal Acc])",
        fontsize=12.5, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    out_a = os.path.join(output_dir, "fig09a_sparsity_vs_sensitivity_epochs.png")
    plt.savefig(out_a, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_a}")


def plot_fig09b_correlation_trajectory(conv_data: dict, abl_data: dict, output_dir: str):
    """
    Figure 09b: Standalone Contemporaneous Multi-Epoch Correlation Trajectory.
    Compares how well Gradient Sparsity (E10) vs. Parameter Count predicts accuracy drop
    contemporaneously across all 20 training epochs.
    """
    conv_keys = ["conv3x3_spatial", "conv1x1_expand", "conv1x1_reduce", "conv1x1_downsample"]
    params = [11.318976, 5.029888, 4.329472, 2.768896]  # MParam

    epochs_data = conv_data["epochs_data"]
    epochs = [d["epoch"] for d in epochs_data]

    r_e10_trajectory = []
    rho_e10_trajectory = []
    r_param_trajectory = []
    rho_param_trajectory = []

    for d in epochs_data:
        ep_num = d["epoch"]
        e10_vals = np.array([d["by_layer_type"][k]["energy10"] for k in conv_keys])
        acc_drops = np.array([get_epoch_accuracy_drop(abl_data, k, ep_num) for k in conv_keys])

        r_e, _ = stats.pearsonr(e10_vals, acc_drops)
        rho_e, _ = stats.spearmanr(e10_vals, acc_drops)
        r_p, _ = stats.pearsonr(params, acc_drops)
        rho_p, _ = stats.spearmanr(params, acc_drops)

        r_e10_trajectory.append(float(r_e))
        rho_e10_trajectory.append(float(rho_e))
        r_param_trajectory.append(float(r_p))
        rho_param_trajectory.append(float(rho_p))

    fig, ax = plt.subplots(figsize=(13.0, 7.0), dpi=300)

    # Plot Parameter Count lines
    ax.plot(epochs, r_param_trajectory, marker="^", color="#2ca02c", linewidth=2.4,
            label="Parameter Count Linear Correlation ($r$) with Accuracy Drop")
    ax.plot(epochs, rho_param_trajectory, marker="v", color="#2ca02c", linewidth=2.0, linestyle=":",
            alpha=0.7, label="Parameter Count Rank Correlation ($\\rho$) with Accuracy Drop")

    # Plot E10 lines
    ax.plot(epochs, r_e10_trajectory, marker="o", color="#1f77b4", linewidth=2.4,
            label="Gradient Energy [E10] Linear Correlation ($r$) with Accuracy Drop")
    ax.plot(epochs, rho_e10_trajectory, marker="s", color="#ff7f0e", linewidth=2.0, linestyle="--",
            label="Gradient Energy [E10] Rank Correlation ($\\rho$) with Accuracy Drop")

    ax.axhline(0, color="black", linestyle=":", linewidth=1.2, alpha=0.7)
    ax.axvspan(0.8, 5.5, color="#1f77b4", alpha=0.08, label="Early Training: E10 Uncorrelated (r ≈ 0); 1x1 Regularization Advantage")
    ax.axvspan(5.5, 20.2, color="#ff7f0e", alpha=0.08, label="Late Training: Spatial vs. Pointwise Separation (E10 r ≈ +0.88)")

    # Detailed regime descriptions in plain English
    box_early = (
        "Early Training (Epochs 1-5):\n"
        "• E10 has near-zero correlation with accuracy drop (r ≈ 0)\n"
        "• Skipping 1x1 convs gives regularization gains (up to +5.67 pp)\n"
        "• Parameter volume reliably separates damage early on (r = 0.85 to 0.97)"
    )
    ax.text(0.03, 0.12, box_early, transform=ax.transAxes, fontsize=9.2,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f4f8", edgecolor="#2b5c8f", alpha=0.9))

    box_late = (
        "Late Training (Epochs 6-20):\n"
        "• Parameter volume consistently predicts accuracy drop (r = 0.98 to 0.996)\n"
        "• E10 correlation rises to r ≈ +0.88 due to binary clustering:\n"
        "  Spatial 3x3 convs (11.3M params) suffer persistent drop (+5.12 pp)\n"
        "  Pointwise 1x1 convs cluster near baseline parity (< 1.0 pp drop)"
    )
    ax.text(0.38, 0.40, box_late, transform=ax.transAxes, fontsize=9.2,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff8f0", edgecolor="#d95f02", alpha=0.9))

    ax.set_title("Evaluating Hypothesis 6: Contemporaneous Correlation Trajectory Across Training Epochs\n"
                 "Comparing Gradient Energy Concentration [E10] vs. Parameter Volume as Predictors of Accuracy Drop",
                 fontweight="bold", fontsize=11.5)
    ax.set_xlabel("Training Epoch", fontweight="bold")
    ax.set_ylabel("Contemporaneous Correlation with Epoch Accuracy Drop [r, ρ]", fontweight="bold")
    ax.set_xticks(epochs)
    ax.set_ylim(-0.7, 1.15)
    ax.grid(True)
    ax.legend(loc="lower right", framealpha=0.92, fontsize=8.8)

    plt.tight_layout()
    out_b = os.path.join(output_dir, "fig09b_correlation_trajectory.png")
    plt.savefig(out_b, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_b}")


def plot_fig09c_active_skipped_e10(conv_data: dict, abl_data: dict, output_dir: str):
    """
    Figure 09c: Standalone Evaluation of Baseline vs. Active Skipped Gradient E10.
    Panel A: 20-Epoch Mean Bar Comparison.
    Panel B: Multi-Epoch Trajectories of Active Skipped vs. Baseline E10 across all 20 epochs.
    """
    conv_keys = ["conv3x3_spatial", "conv1x1_expand", "conv1x1_reduce", "conv1x1_downsample"]
    epochs_data = conv_data["epochs_data"]
    epochs = [d["epoch"] for d in epochs_data]

    # Baseline 20-epoch mean E10
    base_mean_e10 = np.array([
        float(np.mean([d["by_layer_type"][k]["energy10"] for d in epochs_data]))
        for k in conv_keys
    ])

    # Skipped 20-epoch mean E10 and trajectories
    skipped_mean_e10 = []
    skipped_trajectories = {}
    base_trajectories = {}

    for k in conv_keys:
        cond = abl_data.get("conditions", {}).get(k, {})
        epochs_hist = cond.get("epochs_history", [])
        if epochs_hist:
            e10_hist = [ep["by_layer_type"][k]["energy10"] for ep in epochs_hist]
            skipped_mean_e10.append(float(np.mean(e10_hist)))
            skipped_trajectories[k] = e10_hist
        else:
            skipped_mean_e10.append(base_mean_e10[conv_keys.index(k)])
            skipped_trajectories[k] = [d["by_layer_type"][k]["energy10"] for d in epochs_data]
        base_trajectories[k] = [d["by_layer_type"][k]["energy10"] for d in epochs_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.0, 6.4), dpi=300)

    # -------------------------------------------------------------------------
    # SUBPANEL 1: 20-Epoch Mean Bar Comparison
    # -------------------------------------------------------------------------
    x_indices = np.arange(len(conv_keys))
    bar_width = 0.35

    base_bars = ax1.bar(x_indices - bar_width/2, base_mean_e10, bar_width,
                        label="Normal Updates [Baseline E10 20-Epoch Mean]", color="#4682b4", alpha=0.85)
    skip_bars = ax1.bar(x_indices + bar_width/2, skipped_mean_e10, bar_width,
                        label="Active Updates During Skipping [E10 on Non-Skipped Steps]", color="#e7298a", alpha=0.85)

    for rect, lbl in zip(base_bars, base_mean_e10):
        ax1.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.8, f"{lbl:.1f}%",
                 ha="center", va="bottom", fontsize=9, fontweight="medium")

    for rect, lbl in zip(skip_bars, skipped_mean_e10):
        ax1.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.8, f"{lbl:.1f}%",
                 ha="center", va="bottom", fontsize=9, fontweight="bold", color="#b00060")

    clean_xlabels = ["3x3 Spatial\n[conv3x3]", "1x1 Expand\n[conv1x1_exp]", "1x1 Reduce\n[conv1x1_red]", "1x1 Shortcut\n[conv1x1_down]"]
    ax1.set_xticks(x_indices)
    ax1.set_xticklabels(clean_xlabels, fontsize=9.5)
    ax1.set_ylabel("Energy in Top 10% Coordinates [%] [E10]", fontweight="bold")
    ax1.set_ylim(65, 102)
    ax1.set_title("Panel A: Average Gradient Energy Concentration Across 20 Epochs\n"
                  "Comparing Normal Updates vs. Active Updates When Skipping Every Other Step",
                  fontweight="bold", fontsize=11)
    ax1.grid(True, axis="y")
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=9)

    # -------------------------------------------------------------------------
    # SUBPANEL 2: Multi-Epoch Trajectories of Baseline vs Active Skipped E10
    # -------------------------------------------------------------------------
    for k in conv_keys:
        col = COLOR_MAP[k]
        lbl = SHORT_LABEL_MAP[k]
        # Active skipped trajectory
        ax2.plot(epochs, skipped_trajectories[k], marker="o", color=col, linewidth=2.0,
                 label=f"{lbl} (Active Updates)")
        # Baseline trajectory
        ax2.plot(epochs, base_trajectories[k], linestyle="--", color=col, alpha=0.55, linewidth=1.5)

    legend_elements = [
        Line2D([0], [0], color="#ff7f0e", lw=2, marker="o", label="3x3 Spatial Convolutions [conv3x3_spatial]"),
        Line2D([0], [0], color="#2ca02c", lw=2, marker="o", label="1x1 Upscaling Convolutions [conv1x1_expand]"),
        Line2D([0], [0], color="#1f77b4", lw=2, marker="o", label="1x1 Compression Convolutions [conv1x1_reduce]"),
        Line2D([0], [0], color="#17becf", lw=2, marker="o", label="Shortcut Connections [conv1x1_downsample]"),
        Line2D([0], [0], color="#555555", lw=2, linestyle="-", marker="o", label="Solid: Active Updates During Skipping"),
        Line2D([0], [0], color="#555555", lw=1.5, linestyle="--", label="Dashed: Normal Unskipped Updates"),
    ]

    ax2.set_title("Panel B: Energy Concentration Trajectories Across Training\n"
                  "Active Updates Retain High Concentration Across All 20 Epochs",
                  fontweight="bold", fontsize=11)
    ax2.set_xlabel("Training Epoch", fontweight="bold")
    ax2.set_ylabel("Energy in Top 10% Coordinates [%] [E10]", fontweight="bold")
    ax2.set_xticks(epochs)
    ax2.set_ylim(70, 100)
    ax2.grid(True)
    ax2.legend(handles=legend_elements, loc="upper right", framealpha=0.9, fontsize=8.0)

    plt.suptitle(
        "Evaluating Hypothesis 6: Gradient Energy Concentration Under Layer Skipping\n"
        "(Comparing Active Updates During Skipping vs. Normal Unskipped Training in ResNet-50)",
        fontsize=12.5, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    out_c = os.path.join(output_dir, "fig09c_active_skipped_e10.png")
    plt.savefig(out_c, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_c}")


def main():
    parser = argparse.ArgumentParser(description="Plot Hypothesis 6: Multi-Epoch Sparsity vs. Sensitivity Suite")
    parser.add_argument("--convergence-json", type=str, default="training/logs/resnet50_convergence_sparsity.json")
    parser.add_argument("--ablation-json", type=str, default="training/logs/layer_recoverability_with_gradients.json")
    parser.add_argument("--fallback-ablation-json", type=str, default="training/logs/layer_recoverability_ablation.json")
    parser.add_argument("--output-dir", type=str, default="training/figures/exp09_sparsity_vs_sensitivity")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    conv_data, abl_data = load_data(
        conv_path=args.convergence_json,
        abl_path=args.ablation_json,
        fallback_abl_path=args.fallback_ablation_json,
    )

    plot_fig09a_multi_epoch_scatters(conv_data, abl_data, args.output_dir)
    plot_fig09b_correlation_trajectory(conv_data, abl_data, args.output_dir)
    plot_fig09c_active_skipped_e10(conv_data, abl_data, args.output_dir)


if __name__ == "__main__":
    main()
