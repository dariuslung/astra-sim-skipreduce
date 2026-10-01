"""
Ablation Runner for Isolated Layer-Type Compressive Sensing (EXP-10).
Runs 20 epochs for each isolated layer condition under 50% ring skipping (N=4, s=2)
with DCT Compressive Sensing (r=0.15), and consolidates results into a single benchmark log.
"""

import argparse
import copy
import json
import os
import sys
import time

# Ensure repo root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from training.experiments.train_layer_adaptive_cs import train_adaptive, SCHEDULE_PRESETS


ISOLATED_CONDITIONS = [
    "isolate_conv3x3_spatial",
    "isolate_conv1x1_expand",
    "isolate_conv1x1_reduce",
    "isolate_conv1x1_downsample",
]


def load_anchor_runs():
    """Loads pre-existing baseline (s=0) and all-layers-CS (s=2, r=0.15) logs."""
    baseline_path = "training/logs/ranks4_skip0_none_r0.0_noef_1789118323.json"
    all_cs_path = "training/logs/ranks4_skip2_dct_r0.15_noef_1789123956.json"

    baseline_data = None
    all_cs_data = None

    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            baseline_data = json.load(f)

    if os.path.exists(all_cs_path):
        with open(all_cs_path) as f:
            all_cs_data = json.load(f)

    return baseline_data, all_cs_data


def run_ablation(args):
    print("=" * 80)
    print("=== EXP-10: Isolated Layer-Type Compressive Sensing Ablation Probe ===")
    print(f"Ranks: {args.ranks} | Epochs: {args.epochs} | Seed: {args.seed} | LR: {args.lr}")
    print("Conditions to evaluate:")
    for c in ISOLATED_CONDITIONS:
        print(f"  • {c}: {SCHEDULE_PRESETS[c]['description']}")
    print("=" * 80)

    baseline_data, all_cs_data = load_anchor_runs()

    ablation_results = {
        "experiment": "EXP-10: Isolated Layer-Type Compressive Sensing Ablation Probe",
        "model": "resnet50_cifar10",
        "ranks": args.ranks,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "conditions": {}
    }

    if baseline_data:
        base_acc = baseline_data["epochs_data"][-1]["val_acc"]
        ablation_results["conditions"]["baseline"] = {
            "condition": "baseline",
            "description": "Standard Ring AllReduce (0% skip, s=0)",
            "final_val_acc": base_acc,
            "delta_val_acc": 0.0,
            "epochs_history": baseline_data["epochs_data"]
        }
    else:
        base_acc = 92.29

    for cond_name in ISOLATED_CONDITIONS:
        print("\n" + "#" * 80)
        print(f"Running condition: {cond_name}")
        print("#" * 80)

        cond_args = copy.deepcopy(args)
        cond_args.schedule = cond_name
        cond_args.track_sparsity = False  # Keep runs fast

        start_time = time.time()
        cond_history = train_adaptive(cond_args)
        elapsed = time.time() - start_time

        final_acc = cond_history["epochs_data"][-1]["val_acc"]
        delta_acc = round(final_acc - base_acc, 2)

        ablation_results["conditions"][cond_name] = {
            "condition": cond_name,
            "description": SCHEDULE_PRESETS[cond_name]["description"],
            "final_val_acc": final_acc,
            "delta_val_acc": delta_acc,
            "total_time_s": round(elapsed, 2),
            "epochs_history": cond_history["epochs_data"]
        }

        # Save intermediate progress
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(ablation_results, f, indent=2)
        print(f"Intermediate progress saved to {args.output_json}")

    if all_cs_data:
        all_acc = all_cs_data["epochs_data"][-1]["val_acc"]
        ablation_results["conditions"]["all_layers_cs"] = {
            "condition": "all_layers_cs",
            "description": "50% Ring Skip + 15% DCT CS Uniformly Across All Layers",
            "final_val_acc": all_acc,
            "delta_val_acc": round(all_acc - base_acc, 2),
            "epochs_history": all_cs_data["epochs_data"]
        }

    with open(args.output_json, "w") as f:
        json.dump(ablation_results, f, indent=2)

    print("\n" + "=" * 80)
    print("=== FINAL ISOLATED COMPRESSIVE SENSING ABLATION SUMMARY ===")
    print("=" * 80)
    print(f"{'Condition':30s} | {'Final Val Acc':14s} | {'ΔVal Acc vs Base':18s}")
    print("-" * 68)
    for c_name, c_data in ablation_results["conditions"].items():
        print(f"{c_name:30s} | {c_data['final_val_acc']:12.2f}% | {c_data['delta_val_acc']:+16.2f} pp")

    return ablation_results


def main():
    parser = argparse.ArgumentParser(description="Run Isolated Layer-Type Compressive Sensing Ablation")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-json", type=str, default="training/logs/layer_cs_isolated_ablation.json")
    parser.add_argument("--output-dir", type=str, default="training/logs/isolated_cs_runs")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    run_ablation(args)


if __name__ == "__main__":
    main()
