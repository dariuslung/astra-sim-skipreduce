"""
Protocol A: Isolated Layer-Type Skipping Ablation Probe.
Evaluates Hypothesis 5 (Heterogeneous Layer Sensitivity & Recoverability Under Gradient Skipping)
on CIFAR-10 ResNet-50.
"""

import argparse
from collections import defaultdict
import copy
import json
import os
import random
import sys
import time
from typing import Dict, List, Any, Optional

# Ensure repo root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from training.models import get_cifar_resnet50, get_resnet50_layer_metadata


CONDITIONS_CONFIG = {
    "baseline": {
        "name": "baseline",
        "description": "Standard SGD baseline (0% skip)",
        "target_types": [],
        "skip_freq": 0,
    },
    "conv3x3_spatial": {
        "name": "conv3x3_spatial",
        "description": "50% skip on 3x3 spatial convs",
        "target_types": ["conv3x3_spatial"],
        "skip_freq": 2,
    },
    "conv1x1_expand": {
        "name": "conv1x1_expand",
        "description": "50% skip on 1x1 expansion convs",
        "target_types": ["conv1x1_expand"],
        "skip_freq": 2,
    },
    "conv1x1_reduce": {
        "name": "conv1x1_reduce",
        "description": "50% skip on 1x1 reduction convs",
        "target_types": ["conv1x1_reduce"],
        "skip_freq": 2,
    },
    "conv1x1_downsample": {
        "name": "conv1x1_downsample",
        "description": "50% skip on 1x1 shortcut downsamplers",
        "target_types": ["conv1x1_downsample"],
        "skip_freq": 2,
    },
    "classifier_head": {
        "name": "classifier_head",
        "description": "50% skip on linear classifier head",
        "target_types": ["classifier_head"],
        "skip_freq": 2,
    },
    "skip_all": {
        "name": "skip_all",
        "description": "50% skip uniformly across all layers",
        "target_types": ["all"],
        "skip_freq": 2,
    },
}


def set_seed(seed: int = 42):
    """Ensures exact reproducible dataset shuffling and weight initialization."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_cifar10_loaders(data_dir: str, batch_size: int = 128, num_workers: int = 4):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=True
    )

    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return trainloader, testloader


def evaluate(model: nn.Module, testloader, criterion, device: torch.device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return round(val_loss / total, 4), round(100.0 * correct / total, 2)


def run_condition(
    cond_key: str,
    cfg: Dict[str, Any],
    epochs: int,
    batch_size: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    data_dir: str,
    device: torch.device,
    seed: int,
    num_workers: int = 4,
) -> Dict[str, Any]:
    print(f"\n{'='*80}")
    print(f"Condition: [{cond_key}] - {cfg['description']}")
    print(f"Target types: {cfg['target_types']} | Skip Freq: {cfg['skip_freq']} (50% on targets)")
    print(f"{'='*80}")

    # Set identical seed before dataset loaders and model initialization
    set_seed(seed)
    trainloader, testloader = get_cifar10_loaders(data_dir, batch_size=batch_size, num_workers=num_workers)

    set_seed(seed)
    model = get_cifar_resnet50(num_classes=10).to(device)

    # Classify parameters into targeted vs untargeted
    target_params = []
    total_params = 0
    skipped_params = 0
    target_param_names = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        p_count = p.numel()
        total_params += p_count
        meta = get_resnet50_layer_metadata(name)
        is_target = ("all" in cfg["target_types"]) or (meta["layer_type"] in cfg["target_types"])
        if is_target:
            target_params.append(p)
            target_param_names.append(name)
            skipped_params += p_count

    pct_skipped = round(100.0 * skipped_params / total_params, 2)
    print(f"Targeted parameters: {skipped_params:,d} / {total_params:,d} ({pct_skipped}%) across {len(target_params)} tensors")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    skip_freq = cfg["skip_freq"]
    epochs_history = []
    condition_start = time.time()

    print(f"{'Epoch':>5} | {'LR':>7} | {'Train Loss':>10} | {'Val Loss':>8} | {'Val Acc %':>9} | {'Time (s)':>8}")
    print("-" * 65)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Protocol A Skipping: Set targeted parameter gradients to None
            # when batch_idx % skip_freq != 0.
            # In optim.SGD, params with grad=None are bypassed: no momentum update, no weight update.
            if skip_freq > 0 and (batch_idx % skip_freq != 0):
                for p in target_params:
                    p.grad = None

            optimizer.step()
            running_loss += loss.item() * targets.size(0)

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        train_loss = round(running_loss / len(trainloader.dataset), 4)
        val_loss, val_acc = evaluate(model, testloader, criterion, device)
        epoch_time = round(time.time() - epoch_start, 2)

        epochs_history.append({
            "epoch": epoch,
            "lr": round(current_lr, 5),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch_time_s": epoch_time,
        })

        print(f"{epoch:5d} | {current_lr:7.5f} | {train_loss:10.4f} | {val_loss:8.4f} | {val_acc:8.2f}% | {epoch_time:8.2f}s")

    total_time = round(time.time() - condition_start, 2)
    final_val_acc = epochs_history[-1]["val_acc"]
    best_val_acc = max(e["val_acc"] for e in epochs_history)
    final_train_loss = epochs_history[-1]["train_loss"]

    return {
        "condition": cond_key,
        "description": cfg["description"],
        "target_types": cfg["target_types"],
        "skip_freq": cfg["skip_freq"],
        "skipped_params": skipped_params,
        "total_params": total_params,
        "pct_skipped_params": pct_skipped,
        "num_target_tensors": len(target_params),
        "final_val_acc": final_val_acc,
        "best_val_acc": best_val_acc,
        "final_train_loss": final_train_loss,
        "total_time_s": total_time,
        "epochs_history": epochs_history,
    }


def main():
    parser = argparse.ArgumentParser(description="Protocol A: Layer Recoverability Ablation Probe")
    parser.add_argument("--data-dir", type=str, default="./data", help="CIFAR-10 data path")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs per condition")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["all"],
        help="Conditions to run: all, or subset of [baseline, conv3x3_spatial, conv1x1_expand, conv1x1_reduce, conv1x1_downsample, classifier_head, skip_all]",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for identical initial weights & data ordering")
    parser.add_argument("--output-dir", type=str, default="training/logs", help="Output directory")
    parser.add_argument("--output-name", type=str, default="layer_recoverability_ablation.json", help="Log filename")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader num_workers")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing condition results in log file")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"=== Protocol A: Layer Recoverability Ablation Probe ===")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Epochs per condition: {args.epochs} | Batch Size: {args.batch_size} | Seed: {args.seed}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, args.output_name)

    # Determine conditions to run
    if "all" in args.conditions:
        selected_keys = list(CONDITIONS_CONFIG.keys())
    else:
        selected_keys = [k for k in args.conditions if k in CONDITIONS_CONFIG]
        if not selected_keys:
            print(f"Error: No valid conditions selected from {args.conditions}. Valid: {list(CONDITIONS_CONFIG.keys())}")
            sys.exit(1)

    print(f"Selected conditions ({len(selected_keys)}): {selected_keys}")

    # Load existing results if available to support resume/incremental updates
    results_data = {
        "experiment": "Protocol A: Layer Recoverability Ablation Probe",
        "hypothesis": "Hypothesis 5: Heterogeneous Layer Sensitivity & Recoverability Under Gradient Skipping",
        "model": "resnet50_cifar10",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "initial_lr": args.lr,
        "seed": args.seed,
        "conditions": {},
    }
    if os.path.exists(out_path) and not args.overwrite:
        try:
            with open(out_path, "r") as f:
                existing = json.load(f)
                if isinstance(existing.get("conditions"), dict):
                    results_data["conditions"] = existing["conditions"]
                    print(f"Loaded existing results for {list(results_data['conditions'].keys())} from {out_path}")
        except Exception as e:
            print(f"Notice: Could not parse existing results file ({e}). Starting fresh.")

    for cond_key in selected_keys:
        if cond_key in results_data["conditions"] and not args.overwrite:
            print(f"\nSkipping [{cond_key}]: already exists in {out_path} (use --overwrite to rerun)")
            continue

        cfg = CONDITIONS_CONFIG[cond_key]
        cond_res = run_condition(
            cond_key=cond_key,
            cfg=cfg,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            data_dir=args.data_dir,
            device=device,
            seed=args.seed,
            num_workers=args.num_workers,
        )

        results_data["conditions"][cond_key] = cond_res

        # Save immediately after each condition completes so progress is never lost
        with open(out_path, "w") as f:
            json.dump(results_data, f, indent=2)
        print(f"Progress saved to {out_path}")

    # Compute comparative summary metrics relative to baseline
    baseline_res = results_data["conditions"].get("baseline")
    baseline_acc = baseline_res["final_val_acc"] if baseline_res else None

    print("\n" + "=" * 105)
    print("SUMMARY: Evaluating Hypothesis 5 (Layer Recoverability & Sensitivity Under 50% Gradient Skipping)")
    print("=" * 105)
    print(f"{'Condition':<22} | {'Params Skipped':>15} | {'% Model':>8} | {'Final Acc':>10} | {'Delta Acc':>10} | {'Sens Index':>14} | {'Best Acc':>9}")
    print("-" * 105)

    for cond_key, c_data in results_data["conditions"].items():
        val_acc = c_data["final_val_acc"]
        best_acc = c_data["best_val_acc"]
        p_skip = c_data["skipped_params"]
        pct_p = c_data["pct_skipped_params"]

        if baseline_acc is not None:
            delta_acc = round(val_acc - baseline_acc, 2)
            c_data["delta_val_acc"] = delta_acc
            # Sensitivity index: Drop in accuracy per 1M parameters skipped
            # Negative delta is a drop; sensitivity index = |delta_acc| / (params_skipped in M)
            m_params = p_skip / 1e6
            if m_params > 0:
                sens_idx = round(abs(delta_acc) / m_params, 3)
            else:
                sens_idx = 0.0
            c_data["sensitivity_index_drop_per_mparam"] = sens_idx
            delta_str = f"{delta_acc:+6.2f}%"
            sens_str = f"{sens_idx:8.3f} %/M"
        else:
            delta_str = "N/A"
            sens_str = "N/A"

        print(f"{cond_key:<22} | {p_skip:15,d} | {pct_p:7.2f}% | {val_acc:9.2f}% | {delta_str:>10} | {sens_str:>14} | {best_acc:8.2f}%")

    print("=" * 105)

    # Re-save with delta metrics included
    with open(out_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"Final results updated in {out_path}\n")


if __name__ == "__main__":
    main()
