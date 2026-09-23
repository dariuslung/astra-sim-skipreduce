"""
Dedicated ResNet-50 Multi-Epoch Training and Sparsity Profiling Pipeline.
Trains ResNet-50 on CIFAR-10 to convergence (>91% accuracy) while tracking
gradient sparsity evolution (Hoyer index, top-10% energy concentration,
stage-wise and layer-type breakdowns) across all epochs.
"""

import argparse
from collections import defaultdict
import json
import os
import sys
import time
from typing import Dict, List, Any

# Ensure repo root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from training.models import get_cifar_resnet50, get_resnet50_layer_metadata
from training.core.sparsity import (
    compute_hoyer_sparsity,
    compute_energy_concentration,
    compute_relative_threshold_sparsity,
    compute_mask_iou,
    compute_k_energy,
)


def get_cifar10_loaders(data_dir: str, batch_size: int = 128, num_workers: int = 2):
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
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )

    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return trainloader, testloader


def profile_model_gradients(model: nn.Module) -> Dict[str, Any]:
    """Computes mathematical sparsity metrics on the current model gradients."""
    hoyer_list = []
    energy10_list = []
    rel_thresh_list = []

    stage_stats = defaultdict(lambda: {"hoyer": [], "energy10": []})
    type_stats = defaultdict(lambda: {"hoyer": [], "energy10": []})
    masks = {}

    for name, param in model.named_parameters():
        if param.grad is None or not param.requires_grad:
            continue
        if param.dim() < 2:
            continue  # Focus on 2D/4D weight tensors

        meta = get_resnet50_layer_metadata(name)
        stage = meta["stage"]
        layer_type = meta["layer_type"]

        g = param.grad
        hoyer = compute_hoyer_sparsity(g)
        energy10, top_mask = compute_energy_concentration(g, top_fraction=0.10)
        k90 = compute_k_energy(g, energy_threshold=0.90)
        rel_thresh = compute_relative_threshold_sparsity(g, factor=0.05)

        hoyer_list.append(hoyer)
        energy10_list.append(energy10)
        k90_list = stage_stats.setdefault("_k90_list", [])
        k90_list.append(k90)
        rel_thresh_list.append(rel_thresh)

        stage_stats[stage]["hoyer"].append(hoyer)
        stage_stats[stage]["energy10"].append(energy10)
        stage_stats[stage]["k90"].append(k90)

        type_stats[layer_type]["hoyer"].append(hoyer)
        type_stats[layer_type]["energy10"].append(energy10)
        type_stats[layer_type]["k90"].append(k90)

        masks[name] = top_mask

    k90_all = stage_stats.pop("_k90_list", [])
    by_stage = {
        s: {
            "hoyer": round(float(sum(v["hoyer"]) / len(v["hoyer"])), 4),
            "energy10": round(float(sum(v["energy10"]) / len(v["energy10"])), 2),
            "k90": round(float(sum(v["k90"]) / len(v["k90"])), 2),
        }
        for s, v in stage_stats.items() if v["hoyer"]
    }

    by_type = {
        t: {
            "hoyer": round(float(sum(v["hoyer"]) / len(v["hoyer"])), 4),
            "energy10": round(float(sum(v["energy10"]) / len(v["energy10"])), 2),
            "k90": round(float(sum(v["k90"]) / len(v["k90"])), 2),
        }
        for t, v in type_stats.items() if v["hoyer"]
    }

    return {
        "global_hoyer": round(float(sum(hoyer_list) / max(1, len(hoyer_list))), 4),
        "global_energy10": round(float(sum(energy10_list) / max(1, len(energy10_list))), 2),
        "global_k90": round(float(sum(k90_all) / max(1, len(k90_all))), 2),
        "global_rel_thresh": round(float(sum(rel_thresh_list) / max(1, len(rel_thresh_list))), 2),
        "by_stage": by_stage,
        "by_layer_type": by_type,
        "masks": masks,
    }


def evaluate(model: nn.Module, testloader, criterion, device: torch.device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return round(val_loss / total, 4), round(100.0 * correct / total, 2)


def main():
    parser = argparse.ArgumentParser(description="ResNet-50 Multi-Epoch Sparsity Profiling")
    parser.add_argument("--data-dir", type=str, default="./data", help="CIFAR-10 data path")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--sample-per-epoch", type=int, default=5, help="Number of gradient samples per epoch")
    parser.add_argument("--output-dir", type=str, default="training/logs", help="Output directory")
    parser.add_argument("--output-name", type=str, default="resnet50_convergence_sparsity.json", help="Log filename")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"=== Starting ResNet-50 Convergence & Sparsity Profiling ===")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size} | Initial LR: {args.lr}")

    trainloader, testloader = get_cifar10_loaders(args.data_dir, batch_size=args.batch_size)
    model = get_cifar_resnet50(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    num_batches = len(trainloader)
    sample_indices = set([int(round(i * (num_batches - 1) / (args.sample_per_epoch - 1))) for i in range(args.sample_per_epoch)])

    tracked_mask_layers = [
        "conv1.weight",
        "layer1.0.conv2.weight",
        "layer2.1.conv2.weight",
        "layer3.2.conv2.weight",
        "layer4.1.conv2.weight",
        "fc.weight",
    ]
    last_epoch_masks = {}
    epoch_mask_iou_history = defaultdict(list)

    epochs_data = []
    total_start_time = time.time()

    print("\n" + "=" * 90)
    print(f"{'Epoch':>5} | {'LR':>7} | {'Train Loss':>10} | {'Val Loss':>8} | {'Val Acc %':>9} | {'Hoyer':>7} | {'Top-10% E':>9} | {'Time (s)':>8}")
    print("=" * 90)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0

        epoch_hoyers = []
        epoch_energy10s = []
        epoch_rel_threshs = []
        epoch_by_stage_samples = defaultdict(lambda: {"hoyer": [], "energy10": []})
        epoch_by_type_samples = defaultdict(lambda: {"hoyer": [], "energy10": []})
        latest_masks = {}

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Profile gradients at specified sample batches
            if batch_idx in sample_indices:
                prof = profile_model_gradients(model)
                epoch_hoyers.append(prof["global_hoyer"])
                epoch_energy10s.append(prof["global_energy10"])
                epoch_rel_threshs.append(prof["global_rel_thresh"])

                for s, v in prof["by_stage"].items():
                    epoch_by_stage_samples[s]["hoyer"].append(v["hoyer"])
                    epoch_by_stage_samples[s]["energy10"].append(v["energy10"])

                for t, v in prof["by_layer_type"].items():
                    epoch_by_type_samples[t]["hoyer"].append(v["hoyer"])
                    epoch_by_type_samples[t]["energy10"].append(v["energy10"])

                for l_name in tracked_mask_layers:
                    if l_name in prof["masks"]:
                        latest_masks[l_name] = prof["masks"][l_name]

            optimizer.step()
            running_loss += loss.item() * targets.size(0)

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        train_loss = round(running_loss / len(trainloader.dataset), 4)
        val_loss, val_acc = evaluate(model, testloader, criterion, device)
        epoch_time = round(time.time() - epoch_start, 2)

        mean_hoyer = round(float(sum(epoch_hoyers) / len(epoch_hoyers)), 4)
        mean_energy10 = round(float(sum(epoch_energy10s) / len(epoch_energy10s)), 2)
        mean_rel_thresh = round(float(sum(epoch_rel_threshs) / len(epoch_rel_threshs)), 2)

        stage_summary = {
            s: {
                "hoyer": round(float(sum(v["hoyer"]) / len(v["hoyer"])), 4),
                "energy10": round(float(sum(v["energy10"]) / len(v["energy10"])), 2),
            }
            for s, v in epoch_by_stage_samples.items()
        }

        type_summary = {
            t: {
                "hoyer": round(float(sum(v["hoyer"]) / len(v["hoyer"])), 4),
                "energy10": round(float(sum(v["energy10"]) / len(v["energy10"])), 2),
            }
            for t, v in epoch_by_type_samples.items()
        }

        # Track epoch-to-epoch Mask IoU
        epoch_ious = {}
        if last_epoch_masks:
            for l_name in tracked_mask_layers:
                if l_name in latest_masks and l_name in last_epoch_masks:
                    iou = compute_mask_iou(last_epoch_masks[l_name], latest_masks[l_name])
                    epoch_ious[l_name] = round(iou, 4)
                    epoch_mask_iou_history[l_name].append(round(iou, 4))
        last_epoch_masks = latest_masks

        epochs_data.append({
            "epoch": epoch,
            "lr": round(current_lr, 6),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "global_hoyer": mean_hoyer,
            "global_energy10": mean_energy10,
            "global_rel_thresh": mean_rel_thresh,
            "by_stage": stage_summary,
            "by_layer_type": type_summary,
            "mask_iou_from_prev_epoch": epoch_ious,
            "epoch_time_s": epoch_time,
        })

        print(f"{epoch:>5d} | {current_lr:>7.4f} | {train_loss:>10.4f} | {val_loss:>8.4f} | {val_acc:>8.2f}% | {mean_hoyer:>7.4f} | {mean_energy10:>8.2f}% | {epoch_time:>8.2f}s")

    total_time = round(time.time() - total_start_time, 2)
    print("=" * 90)
    print(f"Training complete in {total_time:.2f}s ({total_time/60:.2f} mins)! Final Val Acc: {val_acc}%")

    # Save log
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, args.output_name)
    payload = {
        "model": "resnet50",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "initial_lr": args.lr,
        "final_val_acc": val_acc,
        "final_val_loss": val_loss,
        "total_training_time_s": total_time,
        "epochs_data": epochs_data,
        "epoch_mask_iou_history": epoch_mask_iou_history,
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Profile log written to: {out_path}")


if __name__ == "__main__":
    main()
