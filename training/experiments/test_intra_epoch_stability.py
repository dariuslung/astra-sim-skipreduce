"""
Intra-Epoch Gradient Stability Test for ResNet-50 on CIFAR-10.
Evaluates whether a gradient sample from the first few iterations of an epoch
remains valid across the rest of the epoch (at 25%, 50%, 75%, 100% checkpoints)
for all 20 epochs.
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from training.models import get_cifar_resnet50, get_resnet50_layer_metadata
from training.core.sparsity import (
    compute_hoyer_sparsity,
    compute_energy_concentration,
    compute_relative_threshold_sparsity,
    compute_mask_iou,
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
    parser = argparse.ArgumentParser(description="Test Intra-Epoch Gradient Stability on ResNet-50")
    parser.add_argument("--data-dir", type=str, default="./data", help="Path to CIFAR-10 data")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate")
    parser.add_argument("--window-size", type=int, default=5, help="Number of batches in each checkpoint window")
    parser.add_argument("--output-dir", type=str, default="training/logs", help="Output directory")
    parser.add_argument("--output-name", type=str, default="resnet50_intra_epoch_stability.json", help="Output file")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"=== Starting ResNet-50 Intra-Epoch Gradient Stability Test ===")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size} | Checkpoint Window: {args.window_size} batches")

    trainloader, testloader = get_cifar10_loaders(args.data_dir, batch_size=args.batch_size)
    model = get_cifar_resnet50(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    num_batches = len(trainloader)  # 390
    W = args.window_size

    # Define 5 checkpoints across the 390-batch epoch:
    # T0: Start (0%):    batches 0..4
    # T1: 25%:           batches 95..99
    # T2: 50%:           batches 190..194
    # T3: 75%:           batches 285..289
    # T4: 100% (End):    batches 385..389
    checkpoint_starts = [0, int(num_batches * 0.25) - W // 2, int(num_batches * 0.50) - W // 2,
                         int(num_batches * 0.75) - W // 2, num_batches - W]
    checkpoint_names = ["0% (Start)", "25%", "50% (Mid)", "75%", "100% (End)"]

    checkpoint_batch_to_idx = {}
    for ckpt_idx, start_b in enumerate(checkpoint_starts):
        for b in range(start_b, start_b + W):
            checkpoint_batch_to_idx[b] = ckpt_idx

    tracked_layers = [
        "conv1.weight",
        "layer1.0.conv2.weight",
        "layer2.1.conv2.weight",
        "layer3.2.conv2.weight",
        "layer4.1.conv2.weight",
        "fc.weight",
    ]

    all_epochs_data = []
    total_start_time = time.time()

    print("\n" + "=" * 115)
    print(f"{'Epoch':>5} | {'LR':>7} | {'TrainLoss':>9} | {'ValAcc%':>7} | {'T0 Hoyer':>8} | {'T4 Hoyer':>8} | {'ΔHoyer':>7} | {'Hoyer CV%':>9} | {'T4 IoU':>7} | {'T4 CosSim':>9} | {'Time':>6}")
    print("=" * 115)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0

        # Buffer to accumulate gradient tensors for each checkpoint
        # ckpt_idx -> layer_name -> list of gradient tensors
        checkpoint_grad_buffers = defaultdict(lambda: defaultdict(list))

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            if batch_idx in checkpoint_batch_to_idx:
                ckpt_idx = checkpoint_batch_to_idx[batch_idx]
                for name, param in model.named_parameters():
                    if param.grad is not None and param.dim() >= 2:
                        checkpoint_grad_buffers[ckpt_idx][name].append(param.grad.detach().cpu())

            optimizer.step()
            running_loss += loss.item() * targets.size(0)

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        train_loss = round(running_loss / len(trainloader.dataset), 4)
        val_loss, val_acc = evaluate(model, testloader, criterion, device)
        epoch_time = round(time.time() - epoch_start, 2)

        # Process the 5 checkpoints for this epoch
        checkpoints_data = []
        anchor_masks = {}
        anchor_grads = {}
        anchor_global_hoyer = None

        for ckpt_idx in range(5):
            ckpt_name = checkpoint_names[ckpt_idx]
            ckpt_hoyer_list = []
            ckpt_energy10_list = []
            ckpt_stage_hoyer = defaultdict(list)
            ckpt_type_hoyer = defaultdict(list)
            ckpt_masks = {}
            ckpt_grads = {}

            # Average gradient across the W batches in this window
            for name, grad_list in checkpoint_grad_buffers[ckpt_idx].items():
                avg_grad = torch.stack(grad_list, dim=0).mean(dim=0)
                meta = get_resnet50_layer_metadata(name)
                stage = meta["stage"]
                ltype = meta["layer_type"]

                hoyer = compute_hoyer_sparsity(avg_grad)
                energy10, mask = compute_energy_concentration(avg_grad, top_fraction=0.10)

                ckpt_hoyer_list.append(hoyer)
                ckpt_energy10_list.append(energy10)
                ckpt_stage_hoyer[stage].append(hoyer)
                ckpt_type_hoyer[ltype].append(hoyer)

                if name in tracked_layers:
                    ckpt_masks[name] = mask
                    ckpt_grads[name] = avg_grad.flatten()

            global_h = round(float(np.mean(ckpt_hoyer_list)), 4)
            global_e10 = round(float(np.mean(ckpt_energy10_list)), 2)

            # Checkpoint record
            ckpt_rec = {
                "checkpoint_index": ckpt_idx,
                "name": ckpt_name,
                "global_hoyer": global_h,
                "global_energy10": global_e10,
                "by_stage_hoyer": {s: round(float(np.mean(vals)), 4) for s, vals in ckpt_stage_hoyer.items()},
                "by_type_hoyer": {t: round(float(np.mean(vals)), 4) for t, vals in ckpt_type_hoyer.items()},
            }

            if ckpt_idx == 0:
                # Anchor
                anchor_masks = ckpt_masks
                anchor_grads = ckpt_grads
                anchor_global_hoyer = global_h
                ckpt_rec["mask_iou_from_anchor"] = {l: 1.0 for l in tracked_layers if l in ckpt_masks}
                ckpt_rec["cosine_sim_from_anchor"] = {l: 1.0 for l in tracked_layers if l in ckpt_grads}
                ckpt_rec["delta_hoyer_from_anchor"] = 0.0
            else:
                # Compare with Anchor (T0)
                ious = {}
                cos_sims = {}
                for l in tracked_layers:
                    if l in ckpt_masks and l in anchor_masks:
                        ious[l] = round(compute_mask_iou(anchor_masks[l], ckpt_masks[l]), 4)
                    if l in ckpt_grads and l in anchor_grads:
                        cos = F.cosine_similarity(anchor_grads[l].unsqueeze(0), ckpt_grads[l].unsqueeze(0)).item()
                        cos_sims[l] = round(float(cos), 4)

                ckpt_rec["mask_iou_from_anchor"] = ious
                ckpt_rec["cosine_sim_from_anchor"] = cos_sims
                ckpt_rec["delta_hoyer_from_anchor"] = round(global_h - anchor_global_hoyer, 4)

            checkpoints_data.append(ckpt_rec)

        # Intra-epoch summary statistics
        all_ckpt_hoyers = [c["global_hoyer"] for c in checkpoints_data]
        mean_h = float(np.mean(all_ckpt_hoyers))
        std_h = float(np.std(all_ckpt_hoyers))
        cv_h = round(float(std_h / max(1e-6, mean_h) * 100.0), 2)
        delta_h_t0_t4 = round(checkpoints_data[4]["global_hoyer"] - checkpoints_data[0]["global_hoyer"], 4)

        mean_t4_iou = round(float(np.mean(list(checkpoints_data[4]["mask_iou_from_anchor"].values()))), 4)
        mean_t4_cos = round(float(np.mean(list(checkpoints_data[4]["cosine_sim_from_anchor"].values()))), 4)

        epoch_record = {
            "epoch": epoch,
            "lr": round(current_lr, 6),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "epoch_time_s": epoch_time,
            "intra_epoch_hoyer_mean": round(mean_h, 4),
            "intra_epoch_hoyer_std": round(std_h, 4),
            "intra_epoch_hoyer_cv_pct": cv_h,
            "delta_hoyer_t0_to_t4": delta_h_t0_t4,
            "mean_t4_mask_iou": mean_t4_iou,
            "mean_t4_cosine_sim": mean_t4_cos,
            "checkpoints": checkpoints_data,
        }
        all_epochs_data.append(epoch_record)

        t0_h = checkpoints_data[0]["global_hoyer"]
        t4_h = checkpoints_data[4]["global_hoyer"]

        print(f"{epoch:>5d} | {current_lr:>7.4f} | {train_loss:>9.4f} | {val_acc:>6.2f}% | {t0_h:>8.4f} | {t4_h:>8.4f} | {delta_h_t0_t4:>+7.4f} | {cv_h:>8.2f}% | {mean_t4_iou:>7.4f} | {mean_t4_cos:>9.4f} | {epoch_time:>5.1f}s")

    total_time = round(time.time() - total_start_time, 2)
    print("=" * 115)
    print(f"Intra-Epoch Stability Test completed in {total_time:.1f}s ({total_time/60:.2f} mins)!")

    # Save log
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, args.output_name)
    payload = {
        "test": "intra_epoch_gradient_stability",
        "model": "resnet50",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "checkpoint_window_batches": args.window_size,
        "checkpoint_positions": checkpoint_names,
        "total_runtime_seconds": total_time,
        "epochs_data": all_epochs_data,
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Log successfully written to: {out_path}")


if __name__ == "__main__":
    main()
