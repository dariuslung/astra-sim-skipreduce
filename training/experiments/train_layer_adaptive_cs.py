"""
Multi-Rank Training with Layer-Adaptive Compressive Sensing (EXP-10).
Simulates N=4 virtual ranks on ResNet-50 CIFAR-10 with heterogeneous measurement budgets
across structural layer types (spatial 3x3 convs vs. pointwise 1x1 convs).
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Any

# Ensure repo root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from training.core.ring import simulate_skipreduce_ring_adaptive
from training.core.sparsity import GradientSparsityTracker
from training.models import get_cifar_resnet50, get_resnet50_layer_metadata


SCHEDULE_PRESETS = {
    "uniform_dct_15": {
        "name": "uniform_dct_15",
        "description": "Uniform 15% DCT retention across all layers (s=2, 50% ring skip)",
        "config": {
            "conv3x3_spatial": {"s": 2, "transform": "dct", "retention": 0.15},
            "conv1x1_expand": {"s": 2, "transform": "dct", "retention": 0.15},
            "conv1x1_reduce": {"s": 2, "transform": "dct", "retention": 0.15},
            "conv1x1_downsample": {"s": 2, "transform": "dct", "retention": 0.15},
            "classifier_head": {"s": 0, "transform": "none", "retention": 1.0},
            "default": {"s": 2, "transform": "dct", "retention": 0.15},
        }
    },
    "adaptive_hadamard": {
        "name": "adaptive_hadamard",
        "description": "Layer-Adaptive Hadamard: 20% spatial 3x3, 5% pointwise 1x1, full head (s=2)",
        "config": {
            "conv3x3_spatial": {"s": 2, "transform": "hadamard", "retention": 0.20},
            "conv1x1_expand": {"s": 2, "transform": "hadamard", "retention": 0.05},
            "conv1x1_reduce": {"s": 2, "transform": "hadamard", "retention": 0.05},
            "conv1x1_downsample": {"s": 2, "transform": "hadamard", "retention": 0.05},
            "classifier_head": {"s": 0, "transform": "none", "retention": 1.0},
            "default": {"s": 2, "transform": "hadamard", "retention": 0.10},
        }
    },
    "adaptive_dct": {
        "name": "adaptive_dct",
        "description": "Layer-Adaptive DCT: 20% spatial 3x3, 5% pointwise 1x1, full head (s=2)",
        "config": {
            "conv3x3_spatial": {"s": 2, "transform": "dct", "retention": 0.20},
            "conv1x1_expand": {"s": 2, "transform": "dct", "retention": 0.05},
            "conv1x1_reduce": {"s": 2, "transform": "dct", "retention": 0.05},
            "conv1x1_downsample": {"s": 2, "transform": "dct", "retention": 0.05},
            "classifier_head": {"s": 0, "transform": "none", "retention": 1.0},
            "default": {"s": 2, "transform": "dct", "retention": 0.10},
        }
    },
    "extreme_pointwise_zero": {
        "name": "extreme_pointwise_zero",
        "description": "Extreme Pointwise Zeroing: 20% spatial 3x3, 0% pointwise 1x1 (pure ring skip), full head",
        "config": {
            "conv3x3_spatial": {"s": 2, "transform": "hadamard", "retention": 0.20},
            "conv1x1_expand": {"s": 2, "transform": "none", "retention": 0.0},
            "conv1x1_reduce": {"s": 2, "transform": "none", "retention": 0.0},
            "conv1x1_downsample": {"s": 2, "transform": "none", "retention": 0.0},
            "classifier_head": {"s": 0, "transform": "none", "retention": 1.0},
            "default": {"s": 2, "transform": "none", "retention": 0.0},
        }
    },
    "isolate_conv3x3_spatial": {
        "name": "isolate_conv3x3_spatial",
        "description": "Isolated 3x3 Spatial Convs: 50% ring skip + 15% DCT CS; all other layers full AllReduce (s=0)",
        "config": {
            "conv3x3_spatial": {"s": 2, "transform": "dct", "retention": 0.15},
            "default": {"s": 0, "transform": "none", "retention": 1.0},
        }
    },
    "isolate_conv1x1_expand": {
        "name": "isolate_conv1x1_expand",
        "description": "Isolated 1x1 Expand Convs: 50% ring skip + 15% DCT CS; all other layers full AllReduce (s=0)",
        "config": {
            "conv1x1_expand": {"s": 2, "transform": "dct", "retention": 0.15},
            "default": {"s": 0, "transform": "none", "retention": 1.0},
        }
    },
    "isolate_conv1x1_reduce": {
        "name": "isolate_conv1x1_reduce",
        "description": "Isolated 1x1 Reduce Convs: 50% ring skip + 15% DCT CS; all other layers full AllReduce (s=0)",
        "config": {
            "conv1x1_reduce": {"s": 2, "transform": "dct", "retention": 0.15},
            "default": {"s": 0, "transform": "none", "retention": 1.0},
        }
    },
    "isolate_conv1x1_downsample": {
        "name": "isolate_conv1x1_downsample",
        "description": "Isolated 1x1 Shortcut Convs: 50% ring skip + 15% DCT CS; all other layers full AllReduce (s=0)",
        "config": {
            "conv1x1_downsample": {"s": 2, "transform": "dct", "retention": 0.15},
            "default": {"s": 0, "transform": "none", "retention": 1.0},
        }
    },
}


def get_cifar10_loaders(data_dir: str, micro_batch_size: int, num_ranks: int):
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
    step_batch_size = micro_batch_size * num_ranks
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=step_batch_size, shuffle=True, num_workers=2, drop_last=True
    )
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=2
    )
    return trainloader, testloader


def evaluate(model, testloader, criterion, device):
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

    return val_loss / total, 100.0 * correct / total


def train_adaptive(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"=== EXP-10: Layer-Adaptive Compressive Sensing Multi-Rank Training ===")
    print(f"Device: {device} | Ranks: {args.ranks} | Schedule Preset: {args.schedule}")

    preset = SCHEDULE_PRESETS[args.schedule]
    layer_config = preset["config"]
    print(f"Description: {preset['description']}")

    trainloader, testloader = get_cifar10_loaders(args.data_dir, args.micro_batch_size, args.ranks)
    model = get_cifar_resnet50(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    tracker = GradientSparsityTracker(model, sample_per_epoch=args.sample_per_epoch) if args.track_sparsity else None

    run_id = f"ranks{args.ranks}_{args.schedule}_{int(time.time())}"
    history = {
        "run_id": run_id,
        "experiment": "EXP-10: Layer-Adaptive Compressive Sensing",
        "schedule": args.schedule,
        "schedule_description": preset["description"],
        "layer_config": layer_config,
        "ranks": args.ranks,
        "epochs": args.epochs,
        "lr": args.lr,
        "micro_batch_size": args.micro_batch_size,
        "step_batch_size": args.micro_batch_size * args.ranks,
        "epochs_data": []
    }

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0
        epoch_cos_sims = []
        epoch_rel_errors = []

        start_time = time.time()

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            micro_inputs = torch.chunk(inputs, args.ranks, dim=0)
            micro_targets = torch.chunk(targets, args.ranks, dim=0)

            rank_grads = {name: [] for name, p in model.named_parameters() if p.requires_grad}

            for r in range(args.ranks):
                model.zero_grad()
                outputs = model(micro_inputs[r])
                loss = criterion(outputs, micro_targets[r])
                loss.backward()

                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        rank_grads[name].append(param.grad.detach().clone())

                running_loss += loss.item() * micro_targets[r].size(0)
                total_samples += micro_targets[r].size(0)

            # Adaptive ring reduction per parameter based on layer_type
            step_cos_sims = []
            step_rel_errors = []

            for p_idx, (name, param) in enumerate(model.named_parameters()):
                if not param.requires_grad or len(rank_grads[name]) == 0:
                    continue

                meta = get_resnet50_layer_metadata(name)
                layer_type = meta["layer_type"]

                reduced_grad, stats = simulate_skipreduce_ring_adaptive(
                    grads=rank_grads[name],
                    num_ranks=args.ranks,
                    layer_config=layer_config,
                    layer_type=layer_type,
                    param_id=p_idx,
                )

                param.grad = reduced_grad
                step_cos_sims.append(stats["cos_sim"])
                step_rel_errors.append(stats["rel_l2_error"])

            if tracker and tracker.should_sample(batch_idx):
                tracker.record_step(batch_idx)

            optimizer.step()

            epoch_cos_sims.append(sum(step_cos_sims) / len(step_cos_sims))
            epoch_rel_errors.append(sum(step_rel_errors) / len(step_rel_errors))

        scheduler.step()
        epoch_time = time.time() - start_time

        train_loss = running_loss / total_samples
        val_loss, val_acc = evaluate(model, testloader, criterion, device)
        mean_cos_sim = sum(epoch_cos_sims) / len(epoch_cos_sims)
        mean_rel_error = sum(epoch_rel_errors) / len(epoch_rel_errors)

        print(
            f"Epoch {epoch+1:02d}/{args.epochs:02d} [{epoch_time:.1f}s] | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}% | CosSim: {mean_cos_sim:.4f} | RelErr: {mean_rel_error:.4f}"
        )

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 2),
            "mean_cos_sim": round(mean_cos_sim, 4),
            "mean_rel_error": round(mean_rel_error, 4),
            "epoch_time_s": round(epoch_time, 2)
        }
        history["epochs_data"].append(epoch_record)

    # Save output log
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{run_id}.json")
    with open(out_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete. Saved run log to: {out_path}")
    return history


def main():
    parser = argparse.ArgumentParser(description="Multi-Rank Layer-Adaptive Compressive Sensing Training")
    parser.add_argument("--schedule", type=str, default="adaptive_hadamard",
                        choices=list(SCHEDULE_PRESETS.keys()),
                        help="Pre-configured layer-adaptive schedule")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="training/logs")
    parser.add_argument("--sample-per-epoch", type=int, default=5)
    parser.add_argument("--track-sparsity", action="store_true", default=True)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    train_adaptive(args)


if __name__ == "__main__":
    main()
