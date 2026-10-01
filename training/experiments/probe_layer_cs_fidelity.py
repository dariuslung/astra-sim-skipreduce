"""
Fast Rate-Distortion Profiler for Layer-Adaptive Compressive Sensing (EXP-10).
Evaluates gradient reconstruction fidelity across retention ratios r in [0.00, 0.50]
under 50% ring skipping (N=4 virtual ranks, s=2) across all structural layer types in ResNet-50.
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
import torchvision
import torchvision.transforms as transforms

from training.core.ring import simulate_skipreduce_ring
from training.core.ring.ring_metrics import cosine_similarity, relative_l2_error
from training.models import get_cifar_resnet50, get_resnet50_layer_metadata


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_cifar10_loaders(data_dir: str, micro_batch_size: int = 32, num_ranks: int = 4):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    step_batch_size = micro_batch_size * num_ranks
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=step_batch_size, shuffle=True, num_workers=2, drop_last=True
    )
    return trainloader


def run_profiler(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"=== EXP-10: Layer Compressive Sensing Rate-Distortion Profiler ===")
    print(f"Device: {device} | Ranks: {args.ranks} | Skipped Steps: {args.skip} (50% ring skip)")
    print(f"Transforms: {args.transforms} | Batches: {args.batches}")

    trainloader = get_cifar10_loaders(args.data_dir, args.micro_batch_size, args.ranks)
    model = get_cifar_resnet50(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()

    retention_ratios = args.retention_ratios

    # Data structure: metrics[transform][retention_ratio][layer_type] = {"cos_sim": [], "rel_l2": []}
    metrics = {
        t: {
            r: defaultdict(lambda: {"cos_sim": [], "rel_l2": []})
            for r in retention_ratios
        }
        for t in args.transforms
    }

    start_time = time.time()
    batch_count = 0

    print(f"\nProfiling across {len(retention_ratios)} retention ratios: {retention_ratios}...")

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if batch_count >= args.batches:
            break

        inputs, targets = inputs.to(device), targets.to(device)
        micro_inputs = torch.chunk(inputs, args.ranks, dim=0)
        micro_targets = torch.chunk(targets, args.ranks, dim=0)

        rank_grads = {name: [] for name, p in model.named_parameters() if p.requires_grad}

        # Compute backward gradients for each virtual rank
        for r in range(args.ranks):
            model.zero_grad()
            outputs = model(micro_inputs[r])
            loss = criterion(outputs, micro_targets[r])
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    rank_grads[name].append(param.grad.detach().clone())

        # For each parameter and each retention ratio, evaluate reconstruction
        for name, param in model.named_parameters():
            if not param.requires_grad or len(rank_grads[name]) == 0:
                continue

            meta = get_resnet50_layer_metadata(name)
            layer_type = meta["layer_type"]
            grads = rank_grads[name]

            # Ground truth all-reduce average
            g_true = torch.stack(grads, dim=0).mean(dim=0)

            for trans in args.transforms:
                for r_val in retention_ratios:
                    if r_val == 0.0:
                        # Pure zeroing ring skip
                        reduced_grad, stats = simulate_skipreduce_ring(
                            grads=grads,
                            num_ranks=args.ranks,
                            s=args.skip,
                            transform_type="none",
                            retention_ratio=0.0
                        )
                    else:
                        reduced_grad, stats = simulate_skipreduce_ring(
                            grads=grads,
                            num_ranks=args.ranks,
                            s=args.skip,
                            transform_type=trans,
                            retention_ratio=r_val
                        )

                    metrics[trans][r_val][layer_type]["cos_sim"].append(float(stats["cos_sim"]))
                    metrics[trans][r_val][layer_type]["rel_l2"].append(float(stats["rel_l2_error"]))

        batch_count += 1
        if batch_count % 5 == 0 or batch_count == args.batches:
            elapsed = time.time() - start_time
            print(f"  Processed {batch_count}/{args.batches} batches ({elapsed:.1f}s)")

    # Aggregate results into structured summary
    results = {
        "experiment": "EXP-10: Layer-Adaptive Compressive Sensing Rate-Distortion Profiling",
        "num_ranks": args.ranks,
        "skipped_steps": args.skip,
        "batches_profiled": batch_count,
        "retention_ratios": retention_ratios,
        "transforms": args.transforms,
        "by_transform": {}
    }

    layer_types = ["conv3x3_spatial", "conv1x1_expand", "conv1x1_reduce", "conv1x1_downsample", "classifier_head"]

    for trans in args.transforms:
        results["by_transform"][trans] = {}
        for r_val in retention_ratios:
            # Theoretical network payload ratio vs full allreduce
            full_steps = 2 * (args.ranks - 1)
            saved_steps = args.skip * (1.0 - (r_val if trans != "none" or r_val > 0.0 else 0.0))
            payload_ratio = round((full_steps - saved_steps) / float(full_steps), 4)

            results["by_transform"][trans][str(r_val)] = {
                "retention_ratio": r_val,
                "payload_ratio": payload_ratio,
                "payload_reduction_pct": round((1.0 - payload_ratio) * 100.0, 2),
                "by_layer_type": {}
            }

            for lt in layer_types:
                cos_vals = metrics[trans][r_val][lt]["cos_sim"]
                rel_vals = metrics[trans][r_val][lt]["rel_l2"]
                if cos_vals:
                    results["by_transform"][trans][str(r_val)]["by_layer_type"][lt] = {
                        "mean_cos_sim": round(float(np.mean(cos_vals)), 4),
                        "std_cos_sim": round(float(np.std(cos_vals)), 4),
                        "mean_rel_l2_error": round(float(np.mean(rel_vals)), 4),
                        "std_rel_l2_error": round(float(np.std(rel_vals)), 4),
                    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved structured results to: {args.output_json}")

    # Print summary table for DCT
    primary_trans = "dct" if "dct" in args.transforms else args.transforms[0]
    print(f"\n=== Summary: Cosine Similarity with True Average ({primary_trans.upper()}) ===")
    header = f"{'r (Ret)':8s} | {'Payload':8s} | {'3x3 Spatial':12s} | {'1x1 Expand':12s} | {'1x1 Reduce':12s} | {'1x1 Shortcut':12s} | {'Classifier':12s}"
    print(header)
    print("-" * len(header))
    for r_val in retention_ratios:
        d = results["by_transform"][primary_trans][str(r_val)]
        pay = f"{d['payload_ratio']*100:.1f}%"
        lt_d = d["by_layer_type"]
        s3 = f"{lt_d.get('conv3x3_spatial', {}).get('mean_cos_sim', 0):.4f}"
        e1 = f"{lt_d.get('conv1x1_expand', {}).get('mean_cos_sim', 0):.4f}"
        r1 = f"{lt_d.get('conv1x1_reduce', {}).get('mean_cos_sim', 0):.4f}"
        d1 = f"{lt_d.get('conv1x1_downsample', {}).get('mean_cos_sim', 0):.4f}"
        fc = f"{lt_d.get('classifier_head', {}).get('mean_cos_sim', 0):.4f}"
        print(f"{r_val:8.2f} | {pay:8s} | {s3:12s} | {e1:12s} | {r1:12s} | {d1:12s} | {fc:12s}")

    return results


def main():
    parser = argparse.ArgumentParser(description="EXP-10: Layer Compressive Sensing Rate-Distortion Profiler")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--skip", type=int, default=2)
    parser.add_argument("--micro-batch-size", type=int, default=32)
    parser.add_argument("--batches", type=int, default=25, help="Number of batches to profile")
    parser.add_argument("--transforms", nargs="+", default=["dct", "hadamard"], help="Transforms to evaluate")
    parser.add_argument("--retention-ratios", nargs="+", type=float,
                        default=[0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output-json", type=str, default="training/logs/layer_cs_rate_distortion.json")
    args = parser.parse_args()

    run_profiler(args)


if __name__ == "__main__":
    main()
