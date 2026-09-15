"""
Unified Profiler for Gradient Sparsity and Predictability Analysis.
Profiles CIFAR-10 training across ResNet-50 and Vision Transformer (ViT).
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from ml.sparsity_prediction.metrics import (
    compute_hoyer_sparsity,
    compute_energy_concentration,
    compute_relative_threshold_sparsity,
    compute_mask_iou
)
from ml.sparsity_prediction.models.resnet50 import get_cifar_resnet50, get_resnet50_layer_metadata
from ml.sparsity_prediction.models.vit import get_cifar_vit_tiny, get_vit_layer_metadata
from ml.sparsity_prediction.models.gpt import get_gpt_tiny, get_gpt_layer_metadata


def get_cifar10_loader(data_dir: str, batch_size: int = 128, num_workers: int = 2):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True
    )
    return trainloader


def get_tinyshakespeare_loader(data_dir: str, batch_size: int = 16, block_size: int = 128, steps_per_epoch: int = 390):
    bin_path = os.path.join(data_dir, "tinyshakespeare.pt")
    if not os.path.exists(bin_path):
        txt_path = os.path.join(data_dir, "tinyshakespeare.txt")
        if not os.path.exists(txt_path):
            import urllib.request
            os.makedirs(data_dir, exist_ok=True)
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            urllib.request.urlretrieve(url, txt_path)
        import tiktoken
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        tensor = torch.tensor(tokens, dtype=torch.long)
        torch.save(tensor, bin_path)
    else:
        tensor = torch.load(bin_path)

    # Simple generator for B x T chunks
    n_tokens = len(tensor)
    batches = []
    torch.manual_seed(42)
    for _ in range(steps_per_epoch):
        # Sample B starting points
        ix = torch.randint(0, n_tokens - block_size - 1, (batch_size,))
        x = torch.stack([tensor[i : i + block_size] for i in ix])
        y = torch.stack([tensor[i + 1 : i + block_size + 1] for i in ix])
        batches.append((x, y))
    return batches


def build_model_and_tagger(model_name: str, num_classes: int = 10):
    if model_name.lower() == "resnet50":
        model = get_cifar_resnet50(num_classes=num_classes)
        tagger = get_resnet50_layer_metadata
    elif model_name.lower() == "vit":
        model = get_cifar_vit_tiny(num_classes=num_classes)
        tagger = get_vit_layer_metadata
    elif model_name.lower() == "gpt":
        model = get_gpt_tiny(vocab_size=50257, block_size=128)
        tagger = get_gpt_layer_metadata
    else:
        raise ValueError(f"Unknown model name: {model_name}. Supported: resnet50, vit, gpt")
    return model, tagger


def profile_gradient_step(
    model: nn.Module,
    tagger,
    mask_history: Dict[str, List[torch.Tensor]],
    track_layers: List[str],
) -> Dict[str, Any]:
    """
    Computes mathematical sparsity metrics on current gradients.
    """
    layer_stats = []
    type_aggregates = defaultdict(lambda: {"hoyer": [], "energy10": [], "rel_thresh": [], "count": 0})
    stage_aggregates = defaultdict(lambda: {"hoyer": [], "energy10": [], "rel_thresh": [], "depth_index": None, "count": 0})
    
    for name, param in model.named_parameters():
        if param.grad is None or not param.requires_grad:
            continue
        
        # Focus on weight tensors (ignore 1D biases/BN scales for primary matrix sparsity analysis)
        if param.dim() < 2:
            continue

        meta = tagger(name)
        layer_type = meta["layer_type"]
        stage = meta["stage"]
        depth_idx = meta["depth_index"]
        
        g = param.grad
        hoyer = compute_hoyer_sparsity(g)
        energy10, top_mask = compute_energy_concentration(g, top_fraction=0.10)
        rel_thresh = compute_relative_threshold_sparsity(g, factor=0.05)
        
        # Track mask history for temporal IoU
        if name in track_layers:
            mask_history[name].append(top_mask)
            # Keep history buffer bounded
            if len(mask_history[name]) > 20:
                mask_history[name].pop(0)

        layer_stats.append({
            "name": name,
            "layer_type": layer_type,
            "stage": stage,
            "depth_index": depth_idx,
            "num_params": param.numel(),
            "hoyer": round(hoyer, 4),
            "density": round(1.0 - hoyer, 4),
            "energy10": round(energy10, 2),
            "rel_thresh": round(rel_thresh, 2),
        })

        type_aggregates[layer_type]["hoyer"].append(hoyer)
        type_aggregates[layer_type]["energy10"].append(energy10)
        type_aggregates[layer_type]["rel_thresh"].append(rel_thresh)
        type_aggregates[layer_type]["count"] += 1

        stage_aggregates[stage]["hoyer"].append(hoyer)
        stage_aggregates[stage]["energy10"].append(energy10)
        stage_aggregates[stage]["rel_thresh"].append(rel_thresh)
        stage_aggregates[stage]["depth_index"] = depth_idx
        stage_aggregates[stage]["count"] += 1

    # Summarize per layer type
    layer_type_summary = {}
    for ltype, vals in type_aggregates.items():
        layer_type_summary[ltype] = {
            "mean_hoyer": round(float(sum(vals["hoyer"]) / len(vals["hoyer"])), 4),
            "mean_energy10": round(float(sum(vals["energy10"]) / len(vals["energy10"])), 2),
            "mean_rel_thresh": round(float(sum(vals["rel_thresh"]) / len(vals["rel_thresh"])), 2),
            "num_tensors": vals["count"],
        }

    # Summarize per stage
    stage_summary = {}
    for stage, vals in stage_aggregates.items():
        mean_h = float(sum(vals["hoyer"]) / len(vals["hoyer"]))
        stage_summary[stage] = {
            "depth_index": vals["depth_index"],
            "mean_hoyer": round(mean_h, 4),
            "mean_density": round(1.0 - mean_h, 4),
            "mean_energy10": round(float(sum(vals["energy10"]) / len(vals["energy10"])), 2),
            "mean_rel_thresh": round(float(sum(vals["rel_thresh"]) / len(vals["rel_thresh"])), 2),
            "num_tensors": vals["count"],
        }

    # Global model averages
    all_hoyer = [s["hoyer"] for s in layer_stats]
    all_energy = [s["energy10"] for s in layer_stats]
    global_hoyer = float(sum(all_hoyer) / len(all_hoyer)) if all_hoyer else 0.0
    global_energy = float(sum(all_energy) / len(all_energy)) if all_energy else 0.0

    return {
        "global_hoyer": round(global_hoyer, 4),
        "global_density": round(1.0 - global_hoyer, 4),
        "global_energy10": round(global_energy, 2),
        "by_layer_type": layer_type_summary,
        "by_stage": stage_summary,
        "layer_stats": layer_stats,
    }


def compute_temporal_iou_stats(mask_history: Dict[str, List[torch.Tensor]]) -> Dict[str, Dict[str, float]]:
    """
    Computes average temporal IoU across steps for lag 1, 5, 10.
    """
    iou_results = {}
    for layer_name, masks in mask_history.items():
        n = len(masks)
        lags = [1, 5, 10]
        lag_ious = {}
        for lag in lags:
            if n > lag:
                ious = [compute_mask_iou(masks[i], masks[i + lag]) for i in range(n - lag)]
                lag_ious[f"lag_{lag}"] = round(float(sum(ious) / len(ious)), 4)
            else:
                lag_ious[f"lag_{lag}"] = None
        iou_results[layer_name] = lag_ious
    return iou_results


def run_profiling(
    model_name: str,
    data_dir: str = "./data",
    epochs: int = 1,
    batch_size: int = 128,
    lr: float = None,
    output_dir: str = "ml/sparsity_prediction/logs",
    sample_every: int = 1,
    device: str = "cuda",
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"[{model_name.upper()}] Initializing profiling on {device}...")

    model, tagger = build_model_and_tagger(model_name)
    model = model.to(device)

    # Pick representative layers to track for temporal mask IoU
    criterion = nn.CrossEntropyLoss()
    if model_name.lower() == "resnet50":
        default_lr = 0.05
        optimizer = optim.SGD(model.parameters(), lr=lr or default_lr, momentum=0.9, weight_decay=5e-4)
        track_layers = [
            "conv1.weight",
            "layer1.0.conv2.weight",
            "layer2.1.conv2.weight",
            "layer3.2.conv2.weight",
            "layer4.1.conv2.weight",
            "fc.weight"
        ]
        trainloader = get_cifar10_loader(data_dir=data_dir, batch_size=batch_size)
    elif model_name.lower() == "vit":
        default_lr = 5e-4
        optimizer = optim.AdamW(model.parameters(), lr=lr or default_lr, weight_decay=0.05)
        track_layers = [
            "patch_embed.proj.weight",
            "blocks.0.attn.qkv.weight",
            "blocks.0.mlp.fc1.weight",
            "blocks.3.attn.qkv.weight",
            "blocks.5.mlp.fc2.weight",
            "head.weight"
        ]
        trainloader = get_cifar10_loader(data_dir=data_dir, batch_size=batch_size)
    else:  # gpt
        default_lr = 5e-4
        batch_size = 16 if batch_size == 128 else batch_size
        optimizer = optim.AdamW(model.parameters(), lr=lr or default_lr, weight_decay=0.05)
        track_layers = [
            "transformer.wte.weight",
            "transformer.h.0.attn.c_attn.weight",
            "transformer.h.0.mlp.c_fc.weight",
            "transformer.h.3.attn.c_attn.weight",
            "transformer.h.5.mlp.c_proj.weight",
            "lm_head.weight"
        ]
        trainloader = get_tinyshakespeare_loader(data_dir=data_dir, batch_size=batch_size, block_size=128, steps_per_epoch=390)

    total_steps = len(trainloader) * epochs

    print(f"[{model_name.upper()}] Training batches per epoch: {len(trainloader)} (Total steps: {total_steps})")

    mask_history = defaultdict(list)
    step_records = []
    start_time = time.time()

    step = 0
    model.train()
    for epoch in range(epochs):
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            step += 1
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            if model_name.lower() == "gpt":
                _, loss = model(inputs, targets)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            loss.backward()

            # Profile gradients at specified sample rate
            if step % sample_every == 0 or step == 1 or step == total_steps:
                prof_data = profile_gradient_step(model, tagger, mask_history, track_layers)
                record = {
                    "step": step,
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "loss": round(float(loss.item()), 4),
                    "global_hoyer": prof_data["global_hoyer"],
                    "global_density": prof_data["global_density"],
                    "global_energy10": prof_data["global_energy10"],
                    "by_layer_type": prof_data["by_layer_type"],
                    "by_stage": prof_data["by_stage"],
                }
                # Keep full layer_stats only for first, middle, and last steps to preserve log file size
                if step in (1, total_steps // 2, total_steps):
                    record["full_layer_stats"] = prof_data["layer_stats"]

                step_records.append(record)

            optimizer.step()

            if step % 50 == 0 or step == total_steps:
                elapsed = time.time() - start_time
                print(
                    f"Step [{step}/{total_steps}] | Loss: {loss.item():.4f} | "
                    f"Mean Hoyer: {step_records[-1]['global_hoyer']:.4f} | "
                    f"Mean Energy-10: {step_records[-1]['global_energy10']:.1f}% | "
                    f"Elapsed: {elapsed:.1f}s"
                )

    total_time = time.time() - start_time
    temporal_iou = compute_temporal_iou_stats(mask_history)

    output_payload = {
        "model": model_name,
        "epochs": epochs,
        "total_steps": total_steps,
        "batch_size": batch_size,
        "device": str(device),
        "total_time_seconds": round(total_time, 2),
        "temporal_mask_iou": temporal_iou,
        "trajectory": step_records,
    }

    out_file = os.path.join(output_dir, f"profile_{model_name.lower()}_epoch{epochs}.json")
    with open(out_file, "w") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\n[{model_name.upper()}] Profiling finished in {total_time:.1f}s! Log saved to: {out_file}")
    return out_file


def main():
    parser = argparse.ArgumentParser(description="Profile gradient sparsity patterns across deep models.")
    parser.add_argument("--model", type=str, required=True, choices=["resnet50", "vit", "gpt"], help="Target model architecture")
    parser.add_argument("--data-dir", type=str, default="./data", help="Path to CIFAR-10 data")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs to profile")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (optional override)")
    parser.add_argument("--sample-every", type=int, default=1, help="Sampling frequency (in steps)")
    parser.add_argument("--output-dir", type=str, default="ml/sparsity_prediction/logs", help="Output directory for logs")
    parser.add_argument("--device", type=str, default="cuda", help="Device to execute on (cuda or cpu)")
    args = parser.parse_args()

    run_profiling(
        model_name=args.model,
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
        sample_every=args.sample_every,
        device=args.device,
    )


if __name__ == "__main__":
    main()
