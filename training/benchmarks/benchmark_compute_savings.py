"""
Benchmark script to quantify GPU compute time savings from skipping specific layer types.
Measures exact Forward, Backward, and Optimizer step latency on CUDA.
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from training.models.resnet50 import get_cifar_resnet50
from training.models.vit import get_cifar_vit_tiny
from training.models.gpt import get_gpt_tiny



def get_dummy_batch(model_name: str, device: torch.device):
    """Generates synthetic input/target tensors matching model architecture."""
    if model_name.lower() == "resnet50":
        # CIFAR-10 batch size 128
        inputs = torch.randn(128, 3, 32, 32, device=device)
        targets = torch.randint(0, 10, (128,), device=device)
    elif model_name.lower() == "vit":
        inputs = torch.randn(128, 3, 32, 32, device=device)
        targets = torch.randint(0, 10, (128,), device=device)
    elif model_name.lower() == "gpt":
        # B=16, T=128
        inputs = torch.randint(0, 50257, (16, 128), device=device)
        targets = torch.randint(0, 50257, (16, 128), device=device)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return inputs, targets


def define_skipping_configs(model_name: str):
    """Defines layer skipping configurations and matching rules for each model."""
    if model_name.lower() == "gpt":
        return [
            {"id": "baseline", "name": "Baseline (Full Update)", "skip_fn": lambda n: False},
            {"id": "skip_lm_head", "name": "Skip LM Head", "skip_fn": lambda n: n.startswith("lm_head.")},
            {"id": "skip_wte", "name": "Skip Token Embeddings", "skip_fn": lambda n: n.startswith("transformer.wte.")},
            {"id": "skip_vocab_all", "name": "Skip Vocabulary (Head + Embed)", "skip_fn": lambda n: n.startswith("lm_head.") or n.startswith("transformer.wte.")},
            {"id": "skip_attn_qkv", "name": "Skip All Attn QKV", "skip_fn": lambda n: "attn.c_attn" in n},
            {"id": "skip_attn_all", "name": "Skip All Attention", "skip_fn": lambda n: "attn." in n},
            {"id": "skip_ffn_all", "name": "Skip All FFN (MLP)", "skip_fn": lambda n: "mlp." in n},
            {"id": "skip_all_blocks", "name": "Skip All Transformer Blocks", "skip_fn": lambda n: n.startswith("transformer.h.")},
        ]
    elif model_name.lower() == "resnet50":
        return [
            {"id": "baseline", "name": "Baseline (Full Update)", "skip_fn": lambda n: False},
            {"id": "skip_head", "name": "Skip Classifier Head", "skip_fn": lambda n: n.startswith("fc.")},
            {"id": "skip_conv1x1_reduce", "name": "Skip 1x1 Reduce Convs", "skip_fn": lambda n: "conv1.weight" in n and not n.startswith("conv1.")},
            {"id": "skip_conv1x1_expand", "name": "Skip 1x1 Expand Convs", "skip_fn": lambda n: "conv3.weight" in n},
            {"id": "skip_conv1x1_all", "name": "Skip All 1x1 Convs", "skip_fn": lambda n: ("conv1.weight" in n and not n.startswith("conv1.")) or "conv3.weight" in n or "downsample.0" in n},
            {"id": "skip_conv3x3_spatial", "name": "Skip All 3x3 Spatial Convs", "skip_fn": lambda n: "conv2.weight" in n or n.startswith("conv1.weight")},
            {"id": "skip_stage4", "name": "Skip Deepest Stage (Stage 4)", "skip_fn": lambda n: n.startswith("layer4.")},
        ]
    elif model_name.lower() == "vit":
        return [
            {"id": "baseline", "name": "Baseline (Full Update)", "skip_fn": lambda n: False},
            {"id": "skip_head", "name": "Skip Classifier Head", "skip_fn": lambda n: n.startswith("head.")},
            {"id": "skip_attn_qkv", "name": "Skip All Attn QKV", "skip_fn": lambda n: "attn.qkv" in n},
            {"id": "skip_attn_all", "name": "Skip All Attention", "skip_fn": lambda n: "attn." in n},
            {"id": "skip_ffn_all", "name": "Skip All FFN (MLP)", "skip_fn": lambda n: "mlp." in n},
            {"id": "skip_all_blocks", "name": "Skip All Transformer Blocks", "skip_fn": lambda n: n.startswith("blocks.")},
        ]
    else:
        raise ValueError(f"Unknown model: {model_name}")


def run_benchmark(
    model_name: str,
    iterations: int = 150,
    warmup: int = 25,
    output_dir: str = "training/logs",

):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CUDA required for precise GPU hardware timing!"
    os.makedirs(output_dir, exist_ok=True)

    print(f"[{model_name.upper()}] Initializing compute savings benchmark on {torch.cuda.get_device_name(0)}...")
    print(f"[{model_name.upper()}] Warmup: {warmup} steps | Measured: {iterations} steps\n")

    # Instantiate model
    if model_name.lower() == "resnet50":
        model = get_cifar_resnet50().to(device)
    elif model_name.lower() == "vit":
        model = get_cifar_vit_tiny().to(device)
    elif model_name.lower() == "gpt":
        model = get_gpt_tiny().to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    total_model_params = sum(p.numel() for p in model.parameters())
    criterion = nn.CrossEntropyLoss()
    configs = define_skipping_configs(model_name)

    inputs, targets = get_dummy_batch(model_name, device)

    # CUDA event pairs
    start_fwd, end_fwd = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start_bwd, end_bwd = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start_opt, end_opt = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    results = []
    baseline_bwd_ms = None
    baseline_step_ms = None

    for cfg in configs:
        cfg_id = cfg["id"]
        cfg_name = cfg["name"]
        skip_fn = cfg["skip_fn"]

        # Reset all parameters to requires_grad = True
        skipped_param_count = 0
        active_params = []
        for name, param in model.named_parameters():
            if skip_fn(name):
                param.requires_grad = False
                skipped_param_count += param.numel()
            else:
                param.requires_grad = True
                active_params.append(param)

        pct_skipped = (skipped_param_count / total_model_params) * 100.0

        # Optimizer only tracks active parameters
        if model_name.lower() == "resnet50":
            optimizer = optim.SGD(active_params, lr=0.05, momentum=0.9, weight_decay=5e-4) if active_params else None
        else:
            optimizer = optim.AdamW(active_params, lr=5e-4, weight_decay=0.05) if active_params else None

        # Warmup
        model.train()
        for _ in range(warmup):
            if optimizer:
                optimizer.zero_grad()
            if model_name.lower() == "gpt":
                _, loss = model(inputs, targets)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            loss.backward()
            if optimizer:
                optimizer.step()

        torch.cuda.synchronize()

        # Measurement
        fwd_times = []
        bwd_times = []
        opt_times = []

        for _ in range(iterations):
            if optimizer:
                optimizer.zero_grad()

            # Forward
            start_fwd.record()
            if model_name.lower() == "gpt":
                _, loss = model(inputs, targets)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            end_fwd.record()

            # Backward
            start_bwd.record()
            loss.backward()
            end_bwd.record()

            # Optimizer
            start_opt.record()
            if optimizer:
                optimizer.step()
            end_opt.record()

            torch.cuda.synchronize()

            fwd_times.append(start_fwd.elapsed_time(end_fwd))
            bwd_times.append(start_bwd.elapsed_time(end_bwd))
            opt_times.append(start_opt.elapsed_time(end_opt))

        mean_fwd = float(np.mean(fwd_times))
        mean_bwd = float(np.mean(bwd_times))
        mean_opt = float(np.mean(opt_times))
        mean_step = mean_fwd + mean_bwd + mean_opt

        if cfg_id == "baseline":
            baseline_bwd_ms = mean_bwd
            baseline_step_ms = mean_step
            bwd_saved_ms = 0.0
            bwd_saved_pct = 0.0
            step_saved_ms = 0.0
            step_saved_pct = 0.0
        else:
            bwd_saved_ms = max(0.0, baseline_bwd_ms - mean_bwd)
            bwd_saved_pct = (bwd_saved_ms / baseline_bwd_ms) * 100.0
            step_saved_ms = max(0.0, baseline_step_ms - mean_step)
            step_saved_pct = (step_saved_ms / baseline_step_ms) * 100.0

        record = {
            "id": cfg_id,
            "name": cfg_name,
            "params_skipped": skipped_param_count,
            "pct_params_skipped": round(pct_skipped, 2),
            "fwd_ms": round(mean_fwd, 4),
            "fwd_std": round(float(np.std(fwd_times)), 4),
            "bwd_ms": round(mean_bwd, 4),
            "bwd_std": round(float(np.std(bwd_times)), 4),
            "opt_ms": round(mean_opt, 4),
            "opt_std": round(float(np.std(opt_times)), 4),
            "step_ms": round(mean_step, 4),
            "bwd_saved_ms": round(bwd_saved_ms, 4),
            "bwd_saved_pct": round(bwd_saved_pct, 2),
            "step_saved_ms": round(step_saved_ms, 4),
            "step_saved_pct": round(step_saved_pct, 2),
        }
        results.append(record)

        print(
            f"  {cfg_name:<34} | Step: {mean_step:6.3f} ms (Bwd: {mean_bwd:6.3f} ms) | "
            f"Bwd Saved: {bwd_saved_ms:6.3f} ms ({bwd_saved_pct:5.1f}%) | "
            f"Params Skipped: {pct_skipped:5.1f}%"
        )

    output_payload = {
        "model": model_name,
        "device": torch.cuda.get_device_name(0),
        "total_parameters": total_model_params,
        "iterations": iterations,
        "warmup": warmup,
        "baseline_bwd_ms": round(baseline_bwd_ms, 4),
        "baseline_step_ms": round(baseline_step_ms, 4),
        "configurations": results,
    }

    out_file = os.path.join(output_dir, f"compute_savings_{model_name.lower()}.json")
    with open(out_file, "w") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\n[{model_name.upper()}] Benchmark complete! Saved to {out_file}\n")
    return out_file


def main():
    parser = argparse.ArgumentParser(description="Benchmark GPU compute savings from skipping specific layer types.")
    parser.add_argument("--model", type=str, required=True, choices=["resnet50", "vit", "gpt"], help="Target model")
    parser.add_argument("--iterations", type=int, default=150, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=25, help="Number of warmup iterations")
    parser.add_argument("--output-dir", type=str, default="training/logs", help="Output directory")

    args = parser.parse_args()

    run_benchmark(
        model_name=args.model,
        iterations=args.iterations,
        warmup=args.warmup,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
