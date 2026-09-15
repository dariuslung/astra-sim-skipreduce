# Training, Gradient Sparsity & Algorithmic Simulation Suite

This directory contains the PyTorch-based algorithmic execution and empirical validation testbed for **SkipReduce**, evaluating ring collective synchronization, selective layer-skipping policies, gradient sparsity distributions, and GPU hardware compute savings.

---

## Directory Structure

```text
training/
├── README.md                    # Package documentation and CLI reference
├── __init__.py                  # Top-level package export
├── models/                      # Shared neural network architectures
│   ├── __init__.py              # Factory exports: get_cifar_resnet50, get_cifar_vit_tiny, get_gpt_tiny
│   ├── resnet50.py              # CIFAR ResNet-50 with structural metadata tagging

│   ├── vit.py                   # Vision Transformer with Attention/FFN metadata tagging
│   └── gpt.py                   # Causal Transformer with NLP vocabulary metadata tagging
├── core/                        # Algorithmic building blocks
│   ├── ring/                    # Ring AllReduce simulation (virtual_ring, error_feedback, ring_metrics)
│   ├── sparsity/                # Mathematical sparsity metrics (Hoyer, Energy-10, S_epsilon, Mask IoU)
│   └── transforms/              # Lossy compression transforms (DCT, Walsh-Hadamard)
├── benchmarks/                  # Hardware performance profiling
│   ├── benchmark_compute_savings.py # CUDA Event timing for layer skipping
│   └── benchmark_transform.py   # Transform kernel microbenchmarks
├── experiments/                 # Runnable CLI entrypoints
│   ├── train_cifar.py           # Single-GPU virtual multi-rank CIFAR training
│   ├── run_sweep.py             # Automated grid sweep orchestrator
│   ├── profile_sparsity.py      # Multi-architecture gradient sparsity profiler
│   ├── plot_sparsity.py         # Visualizes Figures 1 to 4 for gradient sparsity
│   └── plot_compute_savings.py  # Visualizes compute latency breakdowns & speedup curves
├── tests/                       # Unified test suite (`python -m unittest discover -s training/tests -v`)
│   ├── test_models.py           # Model architectures and metadata tagging tests
│   ├── test_gpt.py              # Autoregressive transformer test
│   ├── test_sparsity_metrics.py # Mathematical sparsity metric unit tests
│   ├── test_ring.py             # Ring AllReduce simulation tests
│   └── test_transforms.py       # DCT and Walsh-Hadamard invertibility tests
├── logs/                        # Raw machine data (JSON profiling, sweep runs, hardware benchmarks)
└── figures/                     # Publication-ready visualization plots (PNGs)
```

---

## Quick Start

### 1. Run Unified Test Suite
```bash
python -m unittest discover -s training/tests -v
```

### 2. Multi-Rank Virtual Ring Simulation (CIFAR-10 on ResNet-50)
```bash

python training/experiments/train_cifar.py \
    --ranks 4 \
    --skip 1 \
    --transform hadamard \
    --retention 0.15 \
    --epochs 20
```

### 3. Automated SkipReduce Sweep
```bash
python training/experiments/run_sweep.py --epochs 20
```

### 4. Gradient Sparsity & Predictability Profiling
```bash
# Profile 1 epoch across architectures
python training/experiments/profile_sparsity.py --model resnet50 --epochs 1
python training/experiments/profile_sparsity.py --model vit --epochs 1
python training/experiments/profile_sparsity.py --model gpt --epochs 1

# Generate Figures 1 to 4
python training/experiments/plot_sparsity.py --json training/logs/profile_resnet50_epoch1.json
python training/experiments/plot_sparsity.py --json training/logs/profile_vit_epoch1.json
python training/experiments/plot_sparsity.py --json training/logs/profile_gpt_epoch1.json
```
Output plots are saved directly to `training/figures/`:
* `fig1_layer_type_sparsity_<model>.png`: Hypothesis 1 (Layer Type Heterogeneity).
* `fig2_depth_vs_density_<model>.png`: Hypothesis 2 (Controlled depth vs density across identical submodules).
* `fig3_iteration_sparsity_<model>.png`: Hypothesis 3 (Sparsity growth over training steps).
* `fig4_temporal_mask_iou_<model>.png`: Predictability (Top-10% Mask IoU across step lags $\Delta t$).

### 5. Benchmark Hardware Compute Savings
```bash
# Benchmark layer skipping latency on GPU
python training/benchmarks/benchmark_compute_savings.py --model gpt --iterations 150 --warmup 25
python training/benchmarks/benchmark_compute_savings.py --model resnet50 --iterations 150 --warmup 25
python training/benchmarks/benchmark_compute_savings.py --model vit --iterations 150 --warmup 25

# Generate breakdown and speedup plots
python training/experiments/plot_compute_savings.py --json training/logs/compute_savings_gpt.json
python training/experiments/plot_compute_savings.py --json training/logs/compute_savings_resnet50.json
python training/experiments/plot_compute_savings.py --json training/logs/compute_savings_vit.json
```

### 6. Pareto Frontier Analysis
```bash
python analysis/plot_pareto.py
```
Output plots are saved directly to `training/figures/fig4_pareto_accuracy_vs_payload.png`.
