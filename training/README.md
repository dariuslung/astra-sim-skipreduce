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
│   ├── sparsity/                # Mathematical sparsity metrics & tracker (Hoyer, Energy-10, S_epsilon, Mask IoU, GradientSparsityTracker)
│   └── transforms/              # Lossy compression transforms (DCT, Walsh-Hadamard)
├── benchmarks/                  # Hardware performance profiling
│   ├── benchmark_compute_savings.py # CUDA Event timing for layer skipping
│   └── benchmark_transform.py   # Transform kernel microbenchmarks
├── experiments/                 # Runnable CLI entrypoints
│   ├── train_cifar.py           # Single-GPU virtual multi-rank CIFAR training
│   ├── run_sweep.py             # Automated grid sweep orchestrator
│   ├── profile_sparsity.py      # Multi-architecture gradient sparsity profiler
│   ├── plot_sparsity.py         # Visualizes Figures 01 to 04 (EXP-01 - EXP-04)
│   ├── plot_compute_savings.py  # Visualizes compute latency breakdowns & speedup curves (EXP-05: Figures 05a, 05b)
│   ├── plot_resnet50_epoch_sparsity.py # Visualizes multi-epoch convergence & sparsity (EXP-06: Figures 06a - 06d)
│   ├── plot_intra_epoch_stability.py   # Visualizes intra-epoch invariance & stability (EXP-07: Figures 07a - 07e)
│   ├── plot_layer_recoverability.py    # Visualizes layer recoverability & sensitivity (EXP-08: Figures 08a - 08c)
│   └── plot_sparsity_vs_sensitivity.py # Visualizes sparsity vs sensitivity correlation (EXP-09: Figure 09)
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

# Generate Figures 01 to 04 (EXP-01 - EXP-04: 1-Epoch Exploration)
python training/experiments/plot_sparsity.py --json training/logs/profile_resnet50_epoch1.json
python training/experiments/plot_sparsity.py --json training/logs/profile_vit_epoch1.json
python training/experiments/plot_sparsity.py --json training/logs/profile_gpt_epoch1.json

# Multi-Epoch Convergence & Sparsity Profiling (EXP-06: 20 Epochs, ResNet-50)
python training/experiments/train_and_profile_resnet50.py --epochs 20 --lr 0.1 --batch-size 128
python training/experiments/plot_resnet50_epoch_sparsity.py

# Intra-Epoch Gradient Stability Test (EXP-07: 390-Step Horizons across 20 Epochs)
python training/experiments/test_intra_epoch_stability.py --epochs 20 --batch-size 128
python training/experiments/plot_intra_epoch_stability.py

# Protocol A: Layer Recoverability Ablation Probe (EXP-08: Hypothesis 5)
python training/experiments/probe_layer_recoverability.py --epochs 20 --conditions all
python training/experiments/plot_layer_recoverability.py

# Hypothesis 6: Sparsity vs. Sensitivity Correlation Analysis (EXP-09)
python training/experiments/plot_sparsity_vs_sensitivity.py
```

Output plots are organized cleanly into experiment directories under `training/figures/`:
* `training/figures/exp01_layer_type_sparsity/`:
  - `fig01_layer_type_sparsity_<model>.png`: EXP-01: Hypothesis 1 (Layer Type Heterogeneity).
* `training/figures/exp02_depth_vs_density/`:
  - `fig02_depth_vs_density_<model>.png`: EXP-02: Hypothesis 2 (Controlled depth vs density & $E_{10}$ energy across identical submodules — dual-panel).
* `training/figures/exp03_iteration_sparsity/`:
  - `fig03_iteration_sparsity_<model>.png`: EXP-03: Hypothesis 3 (Sparsity & $E_{10}$ energy evolution over training steps — stacked dual-panel).
* `training/figures/exp04_temporal_mask_iou/`:
  - `fig04_temporal_mask_iou_<model>.png`: EXP-04: Temporal Predictability (Top-10% Mask IoU across step lags $\Delta t$).
* `training/figures/exp05_compute_savings/`:
  - `fig05a_compute_breakdown_<model>.png`: EXP-05: Latency breakdown (Forward, Backward, Optimizer).
  - `fig05b_compute_savings_summary_<model>.png`: EXP-05: Speedup vs % parameters skipped.
* `training/figures/exp06_resnet50_convergence/`:
  - `fig06a_resnet50_convergence_vs_sparsity.png`: EXP-06: ResNet-50 Multi-Epoch Convergence vs. Global Sparsity.
  - `fig06b_resnet50_stages_across_epochs.png`: EXP-06: ResNet-50 Stage-Wise Hoyer & $E_{10}$ Evolution Across 20 Epochs (dual-panel).
  - `fig06c_resnet50_layertypes_across_epochs.png`: EXP-06: ResNet-50 Layer-Type Hoyer & $E_{10}$ Evolution Across 20 Epochs (dual-panel).
  - `fig06d_resnet50_mask_iou_across_epochs.png`: EXP-06: ResNet-50 Temporal Mask Persistence Across 20 Epochs.
* `training/figures/exp07_intra_epoch_stability/`:
  - `fig07a_intra_epoch_metric_variance.png`: EXP-07: Intra-epoch Hoyer & $E_{10}$ variance & CV% across 20 epochs ($2\times 2$ grid).
  - `fig07b_intra_epoch_layertype_cv.png`: EXP-07: Intra-epoch CV (%) for individual layer types and stages across 20 epochs.
  - `fig07c_intra_epoch_mask_decay.png`: EXP-07: Long-range Mask IoU decay from $0\% \to 100\%$ of the epoch.
  - `fig07d_intra_epoch_cosine_drift.png`: EXP-07: Intra-epoch gradient direction cosine similarity decay.
  - `fig07e_intra_epoch_layertype_trajectories.png`: EXP-07: Trajectories of individual layer types across checkpoints ($T_0 \to T_4$).
* `training/figures/exp08_layer_recoverability/`:
  - `fig08a_layer_recoverability_accuracy.png`: EXP-08: Protocol A Final Validation Accuracy & ΔAcc across layer-type skipping conditions.
  - `fig08b_layer_recoverability_convergence.png`: EXP-08: Multi-condition convergence trajectories (Val Acc and Train Loss).
  - `fig08c_layer_sensitivity_normalized.png`: EXP-08: Empirical layer sensitivity vs parameter volume (Hypothesis 5 evaluation).
* `training/figures/exp09_sparsity_vs_sensitivity/`:
  - `fig09a_sparsity_vs_sensitivity_epochs.png`: EXP-09: 6-panel multi-epoch scatter plot tracking regime transition (Hypothesis 6 evaluation).
  - `fig09b_correlation_trajectory.png`: EXP-09: Standalone 20-epoch correlation trajectory ($r, \rho$) with shaded regimes and crossover callout.
  - `fig09c_active_skipped_e10.png`: EXP-09: Active skipped vs baseline $E_{10}$ comparison (20-epoch means and trajectories).

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
