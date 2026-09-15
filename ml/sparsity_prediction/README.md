# Gradient Sparsity Pattern Predictability Module

This module provides tools for profiling, analyzing, and predicting gradient sparsity patterns across deep neural networks (supporting both Convolutional and Transformer architectures).

---

## Supported Architectures

1. **ResNet-50 (`models/resnet50.py`)**:
   * 16 bottleneck blocks across 4 depth stages (`layer1` to `layer4`).
   * Categorizes weights into: `1x1 Reduction Conv`, `3x3 Spatial Conv`, `1x1 Expansion Conv`, `Batch Normalization`, and `Classifier Head`.

2. **Vision Transformer / ViT (`models/vit.py`)**:
   * 6 Transformer Encoder blocks with 8 attention heads.
   * Categorizes weights into: `Attention: QKV`, `Attention: Out-Proj`, `FFN: Linear 1 (Up)`, `FFN: Linear 2 (Down)`, `Normalization`, and `Classifier Head`.

## Directory Structure

```
ml/sparsity_prediction/
├── README.md                    # Module documentation and usage
├── metrics.py                   # Scale-invariant Hoyer, E10, S_epsilon, Mask IoU
├── models/
│   ├── __init__.py              # Factory exports
│   ├── resnet50.py              # CIFAR-10 ResNet-50 with layer metadata
│   ├── vit.py                   # CIFAR-10 ViT with Attention/FFN layer metadata
│   └── gpt.py                   # Causal Transformer (GPT-Tiny) with NLP metadata
├── analyze_sparsity.py          # Unified CLI profiler (--model {resnet50, vit, gpt})
├── plot_sparsity.py             # Generates figures (fig1..fig4: layer type, controlled depth, iterations, IoU)
├── benchmark_compute_savings.py # GPU hardware latency profiler for selective layer skipping
├── plot_compute_savings.py      # Generates compute savings and breakdown figures
├── tests/
│   ├── test_metrics.py          # Unit tests for mathematical metrics
│   ├── test_models.py           # Verification for CV models
│   └── test_gpt.py              # Verification for NLP GPT model
└── logs/
    ├── profile_*.json           # Profiler trajectory outputs
    ├── compute_savings_*.json   # Hardware timing benchmark logs
    └── *.png                    # Generated figures
```

---

## Core Hypotheses Tested

1. **Hypothesis 1 (Layer Type)**: Different layer types (e.g. Attention vs. FFN, or 1x1 vs. 3x3 convs) produce structurally distinct gradient sparsity patterns.
2. **Hypothesis 2 (Depth)**: Deepest layers near the loss function exhibit denser gradients than shallow layers. (Evaluated controlling for layer type).
3. **Hypothesis 3 (Iterations)**: Effective gradient sparsity increases monotonically as iterations advance and loss decreases.

---

## Quick Start

### 1. Run Profiling on ResNet-50 (CV Bottleneck CNN)
```bash
python ml/sparsity_prediction/analyze_sparsity.py --model resnet50 --epochs 1
```

### 2. Run Profiling on Vision Transformer (CV Encoder ViT)
```bash
python ml/sparsity_prediction/analyze_sparsity.py --model vit --epochs 1
```

### 3. Run Profiling on Causal Transformer (NLP Autoregressive GPT)
```bash
python ml/sparsity_prediction/analyze_sparsity.py --model gpt --epochs 1
```

### 4. Generate Purpose-Driven Figures
```bash
python ml/sparsity_prediction/plot_sparsity.py --json ml/sparsity_prediction/logs/profile_<model>_epoch1.json
```
Output figures will be saved in `ml/sparsity_prediction/logs/`:
* `fig1_layer_type_sparsity_<model>.png`: Hypothesis 1 (Violin/Bar plots per layer type).
* `fig2_depth_vs_density_<model>.png`: Hypothesis 2 (Depth vs. Density controlled across identical submodules).
* `fig3_iteration_sparsity_<model>.png`: Hypothesis 3 (Step-by-step sparsity growth vs. loss).
* `fig4_temporal_mask_iou_<model>.png`: Predictability (Top-10% Mask IoU across step lags $\Delta t$).

### 5. Benchmark Hardware Compute Savings
```bash
python ml/sparsity_prediction/benchmark_compute_savings.py --model {gpt, resnet50, vit} --iterations 150 --warmup 25
python ml/sparsity_prediction/plot_compute_savings.py --json ml/sparsity_prediction/logs/compute_savings_<model>.json
```

