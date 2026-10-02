# SkipReduce: Ring AllReduce Simulation Testbed

This directory contains the PyTorch-based algorithmic execution and empirical simulation testbed for **SkipReduce**, evaluating ring collective synchronization, step-skipping schedules ($s > 0$), domain transform approximations (1D-DCT, Walsh-Hadamard), and Error Feedback (EF).

---

## Directory Structure

```text
training/
├── README.md                    # Package documentation and CLI reference
├── __init__.py                  # Top-level package export
├── models/                      # Neural network architectures
│   ├── __init__.py              # Factory exports: get_cifar_resnet50, get_cifar_vit_tiny, get_gpt_tiny
│   ├── resnet50.py              # CIFAR ResNet-50
│   ├── vit.py                   # Vision Transformer
│   └── gpt.py                   # Causal Transformer
├── core/                        # Algorithmic building blocks
│   ├── ring/                    # Ring AllReduce simulation (virtual_ring, error_feedback, ring_metrics)
│   └── transforms/              # Domain transforms for skipped steps (DCT, Walsh-Hadamard)
├── benchmarks/                  # Transform kernel microbenchmarks
│   └── benchmark_transform.py
├── experiments/                 # Runnable CLI entrypoints
│   ├── train_cifar.py           # Multi-rank virtual ring simulation with step-skipping (s >= 1)
│   └── run_sweep.py             # Automated grid sweep orchestrator across skipping steps and transforms
├── tests/                       # Unified test suite (python -m unittest discover -s training/tests -v)
│   ├── test_models.py           # Model architectures unit tests
│   ├── test_gpt.py              # Autoregressive transformer tests
│   ├── test_ring.py             # Ring AllReduce simulation & schedule tests
│   └── test_transforms.py       # DCT and Walsh-Hadamard invertibility tests
├── logs/                        # Simulation JSON logs and sweep outputs
└── figures/                     # Publication figures (fig4_pareto_accuracy_vs_payload.png)
```

---

## Quick Start

### 1. Run Unit Tests
```bash
python -m unittest discover -s training/tests -v
```

### 2. Multi-Rank Virtual Ring Simulation (CIFAR-10 on ResNet-50)
Simulate an $N=4$ virtual ring skipping $s=1$ step with Walsh-Hadamard domain transform:
```bash
python training/experiments/train_cifar.py \
    --ranks 4 \
    --skip 1 \
    --transform hadamard \
    --retention 0.15 \
    --epochs 20
```

### 3. Automated SkipReduce Parameter Sweep
Sweep across skipped steps ($s \in \{0, 1, 2\}$) and transform configurations:
```bash
python training/experiments/run_sweep.py --epochs 20
```

### 4. Transform Kernel Microbenchmark
```bash
python training/benchmarks/benchmark_transform.py
```
