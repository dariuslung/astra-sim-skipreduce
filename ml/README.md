# PyTorch Experiments: Domain-Transformed SkipReduce

This module contains the single-GPU virtual multi-rank simulation framework for evaluating **Domain-Transformed SkipReduce** (DCT and Walsh-Hadamard) on deep learning models.

---

## Quick Start

### 1. Environment Requirements
* Python 3.10+
* PyTorch 2.4+ with CUDA support
* torchvision

Activate your conda environment:
```bash
conda activate ml
# Or execute directly via:
/home/dalius/miniconda3/envs/ml/bin/python
```

### 2. Run Unit Tests
```bash
python -m unittest discover -s ml/tests -v
```

### 3. Run GPU Kernel Latency Benchmark
Measure forward and inverse transform latencies on your GPU across tensor sizes:
```bash
python ml/benchmarks/benchmark_transform.py --retention 0.15
```

---

## Running Training Experiments

### Train CIFAR-10 with ResNet-18
```bash
python ml/train_cifar.py \
    --ranks 4 \
    --skip 1 \
    --transform dct \
    --retention 0.15 \
    --use-ef \
    --epochs 20
```

#### Key Arguments
* `--ranks`: Number of virtual ranks $N$ (default: `4`).
* `--skip`: Number of reduction steps skipped $s$ in the reduce-scatter phase (default: `0` for full AllReduce).
* `--transform`: Domain transform applied to skipped partitions: `none`, `dct`, or `hadamard` (default: `none`).
* `--retention`: Fraction $r \in [0.0, 1.0]$ of coefficients retained (default: `0.0`).
* `--use-ef`: Enables local Error Feedback buffer across iterations.
* `--micro-batch-size`: Batch size per virtual rank (effective batch size = `micro_batch_size * ranks`).
* `--epochs`: Total training epochs.

---

## Automated Experiment Sweep

To execute the comparative matrix (Baseline AllReduce, Naive SkipReduce, DCT, DCT+EF, and Hadamard+EF) and automatically record results in `docs/experiments/LOGS.md`:

```bash
python ml/run_sweep.py --epochs 20
```

---

## Summarizing & Plotting Results

Print a summary table of all completed runs and generate convergence curves:
```bash
python analysis/plot_pareto.py --plot
```
Output plot saved to: `ml/logs/convergence_comparison.png`.
