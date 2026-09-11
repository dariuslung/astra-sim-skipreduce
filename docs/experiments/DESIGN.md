# Experiment Design: Domain-Transformed SkipReduce

## 1. Context & Motivation
SkipReduce is a lossy collective communication optimization that reduces AllReduce completion latency by skipping $s$ ring reduction steps during the Reduce-Scatter phase. While this reduces bandwidth demand by $\approx \frac{s}{2(N-1)}$, raw skipping discards gradient partitions from $s$ ranks, causing severe gradient bias, optimization instability, and accuracy degradation in deep networks.

This experiment investigates **Domain-Transformed SkipReduce**:
Instead of completely discarding the $s$ skipped gradient partitions, we apply a domain transform (1D Discrete Cosine Transform [DCT] or Walsh-Hadamard Transform [WHT]) to the skipped chunks, transmit only a small fraction $r \in [5\%, 25\%]$ of low-frequency / low-sequency coefficients, and reconstruct an approximate gradient on all ranks.

---

## 2. Research Hypotheses

* **Hypothesis 1 (Energy Compaction & Accuracy Recovery)**:
  Retaining $r \in [10\%, 20\%]$ of low-frequency DCT coefficients for the $s$ skipped partitions will recover $\ge 95\%$ of baseline validation accuracy on CIFAR-10 compared to raw SkipReduce ($r=0$).

* **Hypothesis 2 (Transform Comparison: DCT vs. Walsh-Hadamard)**:
  * **DCT**: Higher gradient cosine similarity and better accuracy retention at very low retention ($r \le 10\%$) due to strong spectral energy compaction.
  * **Walsh-Hadamard**: Faster computation (additions/subtractions without trigonometric floats), but slightly lower fidelity at extreme compression.

* **Hypothesis 3 (Error Feedback Convergence Guarantee)**:
  Adding a local error feedback buffer $\mathbf{e}_t = \mathbf{g}_t - \mathbf{g}_{t, \text{approx}}$ will eliminate gradient drift, allowing lower retention ratios ($r = 5\%$) to converge to the baseline.

---

## 3. Experimental Setup

### Independent Variables
1. **Virtual Ranks ($N$)**: $N = 4, 8$ (matches standard ASTRA-sim topologies `N-4`, `N-8`).
2. **Skipped Steps ($s$)**:
   - $s = 0$ (Ground truth: standard Ring AllReduce)
   - $s = 1, 2$ (SkipReduce regimes)
3. **Transform Type**:
   - `none` (Raw SkipReduce, $r=0$)
   - `dct` (1D Orthonormal Discrete Cosine Transform)
   - `hadamard` (Fast Walsh-Hadamard Transform)
4. **Retention Ratio ($r$)**: $r \in \{0.05, 0.10, 0.15, 0.25, 0.35\}$.
5. **Error Feedback (EF)**: `Enabled` vs. `Disabled`.

### Dependent / Measured Variables
1. **Validation Accuracy (Top-1 %)**: Model generalization after training.
2. **Training Loss Curve**: Convergence rate and optimization stability.
3. **Gradient Cosine Similarity**:
   $$\text{CosSim}(\mathbf{g}_{\text{sim}}, \mathbf{g}_{\text{true}}) = \frac{\mathbf{g}_{\text{sim}} \cdot \mathbf{g}_{\text{true}}}{\|\mathbf{g}_{\text{sim}}\| \|\mathbf{g}_{\text{true}}\|}$$
4. **Relative Gradient Error**:
   $$\text{RelError} = \frac{\|\mathbf{g}_{\text{sim}} - \mathbf{g}_{\text{true}}\|_2}{\|\mathbf{g}_{\text{true}}\|_2}$$

### Model & Training Hyperparameters
* **Model**: ResNet-18 (modified for CIFAR-10, standard gradient compression benchmark).
* **Dataset**: CIFAR-10 (50k train, 10k test).
* **Optimizer**: SGD with momentum ($0.9$), weight decay ($5 \times 10^{-4}$), initial learning rate $0.1$ with cosine annealing.
* **Effective Batch Size**: 128 (split across $N$ virtual ranks: $128 / N$ per micro-batch).
* **Epochs**: 20 epochs for rapid sweep, 50-100 for final convergence curves.
* **Hardware**: NVIDIA GeForce RTX 4060 Ti (single GPU virtual DDP).
* **Environment**: `/home/dalius/miniconda3/envs/ml/bin/python` (PyTorch 2.4.0).
