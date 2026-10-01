# PR-Reduce Technical Specification & Implementation Guide

This specification extracts the actionable mathematical formulations, system protocols, and architectural models from [PR-Reduce (ICDCS 2026)](file:///home/dalius/Projects/dalius/astra-sim/skipreduce/docs/PR_REDUCE.md) for direct implementation in Astra-Sim collective communication schedules and PyTorch distributed training simulations.

---

## 1. System Abstraction & Problem Formulation

### 1.1 Objective
Mitigate computational and network bandwidth stragglers in distributed synchronous SGD without dropping straggling workers entirely (as in Partial-Reduce). Enable each worker $i \in \{1, \dots, W\}$ to contribute a variable number of gradient coordinates $N_i \le N$ within a bounded per-round soft-timeout $T$.

### 1.2 Heterogeneity Profiles
* **Compute Heterogeneity:** Each worker computes gradients in ascending order of coordinate index:
  $$\mathbf{g}_i = [g_{i1}, g_{i2}, \dots, g_{iN}]^T \in \mathbb{R}^N$$
  Under timeout $T$, worker $i$ completes only $N_i \le N$ coordinates.
* **Network Heterogeneity:** Symmetric pairwise bandwidth matrix $\mathbf{B} \in \mathbb{R}^{W \times W}$, where $b_{ij}$ represents the available bandwidth between workers $i$ and $j$.

---

## 2. Mathematical Mechanics

### 2.1 Coordinate Masking & Partial Gradients
For worker $i$ at timeout $T$, the partial gradient $\mathbf{g}_i' \in \mathbb{R}^N$ is:

$$g_{ij}' = \begin{cases} g_{ij}, & \text{if coordinate } j \le N_i \\ 0, & \text{otherwise} \end{cases}$$

### 2.2 Contribution Weight Vector $\boldsymbol{\gamma}$ (Unequal Contribution Solution)
Let $W_j = \sum_{i=1}^W \mathbb{I}[j \le N_i]$ be the total number of workers that updated coordinate $j$.
To ensure that linear in-network summation reconstructs the true average rather than an unweighted sum, each worker scales coordinates prior to compression using weight vector $\boldsymbol{\gamma} \in \mathbb{R}^N$:

$$\gamma_j = \begin{cases} \frac{1}{W_j}, & \text{if } W_j > 0 \\ 0, & \text{if } W_j = 0 \end{cases}$$

### 2.3 Selective Compressive Sensing (CS) Encoding
Let $\mathbf{\Phi} \in \mathbb{R}^{M \times N}$ be a shared Gaussian or Rademacher measurement matrix with compression ratio $\rho = M / N < 1$.
Worker $i$ computes its compressed measurement vector $\mathbf{y}_i \in \mathbb{R}^M$:

$$\mathbf{y}_i = \mathbf{\Phi} \operatorname{diag}(\boldsymbol{\gamma}) \mathbf{g}_i'$$

When summed across the network ring, the accumulated measurement vector is:

$$\mathbf{y} = \sum_{i=1}^W \mathbf{y}_i = \mathbf{\Phi} \sum_{i=1}^W \operatorname{diag}(\boldsymbol{\gamma}) \mathbf{g}_i' = \mathbf{\Phi} \begin{bmatrix} \frac{1}{W_1} \sum_{i \in \mathcal{W}_1} g_{i1} \\ \frac{1}{W_2} \sum_{i \in \mathcal{W}_2} g_{i2} \\ \vdots \\ \frac{1}{W_N} \sum_{i \in \mathcal{W}_N} g_{iN} \end{bmatrix} = \mathbf{\Phi} \bar{\mathbf{g}}$$

Hence, $\mathbf{y}$ directly represents the compressed sensing projection of the exact coordinate-wise average $\bar{\mathbf{g}}$.

### 2.4 Bandwidth-Aware Row Normalization
When worker $i$ has constrained bandwidth, it transmits only a subset of compressed rows $\mathcal{M}_i \subseteq \{1, \dots, M\}$.
For an aggregated measurement row $m \in \{1, \dots, M\}$ received from subset $\mathcal{W}_m$:

$$\phi'_{mj} = \frac{W_j}{W_{mj}} \phi_{mj}$$

where $W_{mj}$ is the number of active workers in $\mathcal{W}_m$ that contributed to coordinate $j$.
The normalized measurement matrix $\mathbf{\Phi}'$ is used during CS decoding:

$$\mathbf{y}' = \mathbf{\Phi}' \bar{\mathbf{g}}$$

### 2.5 Signal Reconstruction
Each worker reconstructs the global average gradient $\bar{\mathbf{g}}$ locally via $L_1$-minimization or regularized pseudo-inverse:

$$\hat{\mathbf{g}} = \arg\min_{\mathbf{g}} \|\mathbf{g}\|_1 \quad \text{subject to} \quad \mathbf{y}' = \mathbf{\Phi}' \mathbf{g}$$

In high-throughput simulation regimes where gradient sparsity $> 95\%$, the Moore-Penrose pseudo-inverse with soft-thresholding provides low-latency reconstruction:

$$\hat{\mathbf{g}} \approx (\mathbf{\Phi}'^{\dagger}) \mathbf{y}' = (\mathbf{\Phi}'^T (\mathbf{\Phi}' \mathbf{\Phi}'^T)^{-1}) \mathbf{y}'$$

---

## 3. Communication Topology & Ring Scheduling

### 3.1 Max-Min Bottleneck Ring Overlay
Given bandwidth matrix $\mathbf{B}$, find ring permutation $R^* = (w_{\pi(1)}, \dots, w_{\pi(W)}, w_{\pi(1)})$ that maximizes bottleneck link capacity:

$$R^* = \arg\max_{R \in \mathcal{P}} \left( \min_{i=1,\dots,W} b_{\pi(i), \pi(i+1)} \right)$$

### 3.2 Selective Pipelined Ring AllReduce
1. **Reduce-Scatter:**
   - Gradients are divided into $4\text{ KB}$ blocks.
   - Workers pipeline compressed blocks $\mathbf{y}_i$ along $R^*$.
   - Intermediate workers accumulate incoming compressed vectors into local buffers.
   - Workers lacking bandwidth skip forwarding non-critical blocks, directly bypassing to the nearest capable downstream node.
2. **All-Gather:**
   - Aggregated blocks $\mathbf{y}'$ are circulated across $R^*$ so every rank holds the complete aggregated measurement vector.

---

## 4. Empirical Hyperparameters & Operating Points

| Parameter | Recommended Setting | Rationale |
|---|---|---|
| **CS Block Size** | $4\text{ KB}$ | Balances IP/UDP network packet header overhead (64B) against GPU matrix-vector computation time. |
| **Compression Ratio ($M/N$)** | $25\%$ to $50\%$ | Yields gradient reconstruction MSE $\le 10^{-5}$ when sparsity $> 99\%$; accuracy drop $< 1\%$. |
| **Soft-Timeout Threshold ($k$)** | $25\%$ or $50\%$ top ranks | $T = t_{\text{rank}(k)}$. Top-$k$ workers contribute full gradients; remaining $W-k$ contribute partial blocks. |
| **Measurement Matrix $\mathbf{\Phi}$** | Gaussian i.i.d. $\mathcal{N}(0, 1/M)$ or Rademacher ($\pm 1/\sqrt{M}$) | Guarantees Restricted Isometry Property (RIP) with high probability. |
| **Sampling & Recalibration** | Periodic (e.g., initial 5 batches / epoch) | Computation capacities $N_i$ remain stable over short intervals ($\text{CV} < 1\text{--}2\%$). |

---

## 5. Architectural Comparison: PR-Reduce vs. SkipReduce

| Dimension | PR-Reduce (ICDCS 2026) | SkipReduce (This Repository) |
|---|---|---|
| **Core Objective** | Heterogeneity-aware proportional aggregation under stragglers. | Acceleration of Ring AllReduce via early truncation of Reduce-Scatter. |
| **Network Truncation** | Completes all ring steps, but allows workers to send partial blocks / skip congested hops. | Truncates Reduce-Scatter by $s$ steps: executes $(N-1)-s$ steps instead of $N-1$. |
| **Sparsity Exploitation** | Spatial coordinate compressive sensing with pre-scaled weights $\operatorname{diag}(\boldsymbol{\gamma})$. | Domain transforms (1D-DCT / Walsh-Hadamard) + coordinate masks / layer-wise CS budgets. |
| **Straggler Handling** | Proportional participation: workers encode $N_i$ coordinates up to timeout $T$. | Ring path reduction: skips high-latency ranks during chunk aggregation. |
| **Weighting Formulation** | Explicit coordinate weight $\gamma_j = 1/W_j$ matching active contributor counts. | Scaling factor $\frac{N}{N-s}$ combined with domain-transform passband retention $r \in (0, 1]$. |
| **Reconstruction Engine** | $L_1$-minimization or pseudo-inverse $\mathbf{\Phi}'^{\dagger} \mathbf{y}'$. | Inverse domain transform $\mathcal{T}^{-1}$ over retained passband coordinates. |

---

## 6. PyTorch Simulation Blueprint

```python
import torch

class SelectiveCompressiveSensing:
    """
    Implements PR-Reduce Selective CS with proportional weighting and normalization.
    """
    def __init__(self, n_features: int, compression_ratio: float = 0.25, device: str = "cuda"):
        self.N = n_features
        self.M = int(n_features * compression_ratio)
        self.device = device
        
        # Shared measurement matrix Phi ~ N(0, 1/M)
        self.Phi = torch.randn(self.M, self.N, device=device) / (self.M ** 0.5)
        # Precomputed pseudo-inverse for fast reconstruction
        self.Phi_pinv = torch.linalg.pinv(self.Phi)

    def compute_weights(self, completed_coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            completed_coords: [W] tensor where each entry is N_i (number of coords worker i completed)
        Returns:
            gamma: [N] weight vector gamma_j = 1 / W_j
        """
        W = completed_coords.numel()
        # Count contributors per coordinate j
        # Assumes coordinates updated in ascending index order: 0 .. N_i-1
        coord_indices = torch.arange(self.N, device=self.device).unsqueeze(0)  # [1, N]
        active_mask = coord_indices < completed_coords.unsqueeze(1)            # [W, N]
        W_j = active_mask.sum(dim=0).float()                                   # [N]
        
        gamma = torch.zeros(self.N, device=self.device)
        valid = W_j > 0
        gamma[valid] = 1.0 / W_j[valid]
        return gamma

    def encode_local(self, g_i: torch.Tensor, N_i: int, gamma: torch.Tensor) -> torch.Tensor:
        """
        Worker i encodes its partial gradient up to coordinate N_i.
        """
        g_i_partial = torch.zeros_like(g_i)
        g_i_partial[:N_i] = g_i[:N_i]
        
        # Weighted encoding: y_i = Phi @ (diag(gamma) @ g_i_partial)
        weighted_g = gamma * g_i_partial
        y_i = torch.matmul(self.Phi, weighted_g)
        return y_i

    def reconstruct_global(self, y_aggregated: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct average gradient vector from aggregated measurements y.
        """
        # Fast reconstruction using pseudo-inverse
        g_hat = torch.matmul(self.Phi_pinv, y_aggregated)
        return g_hat
```

---

## 7. Astra-Sim Integration Notes

1. **Workload Generator:**
   - In Astra-Sim's workload layer, inject heterogeneous per-rank compute times drawn from Zipf ($s \in [0.2, 0.8]$) or truncated normal distributions.
   - Enforce deadline $T$ at the collective boundary (`AllReduce` collective call).
2. **Network Layer:**
   - Configure `AstraNetworkAPI` with the max-min bottleneck ring topology $R^*$.
   - Scale communication payload by compression ratio $\rho = M/N$ (e.g., $0.25$).
   - Account for the fixed 64-byte IP/UDP header per 4 KB chunk.
3. **Collective Implementation:**
   - Replace standard Ring AllReduce with a 3-phase flow:
     - Rank-local weighted encoding kernel.
     - Pipelined Ring Reduce-Scatter + All-Gather of compressed blocks.
     - Rank-local pseudo-inverse reconstruction kernel.
