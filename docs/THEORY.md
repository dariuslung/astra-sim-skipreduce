# Mathematical Foundations: Ring AllReduce vs. SkipReduce

## 1. Classical Ring AllReduce ($s=0$)

In a cluster of $N$ ranks (ring topology: $0 \to 1 \to \dots \to N-1 \to 0$), a gradient tensor of size $M$ elements is partitioned into $N$ equal chunks: $P_0, P_1, \dots, P_{N-1}$.

### Phase 1: Reduce-Scatter ($N - 1$ steps)
For each chunk $i \in \{0, \dots, N-1\}$, rank $i$ initiates the reduction.
At each step $k \in \{0, \dots, N-2\}$, rank $(i + k) \pmod N$ sends its accumulated chunk $P_i$ to rank $(i + k + 1) \pmod N$, which performs an in-place sum:
$$P_i^{(k+1)} = P_i^{(k)} + \mathbf{g}_{(i + k + 1) \pmod N}^{(i)}$$

After $N - 1$ steps, each rank $r$ holds the fully reduced sum of exactly one chunk:
$$P_i = \sum_{j=0}^{N-1} \mathbf{g}_j^{(i)} \quad \text{at rank } (i + N - 1) \pmod N$$

### Phase 2: All-Gather ($N - 1$ steps)
Each fully reduced chunk is circulated around the ring for $N - 1$ steps until all $N$ ranks hold the complete reduced gradient $\mathbf{g} = [P_0, P_1, \dots, P_{N-1}]$.

---

## 2. SkipReduce ($s > 0$)

In SkipReduce, the **Reduce-Scatter phase is truncated by $s$ steps**:
$$\text{Reduce-Scatter steps executed} = (N - 1) - s$$

### Chunk Reduction Accumulation
Because reduction stops $s$ steps early, each chunk $i$ only visits $(N - s)$ consecutive ranks:
$$\text{Active ranks for chunk } i: \quad \mathcal{A}_i = \{(i + k) \pmod N \mid k = 0, 1, \dots, N - 1 - s\}$$
$$\text{Skipped ranks for chunk } i: \quad \mathcal{S}_i = \{(i + N - s + k) \pmod N \mid k = 0, 1, \dots, s - 1\}$$

The All-Gather phase proceeds normally ($N - 1$ steps), broadcasting whatever partial sum was accumulated to all ranks.

### The Problem with Naive SkipReduce ($r=0$)
In naive SkipReduce, the contributions from ranks in $\mathcal{S}_i$ are discarded ($0$):
$$\tilde{P}_i = \frac{1}{N} \sum_{j \in \mathcal{A}_i} \mathbf{g}_j^{(i)}$$
Because every chunk $i$ has $s$ ranks missing, the effective gradient is scaled by $\frac{N - s}{N}$ and suffers from directional variance, breaking convergence.

---

## 3. Domain-Transformed SkipReduce

Instead of discarding $\mathbf{g}_j^{(i)}$ for $j \in \mathcal{S}_i$, we apply a domain transform $\mathcal{T}$ (e.g. 1D-DCT or Walsh-Hadamard) and retain a fraction $r \in (0, 1]$:

$$\tilde{\mathbf{g}}_j^{(i)} = \mathcal{T}^{-1}\Big(\text{Filter}\big(\mathcal{T}(\mathbf{g}_j^{(i)}), \text{ratio}=r\big)\Big)$$

The accumulated chunk becomes:
$$P_i^{\text{trans}} = \frac{1}{N} \left( \sum_{j \in \mathcal{A}_i} \mathbf{g}_j^{(i)} + \sum_{j \in \mathcal{S}_i} \tilde{\mathbf{g}}_j^{(i)} \right)$$

### Communication Volume / Payload Ratio
* **Full Ring AllReduce volume**:
  $$V_{\text{full}} = 2 \times \frac{N - 1}{N} \times M$$
* **Domain-Transformed SkipReduce volume**:
  $$V_{\text{trans}} = \left( 2 \frac{N - 1}{N} - \frac{s(1 - r)}{N} \right) \times M$$
* **Payload Ratio**:
  $$\rho = \frac{V_{\text{trans}}}{V_{\text{full}}} = 1 - \frac{s(1 - r)}{2(N - 1)}$$

**Example ($N=4, s=1, r=0.15$)**:
$$\rho = 1 - \frac{1 \times (1 - 0.15)}{2 \times 3} = 1 - \frac{0.85}{6} \approx 85.8\% \quad (\mathbf{14.2\% \text{ network payload reduction}})$$

---

## 4. Error Feedback (EF) Mechanism

For the skipped ranks $j \in \mathcal{S}_i$, the discarded high-frequency residual is:
$$\mathbf{e}_j^{(i)}[t] = \mathbf{g}_j^{(i)}[t] - \tilde{\mathbf{g}}_j^{(i)}[t]$$

At step $t + 1$, rank $j$ adds its local residual back before transformation:
$$\mathbf{p}_j^{(i)}[t+1] = \mathbf{g}_j^{(i)}[t+1] + \mathbf{e}_j^{(i)}[t]$$
$$\tilde{\mathbf{g}}_j^{(i)}[t+1] = \mathcal{T}^{-1}\Big(\text{Filter}\big(\mathcal{T}(\mathbf{p}_j^{(i)}[t+1])\big)\Big)$$
$$\mathbf{e}_j^{(i)}[t+1] = \mathbf{p}_j^{(i)}[t+1] - \tilde{\mathbf{g}}_j^{(i)}[t+1]$$

This guarantees that high-frequency gradient information is not destroyed; it accumulates across iterations until it enters the low-frequency passband or until that rank is active for that chunk.
