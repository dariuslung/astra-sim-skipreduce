# PR-Reduce: Heterogeneity-Aware Proportional AllReduce for Distributed Training

**Authors:** Yi-Ching Kuo$^1$, Kate Ching-Ju Lin$^2$  
$^1$*Department of Computer Science, National Yang Ming Chiao Tung University, Hsinchu, Taiwan* (`anniekuo1205.cs12@nycu.edu.tw`)  
$^2$*Department of Computer Science and Information Engineering, National Taiwan University, Taipei, Taiwan* (`katelin@csie.ntu.edu.tw`)  
**Publication:** *2026 IEEE 46th International Conference on Distributed Computing Systems (ICDCS)*  
**DOI:** [10.1109/2575-8411.2026.00136](https://doi.org/10.1109/2575-8411.2026.00136)

---

## Abstract

Distributed Deep Learning (DDL) has become a core technique for training increasingly large and complex models. While parallelizing local training across workers improves scalability, DDL efficiency is often limited by stragglers. Recent studies have proposed partial-reduce schemes that fully or partially exclude slow workers. This, however, wastes their capacity and misses opportunities to improve model accuracy. To address this, we present **PR-Reduce**, a PRoportional Reduce communication framework that allows each worker to contribute gradient updates in proportion to its computing and communication resources. 

The key enabler of PR-Reduce is a selective compressive sensing (CS) design that allows encoding of any number of local gradients while ensuring reliable reconstruction of the true global average, even when each gradient is contributed by a different subset of workers. Simulations demonstrate that PR-Reduce better utilizes heterogeneous resources than partial-reduce, reducing gradient reconstruction error by up to 66.5% without harming model convergence. This benefit grows with greater heterogeneity, demonstrating its effectiveness in mitigating the straggler problem.

**Index Terms:** distributed deep learning, straggler mitigation, gradient sparsity, compressive sensing.

---

## I. Introduction

With the growing complexity of learning tasks, Distributed Deep Learning (DDL) has become the dominant approach for large-scale model training. In DDL, multiple workers use data parallelism to train local models on partitions of the dataset and rely on synchronized collective communication, such as AllReduce, to aggregate updates until the global model converges. Although this collaborative approach shortens overall training time, its efficiency is often limited by stragglers—workers with slower computation or weaker communication capabilities.

Recent studies have explored techniques to mitigate this straggler problem:
1. **Gradient Compression:** Exploits the sparsity of model gradients to compress updates (e.g., quantization, sparsification, low-rank factorization). While this reduces traffic volume across workers, slow workers remain a synchronization bottleneck.
2. **Threshold-based Partial Aggregation:** Excludes workers (entirely or partially) that fail to meet a predefined deadline. Although this strategy reduces round time, it inevitably underutilizes the available computational and communication capacity of straggling workers and misses opportunities to leverage their partial contributions.

To address these limitations, we evaluate two key questions:
1. *Can straggling workers contribute gradients in proportion to their available capacity without violating the desired training deadline?*
2. *Can gradient compression still be effectively applied to reduce traffic volume even when workers contribute different, arbitrary subsets of gradients for aggregation?*

To answer these questions, this paper presents **PR-Reduce**, a proportional reduce collective communication framework that enables workers to compress and contribute partial gradient updates in proportion to their capabilities. Instead of requiring each worker to compress its entire local model, PR-Reduce enables each worker to compress as many gradient elements as it can within a specified time budget and transmit the resulting compressed gradients.

### Key Technical Challenges & Solutions
* **Challenge 1 (In-network aggregation under compression):** DDL relies on in-network gradient aggregation (e.g., Ring AllReduce) where workers forward aggregated results without decoding raw local gradients.
* **Challenge 2 (Unequal contributions):** With proportional contributions, each gradient element may be contributed by a different subset of workers. Standard CS averaging fails because dividing an aggregated vector by the total worker count skews coordinates contributed by fewer workers.
* **Solution (Selective Compressive Sensing):** PR-Reduce introduces selective CS using weighted pre-encoding and measurement matrix normalization, ensuring that decoding the aggregated compressed vector directly yields the correct weighted global average.

Empirical simulations demonstrate that compared to partial-reduce, PR-Reduce's proportional contribution design reduces gradient reconstruction error by up to 66.5% and lowers training latency by up to 38.1% and 34.7% under computational and bandwidth heterogeneity, respectively.

---

## II. Related Work

* **Gradient Compression:** DGC combines gradient clipping with local accumulation; SignSGD transmits only signs; PowerSGD applies low-rank matrix factorization; HiPress overlaps GPU compression with communication; LGC leverages autoencoding. While these methods reduce communication volume, they often degrade accuracy and do not address straggling computation.
* **Sparsity-Aware Collective Communication:** Accurate Top-$k$ AllReduce identifies globally significant gradients with high synchronization cost. Approximate Top-$k$ AllReduce uses majority voting to reduce coordination. However, residual overhead remains high, and stragglers remain unaddressed.
* **Partial Reduce:** Partial-Reduce and FlexEnt exploit stochastic selection or entropy-aware redundancy to drop slow workers. Other schemes integrate straggler mitigation with differential privacy. In contrast, PR-Reduce allows stragglers to contribute partial updates proportionally rather than dropping them.

---

## III. Preliminaries and Motivation

### A. Background of AllReduce and Compressive Sensing (CS)

In standard distributed synchronous SGD with $W$ workers, each worker computes local gradient $\mathbf{g}_i$. The global gradient update is:

$$\bar{\mathbf{g}} = \frac{1}{W} \sum_{i=1}^W \mathbf{g}_i$$

Compressive Sensing (CS) reconstructs a $K$-sparse signal $\mathbf{x} \in \mathbb{R}^N$ ($K \ll N$) from an underdetermined measurement vector $\mathbf{y} \in \mathbb{R}^M$ ($M \ll N$):

$$\mathbf{y} = \mathbf{\Phi} \mathbf{x} \tag{1}$$

where $\mathbf{\Phi} \in \mathbb{R}^{M \times N}$ is a measurement matrix satisfying the Restricted Isometry Property (RIP). The sparse signal is recovered by solving an $L_1$-minimization problem:

$$\hat{\mathbf{x}} = \arg\min_{\mathbf{x}} \|\mathbf{x}\|_1 \quad \text{subject to} \quad \mathbf{y} = \mathbf{\Phi} \mathbf{x} \tag{2}$$

Model gradients in distributed training are known to exhibit extreme empirical sparsity (90% to 99.9%).

### B. Integrating CS with AllReduce

When workers compress their local gradients $\mathbf{y}_i = \mathbf{\Phi} \mathbf{g}_i$, linearity allows in-network aggregation across the ring:

$$\mathbf{y} = \frac{1}{W} \sum_{i=1}^W \mathbf{y}_i = \frac{1}{W} \sum_{i=1}^W \mathbf{\Phi} \mathbf{g}_i = \mathbf{\Phi} \left( \frac{1}{W} \sum_{i=1}^W \mathbf{g}_i \right) = \mathbf{\Phi} \bar{\mathbf{g}} \tag{3}$$

Because cosine similarity between gradient vectors of cooperating workers remains high ($>0.85$ on ResNet-50 / CIFAR-10), the aggregated gradient $\bar{\mathbf{g}}$ maintains sparsity, enabling accurate recovery via CS with mean squared error (MSE) below $10^{-5}$ when gradient sparsity exceeds 99%.

### C. Opportunities and Challenges of Proportional Reduce

Under threshold-based partial reduce (Fig. 1a), workers failing a deadline $T$ (e.g., workers $w_5$ and $w_6$) are completely discarded. However, $w_5$ and $w_6$ may have already computed 3 and 2 gradient elements out of 4. Discarding them discards valid gradient signals.

```
(a) Partial Reduce:
w1: [ g11, g12, g13, g14 ] (Completed - Included)
w2: [ g21, g22, g23, g24 ] (Completed - Included)
w3: [ g31, g32, g33, g34 ] (Completed - Included)
w4: [ g41, g42, g43, g44 ] (Completed - Included)
w5: [ g51, g52, g53, --- ] (Straggler - DISCARDED)
w6: [ g61, g62, ---, --- ] (Straggler - DISCARDED)

(b) Proportional Reduce (PR-Reduce):
w1: [ g11, g12, g13, g14 ] (All 4 elements contributed)
w2: [ g21, g22, g23, g24 ] (All 4 elements contributed)
w3: [ g31, g32, g33, g34 ] (All 4 elements contributed)
w4: [ g41, g42, g43, g44 ] (All 4 elements contributed)
w5: [ g51, g52, g53,  0  ] (3 elements proportionally contributed)
w6: [ g61, g62,  0 ,  0  ] (2 elements proportionally contributed)
```

Naive CS aggregation fails under unequal contributions:
If $w_5$ and $w_6$ encode partial vectors $\mathbf{g}_5' = [g_{51}, g_{52}, g_{53}, 0]^T$ and $\mathbf{g}_6' = [g_{61}, g_{62}, 0, 0]^T$, their sum is:

$$\mathbf{y} = \mathbf{\Phi} \begin{bmatrix} g_{51} + g_{61} \\ g_{52} + g_{62} \\ g_{53} \\ 0 \end{bmatrix} \tag{5}$$

Dividing the decoded sum by $W=2$ yields:

$$\hat{\mathbf{g}} = \begin{bmatrix} (g_{51} + g_{61})/2 \\ (g_{52} + g_{62})/2 \\ g_{53}/2 \\ 0 \end{bmatrix} \neq \bar{\mathbf{g}} \tag{6}$$

Element $g_3$ is distorted because it was divided by 2 despite only having 1 contributor.

---

## IV. Design of PR-Reduce

### A. Overview

PR-Reduce employs **Selective Compressive Sensing** with weighted pre-encoding and measurement matrix normalization, mitigating both computational and network bandwidth heterogeneity.

### B. Selective Compressive Sensing

Gradients are partitioned into blocks of $N \times 1$ vectors: $\mathbf{g}_i = [g_{i1}, g_{i2}, \dots, g_{iN}]^T \in \mathbb{R}^{N \times 1}$. Within timeout $T$, worker $i$ computes a subset of gradient elements:

$$g_{ij}' = \begin{cases} g_{ij}, & \text{if } g_{ij} \text{ is updated before } T \\ 0, & \text{otherwise} \end{cases} \tag{4}$$

### C. Reconstructing Proportional Contributions

#### 1. Contribution Weight Vector $\gamma$
Let $W_j$ denote the number of workers that contribute to gradient element $j$ ($W_j \le W$). We define the contribution weight vector $\boldsymbol{\gamma} \in \mathbb{R}^{N \times 1}$:

$$\gamma_j = \frac{1}{W_j} \quad \text{for } W_j > 0, \quad \gamma_j = 0 \quad \text{if } W_j = 0$$

Each worker pre-scales its partial gradient before CS encoding:

$$\mathbf{y}_i = \mathbf{\Phi} \operatorname{diag}(\boldsymbol{\gamma}) \mathbf{g}_i' \tag{7}$$

Aggregating these encoded vectors across all workers yields:

$$\mathbf{y} = \sum_{i=1}^W \mathbf{y}_i = \sum_{i=1}^W \mathbf{\Phi} \operatorname{diag}(\boldsymbol{\gamma}) \mathbf{g}_i' = \mathbf{\Phi} \begin{bmatrix} \sum_i \gamma_1 g_{i1}' \\ \sum_i \gamma_2 g_{i2}' \\ \vdots \\ \sum_i \gamma_N g_{iN}' \end{bmatrix} = \mathbf{\Phi} \bar{\mathbf{g}} \tag{8}$$

Because $\sum_i \gamma_j g_{ij}' = \frac{1}{W_j} \sum_{i \in \mathcal{W}_j} g_{ij} = \bar{g}_j$, decoding $\mathbf{y}$ directly reconstructs the true global average $\bar{\mathbf{g}}$.

#### 2. Bandwidth Heterogeneity and Measurement Matrix Normalization
When workers with low bandwidth can only forward a subset of compressed elements in $\mathbf{y}_i$, the aggregated vector $\mathbf{y}'$ has different numbers of contributors across rows $m \in \{1, \dots, M\}$.

Let $\mathcal{W}_m$ denote the subset of workers contributing to measurement row $m$, and $W_{mj}$ denote the number of workers in $\mathcal{W}_m$ who contributed to gradient coordinate $j$. To guarantee correct recovery, PR-Reduce normalizes each element $\phi_{mj}$ of the measurement matrix $\mathbf{\Phi}$:

$$\phi'_{mj} = \frac{W_j}{W_{mj}} \phi_{mj}$$

The normalized matrix $\mathbf{\Phi}'$ is used during CS decoding:

$$\mathbf{y}' = \mathbf{\Phi}' \bar{\mathbf{g}}$$

---

### D. System Architecture & Operation

```
+-------------------------------------------------------------------------------+
|                             Phase 1: Initialization                           |
|  - Profile historical link bandwidths Bij -> Build max-min bottleneck ring R* |
|  - Profile computation capacity Ni -> Broadcast & compute weights gamma_j     |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|               Phase 2: Proportional Compression & Ring Reduction              |
|  - Compute local updates up to soft-timeout T (first Ni coordinates)          |
|  - Weighted CS Encoding: y_i = Phi * diag(gamma) * g_i'                       |
|  - Selective pipelined Ring AllReduce (forwarding y_i proportional to B)      |
+---------------------------------------+---------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|               Phase 3: Reconstruction & Model Update                          |
|  - Construct normalized measurement matrix Phi'                               |
|  - Decode average gradient g_hat via L1-minimization or pseudo-inverse        |
|  - Local model parameter update: theta_{t+1} = theta_t - eta * g_hat          |
+-------------------------------------------------------------------------------+
```

#### Ring Topology Construction
To optimize communication under bandwidth heterogeneity, PR-Reduce finds the optimal ring overlay $R^*$ maximizing the bottleneck link capacity:

$$b_{\min}(R) = \min_{i=1,\dots,W} b_{i, i+1} \tag{10}$$

$$R^* = \arg\max_{R \in \mathcal{P}} b_{\min}(R) \tag{11}$$

where $\mathcal{P}$ is the set of all valid ring permutations.

---

### Algorithm 1: Operation of PR-Reduce

```text
Input: Worker set W, bandwidth matrix B, timeout T, CS measurement matrix Phi
Output: Synchronized model parameters theta

// Phase 1: Ring construction and initialization
1: Find optimal ring R* among feasible paths P: R* = argmax_{R in P} b_min(R)
2: Monitor worker computation capacities Ni and learn weight vector gamma = [1/W_1, ..., 1/W_N]^T
3: Learn normalized measurement matrix Phi' for decoding

// Phase 2: Proportional compression and distribution
4: Train local models in parallel until timeout T
5: foreach worker i in W at time T do
6:     Compress current partial gradient g_i' via weighted encoding:
           y_i <- Phi * diag(gamma) * g_i'
7:     Combine received aggregated chunk with local compressed chunk y_i
8:     Forward y to nearest capable worker i' along R*

// Phase 3: Reconstruction from proportional contributions
9: foreach worker i in W after aggregation do
10:    Decode global average g_hat from partial aggregation y' based on:
           g_hat = argmin_g ||g||_1  subject to  y' = Phi' * g
11:    Update local model parameters: theta <- theta - eta * g_hat
12: Repeat Phases 1-3 until convergence
```

---

## V. Performance Evaluation

### Experimental Setup
* **Cluster:** 8 workers deployed across a 2-pod fat-tree network topology with 2 edge switches (4 servers per switch).
* **Hardware:** Intel Core i9-14900K / i7-14700 CPUs, NVIDIA RTX 4080 / RTX 4060 Ti GPUs (16 GB VRAM), 64 GB RAM.
* **Workloads:** ResNet-18 and ResNet-50 trained on CIFAR-10 (SGD, learning rate 0.01, momentum 0.9, cross-entropy loss).
* **Network Parameters:** Mean link bandwidth $\mu = 300\text{ Mbps}$, background traffic variation $\pm 30\% \mu$ (up to $\pm 60\%$).
* **CS Parameters:** Block size $4\text{ KB}$ (balanced between packet header overhead and GPU matrix reconstruction latency), default compression ratio $M/N = 25\%$ or $50\%$.
* **Timeout Threshold:** Soft-timeout $T$ set to the completion time of top-$k$ workers ($k=25\%$ or $50\%$).
* **Baselines Evaluated:**
  1. *All-Reduce:* Standard synchronous ring AllReduce (waits for all workers).
  2. *P-Reduce:* Partial-Reduce (completely drops stragglers past timeout).
  3. *PR-Reduce w/o CS:* Proportional reduce without compressive sensing.
  4. *PR-Reduce:* Full design with selective CS.

---

### Key Empirical Findings

| Metric / Dimension | Observation & Result | Comparative Baseline |
|---|---|---|
| **CS Block Size Trade-off** | $4\text{ KB}$ optimal: $1\text{ KB}$ has excessive packet header overhead (IP/UDP 64B headers); $\ge 8\text{ KB}$ increases GPU reconstruction compute time. | Evaluated across $1, 4, 8, 16\text{ KB}$. |
| **Reconstruction Error (MSE)** | PR-Reduce achieves MSE $\le 10^{-5}$ under $>99\%$ sparsity; reduces gradient MSE by **66.5%** ($k=25\%$) and **10.8%** ($k=50\%$). | Compared to P-Reduce which discards stragglers. |
| **Compute Heterogeneity** | Reduces total training latency by up to **38.1%** under heavy compute skewness (Zipf $s \in [0.4, 0.6]$, Uniform $\pm 100\%$). | Outperforms full AllReduce and P-Reduce. |
| **Bandwidth Heterogeneity** | Reduces training round latency by up to **34.7%** vs. P-Reduce and **59.4%** vs. AllReduce under link variance $\sigma_b = 120\text{ Mbps}$. | Bottleneck-optimal ring $R^*$ avoids congested paths. |
| **Model Convergence** | ResNet-18 and ResNet-50 on CIFAR-10 reach validation accuracy within **$<1\%$** of full AllReduce while maintaining matching loss convergence curves. | Confirms stability of weighted proportional aggregation. |

---

## VI. Conclusion

PR-Reduce provides a heterogeneity-aware collective communication framework that enables distributed deep learning workers to contribute partial gradient updates in proportion to their computing and communication capabilities. By employing selective compressive sensing with contribution-weighted encoding and measurement matrix normalization, PR-Reduce guarantees accurate reconstruction of the true global gradient average while compressing network traffic. Empirical evaluations demonstrate up to 66.5% lower gradient reconstruction error than partial-reduce methods and up to 38.1% lower training latency without compromising model convergence.

---

## References

1. S. Wang, D. Li, and J. Geng, "Geryon: Accelerating distributed CNN training by network-level flow scheduling," in *IEEE INFOCOM*, 2020.
2. Z. Chen, X. Liu, M. Li, Y. Hu, H. Mei, H. Xing, H. Wang, W. Shi, S. Liu, and Y. Xu, "Rina: Enhancing ring-allreduce with in-network aggregation in distributed model training," in *IEEE ICNP*, 2024.
3. Y. Bao, Y. Peng, Y. Chen, and C. Wu, "Preemptive all-reduce scheduling for expediting distributed DNN training," in *IEEE INFOCOM*, 2020.
4. J. Xu, S.-L. Huang, L. Song, and T. Lan, "Live gradient compensation for evading stragglers in distributed learning," in *IEEE INFOCOM*, 2021.
5. E. Warraich, O. Shabtai, K. Manaa, S. Vargaftik, Y. Piasetzky, M. Kadosh, L. Suresh, and M. Shahbaz, "OptiReduce: Resilient and Tail-Optimal AllReduce for distributed deep learning in the cloud," in *USENIX NSDI*, 2025.
6. J. Fei, C.-Y. Ho, A. N. Sahu, M. Canini, and A. Sapio, "Efficient sparse collective communication and its application to accelerate distributed deep learning," in *ACM SIGCOMM*, 2021.
7. H. Xu, K. Kostopoulou, A. Dutta, X. Li, A. Ntoulas, and P. Kalnis, "DeepReduce: A sparse-tensor communication framework for federated deep learning," in *NeurIPS*, 2021.
8. M. Li, R. B. Basat, S. Vargaftik, C. Lao, K. Xu, M. Mitzenmacher, and M. Yu, "THC: accelerating distributed deep learning using tensor homomorphic compression," in *USENIX NSDI*, 2024.
9. Y. Lin, S. Han, H. Mao, Y. Wang, and B. Dally, "Deep gradient compression: Reducing the communication bandwidth for distributed training," in *ICLR*, 2018.
10. J. Bernstein, Y.-X. Wang, K. Azizzadenesheli, and A. Anandkumar, "SignSGD: Compressed optimisation for non-convex problems," in *ICML*, 2018.
11. T. Vogels, S. P. Karimireddy, and M. Jaggi, "PowerSGD: Practical low-rank gradient compression for distributed optimization," in *NeurIPS*, 2019.
12. J. Lin, Z. Jiang, Z. Song, S. Zhao, M. Yu, Z. Wang, C. Wang, Z. Shi, X. Shi, W. Jia, Z. Liu, S. Wang, H. Lin, X. Liu, A. Panda, and J. Li, "Understanding stragglers in large model training using what-if analysis," in *USENIX OSDI*, 2025.
13. X. Miao, X. Nie, Y. Shao, Z. Yang, J. Jiang, L. Ma, and B. Cui, "Heterogeneity-aware distributed machine learning training via partial reduce," in *ACM SIGMOD*, 2021.
14. J. Salamy, A. Sharma, M. Ghobadi, and M. Médard, "FlexEnt: Entropy coding to curb stragglers in large-scale distributed machine learning," in *AI Systems (ACM SOSP Workshop)*, 2019.
15. D. Donoho, "Compressed sensing," *IEEE Transactions on Information Theory*, vol. 52, no. 4, pp. 1289–1306, 2006.
16. E. Candès, J. Romberg, and T. Tao, "Robust uncertainty principles: exact signal reconstruction from highly incomplete frequency information," *IEEE Transactions on Information Theory*, vol. 52, no. 2, pp. 489–509, 2006.
17. Y. Bai, C. Li, Q. Zhou, J. Yi, P. Gong, F. Yan, R. Chen, and Y. Xu, "Gradient compression supercharged high-performance data parallel DNN training," in *ACM SOSP*, 2021.
18. L. Abrahamyan, Y. Chen, G. Bekoulis, and N. Deligiannis, "Learned gradient compression for distributed deep learning," *IEEE TNNLS*, vol. 33, no. 12, pp. 7330–7344, 2021.
19. S. Shi, Q. Wang, K. Zhao, Z. Tang, Y. Wang, X. Huang, and X. Chu, "A distributed synchronous SGD algorithm with global Top-k sparsification for low bandwidth networks," in *IEEE ICDCS*, 2019.
20. S. Li and T. Hoefler, "Near-optimal sparse allreduce for distributed deep learning," in *ACM PPoPP*, 2022.
21. C.-Y. Chen, J. Ni, S. Lu, X. Cui, P.-Y. Chen, X. Sun, N. Wang, S. Venkataramani, V. V. Srinivasan, W. Zhang, and K. Gopalakrishnan, "ScaleCom: Scalable sparsified gradient compression for communication-efficient distributed training," in *NeurIPS*, 2020.
22. Y. Yakimenka, C.-W. Weng, H.-Y. Lin, E. Rosnes, and J. Kliewer, "Straggler-resilient differentially-private decentralized learning," in *IEEE ITW*, 2022.
23. B. Ghazi, Y. Huang, P. Kamath, R. Kumar, P. Manurangsi, A. Sinha, and C. Zhang, "Sparsity-preserving differentially private training of large embedding models," in *NeurIPS*, 2023.
24. Y. Tang, V. Ramanathan, J. Zhang, and N. Li, "Communication-efficient distributed SGD with compressed sensing," *IEEE Control Systems Letters*, vol. 6, pp. 2054–2059, 2022.
