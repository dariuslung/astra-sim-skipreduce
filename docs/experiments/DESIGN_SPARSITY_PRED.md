# Experiment Design: Gradient Sparsity Pattern Predictability Across Layer Types and Depths

## 1. Research Motivation & Core Premise
In distributed training systems, lossy gradient compression (such as Top-$k$ sparsification) reduces communication volume by communicating only coordinates with large gradient magnitudes. However, standard Top-$k$ requires dynamic GPU sorting and coordinate index synchronization on **every single training step**, creating severe compute and metadata overhead.

This research investigates whether **gradient sparsity patterns can be sampled and predicted across training iterations**:
1. Do different layer types (e.g., Attention vs. Feed-Forward Networks, or $1\times 1$ vs. $3\times 3$ convolutions) exhibit structurally distinct sparsity patterns?
2. Are deep layers near the output loss function inherently denser than shallow layers?
3. Does gradient sparsity increase as training iterations progress and the model stabilizes?

By profiling across both a **Convolutional architecture (ResNet-50)** and an **Attention-based architecture (Vision Transformer / ViT)**, we determine whether sparsity predictability is an inherent property of backpropagation dynamics across deep networks.

---

## 2. Research Hypotheses

* **Hypothesis 1 (Layer-Type Heterogeneity)**:
  Different layer types exhibit statistically distinct gradient sparsity profiles:
  * In Transformers, **FFN layers** (activated by non-linearities like GELU/ReLU) exhibit higher gradient sparsity than **Attention layers** (which compute dense token-to-token softmax probability mixtures).
  * In ResNet-50, **$1\times 1$ pointwise expansion/reduction convolutions** exhibit different sparsity distributions than **$3\times 3$ spatial convolutions**.

* **Hypothesis 2 (Depth-Dependent Gradient Density)**:
  Layers closest to the final loss function receive un-attenuated, direct classification error signals:
  $$\frac{\partial \mathcal{L}}{\partial \mathbf{z}_{\text{out}}} = \mathbf{p} - \mathbf{y}$$
  As this error propagates backward through residual connections and normalization layers, it diffuses. Consequently, **the deepest stages and output projection heads exhibit significantly denser gradients** (lower sparsity) than shallow feature-extraction layers.

* **Hypothesis 3 (Temporal Sparsity Evolution Across Iterations)**:
  At early iterations ($t \approx 1$), loss is high, gradients are turbulent, and weights across the entire network receive dense updates. As iterations advance and loss decreases, gradients stabilize: an increasing majority of coordinates receive near-zero updates, causing **effective gradient sparsity to monotonically increase with iteration count**.

---

## 3. Mathematical Sparsity Metrics

To rigorously evaluate sparsity without arbitrary absolute thresholds, we employ three complementary metrics:

### Metric A: Scale-Invariant Hoyer Sparsity Index
For any gradient vector $\mathbf{g} \in \mathbb{R}^d$:
$$\text{Hoyer}(\mathbf{g}) = \frac{\sqrt{d} - \frac{\|\mathbf{g}\|_1}{\|\mathbf{g}\|_2}}{\sqrt{d} - 1} \in [0, 1]$$
* $\text{Hoyer} \approx 0.0$: Completely dense (all coordinates carry equal magnitude).
* $\text{Hoyer} \approx 1.0$: Maximally sparse (energy concentrated in a tiny fraction of elements).
* **Gradient Density** is defined as $1 - \text{Hoyer}(\mathbf{g})$.

### Metric B: Top-10% Energy Concentration ($E_{10}$)
Measures the fraction of total $L_2^2$ gradient energy captured by the top $10\%$ largest coordinates:
$$E_{10}(\mathbf{g}) = \frac{\|\mathbf{g} \odot \mathcal{M}_{\text{top10\%}}\|_2^2}{\|\mathbf{g}\|_2^2} \times 100\%$$
* $E_{10} \approx 10\%$: Uniformly dense (random noise).
* $E_{10} \ge 85\%$: Extremely sparse (top $10\%$ coordinates capture virtually all gradient energy).

### Metric C: Relative Threshold Sparsity ($S_{\epsilon}$)
Fraction of gradient coordinates whose magnitude is negligible relative to the layer's standard deviation:
$$S_{\epsilon}(\mathbf{g}) = \frac{\#\{i \mid |\mathbf{g}[i]| < 0.05 \times \sigma(\mathbf{g})\}}{d} \times 100\%$$

### Metric D: Temporal Jaccard Similarity (Mask IoU)
Measures the persistence of top-$k$ coordinate indices between step $t$ and step $t + \Delta t$:
$$\text{IoU}(\mathcal{M}_t, \mathcal{M}_{t+\Delta t}) = \frac{|\mathcal{M}_t \cap \mathcal{M}_{t+\Delta t}|}{|\mathcal{M}_t \cup \mathcal{M}_{t+\Delta t}|}$$

---

## 4. Architectural Models & Layer Categorization

### Model 1: ResNet-50 (CIFAR-10 Adapted)
* Total Parameters: $\approx 23.5\text{M}$ across 16 bottleneck blocks in 4 progressive stages.
* **Layer Type Categories**:
  1. `Conv 1x1 Reduction`: Bottleneck `conv1` (channel compression)
  2. `Conv 3x3 Spatial`: Bottleneck `conv2` (spatial feature extraction)
  3. `Conv 1x1 Expansion`: Bottleneck `conv3` ($4\times$ channel expansion)
  4. `Batch Normalization`: `bn1, bn2, bn3`
  5. `Classifier Head`: Linear `fc` ($2048 \to 10$)
* **Depth Stages**:
  * `conv1` (Input stem)
  * `layer1` (Stage 1: 3 bottlenecks, 256 ch)
  * `layer2` (Stage 2: 4 bottlenecks, 512 ch)
  * `layer3` (Stage 3: 6 bottlenecks, 1024 ch)
  * `layer4` (Stage 4: 3 bottlenecks, 2048 ch, closest to output)
  * `fc` (Classification head)

### Model 2: Vision Transformer (ViT-Tiny for CIFAR-10)
* Architecture: 6 Transformer Encoder Blocks, 8 Attention Heads, Embedding Dim 256, MLP Dim 1024.
* Total Parameters: $\approx 4.8\text{M}$.
* **Layer Type Categories**:
  1. `Attention: QKV`: Linear projection (`dim -> 3 * dim`)
  2. `Attention: Out-Proj`: Multi-head attention output linear projection
  3. `FFN: Linear 1 (Up)`: First MLP linear expansion (`256 -> 1024`, GELU)
  4. `FFN: Linear 2 (Down)`: Second MLP linear contraction (`1024 -> 256`)
  5. `Normalization`: LayerNorm weights and biases
  6. `Classifier Head`: Final classification linear head (`256 -> 10`)
* **Depth Stages**:
  * `patch_embed` (Input projection)
  * `block_0` to `block_5` (Transformer blocks 0 to 5)
  * `head` (Classification head)

### Model 3: Causal Language Transformer (GPT-Tiny for NLP)
* Architecture: 6 Causal Decoder Blocks, 6 Attention Heads, Embedding Dim 384, Context Window 128 tokens, BPE Vocabulary 50,257 tokens (or character-level 65 tokens).
* Total Parameters: $\approx 29.5\text{M}$ (with $V=50,257$ token embeddings).
* **Layer Type Categories**:
  1. `Embedding`: Token and positional embeddings (`wte`, `wpe`)
  2. `Attention: Causal QKV`: Linear projection (`dim -> 3 * dim`) with lower-triangular causal mask
  3. `Attention: Out-Proj`: Linear projection (`dim -> dim`)
  4. `FFN: Linear 1 (Up)`: First MLP expansion (`384 -> 1536`, GELU)
  5. `FFN: Linear 2 (Down)`: Second MLP contraction (`1536 -> 384`)
  6. `Language Model Head`: Final next-token prediction head (`lm_head`: `384 -> V`)
* **Depth Stages**:
  * `embed` (Token & position embedding)
  * `block_0` to `block_5` (Decoder blocks 0 to 5)
  * `head` (Language modeling head)
* **NLP Hypothesis 2 Premise**: Unlike CV models where output pooling collapses spatial features into a single 1-hot target, autoregressive language modeling computes cross-entropy loss at *every token position* across thousands of target vocabulary words. Concurrently, the embedding matrix receives updates for only the active tokens in the batch ($<5\%$ of vocabulary), creating a natural gradient density gradient from sparse input to dense output!

---

## 5. Experimental Protocol & Directory Structure

```
skipreduce/
├── docs/experiments/
│   ├── DESIGN_SPARSITY_PRED.md          # Detailed specification, math, and architecture definitions
│   ├── LOGS_SPARSITY.md                 # Living log table for profiling trials
│   └── LOGS.md                          # Prior SkipReduce domain transform logs
│
├── training/
│   ├── README.md                        # Package documentation and CLI usage
│   ├── models/                          # Shared model architectures (ResNet-50, ViT, GPT)

│   ├── core/                            # Algorithmic building blocks (ring, sparsity, transforms)
│   ├── benchmarks/                      # Hardware timing and compute savings profilers
│   ├── experiments/                     # CLI entry points (profile_sparsity, plot_sparsity, etc.)
│   ├── tests/                           # Unified test suite
│   ├── logs/                            # Raw machine data (JSON profiling and benchmarks)
│   └── figures/                         # Publication-ready figures (PNGs)
```

1. **Dataset**:
   - CV: CIFAR-10 (50k training images, batch size 128 $\implies$ 390 iterations per epoch).
   - NLP: TinyShakespeare text corpus (1.1MB, tokenized via `tiktoken` BPE, sequence length 128, batch size 16 $\implies$ 390 iterations per epoch).
2. **Execution**:
   - For each model, train for 1 complete epoch.
   - Intercept backward gradients after each step $t \in [1, 390]$.
   - Compute Metric A (Hoyer), Metric B ($E_{10}$), Metric C ($S_{\epsilon}$), and Metric D (IoU) per layer and per block.
3. **Outputs & Diagnostics**:
   - JSON profiling logs in `training/logs/`.
   - Standardized purpose-driven figures validating Hypotheses 1, 2, and 3 in `training/figures/`.

