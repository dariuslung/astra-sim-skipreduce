# Research Decisions & Operational Rules Log

This document records architectural, methodology, and operational decisions established throughout the SkipReduce project.

---

## 1. Scientific Phrasing & Hypothesis Rigor
* **Rule**: All figures, tables, code comments, and documentation must adhere to **Hypothesis evaluation phrasing** rather than claiming definitive proof.
* **Terminology Standard**:
  - Use: `Evaluating Hypothesis <N>`, `Hypothesis 1 Confirmed / Supported`, `Hypothesis 3 Falsified Under Convergence`, `Empirical evidence indicates...`.
  - Avoid: `Proving`, `Proves`, `Definitively proven`.
* **Rationale**: Hypotheses are tested and either supported, rejected, or refined based on empirical observations across architectures and datasets.

---

## 2. Experimental Profiling Cadence & Layer-Wise CV Metric
* **Decision**: Sample the first few iterations ($T_0$, batches 1–5) of each epoch to parameterize layer skipping and Compressive Sensing compression ratios.
* **Metric Choice (Layer Intra-Epoch CV over Sparsity Ratios)**:
  - Rather than indirect sparsity ratios between layer types (which do not reveal whether individual layers drift together), stability is directly quantified by the **Intra-Epoch Coefficient of Variation ($\text{CV}_l^{(e)}$)** for each individual layer type and stage across checkpoints ($T_0 \to T_4$).
  - Post-warmup (Epochs 2–20), all convolutional layer types ($3\times 3$ spatial, $1\times 1$ expand, $1\times 1$ reduce, $1\times 1$ shortcut) maintain an intra-epoch CV strictly below **3%** (consistently under the $5\%$ invariance threshold).
  - Internal representation stages maintain an intra-epoch CV of **0.5%–3.0%**.
  - Therefore, continuous per-step profiling is unnecessary; epoch-onset sampling accurately models each individual layer for the entire epoch with minimal compute overhead.

---

## 3. Mask-Based vs. Direction-Based Skipping Policies
* **Decision**: Skipping policies and Compressive Sensing allocations must be driven by **coordinate masks and layer-type sparsity budgets**, not by directional projections or instantaneous gradient vectors.
* **Empirical Validation**:
  - The top-10% coordinate mask retains an overlap of $\text{IoU} \approx 0.30\text{--}0.35$ across the entire 390-step span ($6\times$ above random chance).
  - Conversely, the directional cosine similarity of instantaneous gradients quickly drops to $\approx 0$ due to mini-batch noise. Coordinate importance is persistent, while instantaneous gradient direction fluctuates.

---

## 4. Figure Layout & Asset Organization
* **Rule**:
  - Figures are organized in structured subdirectories under `training/figures/` matching the experiment indexing (`exp01_` through `exp09_`):
    - `exp01_layer_type_sparsity/` (`fig01_*`)
    - `exp02_depth_vs_density/` (`fig02_*`)
    - `exp03_iteration_sparsity/` (`fig03_*`)
    - `exp04_temporal_mask_iou/` (`fig04_*`)
    - `exp05_compute_savings/` (`fig05a_*`, `fig05b_*`)
    - `exp06_resnet50_convergence/` (`fig06a_*` – `fig06d_*`)
    - `exp07_intra_epoch_stability/` (`fig07a_*` – `fig07e_*`)
    - `exp08_layer_recoverability/` (`fig08a_*` – `fig08c_*`)
    - `exp09_sparsity_vs_sensitivity/` (`fig09_*`)
  - All plots with super-titles must reserve layout headroom using `plt.tight_layout(rect=[0, 0, 1, 0.90])` and save with `bbox_inches="tight"` to prevent any title or label cropping.

---

## 5. Development & Execution Safety
* **Rule**:
  - **Git Operations**: Only stage, commit, or push when explicitly instructed by the user.
  - **Experiment Reruns**: Do not re-run computationally expensive training or profiling runs without explicit user approval.

---

## 6. Sparsity Metrics Framework: Hoyer Index & Top-10% Energy ($E_{10}$)
* **Decision**: Adopt the scale-invariant **Hoyer Index** (for temporal tracking without sorting, $\mathcal{O}(n)$) alongside **Top-10% Energy Concentration ($E_{10}$)** as the primary physical energy metric.
  $$E_{10}(\mathbf{g}) = \frac{\|\mathbf{g} \odot \mathcal{M}_{\text{top10\%}}\|_2^2}{\|\mathbf{g}\|_2^2} \times 100\%$$
* **Omission of $K_{90}$ (Redundancy Elimination)**:
  - $K_{90}$ (measuring what % of coordinates capture 90% energy) was evaluated and omitted due to conceptual and empirical redundancy with $E_{10}$. Both metrics sample the identical underlying cumulative energy (Lorenz) curve.
  - Computing $K_{90}$ requires full vector sorting ($\mathcal{O}(n \log n)$), creating memory and latency overhead on tensors with millions of parameters. In contrast, $E_{10}$ requires only partial selection ($\mathcal{O}(n)$).
  - Furthermore, distributed communication networks operate on fixed buffer sizes and uniform Compressive Sensing measurement rates ($M = m \times N$). $E_{10}$ directly quantifies signal retention under a fixed budget ($r = 0.10$), whereas $K_{90}$ produces variable coordinate allocations that complicate synchronous ring all-reduce schedules.

---

## 7. Isolated Layer-Type Skipping Ablation Protocol (Hypothesis 5)
* **Decision**: Before deploying Compressive Sensing (CS) recovery across layers, evaluate intrinsic layer recoverability and sensitivity via **Protocol A: Isolated Layer-Type Skipping Ablation Probe**.
* **Rationale**: Direct skipping (setting `grad = None` on skipped batches) isolates the exact downstream impact of skipping each structural layer type ($3\times 3$ spatial convs, $1\times 1$ reduce convs, $1\times 1$ expand convs, $1\times 1$ shortcut convs, classifier head), establishing empirical sensitivity coefficients ($\Delta\text{Acc}/\text{MParam}$) to guide non-uniform CS measurement budgets.

---

## 8. Correlation Between Baseline Sparsity and Layer Sensitivity (Hypothesis 6)
* **Decision**: Use baseline gradient sparsity ($E_{10}$ and Hoyer) profiled at epoch onset ($T_0$) as a **zero-cost analytical predictor of layer sensitivity** to determine layer-adaptive skipping and Compressive Sensing budgets.
* **Empirical Validation**:
  - Across all convolutional layers, baseline $E_{10}$ correlates with normalized sensitivity with **Spearman $\rho = -1.0000$** (perfect monotonic inverse ranking) and linear **$R^2 = 0.9183$** ($r = -0.9583, p = 0.0417$).
  - Higher energy concentration directly predicts higher skipping resilience: $1\times 1$ convs ($E_{10} = 93.0\%\text{--}95.8\%$) tolerate 50% update skipping with virtually zero degradation, whereas dense $3\times 3$ spatial convs ($E_{10} = 89.3\%$) suffer a $-5.12\%$ collapse.
  - SkipReduce can assign skipping and compression budgets directly from epoch-onset $E_{10}$ profiles without requiring empirical sensitivity searches.

---

## 9. 1-to-1 Experiment and Figure Indexing Alignment
* **Decision**: Maintain a strict 1-to-1 correspondence across all three organizational layers:
  1. **Experiment designator**: `EXP-01` through `EXP-09`
  2. **Figure filename prefix**: `fig01_` through `fig09_`
  3. **Figure output directory**: `training/figures/exp01_<name>/` through `training/figures/exp09_<name>/`
* **Rationale**: Eliminates ambiguity between multiple independent experiments previously lumped into broad category folders (`convergence/` or `sparsity_profiling/`), allowing automated tracking, modular reproducibility, and clear paper artifact mapping.



