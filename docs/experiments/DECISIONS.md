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
  - Figures are organized in structured subdirectories under `training/figures/`:
    - `sparsity_profiling/`
    - `compute_savings/`
    - `convergence/`
    - `intra_epoch_stability/`
  - All plots with super-titles must reserve layout headroom using `plt.tight_layout(rect=[0, 0, 1, 0.90])` and save with `bbox_inches="tight"` to prevent any title or label cropping.

---

## 5. Development & Execution Safety
* **Rule**:
  - **Git Operations**: Only stage, commit, or push when explicitly instructed by the user.
  - **Experiment Reruns**: Do not re-run computationally expensive training or profiling runs without explicit user approval.

---

## 6. Sparsity Metrics Framework: Dual Physical Energy Metrics ($E_{10}$ and $K_{90}$) alongside Hoyer
* **Decision**: While the scale-invariant **Hoyer Index** is maintained for temporal tracking and convergence optimization ($\mathcal{O}(n)$ computation without sorting), all experiments and reports must evaluate physical energy metrics:
  1. **Top-10% Energy Concentration ($E_{10}$)**:
     $$E_{10}(\mathbf{g}) = \frac{\|\mathbf{g} \odot \mathcal{M}_{\text{top10\%}}\|_2^2}{\|\mathbf{g}\|_2^2} \times 100\%$$
     *Measures*: Given a fixed coordinate budget (10%), what percentage of the total update energy is captured?
  2. **$K_{90}$ Energy Bandwidth**:
     $$K_{90}(\mathbf{g}) = \min \left\{ \frac{k}{d} \times 100\% \;\middle|\; \sum_{i=1}^k g_{(i)}^2 \ge 0.90 \, \|\mathbf{g}\|_2^2 \right\}$$
     *Measures*: Given a target fidelity (90% energy retention), what minimum percentage of parameters must be communicated?
* **Physical Duality**: $E_{10}$ and $K_{90}$ are duals on the cumulative energy curve (Lorenz curve). They provide immediate physical and systems intuition for network payload reduction and Compressive Sensing measurement rates, resolving the intuition gap of unitless inequality indices.

---

## 7. Isolated Layer-Type Skipping Ablation Protocol (Hypothesis 5)
* **Decision**: Before deploying Compressive Sensing (CS) recovery across layers, evaluate intrinsic layer recoverability and sensitivity via **Protocol A: Isolated Layer-Type Skipping Ablation Probe**.
* **Rationale**: Direct skipping (setting `grad = None` on skipped batches) isolates the exact downstream impact of skipping each structural layer type ($3\times 3$ spatial convs, $1\times 1$ reduce convs, $1\times 1$ expand convs, $1\times 1$ shortcut convs, classifier head), establishing empirical sensitivity coefficients ($\Delta\text{Acc}/\text{MParam}$) to guide non-uniform CS measurement budgets.


