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

## 2. Experimental Profiling Cadence
* **Decision**: Sample the first few iterations ($T_0$, batches 1–5) of each epoch to parameterize layer skipping and Compressive Sensing compression ratios.
* **Empirical Validation**:
  - Intra-epoch profiling across 20 full epochs of ResNet-50 (7,800 steps, 390 steps/epoch) showed that gradient Hoyer sparsity remains invariant throughout an epoch with a Coefficient of Variation (CV) between **0.54% and 2.86%** (mean CV = 1.3%).
  - The relative sparsity ratios between layer types (e.g., $3\times 3$ Spatial vs. $1\times 1$ Expand Conv) stay frozen with **$\text{CV} < 1\%$** across all intra-epoch checkpoints.
  - Therefore, continuous per-step profiling is unnecessary; epoch-onset sampling provides an accurate representation with minimal compute overhead.

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
