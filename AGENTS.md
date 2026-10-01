# Agent Directives & User Decisions

Strict operational rules and scientific constraints for the SkipReduce project.

---

## 1. Scientific Phrasing & Terminology
* **Use Hypothesis Phrasing**: Maintain hypothesis evaluation phrasing across all docs, plots, and comments (e.g., `Evaluating Hypothesis 1`, `Hypothesis 1 Confirmed`, `Hypothesis 3 Falsified Under Convergence`).
* **Avoid Proof Claims**: Do NOT use "proving", "proves", or "definitively proven". State empirical observations objectively ("evaluates", "demonstrates", "confirms", "falsifies", "indicates").

---

## 2. Execution & Testing Constraints
* **Do Not Rerun Tests Unprompted**: Do NOT rerun long-running training, sweeps, or profiling unprompted. Use existing logs (`training/logs/`) for analysis.
* **Environment Execution**: Run Python scripts requiring PyTorch/CUDA/Matplotlib with `/home/dalius/miniconda3/envs/ml/bin/python`.

---

## 3. Figure Organization & Visualization Standards
* **Folder Hierarchy**: Save figures into experiment subdirectories under `training/figures/exp{NN}_{topic}/` (e.g., `exp10_layer_cs_budget/`) named `fig{NN}[a-z]_*.png`.
* **Headroom & Margins**: Multi-line supertitles must call `plt.tight_layout(rect=[0, 0, 1, 0.90])` (or adjust `top`). Always export with `plt.savefig(..., bbox_inches="tight")` to prevent clipping.

---

## 4. Key Research & System Decisions
* **Epoch-Onset Profiling**: Profile only initial iterations ($T_0$, batches 1–5) per epoch to fix schedules/budgets; layer sparsity ratios remain invariant ($\text{CV} < 1\text{--}2\%$) across epochs.
* **Coordinate Masks vs. Vector Direction**: Decisions are driven by coordinate masks and layer budgets, not vector angles (which drift with mini-batch noise).

---

## 5. Error Feedback (EF) Policy
* **Do Not Mention or Use Error Feedback (EF)**: Do NOT reference or qualify experiments as "without EF" or "without Error Feedback" unless explicitly asked. SkipReduce operates under direct skipping/CS by default.

---

## 6. Mandatory Gradient Sparsity Logging
All training, probes, and simulations must record mathematical gradient metrics via `training.core.sparsity.GradientSparsityTracker`:
1. **$E_{10}$**: Total $L_2^2$ energy of top 10% coordinates.
2. **Hoyer**: Scale-invariant sparsity in $[0, 1]$.
3. **$S_{0.05}$**: Fraction of coordinates below $0.05 \sigma$.
4. **Structural Breakdowns**: Aggregated by `layer_type` and `stage`.
5. **Sampling**: Sample on non-skipped (active) iterations prior to zeroing/compression.
