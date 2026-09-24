# Agent Directives & User Decisions

This document records explicit user instructions, scientific phrasing rules, and operational constraints for the SkipReduce project. All agents operating in this repository must strictly adhere to these directives.

---

## 1. Scientific Phrasing & Terminology
* **Use Hypothesis Evaluation Phrasing**: Always maintain "hypothesis" phrasing across all documentation, plot titles, super-titles, script comments, and reports (e.g., `Evaluating Hypothesis 1`, `Hypothesis 1 Confirmed`, `Hypothesis 3 Falsified Under Convergence`).
* **Avoid Assertive Proof Claims**: Do NOT use words such as "proving", "proves", or "definitively proven". State empirical observations objectively (e.g., "evaluates", "demonstrates", "confirms", "falsifies", "indicates").

---

## 2. Git & Version Control Protocol
* **Only Commit When Asked**: Do NOT stage, commit, or push changes to git unless the user explicitly requests it.
* Keep intermediate working tree changes available for inspection before committing.

---

## 3. Execution & Testing Constraints
* **Do Not Rerun Tests Unprompted**: Do NOT rerun long-running or computationally expensive experiments, profiling runs, or test suites unless explicitly requested by the user. Use existing logs (`training/logs/`) for analysis and plotting whenever available.
* **Environment Execution**: When running Python scripts that require PyTorch, CUDA, or Matplotlib, run with `/home/dalius/miniconda3/envs/ml/bin/python`.

---

## 4. Figure Organization & Visualization Standards
* **Folder Hierarchy**: All figures must be saved into experiment-specific subdirectories under `training/figures/`:
  - `exp01_layer_type_sparsity/`: Layer-type sparsity sweeps (`fig01_*`).
  - `exp02_depth_vs_density/`: Controlled depth vs density sweeps (`fig02_*`).
  - `exp03_iteration_sparsity/`: Iteration-by-iteration sparsity evolution (`fig03_*`).
  - `exp04_temporal_mask_iou/`: Temporal mask persistence across lags (`fig04_*`).
  - `exp05_compute_savings/`: Hardware latency breakdowns & speedups (`fig05a_*`, `fig05b_*`).
  - `exp06_resnet50_convergence/`: Multi-epoch convergence vs. gradient dynamics (`fig06a_*` - `fig06d_*`).
  - `exp07_intra_epoch_stability/`: Intra-epoch checkpoint trajectories, mask decay, CV% (`fig07a_*` - `fig07e_*`).
  - `exp08_layer_recoverability/`: Protocol A layer skipping sensitivity ablation (`fig08a_*` - `fig08c_*`).
  - `exp09_sparsity_vs_sensitivity/`: Baseline sparsity vs. sensitivity correlation (`fig09_*`).
* **Headroom & Title Clipping Prevention**:
  - Super-titles (`plt.suptitle`) must have sufficient margin. Always call `plt.tight_layout(rect=[0, 0, 1, 0.90])` (or adjust `top`) to leave space for multi-line titles.
  - Always export figures with `plt.savefig(..., bbox_inches="tight")` to ensure labels, titles, and legends are never cropped.

---

## 5. Key Research & System Decisions
* **Epoch-Onset Profiling Strategy**: Because layer-type sparsity distributions and relative ratios remain invariant ($\text{CV} < 1\text{--}2\%$) across 390+ batches within an epoch, SkipReduce profiles only the first few iterations ($T_0$, batches 1–5) per epoch to establish the skipping schedule and Compressive Sensing compression ratios for that entire epoch.
* **Coordinate Masks vs. Vector Direction**: Skipping decisions and compression schedules must be driven by **coordinate masks and layer-type budgets**, rather than directional vector projections (since instantaneous gradient vector angles rotate due to mini-batch noise, while coordinate masks remain $6\times$ above random chance).

---

## 6. Error Feedback (EF) Policy
* **Do Not Mention or Use Error Feedback (EF)**: Do NOT reference, discuss, or qualify experiments as "without EF" or "without Error Feedback" across any documentation, plot titles, super-titles, script comments, or experiment reports unless explicitly prompted by the user. SkipReduce gradient skipping and compression operate under direct skipping by default; do not bring up EF.

---

## 7. Future-Proof Experiment Logging Policy
* **Mandatory Gradient Sparsity Logging**: All future training runs, ablation probes, distributed simulations, and profiling scripts must log mathematical gradient sparsity metrics by default. Specifically, experiments must record:
  1. **Energy Concentration ($E_{10}$)**: Percentage of total $L_2^2$ energy captured by the top 10% coordinates.
  2. **Hoyer Sparsity**: Scale-invariant sparsity bounded in $[0, 1]$.
  3. **Relative Threshold Sparsity ($S_{0.05}$)**: Fraction of coordinates below $0.05 \sigma$.
  4. **Structural Breakdowns**: Aggregated metrics partitioned by `layer_type` and by `stage` (or block).
  5. **Active vs. Skipped Step Sampling**: For experiments with skipping or compression, metrics must be sampled on non-skipped (active) iterations prior to zeroing/compression.
  6. **Standardized Tooling**: Standardize on [`training.core.sparsity.GradientSparsityTracker`](training/core/sparsity/tracker.py) across all training pipelines to record metrics automatically with negligible compute overhead (<0.5%).


