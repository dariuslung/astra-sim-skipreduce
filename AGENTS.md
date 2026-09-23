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
* **Folder Hierarchy**: All figures must be saved into categorized subdirectories under `training/figures/`:
  - `sparsity_profiling/`: Layer-type, depth, and iteration sparsity sweeps.
  - `compute_savings/`: FLOP reductions, speedup breakdowns, and theoretical scaling.
  - `convergence/`: 20-epoch training convergence vs. gradient dynamics.
  - `intra_epoch_stability/`: Intra-epoch checkpoint trajectories, mask decay, and variance analysis.
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

