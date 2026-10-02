# Agent Directives & Operational Rules: SkipReduce

Strict operational rules and scientific constraints for the SkipReduce project.

---

## 1. Project Identity & Scope
* **Core Research Focus**: ASTRA-sim collective communication scheduling and simulation for **SkipReduce** Ring AllReduce with step-skipping ($s > 0$).
* **Collective Architecture**: Truncates the Reduce-Scatter phase from $(N - 1)$ to $(N - 1) - s$ steps along ring topologies using MSCCL schedules (`skipreduce.py`, `output_md_parser.py`).
* **Domain Transforms for Skipped Steps**: Evaluates frequency-domain approximations (1D-DCT, Walsh-Hadamard) and Error Feedback (EF) to mitigate truncation error on skipped ring steps.
* **Separation from Layer-CS**: All Layer-Adaptive Compressive Sensing (PR-CS, random sensing matrices, pure AllReduce without step-skipping, and gradient sparsity profiling) is hosted in `/home/dalius/Projects/dalius/layer-adaptive-cs`. Do not introduce PR-CS code or benchmarks into this repository.

---

## 2. Scientific Phrasing & Terminology
* **Hypothesis Phrasing**: Maintain hypothesis evaluation phrasing across all documentation, plots, and comments (e.g., `Evaluating Hypothesis 1`, `Hypothesis 1 Confirmed`, `Hypothesis 3 Falsified Under Convergence`).
* **Avoid Proof Claims**: Do NOT use "proving", "proves", or "definitively proven". State empirical observations objectively ("evaluates", "demonstrates", "confirms", "falsifies", "indicates").

---

## 3. Execution & Testing Constraints
* **Unprompted Execution**: Do NOT rerun long-running training, parameter sweeps, or Astra-Sim simulations unprompted. Use existing logs (`training/logs/`, `log/`) for analysis. Fast unit tests (<1 min) are permitted.
* **Environment Execution**: Run Python scripts requiring PyTorch/CUDA/Matplotlib with `/home/dalius/miniconda3/envs/ml/bin/python`.
* **Unit Testing**: Run `python -m unittest discover -s training/tests` to verify changes (all 14 tests pass).

---

## 4. Visualization & Logging Standards
* **Figure Organization**: Save SkipReduce figures into `training/figures/` (e.g., `fig4_pareto_accuracy_vs_payload.png`).
* **Headroom & Margins**: Always export with `plt.tight_layout()` and `plt.savefig(..., bbox_inches="tight")` to prevent title or label clipping.
* **Data Logs**: Store simulation and virtual ring logs in structured JSON format under `training/logs/`.
