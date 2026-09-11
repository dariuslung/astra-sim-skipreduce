# Experiment Logs: Domain-Transformed SkipReduce

This file tracks all experimental runs for reproducing results and auditing convergence.

| Run ID | Timestamp | Ranks ($N$) | Skip ($s$) | Transform | Ret. Ratio ($r$) | EF? | Epochs | Final Loss | Top-1 Val Acc (%) | Mean Grad CosSim | Status | Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `ranks4_skip1_dct_r0.15_noef_1789117752` | 2026-09-11 17:04 | 4 | 1 | `dct` | 0.15 | No | 1 | 2.1045 | **40.11%** | 0.9228 | Completed | 1-epoch sanity run on RTX 4060 Ti |
