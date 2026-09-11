# Experiment Logs: Domain-Transformed SkipReduce

This file tracks all experimental runs for reproducing results and auditing convergence.

| Run ID | Timestamp | Ranks ($N$) | Skip ($s$) | Transform | Ret. Ratio ($r$) | EF? | Epochs | Final Loss | Top-1 Val Acc (%) | Mean Grad CosSim | Status | Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `ranks4_skip1_dct_r0.15_noef_1789117752` | 2026-09-11 17:04 | 4 | 1 | `dct` | 0.15 | No | 1 | 2.1045 | **40.11%** | 0.9228 | Completed | 1-epoch sanity run on RTX 4060 Ti |
| `ranks4_skip0_none_r0.0_noef_1789118323` | 2026-09-11 17:25 | 4 | 0 | `none` | 0.00 | No | 20 | 0.0898 | **92.29%** | 1.0000 | Completed | Ground truth AllReduce (100% communication) |
| `ranks4_skip1_none_r0.0_noef_1789120749` | 2026-09-11 18:08 | 4 | 1 | `none` | 0.00 | No | 20 | 0.1160 | **91.90%** | 0.8652 | Completed | Pure SkipReduce s=1, skipped ranks dropped completely |
| `ranks4_skip1_dct_r0.15_noef_1789121341` | 2026-09-11 18:23 | 4 | 1 | `dct` | 0.15 | No | 20 | 0.1180 | **91.54%** | 0.9043 | Completed | DCT 15% retention on skipped partition, no error feedback |
| `ranks4_skip1_dct_r0.15_ef_1789122280` | 2026-09-11 18:39 | 4 | 1 | `dct` | 0.15 | Yes | 20 | 0.1157 | **91.77%** | 0.9038 | Completed | DCT 15% retention with Error Feedback buffer |
| `ranks4_skip2_none_r0.0_noef_1789123362` | 2026-09-11 18:51 | 4 | 2 | `none` | 0.00 | No | 20 | 0.1775 | **90.64%** | 0.7061 | Completed | Heavy SkipReduce s=2 (50% partitions dropped), no transform |
| `ranks4_skip2_dct_r0.15_noef_1789123956` | 2026-09-11 19:12 | 4 | 2 | `dct` | 0.15 | No | 20 | 0.1383 | **91.40%** | 0.7995 | Completed | DCT 15% recovery on s=2 heavy skip, no error feedback |
| `ranks4_skip2_hadamard_r0.15_noef_1789125356` | 2026-09-11 19:58 | 4 | 2 | `hadamard` | 0.15 | No | 20 | 0.1259 | **91.37%** | 0.8816 | Completed | Walsh-Hadamard 15% recovery on s=2 heavy skip, no error feedback |
