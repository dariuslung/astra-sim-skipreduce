"""
Automated experiment sweep orchestrator.
Executes comparison runs across baseline, pure SkipReduce, DCT, and Hadamard,
and automatically updates docs/experiments/LOGS.md with results.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time

# Ensure repo root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


EXPERIMENT_CONFIGS = [
    # 1. Baseline Full AllReduce
    {
        "name": "baseline_allreduce",
        "ranks": 4,
        "skip": 0,
        "transform": "none",
        "retention": 0.0,
        "use_ef": False,
        "notes": "Ground truth AllReduce (100% communication)"
    },
    # 2. Naive SkipReduce (lossy, no transform)
    {
        "name": "naive_skipreduce_s1",
        "ranks": 4,
        "skip": 1,
        "transform": "none",
        "retention": 0.0,
        "use_ef": False,
        "notes": "Pure SkipReduce s=1, skipped ranks dropped completely"
    },
    # 3. DCT Transformed SkipReduce (no EF)
    {
        "name": "dct_s1_r15_noef",
        "ranks": 4,
        "skip": 1,
        "transform": "dct",
        "retention": 0.15,
        "use_ef": False,
        "notes": "DCT 15% retention on skipped partition, no error feedback"
    },
    # 4. DCT Transformed SkipReduce (with EF)
    {
        "name": "dct_s1_r15_ef",
        "ranks": 4,
        "skip": 1,
        "transform": "dct",
        "retention": 0.15,
        "use_ef": True,
        "notes": "DCT 15% retention with Error Feedback buffer"
    },
    # 5. Walsh-Hadamard Transformed SkipReduce (with EF)
    {
        "name": "hadamard_s1_r15_ef",
        "ranks": 4,
        "skip": 1,
        "transform": "hadamard",
        "retention": 0.15,
        "use_ef": True,
        "notes": "Walsh-Hadamard 15% retention with Error Feedback buffer"
    },
    # 6. Heavy Naive SkipReduce (s=2, 50% partitions dropped)
    {
        "name": "naive_skipreduce_s2",
        "ranks": 4,
        "skip": 2,
        "transform": "none",
        "retention": 0.0,
        "use_ef": False,
        "notes": "Heavy SkipReduce s=2 (50% partitions dropped), no transform"
    },
    # 7. Heavy Skip DCT Recovery (s=2, 15% retention)
    {
        "name": "dct_s2_r15_noef",
        "ranks": 4,
        "skip": 2,
        "transform": "dct",
        "retention": 0.15,
        "use_ef": False,
        "notes": "DCT 15% recovery on s=2 heavy skip, no error feedback"
    },
    # 8. Heavy Skip Walsh-Hadamard Recovery (s=2, 15% retention)
    {
        "name": "hadamard_s2_r15_noef",
        "ranks": 4,
        "skip": 2,
        "transform": "hadamard",
        "retention": 0.15,
        "use_ef": False,
        "notes": "Walsh-Hadamard 15% recovery on s=2 heavy skip, no error feedback"
    },
    # 9. Heavy Skip DCT Recovery (s=2, 25% retention)
    {
        "name": "dct_s2_r25_noef",
        "ranks": 4,
        "skip": 2,
        "transform": "dct",
        "retention": 0.25,
        "use_ef": False,
        "notes": "DCT 25% recovery on s=2 heavy skip, no error feedback"
    }
]


def append_to_logs_md(logs_path: str, run_data: dict, config: dict):
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    last_epoch = run_data["epochs_data"][-1]
    
    row = (
        f"| `{run_data['run_id']}` | {now_str} | {config['ranks']} | {config['skip']} | "
        f"`{config['transform']}` | {config['retention']:.2f} | "
        f"{'Yes' if config['use_ef'] else 'No'} | {run_data['epochs']} | "
        f"{last_epoch['train_loss']:.4f} | **{last_epoch['val_acc']:.2f}%** | "
        f"{last_epoch['mean_cos_sim']:.4f} | Completed | {config['notes']} |\n"
    )

    with open(logs_path, "a") as f:
        f.write(row)
    print(f"Appended run record to {logs_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Automated Experiment Grid Sweep")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per run")
    parser.add_argument("--micro-batch-size", type=int, default=32, help="Micro-batch size per virtual rank")
    parser.add_argument("--python-bin", type=str, default=sys.executable, help="Python binary to execute")
    parser.add_argument("--logs-md", type=str, default="docs/experiments/LOGS.md", help="Markdown log notebook path")
    parser.add_argument("--only-run", type=int, default=None, help="Run only specific config index (0-based)")
    args = parser.parse_args()

    configs_to_run = EXPERIMENT_CONFIGS if args.only_run is None else [EXPERIMENT_CONFIGS[args.only_run]]

    print(f"Starting experiment sweep ({len(configs_to_run)} configurations, {args.epochs} epochs each)...")

    for i, cfg in enumerate(configs_to_run):
        print("\n" + "#" * 80)
        print(f"Executing Config [{i+1}/{len(configs_to_run)}]: {cfg['name']}")
        print("#" * 80)

        cmd = [
            args.python_bin, "ml/train_cifar.py",
            "--ranks", str(cfg["ranks"]),
            "--skip", str(cfg["skip"]),
            "--transform", cfg["transform"],
            "--retention", str(cfg["retention"]),
            "--epochs", str(args.epochs),
            "--micro-batch-size", str(args.micro_batch_size)
        ]
        if cfg["use_ef"]:
            cmd.append("--use-ef")

        start = time.time()
        result = subprocess.run(cmd, check=True)
        duration = time.time() - start

        # Find the latest generated log file in ml/logs
        log_files = sorted(
            [os.path.join("ml/logs", f) for f in os.listdir("ml/logs") if f.endswith(".json")],
            key=os.path.getmtime
        )
        if log_files:
            latest_log = log_files[-1]
            with open(latest_log, "r") as f:
                run_data = json.load(f)
            append_to_logs_md(args.logs_md, run_data, cfg)

    print("\n" + "=" * 80)
    print("Experiment sweep successfully completed!")
    print(f"View updated experiment logs in: {args.logs_md}")
    print("=" * 80)


if __name__ == "__main__":
    main()
