# Astra-Sim SkipReduce Simulation Environment

This directory contains a complete pipeline for generating, trace-converting, and simulating the **SkipReduce** collective algorithm using **ASTRA-sim** with both **Analytical** and **ns-3** packet-level network backends.

---

## Key Components

1. **`skipreduce.py`**: A program built using the Microsoft Collective Communication Library (MSCCL) Python DSL. It defines a custom ring topology (locking sequential chains to prevent compiler bypasses), generates the SkipReduce collective (Reduce-Scatter with $s$ steps skipped, followed by an All-Gather phase), and outputs the MSCCL XML.
2. **`run_analytical.sh`**: Helper runner script that generates the SkipReduce workload, compiles Astra-Sim with the Analytical congestion-unaware backend, and runs the simulation.
3. **`run_ns3.sh`**: Helper runner script that generates the SkipReduce workload, configures the ns-3 simulation with output helper files (`flow.txt`, `trace.txt`), compiles/runs the ns-3 packet-level backend, and maps logical dimensions dynamically.
4. **`alt_topologies/`**: Logical and physical topology descriptions:
   * **`analytical/`**: Multi-dimensional YAML topology configurations.
   * **`ns3/`**: Corresponding physical link networks (`*_network.txt`) and logical dimension files (`*_logical.json`).
5. **`output_md_parser.py`**: A helper Python utility to parse standard Astra-Sim stdout logs and print a formatted Markdown table of Wall Time, Communication Time, GPU Time, and Compute-Communication Overlap.

---

## Command Line Interface (CLI)

Both helper runner scripts share an intuitive, unified interface for setting up the simulation dimensions:

```bash
./run_ns3.sh --N <count> [--L1 <count>] [--L2 <count>] [--skip <number>]
./run_analytical.sh --N <count> [--L1 <count>] [--L2 <count>] [--skip <number>]
```

### Argument Reference
* `--N`: Total number of NPUs/GPUs (e.g. `2`, `4`, `8`). **[Required]**
* `--L1`: Local group dimension size (e.g. local nodes under the same L1 switch). **[Optional]**
* `--L2`: Hierarchical dimension size. **[Optional; requires --L1]**
* `--skip`: Number of reduction steps to skip in SkipReduce (defaults to `0`). **[Optional]**

### Examples
To run a flat ring on 2 NPUs:
```bash
./run_ns3.sh --N 2
```

To run a hierarchical configuration on 8 NPUs split across 2 switches of size 4:
```bash
./run_ns3.sh --N 8 --L1 4 --skip 1
```

---

## Topology Structure & Mapping (ns-3 Backend)

Astra-Sim hierarchical topologies mapped from Analytical YAML parameters to ns-3 physical configurations are defined as follows:

* **Flat topologies (`N-<N>`)**:
  All NPUs connect to a single central L1 switch.
* **Direct-connect switches (`L1-<L1>_N-<N>`)**:
  L1 groups of NPUs connect to local switches. If a second layer of `FullyConnected` is specified, the L1 switches are directly connected by point-to-point links.
* **Hierarchical switches (`L2-<L2>_L1-<L1>_N-<N>`)**:
  NPUs connect to local L1 switches. The L1 switches are then connected hierarchically via a central central L2 switch.

---

## Parsing Output Logs

You can pipe or copy the simulation output logs into the parsing utility to format system statistics:

```bash
python3 output_md_parser.py
```
This prints a clean table detailing the wall time, communication execution cycles, and overlap for each simulated system rank.
