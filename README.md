# ASTRA-sim SkipReduce Simulation Environment

This directory contains a complete pipeline for generating, trace-converting, and simulating the **SkipReduce** collective algorithm using **ASTRA-sim** with both **Analytical** and **ns-3** packet-level network backends.


## Environment Setup

Perform the initial setup on the host machine, and then proceed with the dependency installations inside the Docker container.

### 1. Host Machine Setup

#### Clone Repositories
Clone ASTRA-sim and mlcommons/chakra, and clone `astra-sim-skipreduce` into the `astra-sim/skipreduce` directory:
```bash
git clone --recursive https://github.com/astra-sim/astra-sim.git
git clone --recursive https://github.com/mlcommons/chakra.git
cd astra-sim
git clone https://github.com/dariuslung/astra-sim-skipreduce.git skipreduce
git submodule update --init --recursive
```

#### Build Docker Image
From the `astra-sim/` root directory, build the latest Docker image:
```bash
sudo docker build -t astra-sim:latest -f Dockerfile .
```

#### Run and Manage Docker Container
Create and run the Docker container with mounts pointing to the side-by-side repository paths:
```bash
# Run the container
sudo docker run -it --name astra-sim-latest --shm-size=8g \
    -v "$(pwd):/app/astra-sim" \
    -v "$(pwd)/../chakra:/app/chakra" \
    astra-sim:latest bash
```

To manage the container lifecycle:
```bash
# Start container
sudo docker start astra-sim-latest

# Enter container shell
sudo docker exec -it astra-sim-latest bash
```

> [!IMPORTANT]  
> Perform all subsequent installation and execution steps **inside the running Docker container**.

### 2. Container Dependency Installation

#### Install Chakra & Collective API
Inside the container, install Chakra and clone the collective API repository:
```bash
cd /app/chakra
pip install .

git clone https://github.com/astra-sim/collectiveapi.git
cd collectiveapi
```

#### Configure and Install MSCCL Tools
Modify `collectiveapi/.gitmodules` to use the correct `msccl-tools` fork URL:
```ini
[submodule "msccl-tools"]
    path = msccl-tools
    url = https://github.com/jinsun-yoo/msccl-tools.git
```

Sync and install the submodules:
```bash
git submodule sync
git submodule update --init --recursive
cd msccl-tools
pip install .
```

#### Upgrade Protobuf
Upgrade the Python protobuf package to ensure compatibility with modern trace generation:
```bash
pip install protobuf==6.31.1
```

### 3. Configure the Modified Chakra Converter
To use the custom collective converter from this `skipreduce` directory, clean up the default scripts in the collective API path and symlink our modified versions:
```bash
cd /app/chakra/collectiveapi/chakra_converter/
rm et_converter.py mscclang2chakra_converter.py

# Create symlinks pointing to modified converter versions
ln -s /app/astra-sim/skipreduce/chakra_converter/mscclang2chakra_converter.py /app/chakra/collectiveapi/chakra_converter/mscclang2chakra_converter.py
ln -s /app/astra-sim/skipreduce/chakra_converter/et_converter.py /app/chakra/collectiveapi/chakra_converter/et_converter.py
```


## Key Components

1. **`skipreduce.py`**: A program built using the Microsoft Collective Communication Library (MSCCL) Python DSL. It defines a custom ring topology (locking sequential chains to prevent compiler bypasses), generates the SkipReduce collective (Reduce-Scatter with $s$ steps skipped, followed by an All-Gather phase), and outputs the MSCCL XML.
2. **`run_analytical.sh`**: Helper runner script that generates the SkipReduce workload, compiles ASTRA-sim with the Analytical congestion-unaware backend, and runs the simulation.
3. **`run_ns3.sh`**: Helper runner script that generates the SkipReduce workload, configures the ns-3 simulation with output helper files (`flow.txt`, `trace.txt`), compiles/runs the ns-3 packet-level backend, and maps logical dimensions dynamically.
4. **`alt_topologies/`**: Logical and physical topology descriptions:
   * **`analytical/`**: Multi-dimensional YAML topology configurations.
   * **`ns3/`**: Corresponding physical link networks (`*_network.txt`) and logical dimension files (`*_logical.json`).
5. **`output_md_parser.py`**: A helper Python utility to parse standard ASTRA-sim stdout logs and print a formatted Markdown table of Wall Time, Communication Time, GPU Time, and Compute-Communication Overlap.


## Command Line Interface (CLI)

Both helper runner scripts share an intuitive, unified interface for setting up the simulation dimensions (execute from `/app/astra-sim/skipreduce` inside the Docker container):

```bash
cd /app/astra-sim/skipreduce
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


## Trace Debugging & Log Parsing

### Trace Debugging (Optional)
To verify node attributes, dependency links, and payload sizes in generated `.et` trace files:
```bash
export PYTHONPATH=$PYTHONPATH:/app/chakra && python3 -c "
from schema.protobuf.et_def_pb2 import Node, GlobalMetadata
from src.third_party.utils.protolib import decodeMessage
f = open('workload/skipreduce_npus.0.et', 'rb')
decodeMessage(f, GlobalMetadata())
n = Node()
[decodeMessage(f, n) for _ in range(13)]
print(f'Node {n.id} Type: {n.type} Deps: {list(n.data_deps)}')
print(f'Attrs: {[(a.name, a.uint64_val or a.int64_val or a.int32_val) for a in n.attr]}')
"
```

### Parsing Output Logs
You can pipe or copy the simulation output logs into the parsing utility to format system statistics:
```bash
python3 output_md_parser.py
```
This prints a clean table detailing the wall time, communication execution cycles, and overlap for each simulated system rank.

### Example
| System | Wall Time | Comm Time | GPU Time | Comp-Comm Overlap | Exposed Comm |
| --- | --- | --- | --- | --- | --- |
| sys[0] | 163720478 | 163720476 | 8 | 6 | 163720470 |
| sys[1] | 163298155 | 163298155 | 8 | 8 | 163298147 |
| sys[2] | 163844330 | 163844330 | 8 | 8 | 163844322 |
| sys[3] | 163183949 | 163183945 | 8 | 4 | 163183941 |