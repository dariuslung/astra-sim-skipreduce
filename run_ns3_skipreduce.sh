#!/bin/bash
set -e
set -x

# find the absolute path to this script
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_DIR="/app/astra-sim"
NS3_DIR="${PROJECT_DIR}/extern/network_backend/ns-3"

# Binary path (Point to the one you just built)
# Note: Usually named 'ns3' which then runs the scratch script
ASTRA_SIM_EXEC="${NS3_DIR}/ns3"

# Paths to your patched files and configs
WORKLOAD="${SCRIPT_DIR}/workload/skip_reduce_8_gpus"
SYSTEM="${SCRIPT_DIR}/custom_collective.json"
REMOTE_MEMORY="${PROJECT_DIR}/examples/remote_memory/analytical/no_memory_expansion.json"

# ns-3 Specific Configs
# This must be a path ns-3 can resolve (often relative to ns-3 directory or absolute)
NETWORK="simulation/mix/config.txt"
LOGICAL_TOPOLOGY="${SCRIPT_DIR}/ns3_logical_topology.json"

echo "[ASTRA-sim] Running ASTRA-sim SkipReduce with ns-3 Backend..."

# Run ASTRA-sim via the ns-3 runner
# This ensures all packet-level libraries are correctly linked
cd "${NS3_DIR}"
./ns3 run "AstraSimNetwork \
    --workload-configuration=${WORKLOAD} \
    --system-configuration=${SYSTEM} \
    --network-configuration=${NETWORK} \
    --remote-memory-configuration=${REMOTE_MEMORY} \
    --logical-topology-configuration=${LOGICAL_TOPOLOGY} \
    --comm-group-configuration=\"empty\""

echo "[ASTRA-sim] Finished the execution."
