#!/bin/bash
set -e
set -x

# find the absolute path to this script
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_DIR="/app/astra-sim"
EXAMPLE_DIR="${PROJECT_DIR:?}/examples"

# paths
# ASTRA_SIM="${PROJECT_DIR:?}/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware"
ASTRA_SIM="${PROJECT_DIR:?}/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware"
WORKLOAD="${SCRIPT_DIR:?}/workload/skip_reduce_8_gpus"
SYSTEM="${SCRIPT_DIR:?}/custom_collective.json"
NETWORK="${SCRIPT_DIR:?}/topology.yml"
REMOTE_MEMORY="${EXAMPLE_DIR:?}/remote_memory/analytical/no_memory_expansion.json"

# start
echo "[ASTRA-sim] Compiling ASTRA-sim with the Analytical Network Backend..."
echo ""

# Compile
"${PROJECT_DIR:?}"/build/astra_analytical/build.sh

echo "[ASTRA-sim] Compilation finished."
echo "[ASTRA-sim] Running ASTRA-sim SkipReduce with Analytical Network Backend..."

# run ASTRA-sim
"${ASTRA_SIM:?}" \
    --workload-configuration="${WORKLOAD}" \
    --system-configuration="${SYSTEM:?}" \
    --remote-memory-configuration="${REMOTE_MEMORY:?}" \
    --network-configuration="${NETWORK:?}"

# finalize
echo "[ASTRA-sim] Finished the execution."
