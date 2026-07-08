#!/bin/bash
set -euo pipefail
set -x

SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_DIR="/app/astra-sim"
NS3_DIR="${PROJECT_DIR}/extern/network_backend/ns-3"

TOPOLOGY=""
NPUS=""
SKIP="0"

usage() {
    cat <<'EOF'
Usage: ./run_ns3_variant.sh --topology <path> --npus <count> [--skip <number>]

Examples:
  ./run_ns3_variant.sh --topology "alt_topologies/ns3/N-2" --npus 2
  ./run_ns3_variant.sh --topology "alt_topologies/ns3/N-2_network.txt" --npus 2 --skip 0
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --topology)
            TOPOLOGY="${2:-}"
            shift 2
            ;;
        --npus)
            NPUS="${2:-}"
            shift 2
            ;;
        --skip)
            SKIP="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -z "$TOPOLOGY" || -z "$NPUS" ]]; then
    usage >&2
    exit 1
fi

if [[ ! "$NPUS" =~ ^[0-9]+$ ]]; then
    echo "NPUS must be an integer: $NPUS" >&2
    exit 1
fi

if [[ ! "$SKIP" =~ ^[0-9]+$ ]]; then
    echo "SKIP must be an integer: $SKIP" >&2
    exit 1
fi

# --- 1. SET WORKLOAD PATHS (YOUR EXACT FORMAT) ---
TOPOLOGY_SRC="$SCRIPT_DIR/$TOPOLOGY"
WORKLOAD_SRC="$SCRIPT_DIR/alt_workloads/skipreduce_${NPUS}_npus_s${SKIP}.xml"
WORKLOAD_DST_DIR="$SCRIPT_DIR/workload"
WORKLOAD_PREFIX="${WORKLOAD_DST_DIR}/skipreduce_npus"

# --- 2. RESOLVE NS-3 TOPOLOGY FILE SUFFIXES ---
# Strips suffixes if user passes full filename, e.g., N-2_network.txt -> N-2
TOPOLOGY_BASE="${TOPOLOGY_SRC%_network.txt}"
TOPOLOGY_BASE="${TOPOLOGY_BASE%_logical.json}"

NETWORK_FILE="${TOPOLOGY_BASE}_network.txt"
LOGICAL_FILE="${TOPOLOGY_BASE}_logical.json"

if [[ ! -f "$NETWORK_FILE" ]]; then
    echo "ns-3 Network config file not found: $NETWORK_FILE" >&2
    exit 1
fi

if [[ ! -f "$LOGICAL_FILE" ]]; then
    echo "ns-3 Logical topology config file not found: $LOGICAL_FILE" >&2
    exit 1
fi

# --- 3. GENERATE WORKLOAD XML ---
mkdir -p "$SCRIPT_DIR/alt_workloads"

NPUS="$NPUS" SKIP="$SKIP" python "$SCRIPT_DIR/skipreduce.py" "$NPUS" 1 --s "$SKIP" > "$WORKLOAD_SRC"

if [[ ! -f "$WORKLOAD_SRC" ]]; then
    echo "Workload file generation failed: $WORKLOAD_SRC" >&2
    exit 1
fi

# --- 4. CONVERT XML TO CHAKRA TRACES ---
mkdir -p "$WORKLOAD_DST_DIR"
rm -f "${WORKLOAD_PREFIX}"*.et

python /app/chakra/collectiveapi/chakra_converter/et_converter.py \
    --input_filename "$WORKLOAD_SRC" \
    --output_filename "$WORKLOAD_PREFIX" \
    --num_npus "$NPUS"

# --- 5. DYNAMICALLY GENERATE CUSTOM_COLLECTIVE.JSON ---
# Single string array entry pointing to WORKLOAD_PREFIX as expected by ASTRA-sim
SYSTEM_FILE="${SCRIPT_DIR}/custom_collective.json"
cat <<EOF > "$SYSTEM_FILE"
{
  "scheduling-policy": "FIFO",
  "preferred-dataset-splits": 1,
  "all-reduce-implementation-custom": [
    "${WORKLOAD_PREFIX}"
  ],
  "local-mem-bw": 50
}
EOF

# --- 5.5 ENSURE NS-3 OUTPUT FILES AND NETWORK CONFIG GENERATED ---
# Ensure the scratch/output/ directory and default flow/trace files exist
mkdir -p "${NS3_DIR}/scratch/output"
if [[ ! -f "${NS3_DIR}/scratch/output/flow.txt" ]]; then
    echo "0" > "${NS3_DIR}/scratch/output/flow.txt"
fi
if [[ ! -f "${NS3_DIR}/scratch/output/trace.txt" ]]; then
    echo -e "2\n0 1" > "${NS3_DIR}/scratch/output/trace.txt"
fi

NETWORK_CONFIG_FILE="${SCRIPT_DIR}/custom_network_config.txt"
cat <<EOF > "$NETWORK_CONFIG_FILE"
ENABLE_QCN 1
USE_DYNAMIC_PFC_THRESHOLD 1
PACKET_PAYLOAD_SIZE 1000
TOPOLOGY_FILE ${NETWORK_FILE}
FLOW_FILE ${NS3_DIR}/scratch/output/flow.txt
TRACE_FILE ${NS3_DIR}/scratch/output/trace.txt
TRACE_OUTPUT_FILE ${NS3_DIR}/scratch/output/mix.tr
FCT_OUTPUT_FILE ${NS3_DIR}/scratch/output/fct.txt
PFC_OUTPUT_FILE ${NS3_DIR}/scratch/output/pfc.txt
QLEN_MON_FILE ${NS3_DIR}/scratch/output/qlen.txt
QLEN_MON_START 0
QLEN_MON_END 20000
SIMULATOR_STOP_TIME 40000000000000.00
CC_MODE 12
ALPHA_RESUME_INTERVAL 1
RATE_DECREASE_INTERVAL 4
CLAMP_TARGET_RATE 0
RP_TIMER 900 
EWMA_GAIN 0.00390625
FAST_RECOVERY_TIMES 1
RATE_AI 50Mb/s
RATE_HAI 100Mb/s
MIN_RATE 100Mb/s
DCTCP_RATE_AI 1000Mb/s
ERROR_RATE_PER_LINK 0.0000
L2_CHUNK_SIZE 4000
L2_ACK_INTERVAL 1
L2_BACK_TO_ZERO 0
HAS_WIN 1
GLOBAL_T 0
VAR_WIN 1
FAST_REACT 1
U_TARGET 0.95
MI_THRESH 0
INT_MULTI 1
MULTI_RATE 0
SAMPLE_FEEDBACK 0
PINT_LOG_BASE 1.05
PINT_PROB 1.0
NIC_TOTAL_PAUSE_TIME 0
RATE_BOUND 1
ACK_HIGH_PRIO 0
LINK_DOWN 0 0 0
ENABLE_TRACE 1
KMAX_MAP 6 25000000000 400 40000000000 800 100000000000 1600 200000000000 2400 400000000000 3200 2400000000000 3200
KMIN_MAP 6 25000000000 100 40000000000 200 100000000000 400 200000000000 600 400000000000 800 2400000000000 800
PMAX_MAP 6 25000000000 0.2 40000000000 0.2 100000000000 0.2 200000000000 0.2 400000000000 0.2 2400000000000 0.2
BUFFER_SIZE 32
EOF

REMOTE_MEMORY="${PROJECT_DIR}/examples/remote_memory/analytical/no_memory_expansion.json"

# --- 6. RUN SIMULATION VIA NS-3 ---
echo "[ASTRA-sim] Launching SkipReduce (N=${NPUS}, Skip=${SKIP}) with ns-3 Backend..."
echo "  Workload Prefix: $WORKLOAD_PREFIX"
echo "  Network Config:  $NETWORK_CONFIG_FILE"
echo "  Logical Config:  $LOGICAL_FILE"

cd "${NS3_DIR}"
./ns3 run "AstraSimNetwork \
    --workload-configuration=${WORKLOAD_PREFIX} \
    --system-configuration=${SYSTEM_FILE} \
    --network-configuration=${NETWORK_CONFIG_FILE} \
    --remote-memory-configuration=${REMOTE_MEMORY} \
    --logical-topology-configuration=${LOGICAL_FILE} \
    --comm-group-configuration=\"empty\""

echo "[ASTRA-sim] Finished execution."