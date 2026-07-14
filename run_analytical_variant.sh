#!/bin/bash
set -euo pipefail
set -x

SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_DIR="/app/astra-sim"

L2=""
L1=""
N=""
SKIP="0"

usage() {
    cat <<'EOF'
Usage: ./run_analytical_variant.sh --N <count> [--L1 <count>] [--L2 <count>] [--skip <number>]

Examples:
  ./run_analytical_variant.sh --N 2
  ./run_analytical_variant.sh --N 8 --L1 2 --L2 1 --skip 0
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --L2)
            L2="${2:-}"
            shift 2
            ;;
        --L1)
            L1="${2:-}"
            shift 2
            ;;
        --N)
            N="${2:-}"
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

if [[ -z "$N" ]]; then
    echo "Error: --N is required" >&2
    usage >&2
    exit 1
fi

if [[ ! "$N" =~ ^[0-9]+$ ]]; then
    echo "N must be an integer: $N" >&2
    exit 1
fi

if [[ ! "$SKIP" =~ ^[0-9]+$ ]]; then
    echo "SKIP must be an integer: $SKIP" >&2
    exit 1
fi

# Build topology filename
if [[ -n "$L2" ]]; then
    if [[ -z "$L1" ]]; then
        echo "Error: --L2 requires --L1" >&2
        exit 1
    fi
    TOPO_NAME="L2-${L2}_L1-${L1}_N-${N}"
elif [[ -n "$L1" ]]; then
    TOPO_NAME="L1-${L1}_N-${N}"
else
    TOPO_NAME="N-${N}"
fi

NPUS="$N"

TOPOLOGY_SRC="$SCRIPT_DIR/alt_topologies/analytical/${TOPO_NAME}.yml"
WORKLOAD_SRC="$SCRIPT_DIR/alt_workloads/skipreduce_${NPUS}_npus_s${SKIP}.xml"
WORKLOAD_DST_DIR="$SCRIPT_DIR/workload"
WORKLOAD_PREFIX="${WORKLOAD_DST_DIR}/skipreduce_npus"

if [[ ! -f "$TOPOLOGY_SRC" ]]; then
    echo "Topology file not found: $TOPOLOGY_SRC" >&2
    exit 1
fi

# Ensure the output directory for the generation script exists
mkdir -p "$SCRIPT_DIR/alt_workloads"

# Generate alt_workload dynamically using the script variables
NPUS="$NPUS" SKIP="$SKIP" python "$SCRIPT_DIR/skipreduce.py" "$NPUS" 1 --s "$SKIP" > "$WORKLOAD_SRC"

if [[ ! -f "$WORKLOAD_SRC" ]]; then
    echo "Workload file generation failed or file not found: $WORKLOAD_SRC" >&2
    exit 1
fi

mkdir -p "$WORKLOAD_DST_DIR"
cp "$TOPOLOGY_SRC" "$SCRIPT_DIR/topology.yml"
rm -f "${WORKLOAD_PREFIX}"*.et

python /app/chakra/collectiveapi/chakra_converter/et_converter.py \
    --input_filename "$WORKLOAD_SRC" \
    --output_filename "$WORKLOAD_PREFIX" \
    --num_npus "$NPUS"

"$SCRIPT_DIR/run_analytical.sh"
