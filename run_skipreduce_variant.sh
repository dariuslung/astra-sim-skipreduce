#!/bin/bash
set -euo pipefail
set -x

SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_DIR="/app/astra-sim"

TOPOLOGY=""
NPUS=""
SKIP="s0"

usage() {
    cat <<'EOF'
Usage: run_skipreduce_variant.sh --topology <path> --npus <count> [--skip <suffix>]

Examples:
  ./run_skipreduce_variant.sh --topology "alt_topologies/L1x2 Nx4.yml" --npus 4
  ./run_skipreduce_variant.sh --topology "alt_topologies/L2x1 L1x2 Nx8.yml" --npus 8 --skip s0
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

TOPOLOGY_SRC="$SCRIPT_DIR/$TOPOLOGY"
WORKLOAD_SRC="$SCRIPT_DIR/alt_workloads/skipreduce_${NPUS}_npus_${SKIP}.xml"
WORKLOAD_DST_DIR="$SCRIPT_DIR/workload"
WORKLOAD_PREFIX="${WORKLOAD_DST_DIR}/skipreduce_npus"

if [[ ! -f "$TOPOLOGY_SRC" ]]; then
    echo "Topology file not found: $TOPOLOGY_SRC" >&2
    exit 1
fi

if [[ ! -f "$WORKLOAD_SRC" ]]; then
    echo "Workload file not found: $WORKLOAD_SRC" >&2
    exit 1
fi

mkdir -p "$WORKLOAD_DST_DIR"
cp "$TOPOLOGY_SRC" "$SCRIPT_DIR/topology.yml"
rm -f "${WORKLOAD_PREFIX}"*.et

python /app/chakra/collectiveapi/chakra_converter/et_converter.py \
    --input_filename "$WORKLOAD_SRC" \
    --output_filename "$WORKLOAD_PREFIX"

python "$SCRIPT_DIR/patcher.py" \
    --prefix "$WORKLOAD_PREFIX" \
    --npus "$NPUS"

"$SCRIPT_DIR/run_analytical_skipreduce.sh"
