#!/usr/bin/env bash
#BSUB -J compress_json_xz
#BSUB -q hpc
#BSUB -W 02:00
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2000]"
#BSUB -o hpc/output/compress_json_xz_%J.out
#BSUB -e hpc/output/compress_json_xz_%J.err

set -euo pipefail

RESULTS_DIR=${1:-results}
WORKERS=${2:-${LSB_DJOB_NUMPROC:-1}}

if ! command -v xz >/dev/null 2>&1; then
  echo "ERROR: xz is not available on this node."
  exit 2
fi

if [ ! -d "$RESULTS_DIR" ]; then
  echo "ERROR: directory not found: $RESULTS_DIR"
  exit 3
fi

mkdir -p hpc/output

mapfile -d '' JSON_FILES < <(find "$RESULTS_DIR" -type f -name "*.json" -print0)
TOTAL=${#JSON_FILES[@]}

if [ "$TOTAL" -eq 0 ]; then
  echo "No .json files found under $RESULTS_DIR"
  exit 0
fi

echo "Found $TOTAL .json files under $RESULTS_DIR"
echo "Using $WORKERS parallel workers"
echo "Compressing to .json.xz (original .json files are removed by xz)"

printf '%s\0' "${JSON_FILES[@]}" | \
  xargs -0 -n1 -P "$WORKERS" -I{} bash -lc '
    f="$1"
    if [ -f "${f}.xz" ]; then
      echo "SKIP already compressed: ${f}.xz"
      exit 0
    fi
    xz -T1 -9 "$f"
    echo "OK   $f -> ${f}.xz"
  ' _ {}

echo "Done."
