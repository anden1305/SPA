#!/bin/bash
#BSUB -J cv4fold_hmm
#BSUB -q gpuv100
#BSUB -W 24:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=5GB]"
#BSUB -o hpc/output/cv4fold/hmm_raw/%J.out
#BSUB -e hpc/output/cv4fold/hmm_raw/%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

# Full grid: thesis-scale 30k epochs, validate every 1000 — request queue max (24h).
# May need checkpoint resume if a fold does not finish in one slot.
# Usage:
#   bsub hpc/submit/cv4fold/run_cv4fold_hmm_raw.sh src/config/run/hmm/cv4fold/joint/fold_1/hmm_raw.yaml

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "$SPA_ROOT"
mkdir -p hpc/output/cv4fold/hmm_raw

module load cuda/12.8.1
source .venv/bin/activate

CONFIG="${1:-${CONFIG:-}}"
CONFIG="${CONFIG:?Pass config yaml as first argument (or set CONFIG)}"

python3 main.py --method train --config_path "${CONFIG}"

RUN_PREFIX=$(python3 -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['run_name'])")
RESULTS_BASE=$(python3 -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['results_dir'])")

LATEST=$(ls -td "${RESULTS_BASE}/${RUN_PREFIX}"_* 2>/dev/null | head -1 || true)
if [[ -n "${LATEST}" && -f "${LATEST}/config.json" ]]; then
  python3 -m scripts.cv4fold.postprocess_fold --result-root "${LATEST}"
fi
