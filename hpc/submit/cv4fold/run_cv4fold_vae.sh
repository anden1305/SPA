#!/bin/bash
#BSUB -J cv4fold_vae
#BSUB -q gpuv100
#BSUB -W 16:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=5GB]"
#BSUB -o hpc/output/cv4fold/vae/%J.out
#BSUB -e hpc/output/cv4fold/vae/%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

# Full grid: 80 epochs, runs=3, validate every 10 (~8–14 h typical). Queue max 24h.
# Usage:
#   bsub hpc/submit/cv4fold/run_cv4fold_vae.sh src/config/run/cvaeprior/cv4fold/joint/fold_1/cgmvae.yaml

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "$SPA_ROOT"
mkdir -p hpc/output/cv4fold/vae

module load cuda/12.8.1
source .venv/bin/activate

CONFIG="${1:-${CONFIG:-}}"
CONFIG="${CONFIG:?Pass config yaml as first argument (or set CONFIG)}"

python3 main.py --method train_vae --config_path "${CONFIG}"

RUN_PREFIX=$(python3 -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['run_name'])")
RESULTS_BASE=$(python3 -c "import yaml; print(yaml.safe_load(open('${CONFIG}'))['results_dir'])")

LATEST=$(ls -td "${RESULTS_BASE}/${RUN_PREFIX}"_* 2>/dev/null | head -1 || true)
if [[ -n "${LATEST}" && -f "${LATEST}/config.json" ]]; then
  python3 -m scripts.cv4fold.postprocess_fold --result-root "${LATEST}"
  python3 -m scripts.cv4fold.select_best_vae_run --fold-dir "${LATEST}" --key "${LATEST}"
fi
