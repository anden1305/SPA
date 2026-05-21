#!/bin/bash
#BSUB -J hmmgmm_poc_smoke
#BSUB -q gpuv100
#BSUB -W 0:30
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=5GB]"
#BSUB -o hpc/output/hmmgmm/poc/sub039_smoke/%J.out
#BSUB -e hpc/output/hmmgmm/poc/sub039_smoke/%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

# Quick gate: 3 epochs, K=3, crosses warm-schedule phases. Fails fast if broken.

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "$SPA_ROOT"
mkdir -p hpc/output/hmmgmm/poc/sub039_smoke

module load cuda/12.8.1
source .venv/bin/activate

python3 main.py --method train_vae \
  --config_path src/config/run/cvaeprior/hmmgmm/poc/sub039_chmmgmvae_smoke.yaml
