#!/bin/bash
#BSUB -J hmmgmm_poc_stable
#BSUB -q gpuv100
#BSUB -W 3:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=5GB]"
#BSUB -o hpc/output/hmmgmm/poc/sub039_stable/%J.out
#BSUB -e hpc/output/hmmgmm/poc/sub039_stable/%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "$SPA_ROOT"
mkdir -p hpc/output/hmmgmm/poc/sub039_stable

module load cuda/12.8.1
source .venv/bin/activate

python3 main.py --method train_vae \
  --config_path src/config/run/cvaeprior/hmmgmm/ablations/sub039_chmmgmvae_poc_stable.yaml
