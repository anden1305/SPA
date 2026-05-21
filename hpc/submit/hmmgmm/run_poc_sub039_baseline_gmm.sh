#!/bin/bash
#BSUB -J hmmgmm_poc_sub039_gmm
#BSUB -q gpuv100
#BSUB -W 3:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=5GB]"
#BSUB -o hpc/output/hmmgmm/poc/sub039_gmm/%J.out
#BSUB -e hpc/output/hmmgmm/poc/sub039_gmm/%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

# Plain gmm prior (generalization hyperparams) at sequence_length=64 vs cHMMGMVAE POC.

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "$SPA_ROOT"
mkdir -p hpc/output/hmmgmm/poc/sub039_gmm

module load cuda/12.8.1
source .venv/bin/activate

python3 main.py --method train_vae \
  --config_path src/config/run/cvaeprior/hmmgmm/poc/sub039_cgmvae_gmm_seq64_baseline.yaml

# Usage:
# bsub < hpc/submit/hmmgmm/run_poc_sub039_baseline_gmm.sh
