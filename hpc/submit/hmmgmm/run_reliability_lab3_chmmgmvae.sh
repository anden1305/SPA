#!/bin/bash
#BSUB -J hmmgmm_lab3_rel
#BSUB -q gpuv100
#BSUB -W 15:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -o hpc/output/hmmgmm/reliability/lab3/%J.out
#BSUB -e hpc/output/hmmgmm/reliability/lab3/%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

# Lab 3 reliability: cHMM-GMM VAE, 120 epochs, 10 runs (compare to decoder_only reliability).

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "$SPA_ROOT"
mkdir -p hpc/output/hmmgmm/reliability/lab3

module load cuda/12.8.1
source .venv/bin/activate

CONFIG="src/config/run/cvaeprior/hmmgmm/reliability/lab3_chmmgmvae_decoder_only_subject_conditioning.yaml"

python3 main.py --method train_vae --config_path "${CONFIG}"
python3 main.py --method validate_cvae_hmm --config_path "${CONFIG}"

# Usage:
# bsub < hpc/submit/hmmgmm/run_reliability_lab3_chmmgmvae.sh
