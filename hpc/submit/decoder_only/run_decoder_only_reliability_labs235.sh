#!/bin/bash
#BSUB -J dec_only_rel_l235
#BSUB -q gpuv100
#BSUB -W 12:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=12GB]"
#BSUB -o hpc/output/decoder_only/reliability/labs235/%J.out
#BSUB -e hpc/output/decoder_only/reliability/labs235/%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

# Decoder-only cGMVAE reliability: labs 2 + 3 + 5 pooled (20 mice, in-sample).
# train_vae only: per-run checkpoints + GMM validation each seed.

set -euo pipefail

SPA_ROOT="/work3/s204070/SPA"
cd "$SPA_ROOT"

module load cuda/12.8.1
source .venv/bin/activate
mkdir -p hpc/output/decoder_only/reliability/labs235

CONFIG="src/config/run/cvaeprior/decoder_only/reliability/reliability_cgmvae_decoder_only_labs235.yaml"

python3 main.py --method train_vae --config_path "${CONFIG}"
python3 main.py --method validate_cvae_gmm --config_path "${CONFIG}"
