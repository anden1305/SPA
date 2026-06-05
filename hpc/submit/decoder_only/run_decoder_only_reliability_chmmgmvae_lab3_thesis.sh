#!/bin/bash
#BSUB -J dec_chmm_lab3
#BSUB -q gpuv100
#BSUB -W 16:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=12GB]"
#BSUB -o hpc/output/decoder_only/reliability/chmmgmvae_lab3/%J.out
#BSUB -e hpc/output/decoder_only/reliability/chmmgmvae_lab3/%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

# Thesis-style lab_3 cHMMGMVAE (compare to cGMVAE Prior Decoder table row).
# train_vae: 10 seeds; HMM Viterbi validation each run (warm_hmm_gmm).
#
# Submit from repo root:
#   bsub < hpc/submit/decoder_only/run_decoder_only_reliability_chmmgmvae_lab3_thesis.sh

set -euo pipefail

SPA_ROOT="/work3/s204070/SPA"
cd "$SPA_ROOT"

module load cuda/12.8.1
source .venv/bin/activate
mkdir -p hpc/output/decoder_only/reliability/chmmgmvae_lab3

CONFIG="src/config/run/cvaeprior/decoder_only/reliability/reliability_chmmgmvae_decoder_only_lab3_thesis_style.yaml"

python3 main.py --method train_vae --config_path "${CONFIG}"
