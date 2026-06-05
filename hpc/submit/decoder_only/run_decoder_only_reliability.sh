#!/bin/bash
#BSUB -J dec_only_reliability
#BSUB -q gpuv100
#BSUB -W 10:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -o hpc/output/decoder_only/reliability/%J.out
#BSUB -e hpc/output/decoder_only/reliability/%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

# Decoder-only reliability run (population)
# train_vae: 10 seeds, per-run checkpoints under {run_name}/{1..10}/checkpoints/, GMM validate each run.

set -euo pipefail

SPA_ROOT="/work3/s204070/SPA"
cd "$SPA_ROOT"

module load cuda/12.8.1
source .venv/bin/activate
mkdir -p hpc/output/decoder_only/reliability

CONFIG="src/config/run/cvaeprior/decoder_only/reliability/reliability_cgmvae_decoder_only_subject_conditioning.yaml"

python3 main.py --method train_vae --config_path ${CONFIG}
