#!/bin/bash
#BSUB -J dec_only_l2_rel
#BSUB -q gpuv100
#BSUB -W 10:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -o hpc/output/decoder_only_lab_2/reliability/%J.out
#BSUB -e hpc/output/decoder_only_lab_2/reliability/%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

# Lab-2 reliability (6 quality subjects, 2 runs each, subject conditioning).

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "$SPA_ROOT"

module load cuda/12.8.1
source .venv/bin/activate
mkdir -p hpc/output/decoder_only_lab_2/reliability

CONFIG="src/config/run/cvaeprior/decoder_only_lab_2_subject_conditioning/reliability/reliability_cgmvae_decoder_only_subject_conditioning.yaml"

python3 main.py --method train_vae --config_path "${CONFIG}"
python3 main.py --method validate_cvae_gmm --config_path "${CONFIG}"
