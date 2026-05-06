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
# Runs train_vae then validate_cvae_gmm for the reliability config.

set -euo pipefail

module load cuda/12.8.1
source .venv/bin/activate
mkdir -p hpc/output/decoder_only/reliability

CONFIG="src/config/run/cvaeprior/decoder_only/reliability/reliability_cgmvae_decoder_only_mssv_frequency.yaml"

python3 main.py --method train_vae --config_path ${CONFIG}
python3 main.py --method validate_cvae_gmm --config_path ${CONFIG}
