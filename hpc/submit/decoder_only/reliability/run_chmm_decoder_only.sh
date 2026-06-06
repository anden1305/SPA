#!/bin/bash
#BSUB -J do_rel_chmm_dec
#BSUB -q gpuv100
#BSUB -W 6:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -o hpc/output/decoder_only/reliability/chmmgmvae_decoder_only_%J.out
#BSUB -e hpc/output/decoder_only/reliability/chmmgmvae_decoder_only_%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
cd /work3/s204070/SPA
module load cuda/12.8.1
source .venv/bin/activate
mkdir -p hpc/output/decoder_only/reliability
python3 main.py -m train_vae -c src/config/run/cvaemarhmm/decoder_only/reliability/reliability_chmmgmvae_decoder_only_mssv_frequency.yaml
