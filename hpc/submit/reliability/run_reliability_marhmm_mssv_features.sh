#!/bin/bash
#BSUB -J reliability_marhmm_mssv_features                             
#BSUB -q gpuv100            
#BSUB -W 6:00  
#BSUB -n 4
#BSUB -R "span[hosts=1]"                      
#BSUB -R "rusage[mem=5GB]"
#BSUB -o hpc/output/reliability/marhmm_mssv_features/marhmm_mssv_features_%J.out         
#BSUB -e hpc/output/reliability/marhmm_mssv_features/marhmm_mssv_features_%J.err     
#BSUB -gpu "num=1:mode=exclusive_process"  

# Reliability experiment: MAR-HMM on MSSV Features
# Runs the model 10 times with different seeds to test consistency
# Metrics: cross-NMI (consistency), NMI vs true states, loss distribution

module load cuda/12.8.1
source .venv/bin/activate
mkdir -p hpc/output/reliability/marhmm_mssv_features

python3 main.py --method train --config_path src/config/run/marhmm/reliability/reliability_marhmm_mssv_features.yaml

# Usage:
# bsub < hpc/submit/reliability/run_reliability_marhmm_mssv_features.sh
