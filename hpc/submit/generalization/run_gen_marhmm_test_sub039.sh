#!/bin/bash
#BSUB -J gen_marhmm_sub039                             
#BSUB -q gpuv100            
#BSUB -W 12:00  
#BSUB -n 4
#BSUB -R "span[hosts=1]"                      
#BSUB -R "rusage[mem=5GB]"
#BSUB -o hpc/output/generalization/marhmm_test_sub039/marhmm_test_sub039_%J.out        
#BSUB -e hpc/output/generalization/marhmm_test_sub039/marhmm_test_sub039_%J.err     
#BSUB -gpu "num=1:mode=exclusive_process"  

# Generalization experiment: MAR-HMM on MSSV Features - Test on sub-039
# Train on all mice except sub-039, validate on sub-039 runs 1,2,3
# Metrics: Validation NMI, validation predictive likelihood, generalization gap

module load cuda/12.8.1
source .venv/bin/activate
mkdir -p hpc/output/generalization/marhmm_test_sub039

python3 main.py --method train --config_path src/config/run/marhmm/generalization/generalization_marhmm_mssv_features_test_sub039.yaml

# Usage:
# bsub < hpc/submit/generalization/run_gen_marhmm_test_sub039.sh
