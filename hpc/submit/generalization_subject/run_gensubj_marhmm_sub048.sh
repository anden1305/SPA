#!/bin/bash
#BSUB -J gensubj_marhmm_sub048                             
#BSUB -q gpuv100            
#BSUB -W 2:00  
#BSUB -n 4
#BSUB -R "span[hosts=1]"                      
#BSUB -R "rusage[mem=5GB]"
#BSUB -o hpc/output/generalization_subject/marhmm_sub048/marhmm_sub048_%J.out          
#BSUB -e hpc/output/generalization_subject/marhmm_sub048/marhmm_sub048_%J.err     
#BSUB -gpu "num=1:mode=exclusive_process"  

# Generalization Subject experiment: MAR-HMM on sub-048
# Train on runs 1,2, validate on run 3
# Metrics: Within-subject generalization, train-val gap

module load cuda/12.8.1
source .venv/bin/activate
mkdir -p hpc/output/generalization_subject/marhmm_sub048

python3 main.py --method train --config_path src/config/run/marhmm/generalization_subject/generalization_subject_marhmm_sub048.yaml

# Usage:
# bsub < hpc/submit/generalization_subject/run_gensubj_marhmm_sub048.sh
