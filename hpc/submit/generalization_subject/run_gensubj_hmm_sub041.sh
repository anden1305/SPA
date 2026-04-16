#!/bin/bash
#BSUB -J gensubj_hmm_sub041                             
#BSUB -q gpul40s            
#BSUB -W 2:00  
#BSUB -n 8
#BSUB -R "span[hosts=1]"                      
#BSUB -R "rusage[mem=5GB]"
#BSUB -o hpc/output/generalization_subject/hmm_sub041/hmm_sub041_%J.out          
#BSUB -e hpc/output/generalization_subject/hmm_sub041/hmm_sub041_%J.err     
#BSUB -gpu "num=2:mode=exclusive_process"  

# Generalization Subject experiment: HMM on sub-041
# Train on runs 1,2, validate on run 3
# Metrics: Within-subject generalization, train-val gap

module load cuda/12.8.1
source .venv/bin/activate
mkdir -p hpc/output/generalization_subject/hmm_sub041

python3 main.py --method train --config_path src/config/run/hmm/generalization_subject/generalization_subject_hmm_sub041.yaml

# Usage:
# bsub < hpc/submit/generalization_subject/run_gensubj_hmm_sub041.sh
