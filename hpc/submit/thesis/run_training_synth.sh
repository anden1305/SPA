#!/bin/bash
#BSUB -J no_lr_mssv_features                    
#BSUB -q gpuv100            
#BSUB -W 08:00  
#BSUB -n 4
#BSUB -R "span[hosts=1]"                      
#BSUB -R "rusage[mem=4GB]"
#BSUB -o hpc/output/no_lr_mssv_features%J.out          
#BSUB -e hpc/output/no_lr_mssv_features%J.err     
#BSUB -gpu "num=1:mode=exclusive_process"  

# module purge
module load cuda/12.8.1
source .venv/bin/activate
mkdir -p hpc/output
python3 main.py --config src/config/run/hmm/tests/no_lr_mssv_features.yaml
