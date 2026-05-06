#!/bin/bash
#BSUB -J run_full_mssv_features                 
#BSUB -q gpuv100            
#BSUB -W 01:00  
#BSUB -n 4
#BSUB -R "span[hosts=1]"                      
#BSUB -R "rusage[mem=4GB]"
#BSUB -o hpc/output/full_mssv_features_%J.out          
#BSUB -e hpc/output/full_mssv_features_%J.err     
#BSUB -gpu "num=1:mode=exclusive_process"  

# module purge
module load cuda/12.8.1
source .venv/bin/activate
python3 main.py --config src/config/run/hmm/tests/diagvsfull/test_full_mssv_features.yaml
