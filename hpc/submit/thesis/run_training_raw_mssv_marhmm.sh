#!/bin/bash
#BSUB -J raw_mssv_marhmm          
#BSUB -q gpuv100            
#BSUB -W 4:00  
#BSUB -n 4
#BSUB -R "span[hosts=1]"                      
#BSUB -R "rusage[mem=8GB]"
#BSUB -o hpc/output/raw_mssv_marhmm%J.out          
#BSUB -e hpc/output/raw_mssv_marhmm%J.err     
#BSUB -gpu "num=1:mode=exclusive_process"  

# module purge
module load cuda/12.8.1
source .venv/bin/activate
mkdir -p hpc/output
python3 main.py --config src/config/run/marhmm/experiments/experiment_mssv_raw.yaml