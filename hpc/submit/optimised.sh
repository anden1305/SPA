#!/bin/bash
#BSUB -J cuda                             
#BSUB -q gpuv100                
#BSUB -W 00:10  
#BSUB -n 4
#BSUB -R "span[hosts=1]"                      
#BSUB -R "rusage[mem=1GB]"
#BSUB -o hpc/output/optimised_%J.out          
#BSUB -e hpc/output/optimised_%J.err     
#BSUB -gpu "num=1:mode=exclusive_process"  

# module purge
module load cuda/12.8.1
source .venv/bin/activate
mkdir -p output
mkdir -p results
python3 cuda.py
