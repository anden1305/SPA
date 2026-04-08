#!/bin/bash
#BSUB -J vae                             
#BSUB -q gpuv100            
#BSUB -W 12:00  
#BSUB -n 4
#BSUB -R "span[hosts=1]"                      
#BSUB -R "rusage[mem=4GB]"
#BSUB -o hpc/output/cvae_baseline_%J.out          
#BSUB -e hpc/output/cvae_baseline_%J.err     
#BSUB -gpu "num=1:mode=exclusive_process"  

# module purge
module load cuda/12.8.1
source .venv/bin/activate
mkdir -p output
python3 main.py -m train_vae -c src/config/run/cvaemarhmm/final/cvae_final.yaml
