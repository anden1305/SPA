#!/bin/bash
#BSUB -J cvae_test_run                             
#BSUB -q gpuv100            
#BSUB -W 01:00  
#BSUB -n 4
#BSUB -R "span[hosts=1]"                      
#BSUB -R "rusage[mem=4GB]"
#BSUB -o hpc/output/cvae_test_run_%J.out          
#BSUB -e hpc/output/cvae_test_run_%J.err     
#BSUB -gpu "num=1:mode=exclusive_process"  

# module purge
module load cuda/12.8.1
source .venv/bin/activate
mkdir -p output
python main.py -m train_vae -c src/config/run/cvaemarhmm/tests/cvae_test.yaml
