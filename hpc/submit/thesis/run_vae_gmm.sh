#!/bin/bash
#BSUB -J cuda                             
#BSUB -q gpuv100            
#BSUB -W 24:00  
#BSUB -n 4
#BSUB -R "span[hosts=1]"                      
#BSUB -R "rusage[mem=4GB]"
#BSUB -o hpc/output/test_run_%J.out          
#BSUB -e hpc/output/test_run_%J.err     
#BSUB -gpu "num=1:mode=exclusive_process"  

# module purge
module load cuda/12.8.1
source .venv/bin/activate
mkdir -p output
python3 main.py -m train_vae -c src/config/run/cvaemarhmm/tests/cvae_hpc_low_lr_few_epochs_gmm_prior.yaml
