#!/bin/bash
#BSUB -J vae                             
#BSUB -q gpuv100            
#BSUB -W 4:00  
#BSUB -n 4
#BSUB -R "span[hosts=1]"                      
#BSUB -R "rusage[mem=5GB]"
#BSUB -o hpc/output/cvae_bidirectional_%J.out          
#BSUB -e hpc/output/cvae_bidirectional_%J.err     
#BSUB -gpu "num=1:mode=exclusive_process"  

# module purge
module load cuda/12.8.1
source .venv/bin/activate
mkdir -p output

# Run bidirectional cVAE with population-level generalization (leave-one-subject-out)
# Trains on many mice (run 1 only), tests on held-out mouse (runs 2-3)
# This provides the population scope Predictive NMI (generalization metric)

# Run for sub-039 as held-out
python3 main.py -m train_vae -c src/config/run/cvaemarhmm/bidirectional/generalization_cgmvae_bidirectional_mssv_frequency_test_sub039.yaml

# Run for sub-041 as held-out  
# python3 main.py -m train_vae -c src/config/run/cvaemarhmm/bidirectional/generalization_cgmvae_bidirectional_mssv_frequency_test_sub041.yaml

# Run for sub-048 as held-out
# python3 main.py -m train_vae -c src/config/run/cvaemarhmm/bidirectional/generalization_cgmvae_bidirectional_mssv_frequency_test_sub048.yaml
