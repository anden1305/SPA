#!/bin/bash
#BSUB -J cuda                             
#BSUB -q gpuv100            
#BSUB -W 1:00  
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
# python3 main.py -m train_vae -c src/config/run/cvaemarhmm/reliability/cvae_mar_reliability_1.yaml
# python3 main.py -m train_vae -c src/config/run/cvaemarhmm/substages_vae/substages_vae_test_2.yaml
# python3 main.py -m train_vae -c src/config/run/cvaemarhmm/substages_marhmm/substages_marhmm_9.yaml
python3 main.py -m validate_cvae_gmm -c src/config/run/cvaemarhmm/substages_gmm/substages_gmm_population_6.yaml