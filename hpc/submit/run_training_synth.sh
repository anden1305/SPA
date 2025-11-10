#!/bin/bash
#BSUB -J run_training_synth                    
#BSUB -q gpuv100            
#BSUB -W 01:00  
#BSUB -n 4
#BSUB -R "span[hosts=1]"                      
#BSUB -R "rusage[mem=4GB]"
#BSUB -o hpc/output/test_run_synth%J.out          
#BSUB -e hpc/output/test_run_synth%J.err     
#BSUB -gpu "num=1:mode=exclusive_process"  

# module purge
module load cuda/12.8.1
source .venv/bin/activate
mkdir -p output
python3 main.py --config src/config/run/hmm/experiments/experiment_synth_features.yaml 
