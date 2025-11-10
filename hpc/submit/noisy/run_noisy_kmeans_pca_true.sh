#!/bin/bash
#BSUB -J run_noisy_kmeans_pca_true
#BSUB -q gpuv100            
#BSUB -W 02:00  
#BSUB -n 4
#BSUB -R "span[hosts=1]"                      
#BSUB -R "rusage[mem=4GB]"
#BSUB -o hpc/output/noisy/noisy_kmeans_pca_true_%J.out          
#BSUB -e hpc/output/noisy/noisy_kmeans_pca_true_%J.err     
#BSUB -gpu "num=1:mode=exclusive_process"  

module load cuda/12.8.1
source .venv/bin/activate
mkdir -p output/noisy/
python3 main.py --method train --config_path src/config/run/hmm/tests/noisy/kmeans_pca_noisy_true.yaml
