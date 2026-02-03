#!/bin/bash
#BSUB -J reliability_hmm_synth_raw                             
#BSUB -q gpul40s            
#BSUB -W 24:00  
#BSUB -n 4
#BSUB -R "span[hosts=1]"                      
#BSUB -R "rusage[mem=5GB]"
#BSUB -o hpc/output/reliability/hmm_synth_raw/hmm_synth_raw_%J.out          
#BSUB -e hpc/output/reliability/hmm_synth_raw/hmm_synth_raw_%J.err     
#BSUB -gpu "num=1:mode=exclusive_process"  

# Reliability experiment: HMM on Synthetic RAW (no features)
# Runs the model 10 times with different seeds to test consistency
# Metrics: cross-NMI (consistency), NMI vs true states, loss distribution

module load cuda/12.8.1
source .venv/bin/activate
mkdir -p hpc/output/reliability/hmm_synth_raw

python3 main.py --method train --config_path src/config/run/hmm/reliability/reliability_hmm_synth_raw.yaml

# Usage:
# bsub < hpc/submit/reliability/run_reliability_hmm_synth_raw.sh
