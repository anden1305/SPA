#!/bin/bash
#BSUB -J ds006366_process
#BSUB -W 4:00
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -cwd /work3/s204070/SPA
#BSUB -o hpc/output/ds006366_process_%J.out
#BSUB -e hpc/output/ds006366_process_%J.err

set -euo pipefail

mkdir -p hpc/output
source .venv/bin/activate
python3 scripts/data_processing/data_processing.py