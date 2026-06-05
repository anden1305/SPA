#!/bin/bash
#BSUB -J prep_audit
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 2:00
#BSUB -o preprocessing_audit_%J.out
#BSUB -e preprocessing_audit_%J.err

set -euo pipefail
cd /work3/s204070/SPA
source .venv/bin/activate
export PYTHONPATH=.

python -m scripts.preprocessing_audit.run \
  --manifest data/manifests/cv_quality_cohort_v1.yaml \
  --config src/config/run/cvaeprior/cv4fold/templates/cgmvae_base.yaml \
  --out results/preprocessing_audit \
  --max-sequences 500
