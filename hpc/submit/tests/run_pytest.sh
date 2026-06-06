#!/bin/bash
#BSUB -J spa_pytest
#BSUB -q hpc
#BSUB -W 0:30
#BSUB -n 1
#BSUB -R "rusage[mem=4GB]"
#BSUB -o hpc/output/tests/pytest_%J.out
#BSUB -e hpc/output/tests/pytest_%J.err

set -euo pipefail
cd /work3/s204070/SPA
source .venv/bin/activate
mkdir -p hpc/output/tests
export PYTHONPATH=/work3/s204070/SPA
pytest tests/test_hmm_gmm_prior.py tests/test_warm_hmm_gmm_smoke.py -q
