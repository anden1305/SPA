#!/bin/bash
#BSUB -J spa_pytest
#BSUB -q hpc
#BSUB -W 0:30
#BSUB -n 2
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -o hpc/output/pytest/%J.out
#BSUB -e hpc/output/pytest/%J.err

# Run pytest on a compute node — never on the login node.
# Submit from login:  bsub < hpc/submit/run_pytest.sh
# Optional args via LSB_JOBNAME or edit PYTEST_ARGS below.

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "$SPA_ROOT"
mkdir -p hpc/output/pytest

source .venv/bin/activate

PYTEST_ARGS="${PYTEST_ARGS:-tests/ -q}"
python3 -m pytest ${PYTEST_ARGS}
