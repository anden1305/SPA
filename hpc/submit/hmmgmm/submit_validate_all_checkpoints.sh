#!/bin/bash
### Submit validation job (runs on compute node via LSF, not login node).
set -euo pipefail
mkdir -p hpc/output/hmmgmm/validate
bsub < hpc/submit/hmmgmm/run_validate_all_checkpoints.sh
echo "Submitted. Monitor: bjobs"
