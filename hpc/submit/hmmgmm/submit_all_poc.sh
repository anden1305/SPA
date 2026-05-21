#!/bin/bash
### Submit cHMMGMVAE POC + GMM-seq64 baseline together.

set -euo pipefail

echo "Submitting cHMMGMVAE POC (sub-039)..."
bsub < hpc/submit/hmmgmm/run_poc_sub039.sh

echo "Submitting GMM seq64 baseline (sub-039)..."
bsub < hpc/submit/hmmgmm/run_poc_sub039_baseline_gmm.sh

echo "Done. Monitor with: bjobs"
