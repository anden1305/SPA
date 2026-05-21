#!/bin/bash
### Fair GMM vs cHMMGMVAE comparison (matched hyperparams, sub-039, seq64).
### Submit from login node:  bash hpc/submit/hmmgmm/submit_poc_compare.sh

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "$SPA_ROOT"

echo "=== sub-039 seq64: gmm vs warm_hmm_gmm (generalization-matched beta) ==="
echo ""

echo "[1/2] GMM baseline (prior: gmm)..."
JOB_GMM=$(bsub < hpc/submit/hmmgmm/run_poc_sub039_baseline_gmm.sh | sed -n 's/Job <\([^>]*\)>.*/\1/p')
echo "  Job $JOB_GMM"

echo "[2/2] cHMMGMVAE (prior: warm_hmm_gmm)..."
JOB_CHMM=$(bsub < hpc/submit/hmmgmm/run_poc_sub039.sh | sed -n 's/Job <\([^>]*\)>.*/\1/p')
echo "  Job $JOB_CHMM"

echo ""
echo "Results: results/hmmgmm/poc/sub039_cgmvae_gmm_seq64_baseline/"
echo "         results/hmmgmm/poc/sub039_chmmgmvae_poc/"
echo "Optional warm_gmm fallback: bsub < hpc/submit/hmmgmm/run_poc_sub039_warm_gmm.sh"
