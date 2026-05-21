#!/bin/bash
### POC verification: smoke + fair gmm vs hmmgmm pair + stable ablation.
### Submit from login node:  bash hpc/submit/hmmgmm/submit_poc_verify.sh

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "$SPA_ROOT"

echo "=== HMM-GMM POC verification (K=3, sub-039, generalization-matched beta) ==="
echo ""

echo "[1/4] Smoke (3 epochs)..."
JOB_SMOKE=$(bsub < hpc/submit/hmmgmm/run_poc_sub039_smoke.sh | sed -n 's/Job <\([^>]*\)>.*/\1/p')
echo "  Job $JOB_SMOKE"

echo "[2/4] GMM seq64 baseline (prior: gmm, same beta as generalization)..."
JOB_GMM=$(bsub < hpc/submit/hmmgmm/run_poc_sub039_baseline_gmm.sh | sed -n 's/Job <\([^>]*\)>.*/\1/p')
echo "  Job $JOB_GMM"

echo "[3/4] cHMMGMVAE POC (prior: warm_hmm_gmm, matched beta)..."
JOB_CHMM=$(bsub < hpc/submit/hmmgmm/run_poc_sub039.sh | sed -n 's/Job <\([^>]*\)>.*/\1/p')
echo "  Job $JOB_CHMM"

echo "[4/4] cHMMGMVAE stable ablation (conservative schedule if pair is unstable)..."
JOB_STABLE=$(bsub < hpc/submit/hmmgmm/run_poc_sub039_stable.sh | sed -n 's/Job <\([^>]*\)>.*/\1/p')
echo "  Job $JOB_STABLE"

echo ""
echo "Submitted. Monitor:  bjobs -w"
echo "Logs:              hpc/output/hmmgmm/poc/"
echo ""
echo "Fair pair only:    bash hpc/submit/hmmgmm/submit_poc_compare.sh"
echo "warm_gmm fallback: bsub < hpc/submit/hmmgmm/run_poc_sub039_warm_gmm.sh"
