#!/bin/bash
### Lab 3 reliability comparison: cHMM-GMM (120 ep) vs existing decoder-only GMM reliability.
### Submit from login node only (jobs run on compute):
###   bash hpc/submit/hmmgmm/submit_reliability_lab3_compare.sh

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "$SPA_ROOT"

echo "=== Lab 3 reliability comparison ==="
echo ""
echo "Baseline (already in repo):"
echo "  config: src/config/run/cvaeprior/decoder_only/reliability/reliability_cgmvae_decoder_only_subject_conditioning.yaml"
echo "  results: results/decoder_only/reliability/cgmvae_decoder_only/"
echo "  submit:  bsub < hpc/submit/decoder_only/run_decoder_only_reliability.sh"
echo ""

echo "[1/1] cHMM-GMM lab 3 reliability (120 epochs, 10 runs)..."
JOB_CHMM=$(bsub < hpc/submit/hmmgmm/run_reliability_lab3_chmmgmvae.sh | sed -n 's/Job <\([^>]*\)>.*/\1/p')
echo "  Job $JOB_CHMM"
echo ""
echo "New results: results/hmmgmm/reliability/lab3/"
echo "Logs:        hpc/output/hmmgmm/reliability/lab3/"
