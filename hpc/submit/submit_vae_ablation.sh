#!/bin/bash
set -euo pipefail

# Submit baseline and decoder-only ablation jobs.
# Seeding is controlled in YAML files via `seed` and `runs`.

echo "Submitting baseline ablation job"
bsub < hpc/submit/run_vae.sh

echo "Submitting decoder-only ablation job"
bsub < hpc/submit/decoder_only/baseline/run_decoder_only.sh

echo "Done submitting ablation jobs."
