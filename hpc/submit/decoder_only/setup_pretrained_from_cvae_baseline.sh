#!/bin/bash
# Copy latest cvae_baseline run checkpoint to canonical pretrained paths.
#
# Usage (repo root):
#   bash hpc/submit/decoder_only/setup_pretrained_from_cvae_baseline.sh
#   bash hpc/submit/decoder_only/setup_pretrained_from_cvae_baseline.sh /path/to/cvae_final_model.pth

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "$SPA_ROOT"

if [[ -n "${1:-}" ]]; then
  SRC="$1"
else
  SRC="$(ls -t results/decoder_only/cvae_baseline_*/1/checkpoints/cvae_final_model.pth 2>/dev/null | head -1)"
fi

if [[ -z "${SRC}" || ! -f "${SRC}" ]]; then
  echo "No source checkpoint found. Train cvae_baseline first:"
  echo "  uv run main.py -m train_vae -c src/config/run/cvaemarhmm/final/cvae_final.yaml"
  exit 1
fi

mkdir -p results/decoder_only/cvae_final_decoder_only results/decoder_only/cvae_baseline
cp -f "${SRC}" results/decoder_only/cvae_final_decoder_only/cvae_decoder_only_model.pth
cp -f "${SRC}" results/decoder_only/cvae_baseline/cvae_baseline_model.pth

echo "Installed pretrained weights from: ${SRC}"
echo "  -> results/decoder_only/cvae_final_decoder_only/cvae_decoder_only_model.pth"
echo "  -> results/decoder_only/cvae_baseline/cvae_baseline_model.pth"
ls -lh results/decoder_only/cvae_final_decoder_only/cvae_decoder_only_model.pth
