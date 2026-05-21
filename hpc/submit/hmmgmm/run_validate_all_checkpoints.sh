#!/bin/bash
#BSUB -J hmmgmm_validate_all
#BSUB -q gpuv100
#BSUB -W 1:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=5GB]"
#BSUB -o hpc/output/hmmgmm/validate/%J.out
#BSUB -e hpc/output/hmmgmm/validate/%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

# Re-run post-train prior validation on completed hmmgmm checkpoints.
# Do NOT run this on the login node — submit with: bsub < hpc/submit/hmmgmm/run_validate_all_checkpoints.sh
# For interactive debugging: linuxsh  (then activate venv and run scripts manually).

set -euo pipefail
mkdir -p hpc/output/hmmgmm/validate

module load cuda/12.8.1
source .venv/bin/activate

SCRIPT="scripts/hmmgmm_validate_checkpoint.py"

validate_one() {
  local config="$1"
  local ckpt="$2"
  echo "=== $ckpt ==="
  python3 "$SCRIPT" -c "$config" --checkpoint "$ckpt"
}

# POC
validate_one \
  src/config/run/cvaeprior/hmmgmm/poc/sub039_chmmgmvae_poc.yaml \
  results/hmmgmm/poc/sub039_chmmgmvae_poc_20260517-224722/cvae_final_model_run1.pth

validate_one \
  src/config/run/cvaeprior/hmmgmm/poc/sub039_cgmvae_gmm_seq64_baseline.yaml \
  results/hmmgmm/poc/sub039_cgmvae_gmm_seq64_baseline_20260517-224806/cvae_final_model_run1.pth

# LOLO
validate_one \
  src/config/run/cvaeprior/hmmgmm/lolo_subject_lab/generalization_lab_holdout_lab2.yaml \
  results/hmmgmm/lolo_subject_lab/holdout_lab2/chmmgmvae_lolo_subject_lab_holdout_lab2_20260517-224843/cvae_final_model_run1.pth

validate_one \
  src/config/run/cvaeprior/hmmgmm/lolo_subject_lab/generalization_lab_holdout_lab3.yaml \
  results/hmmgmm/lolo_subject_lab/holdout_lab3/chmmgmvae_lolo_subject_lab_holdout_lab3_20260517-224849/cvae_final_model_run1.pth

validate_one \
  src/config/run/cvaeprior/hmmgmm/lolo_subject_lab/generalization_lab_holdout_lab5.yaml \
  results/hmmgmm/lolo_subject_lab/holdout_lab5/chmmgmvae_lolo_subject_lab_holdout_lab5_20260517-224950/cvae_final_model_run1.pth

echo "Done."

# Usage:
#   bsub < hpc/submit/hmmgmm/run_validate_all_checkpoints.sh
