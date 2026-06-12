#!/bin/bash
# Extra gpul40s mirrors (first finish wins). Skips all hmm_raw.
#   bash hpc/submit/cv4fold/submit_l40s_backup_mirrors.sh

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "${SPA_ROOT}"
# shellcheck source=_bsub_train_vae.sh
source "${SPA_ROOT}/hpc/submit/cv4fold/_bsub_train_vae.sh"

QUEUE="gpul40s"

echo "=== Seed reruns (l40s backup) ==="
_bsub_train_vae "${QUEUE}" 1:30 6GB \
  "cv4_seed_rerun_f1_lab3_cgmvae_l40s" \
  "src/config/run/cvaemarhmm/cv4fold/unified_holdout/lab_3/fold_1/cgmvae_locked.yaml" \
  "hpc/output/cv4fold/seed_reruns" "within_f1_lab_3_cgmvae_l40s"
_bsub_train_vae "${QUEUE}" 1:30 6GB \
  "cv4_seed_rerun_f3_lab5_cgmvae_l40s" \
  "src/config/run/cvaemarhmm/cv4fold/unified_holdout/lab_5/fold_3/cgmvae_locked.yaml" \
  "hpc/output/cv4fold/seed_reruns" "within_f3_lab_5_cgmvae_l40s"
_bsub_train_vae "${QUEUE}" 1:30 6GB \
  "cv4_seed_rerun_f4_lab5_cgmvae_l40s" \
  "src/config/run/cvaemarhmm/cv4fold/unified_holdout/lab_5/fold_4/cgmvae_locked.yaml" \
  "hpc/output/cv4fold/seed_reruns" "within_f4_lab_5_cgmvae_l40s"
_bsub_train_vae "${QUEUE}" 2:00 6GB \
  "cv4_seed_rerun_f4_lab5_chmmgmvae_l40s" \
  "src/config/run/cvaemarhmm/cv4fold/unified_holdout/lab_5/fold_4/chmmgmvae_locked.yaml" \
  "hpc/output/cv4fold/seed_reruns" "within_f4_lab_5_chmmgmvae_l40s"

echo ""
echo "=== kx2 fold-4 (l40s backup for v100/a100 PEND K9–K12) ==="
for k in 9 10 11 12; do
  _bsub_train_vae "${QUEUE}" 2:00 6GB \
    "cv4_kx2_f4_K${k}_l40s" \
    "src/config/run/cvaemarhmm/cv4fold/paper_k_sweep/fold_4/chmmgmvae_K${k}_extra2.yaml" \
    "hpc/output/cv4fold/paper_k_sweep" "k_extra2_K${k}_l40s"
done

echo ""
echo "=== population K-sweep (l40s backup for a100 PEND K13–K15) ==="
for k in 13 14 15; do
  _bsub_train_vae "${QUEUE}" 2:30 8GB \
    "cv4_k_sweep_pop_K${k}_l40s" \
    "src/config/run/cvaemarhmm/cv4fold/paper_k_sweep/population/chmmgmvae_K${k}.yaml" \
    "hpc/output/cv4fold/paper_k_sweep/population" "k_sweep_K${k}_l40s"
done

echo ""
echo "Done. 11 gpul40s backup jobs (no hmm_raw)."
echo "Monitor: bjobs -u \$USER | grep l40s"
