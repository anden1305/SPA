#!/bin/bash
# Kill selected PEND gpuv100 jobs and resubmit on gpul40s / gpua10 / gpua100.
#   bash hpc/submit/cv4fold/resubmit_pending_to_spill_queues.sh

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "${SPA_ROOT}"
# shellcheck source=_bsub_train_vae.sh
source "${SPA_ROOT}/hpc/submit/cv4fold/_bsub_train_vae.sh"

KILL_IDS=(
  28628869 28628870 28628871 28628872 28628873 28628874  # pop K3–K8
  28628875 28628876 28628877 28628878                    # pop K9–K12
  28628879 28628880 28628881                              # pop K13–K15
  28627037 28627038 28627039 28627040                    # kx2 K3–K6
  28627041 28627042                                       # kx2 K7–K8
)

echo "=== bkill PEND v100 spill candidates ==="
for id in "${KILL_IDS[@]}"; do
  bkill "${id}" 2>/dev/null && echo "killed ${id}" || echo "skip ${id} (not found or RUN)"
done

echo ""
echo "=== gpul40s: population K3–K8 + kx2 K3–K6 ==="
for k in 3 4 5 6 7 8; do
  _bsub_train_vae gpul40s 2:30 8GB \
    "cv4_k_sweep_pop_K${k}" \
    "src/config/run/cvaemarhmm/cv4fold/paper_k_sweep/population/chmmgmvae_K${k}.yaml" \
    "hpc/output/cv4fold/paper_k_sweep/population" \
    "k_sweep_K${k}"
done
for k in 3 4 5 6; do
  _bsub_train_vae gpul40s 2:00 6GB \
    "cv4_kx2_f4_K${k}" \
    "src/config/run/cvaemarhmm/cv4fold/paper_k_sweep/fold_4/chmmgmvae_K${k}_extra2.yaml" \
    "hpc/output/cv4fold/paper_k_sweep" \
    "k_extra2_K${k}"
done

echo ""
echo "=== gpua10: population K9–K12 ==="
for k in 9 10 11 12; do
  _bsub_train_vae gpua10 2:30 8GB \
    "cv4_k_sweep_pop_K${k}" \
    "src/config/run/cvaemarhmm/cv4fold/paper_k_sweep/population/chmmgmvae_K${k}.yaml" \
    "hpc/output/cv4fold/paper_k_sweep/population" \
    "k_sweep_K${k}"
done

echo ""
echo "=== gpua100: population K13–K15 + kx2 K7–K8 ==="
for k in 13 14 15; do
  _bsub_train_vae gpua100 2:30 8GB \
    "cv4_k_sweep_pop_K${k}" \
    "src/config/run/cvaemarhmm/cv4fold/paper_k_sweep/population/chmmgmvae_K${k}.yaml" \
    "hpc/output/cv4fold/paper_k_sweep/population" \
    "k_sweep_K${k}"
done
for k in 7 8; do
  _bsub_train_vae gpua100 2:00 6GB \
    "cv4_kx2_f4_K${k}" \
    "src/config/run/cvaemarhmm/cv4fold/paper_k_sweep/fold_4/chmmgmvae_K${k}_extra2.yaml" \
    "hpc/output/cv4fold/paper_k_sweep" \
    "k_extra2_K${k}"
done

echo ""
echo "Done. Left on gpuv100 PEND: hmm_raw f4, seed reruns, kx2 K9–K15."
echo "Monitor: bjobs -u \$USER | egrep 'k_sweep_pop|kx2|hmm_raw|seed_rerun'"
