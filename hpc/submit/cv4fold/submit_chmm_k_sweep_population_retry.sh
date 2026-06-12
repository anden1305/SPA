#!/bin/bash
# Retry population K-sweep: failed OOM K + top-up seeds (K12→2, K15→1).
# gpuv100, -n 4, rusage[mem=6GB] per slot (24GB total host RAM).
#
#   PYTHONPATH=. python3 scripts/cv4fold/generate_chmm_k_sweep_population_retry.py
#   bash hpc/submit/cv4fold/submit_chmm_k_sweep_population_retry.sh

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "${SPA_ROOT}"
# shellcheck source=_bsub_train_vae.sh
source "${SPA_ROOT}/hpc/submit/cv4fold/_bsub_train_vae.sh"

QUEUE="gpuv100"
MEM="6GB"
CPUS="4"
WALLTIME="4:00"
CONFIG_DIR="src/config/run/cvaemarhmm/cv4fold/paper_k_sweep/population_retry"
LOG_DIR="hpc/output/cv4fold/paper_k_sweep/population"

PYTHONPATH="${SPA_ROOT}" python3 scripts/cv4fold/generate_chmm_k_sweep_population_retry.py

FULL_K=(3 4 5 6 7 8 9 10 11 13 14)
for k in "${FULL_K[@]}"; do
  config="${CONFIG_DIR}/chmmgmvae_K${k}.yaml"
  [[ -f "${config}" ]] || { echo "Missing ${config}" >&2; exit 1; }
  _bsub_train_vae "${QUEUE}" "${WALLTIME}" "${MEM}" \
    "cv4_k_sweep_pop_K${k}" \
    "${config}" \
    "${LOG_DIR}" "k_sweep_K${k}"
done

for spec in "12:2" "15:1"; do
  k="${spec%%:*}"
  config="${CONFIG_DIR}/chmmgmvae_K${k}.yaml"
  [[ -f "${config}" ]] || { echo "Missing ${config}" >&2; exit 1; }
  _bsub_train_vae "${QUEUE}" "${WALLTIME}" "${MEM}" \
    "cv4_k_sweep_pop_K${k}" \
    "${config}" \
    "${LOG_DIR}" "k_sweep_K${k}"
done

echo ""
echo "Done: ${#FULL_K[@]} full + 2 top-up on ${QUEUE} (-n ${CPUS}, ${MEM}/slot, ${WALLTIME})."
echo "Monitor: bjobs -u \$USER | grep cv4_k_sweep_pop"
