#!/bin/bash
# Retry population K-sweep: failed OOM K + top-up seeds (K12→2, K15→1).
# gpua100 (40 GB VRAM), -n 4, rusage[mem=8GB] per slot, batch 64 in configs.
#
#   PYTHONPATH=. python3 scripts/cv4fold/generate_chmm_k_sweep_population_retry.py
#   bash hpc/submit/cv4fold/submit_chmm_k_sweep_population_retry.sh

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "${SPA_ROOT}"
# shellcheck source=_bsub_train_vae.sh
source "${SPA_ROOT}/hpc/submit/cv4fold/_bsub_train_vae.sh"

QUEUE="gpua100"
MEM="8GB"
CPUS="4"
WALLTIME="4:00"
CONFIG_DIR="src/config/run/cvaemarhmm/cv4fold/paper_k_sweep/population_retry"
LOG_DIR="hpc/output/cv4fold/paper_k_sweep/population"

PYTHONPATH="${SPA_ROOT}" python3 scripts/cv4fold/generate_chmm_k_sweep_population_retry.py

# Submit priority (user order): 6,5,4,9,10,7,8 then rest
ORDERED_K=(6 5 4 9 10 7 8 3 11 13 14 12 15)
for k in "${ORDERED_K[@]}"; do
  config="${CONFIG_DIR}/chmmgmvae_K${k}.yaml"
  [[ -f "${config}" ]] || { echo "Missing ${config}" >&2; exit 1; }
  _bsub_train_vae "${QUEUE}" "${WALLTIME}" "${MEM}" \
    "cv4_k_sweep_pop_K${k}" \
    "${config}" \
    "${LOG_DIR}" "k_sweep_K${k}"
done

echo ""
echo "Done: ${#ORDERED_K[@]} jobs on ${QUEUE} (-n ${CPUS}, ${MEM}/slot, ${WALLTIME}), order: ${ORDERED_K[*]}."
echo "Monitor: bjobs -u \$USER | grep cv4_k_sweep_pop"
