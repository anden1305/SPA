#!/bin/bash
# Spill backups: 9 jobs (K7–K15), one per K, rotated a100/a10/l40s. First finish wins.
# Seeds: a100=599, a10=699, l40s=799 (v100 primary uses 499).
#
#   bash hpc/submit/cv4fold/submit_chmm_k_sweep_population_spill.sh

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "${SPA_ROOT}"
# shellcheck source=_bsub_train_vae.sh
source "${SPA_ROOT}/hpc/submit/cv4fold/_bsub_train_vae.sh"

MEM="8GB"
WALLTIME="4:00"
LOG_DIR="hpc/output/cv4fold/paper_k_sweep/population"

# K:queue pairs (3 per spill queue)
declare -a PAIRS=(
  "7:gpua100"
  "8:gpua10"
  "9:gpul40s"
  "10:gpua100"
  "11:gpua10"
  "12:gpul40s"
  "13:gpua100"
  "14:gpua10"
  "15:gpul40s"
)

SPILL_LABEL_for_queue() {
  case "$1" in
    gpua100) echo a100 ;;
    gpua10) echo a10 ;;
    gpul40s) echo l40s ;;
    *) echo "unknown queue $1" >&2; return 1 ;;
  esac
}

echo "=== 9 mixed spill backups (K7–K15) ==="
for pair in "${PAIRS[@]}"; do
  k="${pair%%:*}"
  queue="${pair##*:}"
  spill="$(SPILL_LABEL_for_queue "${queue}")"
  config_dir="src/config/run/cvaemarhmm/cv4fold/paper_k_sweep/population_spill/${spill}"

  PYTHONPATH="${SPA_ROOT}" python3 scripts/cv4fold/generate_chmm_k_sweep_population_spill.py \
    --spill "${spill}" --k "${k}"

  config="${config_dir}/chmmgmvae_K${k}.yaml"
  [[ -f "${config}" ]] || { echo "Missing ${config}" >&2; exit 1; }
  _bsub_train_vae "${queue}" "${WALLTIME}" "${MEM}" \
    "cv4_k_sweep_pop_K${k}_${spill}" \
    "${config}" \
    "${LOG_DIR}" "k_sweep_K${k}_${spill}"
done

echo ""
echo "Done: 9 spill jobs (3×a100, 3×a10, 3×l40s)."
echo "Monitor: bjobs -u \$USER | grep -E 'pop_K.*_(a100|a10|l40s)'"
