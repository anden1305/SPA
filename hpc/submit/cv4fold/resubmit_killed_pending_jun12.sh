#!/bin/bash
# Resubmit spill/l40s mirrors on gpuv100 (-n 4, 6GB/slot).
#   bash hpc/submit/cv4fold/resubmit_killed_pending_jun12.sh

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "${SPA_ROOT}"
# shellcheck source=_bsub_train_vae.sh
source "${SPA_ROOT}/hpc/submit/cv4fold/_bsub_train_vae.sh"

QUEUE="gpuv100"
MEM="6GB"
CPUS="4"

_submit_hmm_raw_within() {
  local fold="$1" lab="$2"
  bsub <<EOF
#!/bin/bash
#BSUB -J cv4_hmm_raw_f${fold}_${lab}
#BSUB -q ${QUEUE}
#BSUB -W 12:00
#BSUB -n ${CPUS}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=${MEM}]"
#BSUB -o hpc/output/cv4fold/hmm_raw/f${fold}_${lab}_%J.out
#BSUB -e hpc/output/cv4fold/hmm_raw/f${fold}_${lab}_%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
cd /work3/s204070/SPA
module load cuda/12.8.1
source .venv/bin/activate
python3 main.py --method train --config_path src/config/run/hmm/cv4fold/unified_holdout/${lab}/fold_${fold}/hmm_raw.yaml
EOF
  echo "Submitted cv4_hmm_raw_f${fold}_${lab} (${QUEUE}, -n ${CPUS}, ${MEM}/slot)"
}

_submit_hmm_raw_within 4 lab_5

_bsub_train_vae "${QUEUE}" 1:30 "${MEM}" \
  "cv4_seed_rerun_f3_lab5_cgmvae" \
  "src/config/run/cvaemarhmm/cv4fold/unified_holdout/lab_5/fold_3/cgmvae_locked.yaml" \
  "hpc/output/cv4fold/seed_reruns" "within_f3_lab_5_cgmvae"

_bsub_train_vae "${QUEUE}" 1:30 "${MEM}" \
  "cv4_seed_rerun_f4_lab5_cgmvae" \
  "src/config/run/cvaemarhmm/cv4fold/unified_holdout/lab_5/fold_4/cgmvae_locked.yaml" \
  "hpc/output/cv4fold/seed_reruns" "within_f4_lab_5_cgmvae"

_bsub_train_vae "${QUEUE}" 2:00 "${MEM}" \
  "cv4_seed_rerun_f4_lab5_chmmgmvae" \
  "src/config/run/cvaemarhmm/cv4fold/unified_holdout/lab_5/fold_4/chmmgmvae_locked.yaml" \
  "hpc/output/cv4fold/seed_reruns" "within_f4_lab_5_chmmgmvae"

for k in 9 10 11 12; do
  _bsub_train_vae "${QUEUE}" 2:00 "${MEM}" \
    "cv4_kx2_f4_K${k}" \
    "src/config/run/cvaemarhmm/cv4fold/paper_k_sweep/fold_4/chmmgmvae_K${k}_extra2.yaml" \
    "hpc/output/cv4fold/paper_k_sweep" "k_extra2_K${k}"
done

echo ""
echo "Done: 8 jobs on ${QUEUE} (-n ${CPUS}, ${MEM}/slot)."
