#!/bin/bash
# Kill remaining PEND cv4fold jobs and resubmit 11× gpuv100 + 2× gpul40s.
#
#   bash hpc/submit/cv4fold/submit_resubmit_pending_v100.sh

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "${SPA_ROOT}"

MEM="6GB"
CPUS="4"
WALL_CHMM="2:00"
WALL_CGMVAE="1:30"

pending_ids=$(bjobs -u "$USER" -p -noheader 2>/dev/null | awk '$1 ~ /^[0-9]+$/ {print $1}')
if [[ -n "${pending_ids}" ]]; then
  echo "Killing ${pending_ids//$'\n'/ }..."
  while read -r jid; do
    [[ -n "${jid}" ]] && bkill "${jid}" 2>/dev/null || true
  done <<< "${pending_ids}"
  sleep 3
fi

_submit_train() {
  local tag="$1"
  local queue="$2"
  local wall="$3"
  local config="$4"
  local logdir="$5"
  local logstem="$6"

  [[ -f "${config}" ]] || { echo "Missing ${config}" >&2; exit 1; }

  bsub <<EOF
#!/bin/bash
#BSUB -J ${tag}
#BSUB -q ${queue}
#BSUB -W ${wall}
#BSUB -n ${CPUS}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=${MEM}]"
#BSUB -o ${logdir}/${logstem}_%J.out
#BSUB -e ${logdir}/${logstem}_%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
cd /work3/s204070/SPA
module load cuda/12.8.1
source .venv/bin/activate
python3 main.py --method train_vae --config_path ${config}
EOF
  echo "Submitted: ${tag} → ${queue} (${wall})"
}

mkdir -p hpc/output/cv4fold/joint_holdout hpc/output/cv4fold/unified_holdout

echo "=== 11× gpuv100 ==="
_submit_train "cv4_joint_locked_f1_chmmgmvae" gpuv100 "${WALL_CHMM}" \
  "src/config/run/cvaemarhmm/cv4fold/joint_holdout/fold_1/chmmgmvae_locked.yaml" \
  "hpc/output/cv4fold/joint_holdout" "f1_chmmgmvae_locked"
_submit_train "cv4_joint_locked_f2_cgmvae" gpuv100 "${WALL_CGMVAE}" \
  "src/config/run/cvaemarhmm/cv4fold/joint_holdout/fold_2/cgmvae_locked.yaml" \
  "hpc/output/cv4fold/joint_holdout" "f2_cgmvae_locked"
_submit_train "cv4_joint_locked_f2_chmmgmvae" gpuv100 "${WALL_CHMM}" \
  "src/config/run/cvaemarhmm/cv4fold/joint_holdout/fold_2/chmmgmvae_locked.yaml" \
  "hpc/output/cv4fold/joint_holdout" "f2_chmmgmvae_locked"
_submit_train "cv4_within_locked_f1_lab_3_chmmgmvae" gpuv100 "${WALL_CHMM}" \
  "src/config/run/cvaemarhmm/cv4fold/unified_holdout/lab_3/fold_1/chmmgmvae_locked.yaml" \
  "hpc/output/cv4fold/unified_holdout" "f1_lab_3_chmmgmvae_locked"
_submit_train "cv4_within_locked_f1_lab_5_cgmvae" gpuv100 "${WALL_CGMVAE}" \
  "src/config/run/cvaemarhmm/cv4fold/unified_holdout/lab_5/fold_1/cgmvae_locked.yaml" \
  "hpc/output/cv4fold/unified_holdout" "f1_lab_5_cgmvae_locked"
_submit_train "cv4_within_locked_f1_lab_5_chmmgmvae" gpuv100 "${WALL_CHMM}" \
  "src/config/run/cvaemarhmm/cv4fold/unified_holdout/lab_5/fold_1/chmmgmvae_locked.yaml" \
  "hpc/output/cv4fold/unified_holdout" "f1_lab_5_chmmgmvae_locked"
_submit_train "cv4_within_locked_f2_lab_2_chmmgmvae" gpuv100 "${WALL_CHMM}" \
  "src/config/run/cvaemarhmm/cv4fold/unified_holdout/lab_2/fold_2/chmmgmvae_locked.yaml" \
  "hpc/output/cv4fold/unified_holdout" "f2_lab_2_chmmgmvae_locked"
_submit_train "cv4_within_locked_f3_lab_2_chmmgmvae" gpuv100 "${WALL_CHMM}" \
  "src/config/run/cvaemarhmm/cv4fold/unified_holdout/lab_2/fold_3/chmmgmvae_locked.yaml" \
  "hpc/output/cv4fold/unified_holdout" "f3_lab_2_chmmgmvae_locked"
_submit_train "cv4_within_locked_f3_lab_3_cgmvae" gpuv100 "${WALL_CGMVAE}" \
  "src/config/run/cvaemarhmm/cv4fold/unified_holdout/lab_3/fold_3/cgmvae_locked.yaml" \
  "hpc/output/cv4fold/unified_holdout" "f3_lab_3_cgmvae_locked"
_submit_train "cv4_within_locked_f3_lab_3_chmmgmvae" gpuv100 "${WALL_CHMM}" \
  "src/config/run/cvaemarhmm/cv4fold/unified_holdout/lab_3/fold_3/chmmgmvae_locked.yaml" \
  "hpc/output/cv4fold/unified_holdout" "f3_lab_3_chmmgmvae_locked"
_submit_train "cv4_within_locked_f3_lab_5_chmmgmvae" gpuv100 "${WALL_CHMM}" \
  "src/config/run/cvaemarhmm/cv4fold/unified_holdout/lab_5/fold_3/chmmgmvae_locked.yaml" \
  "hpc/output/cv4fold/unified_holdout" "f3_lab_5_chmmgmvae_locked"

echo "=== 2× gpul40s ==="
_submit_train "cv4_joint_locked_f3_chmmgmvae" gpul40s "${WALL_CHMM}" \
  "src/config/run/cvaemarhmm/cv4fold/joint_holdout/fold_3/chmmgmvae_locked.yaml" \
  "hpc/output/cv4fold/joint_holdout" "f3_chmmgmvae_locked"
_submit_train "cv4_within_locked_f4_lab_5_chmmgmvae" gpul40s "${WALL_CHMM}" \
  "src/config/run/cvaemarhmm/cv4fold/unified_holdout/lab_5/fold_4/chmmgmvae_locked.yaml" \
  "hpc/output/cv4fold/unified_holdout" "f4_lab_5_chmmgmvae_locked"

echo ""
echo "Done: 13 jobs (11 v100 + 2 l40s). Monitor:"
echo "  bjobs -u \$USER | grep cv4_"
