#!/bin/bash
# Kill PEND cv4fold jobs and resubmit across gpuv100 / gpua100 / gpua10 / gpul40s / gpua40.
# Mix per 10 jobs: 5× v100, 2× a100, 1× a10, 1× l40s, 1× a40 (_queue_mix.sh).
# Walltimes ≈ 2× observed runtimes (shorter wall → faster scheduling).
#
#   bash hpc/submit/cv4fold/submit_redistribute_pending_queues.sh

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "${SPA_ROOT}"
source "${SPA_ROOT}/hpc/submit/cv4fold/_queue_mix.sh"

MEM="6GB"
CPUS="4"
JOB_IDX=0

pending_ids=$(bjobs -u "$USER" -p -noheader 2>/dev/null | awk '$1 ~ /^[0-9]+$/ {print $1}')
if [[ -n "${pending_ids}" ]]; then
  echo "Killing pending jobs..."
  while read -r jid; do
    [[ -n "${jid}" ]] && bkill "${jid}" 2>/dev/null || true
  done <<< "${pending_ids}"
  sleep 3
fi

_submit_train() {
  local tag="$1"
  local wall="$2"
  local config="$3"
  local logdir="$4"
  local logstem="$5"
  _pick_queue
  local queue="${PICKED_QUEUE}"

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

mkdir -p hpc/output/cv4fold/joint_holdout hpc/output/cv4fold/unified_holdout hpc/output/cv4fold/ablation_joint_chmm

echo "=== fold-4 one-offs ==="
WALL_CHMM="2:00"    # ~60 min observed × 2
WALL_CGMVAE="1:30"  # ~45 min estimated × 2

_submit_train "abl_jchmm_f4_warm_emb8_sticky92" "${WALL_CHMM}" \
  "src/config/run/cvaemarhmm/cv4fold/ablation_joint_chmm/fold_4/warm_emb8_sticky92.yaml" \
  "hpc/output/cv4fold/ablation_joint_chmm" "f4_warm_emb8_sticky92"
_submit_train "cv4_joint_f4_cgmvae_emb8" "${WALL_CGMVAE}" \
  "src/config/run/cvaemarhmm/cv4fold/joint_holdout/fold_4/cgmvae_emb8.yaml" \
  "hpc/output/cv4fold/joint_holdout" "f4_cgmvae_emb8"
_submit_train "cv4_lrswap_chmm_f4" "${WALL_CHMM}" \
  "src/config/run/cvaemarhmm/cv4fold/ablation_joint_chmm/fold_4/warm_emb8_sticky92_lr3e4.yaml" \
  "hpc/output/cv4fold/ablation_joint_chmm" "f4_warm_emb8_sticky92_lr3e4"
_submit_train "cv4_lrswap_cgmvae_f4" "${WALL_CGMVAE}" \
  "src/config/run/cvaemarhmm/cv4fold/joint_holdout/fold_4/cgmvae_emb8_lr1p3e3.yaml" \
  "hpc/output/cv4fold/joint_holdout" "f4_cgmvae_emb8_lr1p3e3"

echo "=== joint locked folds 1–4 (fold 4 last) ==="
for fold in 1 2 3 4; do
  for model in cgmvae chmmgmvae; do
    wall="${WALL_CHMM}"
    [[ "${model}" == "cgmvae" ]] && wall="${WALL_CGMVAE}"
    _submit_train "cv4_joint_locked_f${fold}_${model}" "${wall}" \
      "src/config/run/cvaemarhmm/cv4fold/joint_holdout/fold_${fold}/${model}_locked.yaml" \
      "hpc/output/cv4fold/joint_holdout" "f${fold}_${model}_locked"
  done
done

echo "=== within-lab locked folds 1–3, then fold 4 ==="
for fold in 1 2 3; do
  for lab in lab_2 lab_3 lab_5; do
    for model in cgmvae chmmgmvae; do
      wall="${WALL_CHMM}"
      [[ "${model}" == "cgmvae" ]] && wall="${WALL_CGMVAE}"
      _submit_train "cv4_within_locked_f${fold}_${lab}_${model}" "${wall}" \
        "src/config/run/cvaemarhmm/cv4fold/unified_holdout/${lab}/fold_${fold}/${model}_locked.yaml" \
        "hpc/output/cv4fold/unified_holdout" "f${fold}_${lab}_${model}_locked"
    done
  done
done
for lab in lab_2 lab_3 lab_5; do
  for model in cgmvae chmmgmvae; do
    wall="${WALL_CHMM}"
    [[ "${model}" == "cgmvae" ]] && wall="${WALL_CGMVAE}"
    _submit_train "cv4_within_locked_f4_${lab}_${model}" "${wall}" \
      "src/config/run/cvaemarhmm/cv4fold/unified_holdout/${lab}/fold_4/${model}_locked.yaml" \
      "hpc/output/cv4fold/unified_holdout" "f4_${lab}_${model}_locked"
  done
done

echo ""
echo "Done: 36 jobs across mixed queues. Monitor:"
echo "  bjobs -u \$USER | grep -E 'cv4_|abl_'"
