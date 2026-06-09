#!/bin/bash
# Joint locked holdout: cgmvae_locked + hmmgmvae_locked + chmmgmvae_locked, folds 1–4.
# Submits fold 4 last (queue order only — no LSF dependencies).
#
# Generate:
#   PYTHONPATH=. python3 scripts/cv4fold/generate_joint_locked_holdout.py
# Submit:
#   bash hpc/submit/cv4fold/submit_joint_locked_holdout.sh

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "${SPA_ROOT}"
# shellcheck source=_queue_mix.sh
source "${SPA_ROOT}/hpc/submit/cv4fold/_queue_mix.sh"

JOB_IDX="${JOB_IDX:-0}"
MEM="6GB"
CPUS="4"
WALLTIME_CGMVAE="1:30"  # ~45 min est × 2
WALLTIME_HMM="2:00"     # HMM models ~60 min observed × 2

mkdir -p hpc/output/cv4fold/joint_holdout

_submit() {
  local fold="$1"
  local model="$2"
  local wall="$3"
  _pick_queue
  local queue="${PICKED_QUEUE}"

  local config="src/config/run/cvaemarhmm/cv4fold/joint_holdout/fold_${fold}/${model}_locked.yaml"
  if [[ ! -f "${config}" ]]; then
    echo "Missing ${config} — run generate_joint_locked_holdout.py" >&2
    exit 1
  fi

  local tag="cv4_joint_locked_f${fold}_${model}"

  bsub <<EOF
#!/bin/bash
#BSUB -J ${tag}
#BSUB -q ${queue}
#BSUB -W ${wall}
#BSUB -n ${CPUS}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=${MEM}]"
#BSUB -o hpc/output/cv4fold/joint_holdout/f${fold}_${model}_locked_%J.out
#BSUB -e hpc/output/cv4fold/joint_holdout/f${fold}_${model}_locked_%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
cd /work3/s204070/SPA
module load cuda/12.8.1
source .venv/bin/activate
python3 main.py --method train_vae --config_path ${config}
EOF

  echo "Submitted: ${tag} (fold ${fold}, ${queue}, ${wall})"
}

for fold in 1 2 3 4; do
  _submit "${fold}" "cgmvae" "${WALLTIME_CGMVAE}"
  _submit "${fold}" "hmmgmvae" "${WALLTIME_HMM}"
  _submit "${fold}" "chmmgmvae" "${WALLTIME_HMM}"
done

echo ""
echo "Done. Submitted 12 jobs (3 models × 4 folds). Fold 4 last in submission order."
echo "Monitor: bjobs -u \$USER | grep cv4_joint_locked"
echo "Logs: hpc/output/cv4fold/joint_holdout/f<FOLD>_<MODEL>_locked_%J.out"
