#!/bin/bash
# Within-lab locked holdout: cgmvae_locked + hmmgmvae_locked + chmmgmvae_locked.
# 3 labs × 4 folds × 3 models = 36 jobs.
# Folds 1–3 submitted first, then fold 4 (submission order only, no dependencies).
#
# Generate:
#   PYTHONPATH=. python3 scripts/cv4fold/generate_within_lab_locked_holdout.py
# Submit:
#   bash hpc/submit/cv4fold/submit_within_lab_locked_holdout.sh

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "${SPA_ROOT}"
# shellcheck source=_queue_mix.sh
source "${SPA_ROOT}/hpc/submit/cv4fold/_queue_mix.sh"

JOB_IDX="${JOB_IDX:-0}"
LABS=(lab_2 lab_3 lab_5)
MODELS=(cgmvae hmmgmvae chmmgmvae)
MEM="6GB"
CPUS="4"
WALLTIME_CGMVAE="1:30"
WALLTIME_CHMM="2:00"

mkdir -p hpc/output/cv4fold/unified_holdout

_submit() {
  local fold="$1"
  local lab="$2"
  local model="$3"
  _pick_queue
  local queue="${PICKED_QUEUE}"

  local config="src/config/run/cvaemarhmm/cv4fold/unified_holdout/${lab}/fold_${fold}/${model}_locked.yaml"
  if [[ ! -f "${config}" ]]; then
    echo "Missing ${config} — run generate_within_lab_locked_holdout.py" >&2
    exit 1
  fi

  local tag="cv4_within_locked_f${fold}_${lab}_${model}"
  local wall="${WALLTIME_CHMM}"
  if [[ "${model}" == "cgmvae" ]]; then
    wall="${WALLTIME_CGMVAE}"
  fi

  bsub <<EOF
#!/bin/bash
#BSUB -J ${tag}
#BSUB -q ${queue}
#BSUB -W ${wall}
#BSUB -n ${CPUS}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=${MEM}]"
#BSUB -o hpc/output/cv4fold/unified_holdout/f${fold}_${lab}_${model}_locked_%J.out
#BSUB -e hpc/output/cv4fold/unified_holdout/f${fold}_${lab}_${model}_locked_%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
cd /work3/s204070/SPA
module load cuda/12.8.1
source .venv/bin/activate
python3 main.py --method train_vae --config_path ${config}
EOF

  echo "Submitted: fold ${fold} ${lab} ${model}_locked (${queue}, ${wall})"
}

echo "Submitting within-lab locked holdout (folds 1–3)..."

for fold in 1 2 3; do
  for lab in "${LABS[@]}"; do
    for model in "${MODELS[@]}"; do
      _submit "${fold}" "${lab}" "${model}"
    done
  done
done

echo "Submitting fold 4 (last)..."

for lab in "${LABS[@]}"; do
  for model in "${MODELS[@]}"; do
    _submit 4 "${lab}" "${model}"
  done
done

echo "Done. 36 jobs. Monitor: bjobs -u \$USER | grep cv4_within_locked"
echo "Logs: hpc/output/cv4fold/unified_holdout/f<FOLD>_<LAB>_<MODEL>_locked_%J.out"
