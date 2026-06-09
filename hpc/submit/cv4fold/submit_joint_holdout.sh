#!/bin/bash
# Joint cross-lab holdout — paper line (unified recipes + per-lab cvae_overrides).
#
# Generate, e.g. Phase 0 pilot:
#   PYTHONPATH=. python3 scripts/cv4fold/generate_configs.py \
#     --phase joint_holdout --fold 4 --models chmmgmvae
#
# Submit:
#   bash hpc/submit/cv4fold/submit_joint_holdout.sh --fold 4 --models chmmgmvae
#   bash hpc/submit/cv4fold/submit_joint_holdout.sh --fold all

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "${SPA_ROOT}"

FOLDS=(4)
MODELS=(cgmvae hmmgmvae chmmgmvae)
QUEUE="gpuv100"
MEM="6GB"
CPUS="4"
SKIP_FOLD=""
SKIP_MODEL=""

walltime_for_model() {
  case "$1" in
    cgmvae) echo "4:00" ;;
    *) echo "6:00" ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fold)
      shift
      FOLDS=()
      if [[ $# -gt 0 && "$1" == "all" ]]; then
        FOLDS=(1 2 3 4)
        shift
      else
        while [[ $# -gt 0 && "$1" != --* ]]; do FOLDS+=("$1"); shift; done
      fi
      ;;
    --models)
      shift
      MODELS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do MODELS+=("$1"); shift; done
      ;;
    --skip-fold) SKIP_FOLD="$2"; shift 2 ;;
    --skip-model) SKIP_MODEL="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "hpc/output/cv4fold/joint_holdout"

echo "Submitting joint holdout: folds=${FOLDS[*]} models=${MODELS[*]}..."

for fold in "${FOLDS[@]}"; do
  for model in "${MODELS[@]}"; do
    if [[ -n "${SKIP_FOLD}" && "${fold}" == "${SKIP_FOLD}" && -n "${SKIP_MODEL}" && "${model}" == "${SKIP_MODEL}" ]]; then
      echo "Skip (already run): fold ${fold} ${model}"
      continue
    fi

    CONFIG="src/config/run/cvaemarhmm/cv4fold/joint_holdout/fold_${fold}/${model}.yaml"
    if [[ ! -f "${CONFIG}" ]]; then
      echo "Skip missing ${CONFIG} — run generate_configs.py --phase joint_holdout" >&2
      continue
    fi

    WALLTIME="$(walltime_for_model "${model}")"
    JOB_TAG="cv4_joint_f${fold}_${model}"

    bsub <<EOF
#!/bin/bash
#BSUB -J ${JOB_TAG}
#BSUB -q ${QUEUE}
#BSUB -W ${WALLTIME}
#BSUB -n ${CPUS}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=${MEM}]"
#BSUB -o hpc/output/cv4fold/joint_holdout/f${fold}_${model}_%J.out
#BSUB -e hpc/output/cv4fold/joint_holdout/f${fold}_${model}_%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
cd /work3/s204070/SPA
module load cuda/12.8.1
source .venv/bin/activate
python3 main.py --method train_vae --config_path ${CONFIG}
EOF

    echo "Submitted: fold ${fold} ${model} (${WALLTIME})"
  done
done

echo "Done. Monitor: bjobs -u \$USER | grep cv4_joint"
echo "Logs: hpc/output/cv4fold/joint_holdout/f<fold>_<model>_%J.out"
