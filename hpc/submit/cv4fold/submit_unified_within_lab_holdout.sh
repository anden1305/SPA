#!/bin/bash
# Within-lab holdout — paper line (unified recipes + per-lab cvae_overrides).
#
# Generate:
#   PYTHONPATH=. python3 scripts/cv4fold/generate_configs.py \
#     --phase within_lab_holdout --fold all
#
# Submit:
#   bash hpc/submit/cv4fold/submit_unified_within_lab_holdout.sh --fold all

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "${SPA_ROOT}"

FOLDS=(4)
LABS=(lab_2 lab_3 lab_5)
MODELS=(cgmvae hmmgmvae chmmgmvae)
QUEUE="gpuv100"
MEM="6GB"
CPUS="4"

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
    --labs)
      shift
      LABS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do LABS+=("$1"); shift; done
      ;;
    --models)
      shift
      MODELS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do MODELS+=("$1"); shift; done
      ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "hpc/output/cv4fold/unified_holdout"

echo "Submitting unified within-lab holdout: folds=${FOLDS[*]} labs=${LABS[*]} models=${MODELS[*]}..."

for fold in "${FOLDS[@]}"; do
  for lab in "${LABS[@]}"; do
    for model in "${MODELS[@]}"; do
      CONFIG="src/config/run/cvaemarhmm/cv4fold/unified_holdout/${lab}/fold_${fold}/${model}.yaml"
      if [[ ! -f "${CONFIG}" ]]; then
        echo "Skip missing ${CONFIG}" >&2
        continue
      fi

      WALLTIME="$(walltime_for_model "${model}")"
      JOB_TAG="cv4_uho_f${fold}_${lab}_${model}"

      bsub <<EOF
#!/bin/bash
#BSUB -J ${JOB_TAG}
#BSUB -q ${QUEUE}
#BSUB -W ${WALLTIME}
#BSUB -n ${CPUS}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=${MEM}]"
#BSUB -o hpc/output/cv4fold/unified_holdout/f${fold}_${lab}_${model}_%J.out
#BSUB -e hpc/output/cv4fold/unified_holdout/f${fold}_${lab}_${model}_%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
cd /work3/s204070/SPA
module load cuda/12.8.1
source .venv/bin/activate
python3 main.py --method train_vae --config_path ${CONFIG}
EOF

      echo "Submitted: fold ${fold} ${lab} ${model} (${WALLTIME})"
    done
  done
done

echo "Done. Monitor: bjobs -u \$USER | grep cv4_uho"
echo "Logs: hpc/output/cv4fold/unified_holdout/f<fold>_<lab>_<model>_%J.out"
