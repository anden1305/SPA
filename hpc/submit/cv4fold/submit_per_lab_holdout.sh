#!/bin/bash
# Per-lab within-lab holdout cv4fold (cGMVAE / cHMM-GMVAE).
#
# Generate configs first, e.g. fold-4 pilot:
#   source .venv/bin/activate
#   PYTHONPATH=. python3 scripts/cv4fold/generate_configs.py \
#     --phase per_lab_holdout --fold 4 --labs lab_2 lab_3 lab_5 \
#     --models cgmvae chmmgmvae
#
# Submit (user runs when ready):
#   bash hpc/submit/cv4fold/submit_per_lab_holdout.sh --fold 4

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "${SPA_ROOT}"

FOLD=4
LABS=(lab_2 lab_3 lab_5)
MODELS=(cgmvae chmmgmvae)
QUEUE="gpuv100"
WALLTIME="6:00"
MEM="6GB"
CPUS="4"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fold) FOLD="$2"; shift 2 ;;
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

mkdir -p "hpc/output/cv4fold/per_lab_holdout"

echo "Submitting per-lab holdout fold ${FOLD}: ${#LABS[@]} labs × ${#MODELS[@]} models..."

for lab in "${LABS[@]}"; do
  for model in "${MODELS[@]}"; do
    CONFIG="src/config/run/cvaemarhmm/cv4fold/per_lab_holdout/${lab}/fold_${FOLD}/${model}.yaml"
    if [[ ! -f "${CONFIG}" ]]; then
      echo "Missing ${CONFIG} — run generate_configs.py --phase per_lab_holdout --fold ${FOLD}" >&2
      exit 1
    fi

    bsub <<EOF
#!/bin/bash
#BSUB -J cv4_ho_f${FOLD}_${lab}_${model}
#BSUB -q ${QUEUE}
#BSUB -W ${WALLTIME}
#BSUB -n ${CPUS}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=${MEM}]"
#BSUB -o hpc/output/cv4fold/per_lab_holdout/f${FOLD}_${lab}_${model}_%J.out
#BSUB -e hpc/output/cv4fold/per_lab_holdout/f${FOLD}_${lab}_${model}_%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
cd /work3/s204070/SPA
module load cuda/12.8.1
source .venv/bin/activate
python3 main.py --method train_vae --config_path ${CONFIG}
EOF

    echo "Submitted: fold ${FOLD} ${lab} ${model}"
  done
done

echo "Done. Monitor: bjobs -u \$USER | grep cv4_ho"
echo ""
echo "When ALL jobs finish, run postprocess (copy-paste from docs/cv4fold/per_lab_holdout_pilot.md"
echo "  section 'When jobs finish (postprocess)' or docs/cv4fold/postprocess.md)."
