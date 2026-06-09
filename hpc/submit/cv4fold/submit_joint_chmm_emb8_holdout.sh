#!/bin/bash
# Joint holdout folds 1–3: cHMM-GMVAE emb8 winner (fold-4 ablation lock).
#
# Generate:
#   PYTHONPATH=. python3 scripts/cv4fold/generate_joint_chmm_emb8_holdout.py
# Submit:
#   bash hpc/submit/cv4fold/submit_joint_chmm_emb8_holdout.sh

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "${SPA_ROOT}"

FOLDS=(1 2 3)
QUEUE="gpuv100"
MEM="6GB"
CPUS="4"
WALLTIME="6:00"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fold)
      shift
      FOLDS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do FOLDS+=("$1"); shift; done
      ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

mkdir -p hpc/output/cv4fold/joint_holdout

for fold in "${FOLDS[@]}"; do
  config="src/config/run/cvaemarhmm/cv4fold/joint_holdout/fold_${fold}/chmmgmvae_emb8.yaml"
  if [[ ! -f "${config}" ]]; then
    echo "Missing ${config} — run generate_joint_chmm_emb8_holdout.py" >&2
    exit 1
  fi

  tag="cv4_joint_f${fold}_chmm_emb8"

  bsub <<EOF
#!/bin/bash
#BSUB -J ${tag}
#BSUB -q ${QUEUE}
#BSUB -W ${WALLTIME}
#BSUB -n ${CPUS}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=${MEM}]"
#BSUB -o hpc/output/cv4fold/joint_holdout/f${fold}_chmm_emb8_%J.out
#BSUB -e hpc/output/cv4fold/joint_holdout/f${fold}_chmm_emb8_%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
cd /work3/s204070/SPA
module load cuda/12.8.1
source .venv/bin/activate
python3 main.py --method train_vae --config_path ${config}
EOF

  echo "Submitted: fold ${fold} chmmgmvae_emb8 (${QUEUE}, ${WALLTIME})"
done

echo "Done. Monitor: bjobs -u \$USER | grep cv4_joint_f"
