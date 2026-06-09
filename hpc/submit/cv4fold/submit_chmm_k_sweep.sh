#!/bin/bash
# K substage sweep — locked cHMM on joint holdout fold 4.
#
# Generate:
#   PYTHONPATH=. python3 scripts/cv4fold/generate_chmm_k_sweep.py --fold 4
# Submit:
#   bash hpc/submit/cv4fold/submit_chmm_k_sweep.sh

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "${SPA_ROOT}"
# shellcheck source=_queue_mix.sh
source "${SPA_ROOT}/hpc/submit/cv4fold/_queue_mix.sh"

JOB_IDX="${JOB_IDX:-0}"
FOLD=4
MEM="6GB"
CPUS="4"
WALLTIME="2:00"
K_VALUES=(3 5 7 9 11 13 15)

mkdir -p hpc/output/cv4fold/paper_k_sweep

for k in "${K_VALUES[@]}"; do
  _pick_queue
  queue="${PICKED_QUEUE}"
  config="src/config/run/cvaemarhmm/cv4fold/paper_k_sweep/fold_${FOLD}/chmmgmvae_K${k}.yaml"
  if [[ ! -f "${config}" ]]; then
    echo "Missing ${config} — run generate_chmm_k_sweep.py" >&2
    exit 1
  fi
  tag="cv4_k_sweep_f${FOLD}_K${k}"

  bsub <<EOF
#!/bin/bash
#BSUB -J ${tag}
#BSUB -q ${queue}
#BSUB -W ${WALLTIME}
#BSUB -n ${CPUS}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=${MEM}]"
#BSUB -o hpc/output/cv4fold/paper_k_sweep/k_sweep_K${k}_%J.out
#BSUB -e hpc/output/cv4fold/paper_k_sweep/k_sweep_K${k}_%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
cd /work3/s204070/SPA
module load cuda/12.8.1
source .venv/bin/activate
python3 main.py --method train_vae --config_path ${config}
EOF

  echo "Submitted: ${tag} (${queue}, ${WALLTIME})"
done

echo ""
echo "Done. ${#K_VALUES[@]} K-sweep jobs."
echo "Monitor: bjobs -u \$USER | grep cv4_k_sweep"
