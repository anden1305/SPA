#!/bin/bash
# lab_2 REM ablations: preprocessing / VAE-input tweaks (encoder unchanged).
# Generate configs first:
#   source .venv/bin/activate
#   PYTHONPATH=. python3 scripts/cv4fold/generate_ablation_lab2_rem.py
# Submit (user runs, or confirm with agent):
#   bash hpc/submit/cv4fold/submit_ablation_lab2_rem.sh

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "${SPA_ROOT}"

# Student GPU queues: gpuv100 (fastest, most jobs), gpua10 / gpua100 / gpul40s ×1 each.
# gpul40s & gpua100 often have long PEND — check: bqueues | egrep 'gpuv100|gpua10|gpua100|gpul40s'
declare -A JOB_QUEUES=(
  ["lab_2_rem_winner"]="gpuv100"
  ["lab_2_rem_paper_robust"]="gpuv100"
  ["lab_2_rem_emg_wide"]="gpuv100"
  ["lab_2_rem_emg_low"]="gpuv100"
  ["lab_2_rem_append_rms"]="gpuv100"
  ["lab_2_rem_paper_robust_emg_wide"]="gpua10"
  ["lab_2_rem_winner_append_rms"]="gpua100"
  ["lab_2_rem_paper_robust_append_rms"]="gpul40s"
)

JOBS=(
  "lab_2_rem_winner:src/config/run/cvaemarhmm/cv4fold/ablation_rem/lab_2/rem_winner.yaml"
  "lab_2_rem_paper_robust:src/config/run/cvaemarhmm/cv4fold/ablation_rem/lab_2/rem_paper_robust.yaml"
  "lab_2_rem_emg_wide:src/config/run/cvaemarhmm/cv4fold/ablation_rem/lab_2/rem_emg_wide.yaml"
  "lab_2_rem_emg_low:src/config/run/cvaemarhmm/cv4fold/ablation_rem/lab_2/rem_emg_low.yaml"
  "lab_2_rem_paper_robust_emg_wide:src/config/run/cvaemarhmm/cv4fold/ablation_rem/lab_2/rem_paper_robust_emg_wide.yaml"
  "lab_2_rem_append_rms:src/config/run/cvaemarhmm/cv4fold/ablation_rem/lab_2/rem_append_rms.yaml"
  "lab_2_rem_winner_append_rms:src/config/run/cvaemarhmm/cv4fold/ablation_rem/lab_2/rem_winner_append_rms.yaml"
  "lab_2_rem_paper_robust_append_rms:src/config/run/cvaemarhmm/cv4fold/ablation_rem/lab_2/rem_paper_robust_append_rms.yaml"
)

WALLTIME="4:00"
MEM="6GB"
CPUS="4"

mkdir -p "hpc/output/cv4fold/ablation_rem"
echo "Submitting ${#JOBS[@]} lab_2 REM ablation jobs (gpuv100×5, gpua10×1, gpua100×1, gpul40s×1)..."

for entry in "${JOBS[@]}"; do
  TAG="${entry%%:*}"
  CONFIG="${entry#*:}"
  QUEUE="${JOB_QUEUES[$TAG]:-gpuv100}"
  if [[ ! -f "${CONFIG}" ]]; then
    echo "Missing ${CONFIG} — run generate_ablation_lab2_rem.py first" >&2
    exit 1
  fi

  bsub <<EOF
#!/bin/bash
#BSUB -J abl_${TAG}
#BSUB -q ${QUEUE}
#BSUB -W ${WALLTIME}
#BSUB -n ${CPUS}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=${MEM}]"
#BSUB -o hpc/output/cv4fold/ablation_rem/${TAG}_%J.out
#BSUB -e hpc/output/cv4fold/ablation_rem/${TAG}_%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
cd /work3/s204070/SPA
module load cuda/12.8.1
source .venv/bin/activate
python3 main.py --method train_vae --config_path ${CONFIG}
EOF

  echo "Submitted: ${TAG} (queue: ${QUEUE})"
done

echo "Done. Monitor: bjobs -u \$USER | grep abl_lab_2_rem"
