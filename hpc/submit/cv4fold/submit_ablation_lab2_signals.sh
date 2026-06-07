#!/bin/bash
# lab_2 signal montage: EEG1+EEG3 (baseline) vs EEG1+EEG4 on no_postnorm_widebp winner.
# Generate configs first:
#   source .venv/bin/activate
#   PYTHONPATH=. python3 scripts/cv4fold/generate_ablation_lab2_signals.py
# Submit:
#   bash hpc/submit/cv4fold/submit_ablation_lab2_signals.sh

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "${SPA_ROOT}"

declare -A JOB_QUEUES=(
  ["lab_2_baseline_eeg1_eeg3"]="gpuv100"
  ["lab_2_eeg1_eeg4"]="gpua10"
)

JOBS=(
  "lab_2_baseline_eeg1_eeg3:src/config/run/cvaemarhmm/cv4fold/ablation_signals/lab_2/baseline_eeg1_eeg3.yaml"
  "lab_2_eeg1_eeg4:src/config/run/cvaemarhmm/cv4fold/ablation_signals/lab_2/eeg1_eeg4.yaml"
)

WALLTIME="4:00"
MEM="6GB"
CPUS="4"

mkdir -p "hpc/output/cv4fold/ablation_signals"
echo "Submitting ${#JOBS[@]} lab_2 signal ablation jobs..."

for entry in "${JOBS[@]}"; do
  TAG="${entry%%:*}"
  CONFIG="${entry#*:}"
  QUEUE="${JOB_QUEUES[$TAG]:-gpuv100}"
  if [[ ! -f "${CONFIG}" ]]; then
    echo "Missing ${CONFIG} — run generate_ablation_lab2_signals.py first" >&2
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
#BSUB -o hpc/output/cv4fold/ablation_signals/${TAG}_%J.out
#BSUB -e hpc/output/cv4fold/ablation_signals/${TAG}_%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
cd /work3/s204070/SPA
module load cuda/12.8.1
source .venv/bin/activate
python3 main.py --method train_vae --config_path ${CONFIG}
EOF

  echo "Submitted: ${TAG} (queue: ${QUEUE})"
done

echo "Done. Monitor: bjobs -u \$USER | grep abl_lab_2"
