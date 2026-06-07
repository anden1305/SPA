#!/bin/bash
# lab_2: 50 Hz notch on locked winner rem_emg_wide_eeg4.
# Generate first:
#   source .venv/bin/activate
#   PYTHONPATH=. python3 scripts/cv4fold/generate_ablation_lab2_notch.py
# Submit:
#   bash hpc/submit/cv4fold/submit_ablation_lab2_notch.sh

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "${SPA_ROOT}"

CONFIG="src/config/run/cvaemarhmm/cv4fold/ablation_rem/lab_2/rem_emg_wide_eeg4_notch50.yaml"
WALLTIME="4:00"
MEM="6GB"
CPUS="4"
QUEUE="gpuv100"
TAG="lab_2_rem_emg_wide_eeg4_notch50"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing ${CONFIG} — run generate_ablation_lab2_notch.py first" >&2
  exit 1
fi

mkdir -p hpc/output/cv4fold/ablation_rem/lab_2
echo "Submitting ${TAG} on ${QUEUE}..."

bsub <<EOF
#!/bin/bash
#BSUB -J abl_${TAG}
#BSUB -q ${QUEUE}
#BSUB -W ${WALLTIME}
#BSUB -n ${CPUS}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=${MEM}]"
#BSUB -o hpc/output/cv4fold/ablation_rem/lab_2/notch50_%J.out
#BSUB -e hpc/output/cv4fold/ablation_rem/lab_2/notch50_%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
cd /work3/s204070/SPA
module load cuda/12.8.1
source .venv/bin/activate
python3 main.py --method train_vae --config_path ${CONFIG}
EOF

echo "Submitted: ${TAG} (queue: ${QUEUE})"
echo "Monitor: bjobs -J abl_${TAG}"
