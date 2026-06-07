#!/bin/bash
# lab_2 REM ablations on EEG1+EEG4+EMG montage (after signal ablation favours EEG4).
# Generate first:
#   PYTHONPATH=. python3 scripts/cv4fold/generate_ablation_lab2_rem.py --montage eeg4 --variants rem_emg_low rem_emg_wide

set -euo pipefail
cd /work3/s204070/SPA

JOBS=(
  "lab_2_rem_emg_low_eeg4:src/config/run/cvaemarhmm/cv4fold/ablation_rem/lab_2/rem_emg_low_eeg4.yaml"
  "lab_2_rem_emg_wide_eeg4:src/config/run/cvaemarhmm/cv4fold/ablation_rem/lab_2/rem_emg_wide_eeg4.yaml"
)

WALLTIME="4:00"
MEM="6GB"
CPUS="4"

mkdir -p hpc/output/cv4fold/ablation_rem
echo "Submitting ${#JOBS[@]} lab_2 REM (EEG4 montage) jobs..."

for entry in "${JOBS[@]}"; do
  TAG="${entry%%:*}"
  CONFIG="${entry#*:}"
  if [[ ! -f "${CONFIG}" ]]; then
    echo "Missing ${CONFIG}" >&2
    exit 1
  fi

  bsub <<EOF
#!/bin/bash
#BSUB -J abl_${TAG}
#BSUB -q gpuv100
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

  echo "Submitted: ${TAG}"
done

echo "Done. Monitor: bjobs -u \$USER | grep rem_emg"
