#!/bin/bash
# lab_5: EMG-wide (+ optional 50 Hz notch) on locked wide_mlp.
# Generate first:
#   source .venv/bin/activate
#   PYTHONPATH=. python3 scripts/cv4fold/generate_ablation_lab5_emg.py
# Submit (batch — long PEND on gpuv100/gpua10 is normal):
#   bash hpc/submit/cv4fold/submit_ablation_lab5_emg.sh
#
# Interactive fallback (after bkill on PEND jobs):
#   voltash   # or sxm2sh / a100sh — one shell per variant
#   bash hpc/submit/cv4fold/run_ablation_lab5_emg_interactive.sh wide_mlp_emg_wide
#   bash hpc/submit/cv4fold/run_ablation_lab5_emg_interactive.sh wide_mlp_emg_wide_notch50

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "${SPA_ROOT}"

WALLTIME="4:00"
MEM="6GB"
CPUS="4"
QUEUE="gpuv100"

JOBS=(
  "lab_5_wide_mlp_emg_wide:src/config/run/cvaemarhmm/cv4fold/ablation_emg/lab_5/wide_mlp_emg_wide.yaml"
  "lab_5_wide_mlp_emg_wide_notch50:src/config/run/cvaemarhmm/cv4fold/ablation_emg/lab_5/wide_mlp_emg_wide_notch50.yaml"
)

mkdir -p hpc/output/cv4fold/ablation_emg/lab_5
echo "Submitting ${#JOBS[@]} lab_5 EMG ablation jobs (${QUEUE})..."

for entry in "${JOBS[@]}"; do
  TAG="${entry%%:*}"
  CONFIG="${entry#*:}"
  if [[ ! -f "${CONFIG}" ]]; then
    echo "Missing ${CONFIG} — run generate_ablation_lab5_emg.py first" >&2
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
#BSUB -o hpc/output/cv4fold/ablation_emg/lab_5/${TAG}_%J.out
#BSUB -e hpc/output/cv4fold/ablation_emg/lab_5/${TAG}_%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
cd /work3/s204070/SPA
module load cuda/12.8.1
source .venv/bin/activate
python3 main.py --method train_vae --config_path ${CONFIG}
EOF

  echo "Submitted: ${TAG} (queue: ${QUEUE})"
done

echo "Done. Monitor: bjobs -u \$USER | grep abl_lab_5_wide_mlp_emg"
