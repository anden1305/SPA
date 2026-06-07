#!/bin/bash
# Per-lab no_beta_epochs ablation (0 vs 10) on each lab's locked best-NMI recipe.
# Generate first:
#   source .venv/bin/activate
#   PYTHONPATH=. python3 scripts/cv4fold/generate_ablation_no_beta_epochs.py
# Default emits no_beta_epochs=0 only (control 10 = existing winner runs).
# Full 0+10 re-run:
#   PYTHONPATH=. python3 scripts/cv4fold/generate_ablation_no_beta_epochs.py --values 0 10

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "${SPA_ROOT}"

WALLTIME="4:00"
MEM="6GB"
CPUS="4"
QUEUE="gpuv100"

# lab:config_path relative to repo (after generate)
declare -a JOBS=(
  "lab_2:src/config/run/cvaemarhmm/cv4fold/ablation_beta/lab_2/no_beta_epochs_0_rem_emg_wide_eeg4.yaml"
  "lab_3:src/config/run/cvaemarhmm/cv4fold/ablation_beta/lab_3/no_beta_epochs_0_wide_mlp.yaml"
  "lab_5:src/config/run/cvaemarhmm/cv4fold/ablation_beta/lab_5/no_beta_epochs_0_wide_mlp.yaml"
)

# Optional second wave if generated with --values 0 10:
EXTRA=(
  "lab_2:src/config/run/cvaemarhmm/cv4fold/ablation_beta/lab_2/no_beta_epochs_10_rem_emg_wide_eeg4.yaml"
  "lab_3:src/config/run/cvaemarhmm/cv4fold/ablation_beta/lab_3/no_beta_epochs_10_wide_mlp.yaml"
  "lab_5:src/config/run/cvaemarhmm/cv4fold/ablation_beta/lab_5/no_beta_epochs_10_wide_mlp.yaml"
)
for e in "${EXTRA[@]}"; do
  cfg="${SPA_ROOT}/${e#*:}"
  [[ -f "${cfg}" ]] && JOBS+=("${e}")
done

mkdir -p hpc/output/cv4fold/ablation_beta
echo "Submitting ${#JOBS[@]} no_beta_epochs jobs (${QUEUE})..."

for entry in "${JOBS[@]}"; do
  LAB="${entry%%:*}"
  CONFIG="${entry#*:}"
  TAG="${LAB}_$(basename "${CONFIG}" .yaml)"
  if [[ ! -f "${CONFIG}" ]]; then
    echo "Missing ${CONFIG} — run generate_ablation_no_beta_epochs.py" >&2
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
#BSUB -o hpc/output/cv4fold/ablation_beta/${TAG}_%J.out
#BSUB -e hpc/output/cv4fold/ablation_beta/${TAG}_%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
cd /work3/s204070/SPA
module load cuda/12.8.1
source .venv/bin/activate
python3 main.py --method train_vae --config_path ${CONFIG}
EOF

  echo "Submitted: ${TAG}"
done

echo "Done. Monitor: bjobs -u \$USER | grep no_beta_epochs"
