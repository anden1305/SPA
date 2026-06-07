#!/bin/bash
# Round 4: EEG 0.3-25 Hz bandpass on lab_3 and lab_5 (mid band vs locked 0-20).
# Generate configs first:
#   cd /work3/s204070/SPA && source .venv/bin/activate
#   PYTHONPATH=. python3 scripts/cv4fold/generate_ablation_prepro.py
# Run with bash (wrapper calls bsub in a loop), NOT `bsub < ...`.
# See docs/cv4fold/ablations/ablation_prepro_lab2_lab5.md

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "${SPA_ROOT}"

JOBS=(
  "lab_3_bp25:src/config/run/cvaemarhmm/cv4fold/ablation_prepro/lab_3/bp25.yaml"
  "lab_3_no_postnorm_bp25:src/config/run/cvaemarhmm/cv4fold/ablation_prepro/lab_3/no_postnorm_bp25.yaml"
  "lab_5_bp25:src/config/run/cvaemarhmm/cv4fold/ablation_prepro/lab_5/bp25.yaml"
  "lab_5_no_postnorm_bp25:src/config/run/cvaemarhmm/cv4fold/ablation_prepro/lab_5/no_postnorm_bp25.yaml"
)

mkdir -p "hpc/output/cv4fold/ablation_prepro"
echo "Submitting ${#JOBS[@]} round-4 bp25 ablation jobs..."

for entry in "${JOBS[@]}"; do
  TAG="${entry%%:*}"
  CONFIG="${entry#*:}"
  if [[ ! -f "${CONFIG}" ]]; then
    echo "Missing ${CONFIG} — run generate_ablation_prepro.py first" >&2
    exit 1
  fi

  bsub <<EOF
#!/bin/bash
#BSUB -J abl_${TAG}
#BSUB -q gpuv100
#BSUB -W 4:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=6GB]"
#BSUB -o hpc/output/cv4fold/ablation_prepro/${TAG}_%J.out
#BSUB -e hpc/output/cv4fold/ablation_prepro/${TAG}_%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
cd /work3/s204070/SPA
module load cuda/12.8.1
source .venv/bin/activate
python3 main.py --method train_vae --config_path ${CONFIG}
EOF

  echo "Submitted: ${TAG} (${CONFIG})"
done

echo "Done. Monitor: bjobs -u \$USER | grep abl_lab_[35]_"
