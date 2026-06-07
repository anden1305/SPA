#!/bin/bash
# Round 3 ablation: paper-style lab-agnostic front-end (median/IQR robust
# scaling + 0.3-35 Hz EEG band, post/pre-norm off) on lab_2/3/5.
# Generate configs first:
#   cd /work3/s204070/SPA && source .venv/bin/activate
#   PYTHONPATH=. python3 scripts/cv4fold/generate_ablation_prepro.py
# Run with bash (wrapper that calls bsub in a loop), NOT `bsub < ...`.
# References (baseline/no_postnorm) are from rounds 1-2 — not resubmitted here.
# See docs/cv4fold/ablations/ablation_prepro_lab2_lab5.md

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "${SPA_ROOT}"

JOBS=(
  "lab_2_paper_robust:src/config/run/cvaemarhmm/cv4fold/ablation_prepro/lab_2/paper_robust.yaml"
  "lab_3_paper_robust:src/config/run/cvaemarhmm/cv4fold/ablation_prepro/lab_3/paper_robust.yaml"
  "lab_5_paper_robust:src/config/run/cvaemarhmm/cv4fold/ablation_prepro/lab_5/paper_robust.yaml"
)

mkdir -p "hpc/output/cv4fold/ablation_prepro"
echo "Submitting ${#JOBS[@]} paper_robust ablation jobs..."

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

echo "Done. Monitor: bjobs -u \$USER | grep abl_"
