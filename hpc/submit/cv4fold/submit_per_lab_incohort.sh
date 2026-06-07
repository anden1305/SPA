#!/bin/bash
# Per-lab in-cohort cGMVAE (HQ cohort only): lab_5 then lab_2.
# Generate configs first:
#   cd /work3/s204070/SPA && source .venv/bin/activate
#   PYTHONPATH=. python3 scripts/cv4fold/generate_configs.py --phase per_lab_incohort --labs lab_5 lab_2

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "${SPA_ROOT}"

LABS=("lab_5" "lab_2")

echo "Submitting per-lab in-cohort cGMVAE jobs (HQ cohort only)..."

for lab in "${LABS[@]}"; do
  CONFIG="src/config/run/cvaemarhmm/cv4fold/per_lab_incohort/${lab}/cgmvae.yaml"
  if [[ ! -f "${CONFIG}" ]]; then
    echo "Missing ${CONFIG} — run generate_configs.py first" >&2
    exit 1
  fi

  mkdir -p "hpc/output/cv4fold/per_lab_incohort"

  bsub <<EOF
#!/bin/bash
#BSUB -J cv4_pl_${lab}
#BSUB -q gpuv100
#BSUB -W 6:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=6GB]"
#BSUB -o hpc/output/cv4fold/per_lab_incohort/${lab}_%J.out
#BSUB -e hpc/output/cv4fold/per_lab_incohort/${lab}_%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
cd /work3/s204070/SPA
module load cuda/12.8.1
source .venv/bin/activate
python3 main.py --method train_vae --config_path ${CONFIG}
EOF

  echo "Submitted: ${lab} (${CONFIG})"
done

echo "Done. Monitor: bjobs -u \$USER | grep cv4_pl"
