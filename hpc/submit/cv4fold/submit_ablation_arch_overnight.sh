#!/bin/bash
# Overnight architecture ablation (latent dim + wider MLP) on per-lab best prepro.
# Generate configs first:
#   cd /work3/s204070/SPA && source .venv/bin/activate
#   PYTHONPATH=. python3 scripts/cv4fold/generate_ablation_arch.py
# Run: bash hpc/submit/cv4fold/submit_ablation_arch_overnight.sh
# See docs/cv4fold/overnight_experiments_20260606.md

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "${SPA_ROOT}"

JOBS=()
for lab in lab_2 lab_3 lab_5; do
  for variant in lat4 lat16 wide_mlp wide_lat16; do
    JOBS+=("${lab}_${variant}:src/config/run/cvaemarhmm/cv4fold/ablation_arch/${lab}/${variant}.yaml")
  done
done

mkdir -p "hpc/output/cv4fold/ablation_arch"
echo "Submitting ${#JOBS[@]} architecture ablation jobs..."

for entry in "${JOBS[@]}"; do
  TAG="${entry%%:*}"
  CONFIG="${entry#*:}"
  if [[ ! -f "${CONFIG}" ]]; then
    echo "Missing ${CONFIG} — run generate_ablation_arch.py first" >&2
    exit 1
  fi

  bsub <<EOF
#!/bin/bash
#BSUB -J arch_${TAG}
#BSUB -q gpuv100
#BSUB -W 6:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -o hpc/output/cv4fold/ablation_arch/${TAG}_%J.out
#BSUB -e hpc/output/cv4fold/ablation_arch/${TAG}_%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
cd /work3/s204070/SPA
module load cuda/12.8.1
source .venv/bin/activate
python3 main.py --method train_vae --config_path ${CONFIG}
EOF

  echo "Submitted: arch_${TAG}"
done

echo "Done. Monitor: bjobs -u \$USER | grep arch_"
