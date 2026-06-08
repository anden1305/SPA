#!/bin/bash
# Phase 3 cHMM prepro deltas (conditional — run when baseline prior lags cGMVAE lock).
# Generate configs first:
#   PYTHONPATH=. python3 scripts/cv4fold/generate_ablation_chmm_incohort.py --phase prepro_delta --prior-tier simple
# Optional warm tier if prior lock still ambiguous:
#   PYTHONPATH=. python3 scripts/cv4fold/generate_ablation_chmm_incohort.py --phase prepro_delta --prior-tier both
# Run with bash, NOT `bsub < ...`.

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "${SPA_ROOT}"

# lab : yaml stem (without .yaml)
declare -a JOBS=()

while IFS= read -r -d '' f; do
  lab="$(basename "$(dirname "$f")")"
  stem="$(basename "$f" .yaml)"
  if [[ "${stem}" == *_locked ]]; then
    continue
  fi
  JOBS+=("${lab}:${stem}")
done < <(find src/config/run/cvaemarhmm/cv4fold/ablation_chmm -name '*.yaml' -print0 2>/dev/null || true)

if [[ ${#JOBS[@]} -eq 0 ]]; then
  echo "No prepro_delta configs found under ablation_chmm/ — run generate_ablation_chmm_incohort.py --phase prepro_delta" >&2
  exit 1
fi

mkdir -p "hpc/output/cv4fold/ablation_chmm"
echo "Submitting ${#JOBS[@]} cHMM prepro_delta jobs..."

for entry in "${JOBS[@]}"; do
  lab="${entry%%:*}"
  stem="${entry#*:}"
  config="src/config/run/cvaemarhmm/cv4fold/ablation_chmm/${lab}/${stem}.yaml"
  tag="cv4_chmm_${lab}_${stem}"

  bsub <<EOF
#!/bin/bash
#BSUB -J ${tag}
#BSUB -q gpuv100
#BSUB -W 6:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=6GB]"
#BSUB -o hpc/output/cv4fold/ablation_chmm/${lab}_${stem}_%J.out
#BSUB -e hpc/output/cv4fold/ablation_chmm/${lab}_${stem}_%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
cd /work3/s204070/SPA
module load cuda/12.8.1
source .venv/bin/activate
python3 main.py --method train_vae --config_path ${config}
EOF

  echo "Submitted: ${tag}"
done

echo "Done. Monitor: bjobs -u \$USER | grep cv4_chmm"
