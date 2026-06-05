#!/bin/bash
# cGMVAE interactive tests — run on a compute node (linuxsh), NOT hpclogin.
#
#   linuxsh
#   cd /work3/s204070/SPA && source .venv/bin/activate
#   bash hpc/submit/cv4fold/interactive_cgmvae_linuxsh.sh cgmvae_lab3_seq1_fold4
#
# Generate config:
#   PYTHONPATH=. python3 scripts/cv4fold/generate_configs.py --phase interactive --variants cgmvae_lab3_seq1_fold4

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "$SPA_ROOT"

VARIANT="${1:-cgmvae_lab3_seq1_fold4}"
CFG="src/config/run/cvaeprior/cv4fold/interactive/${VARIANT}.yaml"

if [[ ! -f "$CFG" ]]; then
  echo "Missing $CFG — run:"
  echo "  PYTHONPATH=. python3 scripts/cv4fold/generate_configs.py --phase interactive --variants ${VARIANT}"
  exit 1
fi

module load cuda/12.8.1 2>/dev/null || true
source .venv/bin/activate

echo "=== Train: $VARIANT ==="
python3 main.py --method train_vae --config_path "$CFG"

echo "=== Validate GMM (metrics.txt parity with decoder-only reliability): $VARIANT ==="
python3 main.py --method validate_cvae_gmm --config_path "$CFG"

RUN_PREFIX=$(python3 -c "import yaml; print(yaml.safe_load(open('$CFG'))['run_name'])")
RESULTS_BASE=$(python3 -c "import yaml; print(yaml.safe_load(open('$CFG'))['results_dir'])")
LATEST=$(ls -td "${RESULTS_BASE}/${RUN_PREFIX}"_* 2>/dev/null | head -1)

if [[ -n "${LATEST}" ]]; then
  echo "=== Postprocess: ${LATEST} ==="
  python3 -m scripts.cv4fold.postprocess_fold --result-root "${LATEST}"
fi

echo ""
echo "=== Key outputs ==="
echo "  metrics:  ${LATEST}/1/plots/metrics.txt"
echo "  tripanel: ${LATEST}/1/plots/hmm_tripanel_pc1_pc2.png"
echo ""
grep -E '^NMI:|^Likelihood:' "${LATEST}/1/plots/metrics.txt" 2>/dev/null || true
