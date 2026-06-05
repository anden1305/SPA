#!/bin/bash
# Interactive cHMM-GMVAE tests — run on a compute node (linuxsh), NOT hpclogin.
#
#   linuxsh
#   cd /work3/s204070/SPA && source .venv/bin/activate
#   bash hpc/submit/cv4fold/interactive_chmmgmvae_linuxsh.sh quick_joint_fold4
#
# Variants (generate all: PYTHONPATH=. python3 scripts/cv4fold/generate_configs.py --phase interactive)
#   quick_joint_fold4        ~40 ep, fast sanity (~1–2 h)
#   stable_joint_fold4       120 ep, gentle HMM schedule (recommended)
#   tune_winner_joint_fold4  120 ep, lr=3e-4 (grid tune winner settings)
#   stable_per_lab3_fold4      120 ep, single lab — easier than joint

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "$SPA_ROOT"

VARIANT="${1:-quick_joint_fold4}"
CFG="src/config/run/cvaeprior/cv4fold/interactive/${VARIANT}.yaml"

if [[ ! -f "$CFG" ]]; then
  echo "Missing $CFG — run:"
  echo "  PYTHONPATH=. python3 scripts/cv4fold/generate_configs.py --phase interactive"
  exit 1
fi

module load cuda/12.8.1 2>/dev/null || true
source .venv/bin/activate

echo "=== Train: $VARIANT ==="
python3 main.py --method train_vae --config_path "$CFG"

RUN_PREFIX=$(python3 -c "import yaml; print(yaml.safe_load(open('$CFG'))['run_name'])")
RESULTS_BASE=$(python3 -c "import yaml; print(yaml.safe_load(open('$CFG'))['results_dir'])")
LATEST=$(ls -td "${RESULTS_BASE}/${RUN_PREFIX}"_* 2>/dev/null | head -1)

if [[ -z "${LATEST}" ]]; then
  echo "No result dir under ${RESULTS_BASE}/"
  exit 1
fi

echo "=== Postprocess: ${LATEST} ==="
python3 -m scripts.cv4fold.postprocess_fold --result-root "${LATEST}"

echo ""
echo "=== Key outputs ==="
echo "  metrics:  ${LATEST}/1/plots/metrics.txt"
echo "  tripanel: ${LATEST}/1/plots/hmm_tripanel_pc1_pc2.png"
echo "  per-mouse: ${LATEST}/per_mouse/run_1/"
echo ""
grep -E '^NMI:|^Predicted unique states:|^HMM switch' "${LATEST}/1/plots/metrics.txt" 2>/dev/null || true
