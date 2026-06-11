#!/bin/bash
# CPU job: paper substage figures → results/cv4fold/paper_figures/ (isolated from results/substages*)
#
# Prereq: K-sweep finished (results/cv4fold/paper_k_sweep/fold_4/K*/.../plots/*/results.npz)
# Submit:
#   bsub < hpc/submit/paper/run_substage_figures.sh
#
# Override winner npz:
#   NPZ=results/cv4fold/paper_k_sweep/fold_4/K4/.../plots/3/results.npz bsub < ...

#BSUB -J paper_substage_fig
#BSUB -q hpc
#BSUB -W 0:45
#BSUB -n 2
#BSUB -R "rusage[mem=12GB]"
#BSUB -o hpc/output/paper/substage_figures_%J.out
#BSUB -e hpc/output/paper/substage_figures_%J.err

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "${SPA_ROOT}"
source .venv/bin/activate
export PYTHONPATH="${SPA_ROOT}"
mkdir -p hpc/output/paper results/cv4fold/paper_figures

NPZ="${NPZ:-results/cv4fold/paper_k_sweep/fold_4/K4/joint_k_sweep_f4_K4_20260609-183717/plots/3/results.npz}"

if [[ ! -f "${NPZ}" ]]; then
  echo "Missing ${NPZ} — set NPZ=.../results.npz or finish K-sweep first" >&2
  exit 1
fi

python3 scripts/paper/run_paper_substage_figures.py \
  --out-root results/cv4fold/paper_figures \
  --k-root results/cv4fold/paper_k_sweep/fold_4 \
  --npz "${NPZ}" \
  --copy-to-paper

echo "Done. Outputs under results/cv4fold/paper_figures/"
find results/cv4fold/paper_figures -type f \( -name '*.pdf' -o -name '*.png' -o -name '*.json' \) | sort
