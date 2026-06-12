#!/bin/bash
# CPU job: Fig 27 + t-SNE for any population K that already has results.npz.
# Safe to run while GPU K-sweep jobs are still pending (read-only on training dirs).
#
#   bsub < hpc/submit/paper/run_population_biology_meeting_incremental.sh

#BSUB -J paper_bio_pop_inc
#BSUB -q hpc
#BSUB -W 0:45
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -o hpc/output/paper/population_biology_meeting_incremental_%J.out
#BSUB -e hpc/output/paper/population_biology_meeting_incremental_%J.err

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "${SPA_ROOT}"
source .venv/bin/activate
export PYTHONPATH="${SPA_ROOT}"

python3 scripts/paper/run_biology_meeting_pack.py \
  --root results/cv4fold/paper_k_sweep/population \
  --out-dir results/cv4fold/paper_figures/biology_meeting_population \
  --protocol population \
  --professor-dir docs/paper/figures/professor_meeting_population \
  --fig27-tsne-only

echo "Done → results/cv4fold/paper_figures/biology_meeting_population/ (Fig27 + t-SNE for available K)"
