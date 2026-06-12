#!/bin/bash
# CPU job: full biology meeting pack from population K-sweep (all 20 mice).
# Run after population training jobs finish.
#
#   bsub < hpc/submit/paper/run_population_biology_meeting.sh

#BSUB -J paper_bio_pop
#BSUB -q hpc
#BSUB -W 2:00
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -o hpc/output/paper/population_biology_meeting_%J.out
#BSUB -e hpc/output/paper/population_biology_meeting_%J.err

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "${SPA_ROOT}"
source .venv/bin/activate
export PYTHONPATH="${SPA_ROOT}"

python3 scripts/paper/run_biology_meeting_pack.py \
  --root results/cv4fold/paper_k_sweep/population \
  --out-dir results/cv4fold/paper_figures/biology_meeting_population \
  --protocol population \
  --professor-dir docs/paper/figures/professor_meeting_population

echo "Done → results/cv4fold/paper_figures/biology_meeting_population/"
