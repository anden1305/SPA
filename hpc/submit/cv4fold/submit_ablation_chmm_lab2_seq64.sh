#!/bin/bash
# lab_2 cHMM seq64 — 3 pruned stability jobs (see ablation_chmm_lab2_seq64.md).
# Generate: PYTHONPATH=. python3 scripts/cv4fold/generate_ablation_chmm_lab2_seq64.py
# Submit:   bash hpc/submit/cv4fold/submit_ablation_chmm_lab2_seq64.sh

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "${SPA_ROOT}"

QUEUE="gpuv100"
MEM="6GB"
CPUS="4"
ROOT="src/config/run/cvaemarhmm/cv4fold/ablation_chmm_seq64/lab_2"
LOG_ROOT="hpc/output/cv4fold/ablation_chmm_seq64/lab_2"

# Walltimes from completed LSF runs (Run time in *.out):
#   lab_2 hmm_gmm seq64: 971s (~16 min)  — 28609769
#   lab_2 hmm_gmm seq1:  3948s (~66 min)
#   lab_2 notch50_warm seq1: 11182s (~3.1 h) — 28609381
# seq64 ≈ 0.25× seq1 for simple prior → notch50_warm seq64 est. ~45–60 min.
declare -A WALLTIME=(
  ["sweep_chmm_winner_seq64"]="1:30"
  ["beta_anneal_seq64"]="1:30"
  ["notch50_warm_seq64"]="2:30"
)

CONFIGS=(
  sweep_chmm_winner_seq64
  notch50_warm_seq64
  beta_anneal_seq64
)

mkdir -p "${LOG_ROOT}"
echo "Submitting ${#CONFIGS[@]} lab_2 seq64 cHMM jobs (${QUEUE})..."

for stem in "${CONFIGS[@]}"; do
  config="${ROOT}/${stem}.yaml"
  wall="${WALLTIME[$stem]:-2:00}"
  if [[ ! -f "${config}" ]]; then
    echo "Missing ${config} — run generate_ablation_chmm_lab2_seq64.py" >&2
    exit 1
  fi

  bsub <<EOF
#!/bin/bash
#BSUB -J cv4_l2_s64_${stem%%_*}
#BSUB -q ${QUEUE}
#BSUB -W ${wall}
#BSUB -n ${CPUS}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=${MEM}]"
#BSUB -o ${LOG_ROOT}/${stem}_%J.out
#BSUB -e ${LOG_ROOT}/${stem}_%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
cd /work3/s204070/SPA
module load cuda/12.8.1
source .venv/bin/activate
python3 main.py --method train_vae --config_path ${config}
EOF

  echo "Submitted: ${stem} (${wall})"
done

echo "Done. Monitor: bjobs -u \$USER | grep cv4_l2_s64"
