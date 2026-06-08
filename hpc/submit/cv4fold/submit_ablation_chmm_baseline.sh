#!/bin/bash
# Incohort cHMM-GMVAE baseline ablations (scratch, locked cGMVAE prepro/arch).
# Generate configs first:
#   cd /work3/s204070/SPA && source .venv/bin/activate
#   PYTHONPATH=. python3 scripts/cv4fold/generate_ablation_chmm_incohort.py --phase all_baselines
# Run with bash (calls bsub in a loop), NOT `bsub < ...`.
# See docs/cv4fold/ablations/ablation_chmm_incohort.md

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "${SPA_ROOT}"

PHASE="${1:-all}"
case "${PHASE}" in
  simple|warm|all) ;;
  *)
    echo "Usage: bash hpc/submit/cv4fold/submit_ablation_chmm_baseline.sh [simple|warm|all]" >&2
    exit 1
    ;;
esac

LABS=(lab_2 lab_3 lab_5)
mkdir -p "hpc/output/cv4fold/ablation_chmm"

submit_one() {
  local lab="$1"
  local variant="$2"
  local tag="$3"
  local config="src/config/run/cvaemarhmm/cv4fold/ablation_chmm/${lab}/${variant}.yaml"
  if [[ ! -f "${config}" ]]; then
    echo "Missing ${config} — run generate_ablation_chmm_incohort.py first" >&2
    exit 1
  fi

  bsub <<EOF
#!/bin/bash
#BSUB -J ${tag}
#BSUB -q gpuv100
#BSUB -W 6:00
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=6GB]"
#BSUB -o hpc/output/cv4fold/ablation_chmm/${lab}_${tag#cv4_chmm_}_%J.out
#BSUB -e hpc/output/cv4fold/ablation_chmm/${lab}_${tag#cv4_chmm_}_%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
cd /work3/s204070/SPA
module load cuda/12.8.1
source .venv/bin/activate
python3 main.py --method train_vae --config_path ${config}
EOF

  echo "Submitted: ${tag} (${config})"
}

if [[ "${PHASE}" == "simple" || "${PHASE}" == "all" ]]; then
  for lab in "${LABS[@]}"; do
    submit_one "${lab}" "hmm_gmm_locked" "cv4_chmm_${lab}_hmm_gmm"
  done
fi

if [[ "${PHASE}" == "warm" || "${PHASE}" == "all" ]]; then
  for lab in "${LABS[@]}"; do
    submit_one "${lab}" "warm_hmm_gmm_locked" "cv4_chmm_${lab}_warm"
  done
fi

echo "Done. Monitor: bjobs -u \$USER | grep cv4_chmm"
