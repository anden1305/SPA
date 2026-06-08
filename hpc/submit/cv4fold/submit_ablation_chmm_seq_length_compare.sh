#!/bin/bash
# Submit seq-length compare: locked winner recipe only (3 labs × T, default T=32 64 128).
# Generate first:
#   PYTHONPATH=. python3 scripts/cv4fold/generate_ablation_chmm_incohort.py \
#     --phase seq_length_compare --sequence-lengths 32 64 128 --prior-tier simple
# Run with bash, NOT `bsub < ...`.

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "${SPA_ROOT}"

MEM="6GB"
CPUS="4"
PRIOR_TIER="${PRIOR_TIER:-simple}"
if [[ "${PRIOR_TIER}" == "simple" ]]; then
  STEM="hmm_gmm_locked"
else
  STEM="warm_hmm_gmm_locked"
fi

declare -A WALLTIME=(
  ["lab_2"]="8:00"
  ["lab_3"]="12:00"
  ["lab_5"]="12:00"
)
declare -A WALLTIME_64=(
  ["lab_2"]="12:00"
  ["lab_3"]="24:00"
  ["lab_5"]="24:00"
)
declare -A WALLTIME_128=(
  ["lab_2"]="24:00"
  ["lab_3"]="24:00"
  ["lab_5"]="24:00"
)

declare -A JOB_QUEUES=(
  ["lab_2"]="gpuv100"
  ["lab_3"]="gpua100"
  ["lab_5"]="gpuv100"
)

if [[ $# -eq 0 ]]; then
  SEQ_LENGTHS=(32 64 128)
else
  SEQ_LENGTHS=("$@")
fi

for SEQ_LEN in "${SEQ_LENGTHS[@]}"; do
  ROOT="src/config/run/cvaemarhmm/cv4fold/ablation_chmm_seq${SEQ_LEN}"
  LOG_ROOT="hpc/output/cv4fold/ablation_chmm_seq${SEQ_LEN}"
  mkdir -p "${LOG_ROOT}"
  stem="${STEM}_seq${SEQ_LEN}"

  for lab in lab_2 lab_3 lab_5; do
    config="${ROOT}/${lab}/${stem}.yaml"
    if [[ ! -f "${config}" ]]; then
      echo "Missing ${config} — run generate_ablation_chmm_incohort.py --phase seq_length_compare" >&2
      exit 1
    fi

    case "${SEQ_LEN}" in
      64) wall="${WALLTIME_64[$lab]:-12:00}" ;;
      128) wall="${WALLTIME_128[$lab]:-24:00}" ;;
      *) wall="${WALLTIME[$lab]:-8:00}" ;;
    esac
    queue="${JOB_QUEUES[$lab]:-gpuv100}"
    tag="cv4_chmm_seq${SEQ_LEN}_${lab}"

    bsub <<EOF
#!/bin/bash
#BSUB -J ${tag}
#BSUB -q ${queue}
#BSUB -W ${wall}
#BSUB -n ${CPUS}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=${MEM}]"
#BSUB -o ${LOG_ROOT}/${lab}_${stem}_%J.out
#BSUB -e ${LOG_ROOT}/${lab}_${stem}_%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
cd /work3/s204070/SPA
module load cuda/12.8.1
source .venv/bin/activate
python3 main.py --method train_vae --config_path ${config}
EOF

    echo "Submitted: ${tag} (${queue}, ${wall})"
  done
done

echo "Done. Monitor: bjobs -u \$USER | grep cv4_chmm_seq"
