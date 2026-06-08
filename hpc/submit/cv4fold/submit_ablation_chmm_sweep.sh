#!/bin/bash
# Submit full incohort cHMM-GMVAE ablation sweep (all YAMLs under ablation_chmm/).
# Generate first:
#   PYTHONPATH=. python3 scripts/cv4fold/generate_ablation_chmm_incohort.py --phase all
# Run with bash, NOT `bsub < ...`.
# Queue mix: gpuv100×8 (primary), gpua10×1, gpua100×2, gpul40s×1
# See docs/cv4fold/ablations/ablation_chmm_incohort.md

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "${SPA_ROOT}"

MEM="6GB"
CPUS="4"
WALLTIME_DEFAULT="6:00"
WALLTIME_LAB5="6:00"

# Key: lab/stem (relative to ablation_chmm/)
declare -A JOB_QUEUES=(
  ["lab_2/hmm_gmm_locked"]="gpuv100"
  ["lab_2/warm_hmm_gmm_locked"]="gpuv100"
  ["lab_2/notch50_simple"]="gpuv100"
  ["lab_2/notch50_warm"]="gpua10"
  ["lab_3/hmm_gmm_locked"]="gpuv100"
  ["lab_3/warm_hmm_gmm_locked"]="gpua100"
  ["lab_5/hmm_gmm_locked"]="gpuv100"
  ["lab_5/warm_hmm_gmm_locked"]="gpuv100"
  ["lab_5/emg_wide_simple"]="gpuv100"
  ["lab_5/emg_wide_warm"]="gpua100"
  ["lab_5/emg_wide_notch50_simple"]="gpuv100"
  ["lab_5/emg_wide_notch50_warm"]="gpul40s"
)

mkdir -p hpc/output/cv4fold/ablation_chmm

shopt -s nullglob
configs=(src/config/run/cvaemarhmm/cv4fold/ablation_chmm/*/*.yaml)
if [[ ${#configs[@]} -eq 0 ]]; then
  echo "No configs — run: PYTHONPATH=. python3 scripts/cv4fold/generate_ablation_chmm_incohort.py --phase all" >&2
  exit 1
fi

echo "Submitting ${#configs[@]} cHMM jobs (gpuv100×8, gpua10×1, gpua100×2, gpul40s×1)..."

for config in "${configs[@]}"; do
  lab="$(basename "$(dirname "${config}")")"
  stem="$(basename "${config}" .yaml)"
  key="${lab}/${stem}"
  tag="cv4_chmm_${lab}_${stem}"
  queue="${JOB_QUEUES[$key]:-gpuv100}"

  if [[ "${lab}" == "lab_5" ]]; then
    wall="${WALLTIME_LAB5}"
  else
    wall="${WALLTIME_DEFAULT}"
  fi

  bsub <<EOF
#!/bin/bash
#BSUB -J ${tag}
#BSUB -q ${queue}
#BSUB -W ${wall}
#BSUB -n ${CPUS}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=${MEM}]"
#BSUB -o hpc/output/cv4fold/ablation_chmm/${lab}_${stem}_%J.out
#BSUB -e hpc/output/cv4fold/ablation_chmm/${lab}_${stem}_%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
cd /work3/s204070/SPA
module load cuda/12.8.1
source .venv/bin/activate
python3 main.py --method train_vae --config_path ${config}
EOF

  echo "Submitted: ${tag} (${queue}, ${wall})"
done

echo "Done. Monitor: bjobs -u \$USER | grep cv4_chmm"
