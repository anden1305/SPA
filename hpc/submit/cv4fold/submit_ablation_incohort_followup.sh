#!/bin/bash
# Incohort follow-up cGMVAE ablations (analysis matrix 2026-06-08).
# Generate first:
#   source .venv/bin/activate
#   PYTHONPATH=. python3 scripts/cv4fold/generate_ablation_incohort_followup.py
# Submit:
#   bash hpc/submit/cv4fold/submit_ablation_incohort_followup.sh

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "${SPA_ROOT}"

QUEUE="gpuv100"
MEM="6GB"
CPUS="4"

JOBS=(
  "followup_lab_5_emg_wide_notch50|6:00|lab_5|src/config/run/cvaemarhmm/cv4fold/ablation_followup/lab_5/emg_wide_notch50.yaml"
  "followup_lab_2_wide_mlp|4:00|lab_2|src/config/run/cvaemarhmm/cv4fold/ablation_followup/lab_2/wide_mlp.yaml"
  "followup_lab_2_epochs300|6:00|lab_2|src/config/run/cvaemarhmm/cv4fold/ablation_followup/lab_2/epochs300.yaml"
  "followup_lab_2_rem_recall_ckpt|4:00|lab_2|src/config/run/cvaemarhmm/cv4fold/ablation_followup/lab_2/rem_recall_ckpt.yaml"
)

mkdir -p hpc/output/cv4fold/ablation_followup/lab_2
mkdir -p hpc/output/cv4fold/ablation_followup/lab_5
echo "Submitting ${#JOBS[@]} incohort follow-up jobs (${QUEUE})..."

for entry in "${JOBS[@]}"; do
  IFS='|' read -r TAG WALLTIME LAB CONFIG <<< "${entry}"
  if [[ ! -f "${CONFIG}" ]]; then
    echo "Missing ${CONFIG} — run generate_ablation_incohort_followup.py first" >&2
    exit 1
  fi

  bsub <<EOF
#!/bin/bash
#BSUB -J abl_${TAG}
#BSUB -q ${QUEUE}
#BSUB -W ${WALLTIME}
#BSUB -n ${CPUS}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=${MEM}]"
#BSUB -o hpc/output/cv4fold/ablation_followup/${LAB}/${TAG}_%J.out
#BSUB -e hpc/output/cv4fold/ablation_followup/${LAB}/${TAG}_%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
cd /work3/s204070/SPA
module load cuda/12.8.1
source .venv/bin/activate
python3 main.py --method train_vae --config_path ${CONFIG}
EOF

  echo "Submitted: ${TAG} (${WALLTIME}, ${QUEUE})"
done

echo "Done. Monitor: bjobs -u \$USER | grep abl_followup"
