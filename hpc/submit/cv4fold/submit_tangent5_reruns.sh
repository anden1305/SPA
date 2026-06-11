#!/bin/bash
# Best-of-5 seed reruns for weak locked holdout cells (high ROI only).
# Skips: lab_2/f2 (tangent done ~0.30/0.37), lab_3/f2 (tangent won 0.55), lab_5/f2 (tangent ~0.42).
#
# Submit:
#   bash hpc/submit/cv4fold/submit_tangent5_reruns.sh

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "${SPA_ROOT}"
# shellcheck source=_queue_mix.sh
source "${SPA_ROOT}/hpc/submit/cv4fold/_queue_mix.sh"

JOB_IDX="${JOB_IDX:-0}"
MEM="6GB"
CPUS="4"
WALLTIME_CGMVAE="1:30"
WALLTIME_CHMM="2:00"

JOBS=(
  "joint_f1_cgmvae|src/config/run/cvaemarhmm/cv4fold/joint_holdout/fold_1/cgmvae_locked_tangent5.yaml|${WALLTIME_CGMVAE}"
  "joint_f2_cgmvae|src/config/run/cvaemarhmm/cv4fold/joint_holdout/fold_2/cgmvae_locked_tangent5.yaml|${WALLTIME_CGMVAE}"
  "joint_f2_chmm|src/config/run/cvaemarhmm/cv4fold/joint_holdout/fold_2/chmmgmvae_locked_tangent5.yaml|${WALLTIME_CHMM}"
)

mkdir -p hpc/output/cv4fold/tangent5

for entry in "${JOBS[@]}"; do
  IFS='|' read -r TAG CONFIG WALL <<< "${entry}"

  if [[ ! -f "${CONFIG}" ]]; then
    echo "Missing ${CONFIG}" >&2
    exit 1
  fi

  _pick_queue
  queue="${PICKED_QUEUE}"

  bsub <<EOF
#!/bin/bash
#BSUB -J cv4_t5_${TAG}
#BSUB -q ${queue}
#BSUB -W ${WALL}
#BSUB -n ${CPUS}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=${MEM}]"
#BSUB -o hpc/output/cv4fold/tangent5/${TAG}_%J.out
#BSUB -e hpc/output/cv4fold/tangent5/${TAG}_%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
cd /work3/s204070/SPA
module load cuda/12.8.1
source .venv/bin/activate
python3 main.py --method train_vae --config_path ${CONFIG}
EOF

  echo "Submitted: cv4_t5_${TAG} (${queue}, ${WALL})"
done

echo ""
echo "Done. ${#JOBS[@]} tangent5 jobs."
echo "Monitor: bjobs -u \$USER | grep cv4_t5"
echo "Logs: hpc/output/cv4fold/tangent5/<tag>_%J.out"
