#!/bin/bash
# Rerun holdout configs whose latest timestamp dir is missing seed(s).
# Full 3-seed jobs; summarize_holdout_ladder.py picks newest complete run.
#
#   bash hpc/submit/cv4fold/submit_incomplete_seed_reruns.sh

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "${SPA_ROOT}"

QUEUE="gpuv100"
MEM="6GB"
CPUS="4"
WALLTIME_CGMVAE="1:30"
WALLTIME_CHMM="2:00"

mkdir -p hpc/output/cv4fold/seed_reruns

_submit() {
  local tag="$1"
  local config="$2"
  local wall="$3"
  local log_stem="$4"

  bsub <<EOF
#!/bin/bash
#BSUB -J ${tag}
#BSUB -q ${QUEUE}
#BSUB -W ${wall}
#BSUB -n ${CPUS}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=${MEM}]"
#BSUB -o hpc/output/cv4fold/seed_reruns/${log_stem}_%J.out
#BSUB -e hpc/output/cv4fold/seed_reruns/${log_stem}_%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
cd /work3/s204070/SPA
module load cuda/12.8.1
source .venv/bin/activate
python3 main.py --method train_vae --config_path ${config}
EOF

  echo "Submitted: ${tag} (${QUEUE}, ${wall})"
}

echo "=== Joint cGMVAE f1–f4 (missing seed 3) ==="
for fold in 1 2 3 4; do
  _submit \
    "cv4_seed_rerun_f${fold}_cgmvae" \
    "src/config/run/cvaemarhmm/cv4fold/joint_holdout/fold_${fold}/cgmvae_locked.yaml" \
    "${WALLTIME_CGMVAE}" \
    "joint_f${fold}_cgmvae"
done

echo "=== Within-lab cGMVAE (missing seed 3) ==="
_submit \
  "cv4_seed_rerun_f1_lab3_cgmvae" \
  "src/config/run/cvaemarhmm/cv4fold/unified_holdout/lab_3/fold_1/cgmvae_locked.yaml" \
  "${WALLTIME_CGMVAE}" \
  "within_f1_lab_3_cgmvae"

_submit \
  "cv4_seed_rerun_f3_lab5_cgmvae" \
  "src/config/run/cvaemarhmm/cv4fold/unified_holdout/lab_5/fold_3/cgmvae_locked.yaml" \
  "${WALLTIME_CGMVAE}" \
  "within_f3_lab_5_cgmvae"

_submit \
  "cv4_seed_rerun_f4_lab5_cgmvae" \
  "src/config/run/cvaemarhmm/cv4fold/unified_holdout/lab_5/fold_4/cgmvae_locked.yaml" \
  "${WALLTIME_CGMVAE}" \
  "within_f4_lab_5_cgmvae"

echo "=== Within f4 lab_5 cHMMGMVAE (missing seeds 2–3) ==="
_submit \
  "cv4_seed_rerun_f4_lab5_chmmgmvae" \
  "src/config/run/cvaemarhmm/cv4fold/unified_holdout/lab_5/fold_4/chmmgmvae_locked.yaml" \
  "${WALLTIME_CHMM}" \
  "within_f4_lab_5_chmmgmvae"

echo ""
echo "Done. 8 jobs on ${QUEUE}."
echo "Monitor: bjobs -u \$USER | grep cv4_seed_rerun"
echo "Logs: hpc/output/cv4fold/seed_reruns/*.out"
