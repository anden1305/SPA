#!/bin/bash
# Duplicate-submit PEND gpuv100 cv4fold jobs on spill queues (no bkill).
# First finish wins; kill extras manually once results exist.
#
# Queues: gpua100, gpua10, gpul40s, gpua40 (rotating).
#
#   bash hpc/submit/cv4fold/submit_mirror_pending_spill_queues.sh

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "${SPA_ROOT}"

CPUS="4"
SPILL_QUEUES=(gpua100 gpua10 gpul40s gpua40)
SPILL_IDX=0

_pick_spill_queue() {
  PICKED_QUEUE="${SPILL_QUEUES[$(( SPILL_IDX % ${#SPILL_QUEUES[@]} ))]}"
  SPILL_IDX=$(( SPILL_IDX + 1 ))
}

_submit_hmm_raw_joint() {
  local fold="$1"
  _pick_spill_queue
  local queue="${PICKED_QUEUE}"
  local config="src/config/run/hmm/cv4fold/joint_holdout/fold_${fold}/hmm_raw.yaml"
  local tag="cv4_hmm_raw_joint_f${fold}_spill"

  bsub <<EOF
#!/bin/bash
#BSUB -J ${tag}
#BSUB -q ${queue}
#BSUB -W 12:00
#BSUB -n ${CPUS}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -o hpc/output/cv4fold/hmm_raw/joint_f${fold}_spill_%J.out
#BSUB -e hpc/output/cv4fold/hmm_raw/joint_f${fold}_spill_%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
cd /work3/s204070/SPA
module load cuda/12.8.1
source .venv/bin/activate
python3 main.py --method train --config_path ${config}
EOF
  echo "Submitted HMM raw joint f${fold} → ${queue}"
}

_submit_hmm_raw_within() {
  local fold="$1"
  local lab="$2"
  _pick_spill_queue
  local queue="${PICKED_QUEUE}"
  local config="src/config/run/hmm/cv4fold/unified_holdout/${lab}/fold_${fold}/hmm_raw.yaml"
  local tag="cv4_hmm_raw_f${fold}_${lab}_spill"

  bsub <<EOF
#!/bin/bash
#BSUB -J ${tag}
#BSUB -q ${queue}
#BSUB -W 12:00
#BSUB -n ${CPUS}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -o hpc/output/cv4fold/hmm_raw/f${fold}_${lab}_spill_%J.out
#BSUB -e hpc/output/cv4fold/hmm_raw/f${fold}_${lab}_spill_%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
cd /work3/s204070/SPA
module load cuda/12.8.1
source .venv/bin/activate
python3 main.py --method train --config_path ${config}
EOF
  echo "Submitted HMM raw f${fold} ${lab} → ${queue}"
}

_submit_seed_rerun() {
  local tag="$1"
  local config="$2"
  local wall="$3"
  local log_stem="$4"
  _pick_spill_queue
  local queue="${PICKED_QUEUE}"

  bsub <<EOF
#!/bin/bash
#BSUB -J ${tag}_spill
#BSUB -q ${queue}
#BSUB -W ${wall}
#BSUB -n ${CPUS}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=6GB]"
#BSUB -o hpc/output/cv4fold/seed_reruns/${log_stem}_spill_%J.out
#BSUB -e hpc/output/cv4fold/seed_reruns/${log_stem}_spill_%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
cd /work3/s204070/SPA
module load cuda/12.8.1
source .venv/bin/activate
python3 main.py --method train_vae --config_path ${config}
EOF
  echo "Submitted ${tag} → ${queue} (${wall})"
}

mkdir -p hpc/output/cv4fold/hmm_raw hpc/output/cv4fold/seed_reruns

echo "=== HMM raw joint (PEND on v100: f2, f3) ==="
for fold in 2 3; do
  _submit_hmm_raw_joint "${fold}"
done

echo "=== HMM raw within-lab (PEND on v100) ==="
for spec in \
  "1|lab_2" "1|lab_3" "1|lab_5" \
  "2|lab_2" "2|lab_3" \
  "3|lab_3" "3|lab_5" \
  "4|lab_2" "4|lab_3" "4|lab_5"; do
  fold="${spec%%|*}"
  lab="${spec##*|}"
  _submit_hmm_raw_within "${fold}" "${lab}"
done

echo "=== Seed reruns (PEND on v100) ==="
_submit_seed_rerun "cv4_seed_rerun_f3_cgmvae" \
  "src/config/run/cvaemarhmm/cv4fold/joint_holdout/fold_3/cgmvae_locked.yaml" \
  "1:30" "joint_f3_cgmvae"
_submit_seed_rerun "cv4_seed_rerun_f4_cgmvae" \
  "src/config/run/cvaemarhmm/cv4fold/joint_holdout/fold_4/cgmvae_locked.yaml" \
  "1:30" "joint_f4_cgmvae"
_submit_seed_rerun "cv4_seed_rerun_f1_lab3_cgmvae" \
  "src/config/run/cvaemarhmm/cv4fold/unified_holdout/lab_3/fold_1/cgmvae_locked.yaml" \
  "1:30" "within_f1_lab_3_cgmvae"
_submit_seed_rerun "cv4_seed_rerun_f3_lab5_cgmvae" \
  "src/config/run/cvaemarhmm/cv4fold/unified_holdout/lab_5/fold_3/cgmvae_locked.yaml" \
  "1:30" "within_f3_lab_5_cgmvae"
_submit_seed_rerun "cv4_seed_rerun_f4_lab5_cgmvae" \
  "src/config/run/cvaemarhmm/cv4fold/unified_holdout/lab_5/fold_4/cgmvae_locked.yaml" \
  "1:30" "within_f4_lab_5_cgmvae"
_submit_seed_rerun "cv4_seed_rerun_f4_lab5_chmmgmvae" \
  "src/config/run/cvaemarhmm/cv4fold/unified_holdout/lab_5/fold_4/chmmgmvae_locked.yaml" \
  "2:00" "within_f4_lab_5_chmmgmvae"

echo ""
echo "Done: 18 spill mirrors (v100 PEND jobs unchanged)."
echo "Queue mix: ${SPILL_QUEUES[*]}"
echo "Monitor: bjobs -u \$USER | grep -E 'spill|cv4_hmm_raw|cv4_seed_rerun'"
