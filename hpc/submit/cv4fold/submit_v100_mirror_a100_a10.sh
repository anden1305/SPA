#!/bin/bash
# Mirror PEND gpua100/gpua10 jobs on gpuv100 (duplicate submit; first finish wins).
# Run after checking: bjobs -u $USER -noheader -o "jobid queue stat job_name" | awk '$2 ~ /gpua100|gpua10/ && $3=="PEND"'
#
#   bash hpc/submit/cv4fold/submit_v100_mirror_a100_a10.sh

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "${SPA_ROOT}"

QUEUE="gpuv100"
CPUS="4"

_submit_vae() {
  local tag="$1"
  local config="$2"
  local wall="$3"
  local mem="$4"
  local out_dir="$5"
  local out_stem="$6"

  bsub <<EOF
#!/bin/bash
#BSUB -J ${tag}
#BSUB -q ${QUEUE}
#BSUB -W ${wall}
#BSUB -n ${CPUS}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=${mem}]"
#BSUB -o ${out_dir}/${out_stem}_v100_%J.out
#BSUB -e ${out_dir}/${out_stem}_v100_%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
cd /work3/s204070/SPA
module load cuda/12.8.1
source .venv/bin/activate
python3 main.py --method train_vae --config_path ${config}
EOF
  echo "Submitted v100 mirror: ${tag}"
}

_submit_train() {
  local tag="$1"
  local config="$2"
  local wall="$3"
  local mem="$4"
  local out_dir="$5"
  local out_stem="$6"

  bsub <<EOF
#!/bin/bash
#BSUB -J ${tag}
#BSUB -q ${QUEUE}
#BSUB -W ${wall}
#BSUB -n ${CPUS}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=${mem}]"
#BSUB -o ${out_dir}/${out_stem}_v100_%J.out
#BSUB -e ${out_dir}/${out_stem}_v100_%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
cd /work3/s204070/SPA
module load cuda/12.8.1
source .venv/bin/activate
python3 main.py --method train --config_path ${config}
EOF
  echo "Submitted v100 mirror: ${tag}"
}

echo "=== Joint locked (a100/a10 mirrors) ==="
_submit_vae "cv4_joint_locked_f2_chmmgmvae_v100" \
  "src/config/run/cvaemarhmm/cv4fold/joint_holdout/fold_2/chmmgmvae_locked.yaml" \
  "2:00" "6GB" "hpc/output/cv4fold/joint_holdout" "f2_chmmgmvae_locked"
_submit_vae "cv4_joint_locked_f3_cgmvae_v100" \
  "src/config/run/cvaemarhmm/cv4fold/joint_holdout/fold_3/cgmvae_locked.yaml" \
  "1:30" "6GB" "hpc/output/cv4fold/joint_holdout" "f3_cgmvae_locked"
_submit_vae "cv4_joint_locked_f3_hmmgmvae_v100" \
  "src/config/run/cvaemarhmm/cv4fold/joint_holdout/fold_3/hmmgmvae_locked.yaml" \
  "2:00" "6GB" "hpc/output/cv4fold/joint_holdout" "f3_hmmgmvae_locked"

echo "=== Within-lab HMMGMVAE (a100/a10 mirrors) ==="
for spec in \
  "1|lab_2" "1|lab_3" "1|lab_5" "4|lab_3" "4|lab_5"; do
  fold="${spec%%|*}"
  lab="${spec##*|}"
  _submit_vae "cv4_within_locked_f${fold}_${lab}_hmmgmvae_v100" \
    "src/config/run/cvaemarhmm/cv4fold/unified_holdout/${lab}/fold_${fold}/hmmgmvae_locked.yaml" \
    "2:00" "6GB" "hpc/output/cv4fold/unified_holdout" "f${fold}_${lab}_hmmgmvae_locked"
done

echo "=== HMM raw (a100/a10 mirrors) ==="
for spec in \
  "1|lab_2" "1|lab_3" "1|lab_5" "4|lab_3" "4|lab_5"; do
  fold="${spec%%|*}"
  lab="${spec##*|}"
  _submit_train "cv4_hmm_raw_f${fold}_${lab}_v100" \
    "src/config/run/hmm/cv4fold/unified_holdout/${lab}/fold_${fold}/hmm_raw.yaml" \
    "12:00" "8GB" "hpc/output/cv4fold/hmm_raw" "f${fold}_${lab}"
done

echo ""
echo "Done. 13 v100 mirror jobs submitted."
echo "Monitor: bjobs -u \$USER | grep _v100"
