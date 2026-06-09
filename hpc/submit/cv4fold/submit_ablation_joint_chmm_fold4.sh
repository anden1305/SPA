#!/bin/bash
# Joint holdout fold-4 cHMM-GMVAE ablations (~3h wall, 9 jobs).
# Generate first:
#   PYTHONPATH=. python3 scripts/cv4fold/generate_ablation_joint_chmm_fold4.py
# Run with bash, NOT `bsub < ...`.
# Queue mix: gpuv100×5, gpua100×1, gpua10×1, gpul40s×1, gpua40×1

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "${SPA_ROOT}"

MEM="6GB"
CPUS="4"
WALLTIME="3:00"

declare -A JOB_QUEUES=(
  ["warm_prior"]="gpuv100"
  ["seq32"]="gpuv100"
  ["warm_seq32"]="gpuv100"
  ["lr8e4"]="gpuv100"
  ["sticky92"]="gpuv100"
  ["latent8"]="gpua100"
  ["no_beta20"]="gpua10"
  ["emb8"]="gpul40s"
  ["warm_latent8"]="gpua40"
)

ABLATIONS=(
  warm_prior
  seq32
  warm_seq32
  lr8e4
  sticky92
  latent8
  no_beta20
  emb8
  warm_latent8
)

mkdir -p hpc/output/cv4fold/ablation_joint_chmm

for abl_id in "${ABLATIONS[@]}"; do
  config="src/config/run/cvaemarhmm/cv4fold/ablation_joint_chmm/fold_4/${abl_id}.yaml"
  if [[ ! -f "${config}" ]]; then
    echo "Missing ${config} — run generate_ablation_joint_chmm_fold4.py" >&2
    exit 1
  fi

  queue="${JOB_QUEUES[$abl_id]:-gpuv100}"
  tag="abl_jchmm_f4_${abl_id}"

  bsub <<EOF
#!/bin/bash
#BSUB -J ${tag}
#BSUB -q ${queue}
#BSUB -W ${WALLTIME}
#BSUB -n ${CPUS}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=${MEM}]"
#BSUB -o hpc/output/cv4fold/ablation_joint_chmm/f4_${abl_id}_%J.out
#BSUB -e hpc/output/cv4fold/ablation_joint_chmm/f4_${abl_id}_%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
cd /work3/s204070/SPA
module load cuda/12.8.1
source .venv/bin/activate
python3 main.py --method train_vae --config_path ${config}
EOF

  echo "Submitted: ${tag} (${queue}, ${WALLTIME})"
done

echo "Done. Monitor: bjobs -u \$USER | grep abl_jchmm_f4"
