#!/bin/bash
# Joint fold 4: LR swap sensitivity (cHMM @ 3e-4, cGMVAE @ 1.3e-3).
# Generate: PYTHONPATH=. python3 scripts/cv4fold/generate_joint_lr_swap_fold4.py

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "${SPA_ROOT}"

MEM="6GB"
CPUS="4"
QUEUE="gpuv100"
FOLD=4

mkdir -p hpc/output/cv4fold/joint_holdout
mkdir -p hpc/output/cv4fold/ablation_joint_chmm

submit_job() {
  local tag="$1"
  local config="$2"
  local wall="$3"
  local logdir="$4"
  local logstem="$5"

  bsub <<EOF
#!/bin/bash
#BSUB -J ${tag}
#BSUB -q ${QUEUE}
#BSUB -W ${wall}
#BSUB -n ${CPUS}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=${MEM}]"
#BSUB -o ${logdir}/${logstem}_%J.out
#BSUB -e ${logdir}/${logstem}_%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
cd /work3/s204070/SPA
module load cuda/12.8.1
source .venv/bin/activate
python3 main.py --method train_vae --config_path ${config}
EOF

  echo "Submitted: ${tag} (${QUEUE}, ${wall})"
}

chmm_cfg="src/config/run/cvaemarhmm/cv4fold/ablation_joint_chmm/fold_${FOLD}/warm_emb8_sticky92_lr3e4.yaml"
cgmvae_cfg="src/config/run/cvaemarhmm/cv4fold/joint_holdout/fold_${FOLD}/cgmvae_emb8_lr1p3e3.yaml"

[[ -f "${chmm_cfg}" ]] || { echo "Missing ${chmm_cfg}" >&2; exit 1; }
[[ -f "${cgmvae_cfg}" ]] || { echo "Missing ${cgmvae_cfg}" >&2; exit 1; }

submit_job "cv4_lrswap_chmm_f4" "${chmm_cfg}" "3:00" "hpc/output/cv4fold/ablation_joint_chmm" "f4_warm_emb8_sticky92_lr3e4"
submit_job "cv4_lrswap_cgmvae_f4" "${cgmvae_cfg}" "4:00" "hpc/output/cv4fold/joint_holdout" "f4_cgmvae_emb8_lr1p3e3"

echo "Done. Monitor: bjobs -u \$USER | grep cv4_lrswap"
