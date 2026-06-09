#!/bin/bash
# Joint fold-4 cHMM lock candidate: warm + emb8 + sticky92 + T=64
# Generate: PYTHONPATH=. python3 scripts/cv4fold/generate_ablation_joint_chmm_fold4.py --round 3

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "${SPA_ROOT}"

MEM="6GB"
CPUS="4"
WALLTIME="3:00"
QUEUE="gpuv100"
ABL_ID="warm_emb8_sticky92"

config="src/config/run/cvaemarhmm/cv4fold/ablation_joint_chmm/fold_4/${ABL_ID}.yaml"
if [[ ! -f "${config}" ]]; then
  echo "Missing ${config} — run generate_ablation_joint_chmm_fold4.py --round 3" >&2
  exit 1
fi

mkdir -p hpc/output/cv4fold/ablation_joint_chmm

bsub <<EOF
#!/bin/bash
#BSUB -J abl_jchmm_f4_${ABL_ID}
#BSUB -q ${QUEUE}
#BSUB -W ${WALLTIME}
#BSUB -n ${CPUS}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=${MEM}]"
#BSUB -o hpc/output/cv4fold/ablation_joint_chmm/f4_${ABL_ID}_%J.out
#BSUB -e hpc/output/cv4fold/ablation_joint_chmm/f4_${ABL_ID}_%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
cd /work3/s204070/SPA
module load cuda/12.8.1
source .venv/bin/activate
python3 main.py --method train_vae --config_path ${config}
EOF

echo "Submitted: abl_jchmm_f4_${ABL_ID} (${QUEUE}, ${WALLTIME})"
