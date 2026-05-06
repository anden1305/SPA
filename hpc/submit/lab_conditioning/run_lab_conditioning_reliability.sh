#!/bin/bash
### Lab-conditioning cGMVAE reliability experiments

set -euo pipefail

CONFIGS=(
  "src/config/run/cvaeprior/lab_conditioning/reliability/lab3_cgmvae_lab_conditioning_mssv_frequency.yaml"
  "src/config/run/cvaeprior/lab_conditioning/reliability/lab5_cgmvae_lab_conditioning_mssv_frequency.yaml"
)

QUEUE="gpuv100"
WALLTIME="4:00"
MEM="5GB"
CPUS="4"

echo "Submitting lab-conditioning reliability jobs..."

for CONFIG in "${CONFIGS[@]}"; do
  LAB=$(basename "$CONFIG" | grep -oP 'lab\d' || echo "unknown")
  OUTPUT_DIR="hpc/output/lab_conditioning/reliability/${LAB}"

  mkdir -p "$OUTPUT_DIR"

  bsub <<EOF
#!/bin/bash
#BSUB -J lab_cond_rel_${LAB}
#BSUB -q ${QUEUE}
#BSUB -W ${WALLTIME}
#BSUB -n ${CPUS}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=${MEM}]"
#BSUB -o ${OUTPUT_DIR}/%J.out
#BSUB -e ${OUTPUT_DIR}/%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

module load cuda/12.8.1
source .venv/bin/activate

python3 main.py --method train_vae --config_path ${CONFIG}
python3 main.py --method validate_cvae_gmm --config_path ${CONFIG}
EOF

  echo "Submitted: ${CONFIG}"
done

echo "Done. Monitor with: bjobs"
