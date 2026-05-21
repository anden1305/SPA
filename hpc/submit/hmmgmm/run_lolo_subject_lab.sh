#!/bin/bash
### cHMMGMVAE leave-one-lab-out (LOLO) experiments with subject + lab conditioning
### Train on two labs, validate on the held-out lab. Mirrors
### hpc/submit/lab_conditioning/run_lab_conditioning_lab_only_LOLO.sh.

set -euo pipefail

LABS=("lab2" "lab3" "lab5")
QUEUE="gpuv100"
WALLTIME="4:00"
MEM="5GB"
CPUS="4"

echo "Submitting cHMMGMVAE LOLO (subject_lab) jobs..."

for lab in "${LABS[@]}"; do
  CONFIG="src/config/run/cvaeprior/hmmgmm/lolo_subject_lab/generalization_lab_holdout_${lab}.yaml"
  OUTPUT_DIR="hpc/output/hmmgmm/lolo_subject_lab/${lab}"

  mkdir -p "$OUTPUT_DIR"

  bsub <<EOF
#!/bin/bash
#BSUB -J hmmgmm_lolo_subject_lab_${lab}
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
EOF

  echo "Submitted: ${lab}"
done

echo "Done. Monitor with: bjobs"
