#!/bin/bash
### Lab-conditioning cGMVAE Leave-One-Lab-Out (LOLO) experiments with lab+subject conditioning
### Trains on two labs, tests on held-out lab (with subject conditioning for domain robustness)

set -euo pipefail

LABS=("lab2" "lab3" "lab5")
QUEUE="gpua100"
WALLTIME="4:00"
MEM="5GB"
CPUS="4"

echo "Submitting lab-conditioning lab+subject LOLO jobs..."

for lab in "${LABS[@]}"; do
  CONFIG="src/config/run/cvaeprior/lab_conditioning/lab_and_subject_conditioning_LOLO/generalization_lab_holdout_${lab}.yaml"
  OUTPUT_DIR="hpc/output/lab_conditioning/lab_and_subject_LOLO/${lab}"

  mkdir -p "$OUTPUT_DIR"

  bsub <<EOF
#!/bin/bash
#BSUB -J lab_cond_lolo_lab_subj_${lab}
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

python3 main.py --method train --config_path ${CONFIG}
EOF

  echo "Submitted: ${lab}"
done

echo "Done. Monitor with: bjobs"
