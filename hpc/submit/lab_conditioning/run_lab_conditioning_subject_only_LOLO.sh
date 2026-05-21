#!/bin/bash
### Lab-conditioning cGMVAE leave-one-lab-out (LOLO) with subject-only decoder conditioning
### Same lab holdouts as lab-only and lab+subject LOLO; trains on two labs, tests on held-out lab

set -euo pipefail

LABS=("lab2" "lab3" "lab5")
QUEUE="gpuv100"
WALLTIME="4:00"
MEM="5GB"
CPUS="4"

echo "Submitting lab-conditioning subject-only LOLO jobs..."

for lab in "${LABS[@]}"; do
  CONFIG="src/config/run/cvaeprior/lab_conditioning/subject_conditioning_not_lab_LOLO/generalization_lab_holdout_${lab}.yaml"
  OUTPUT_DIR="hpc/output/lab_conditioning/subject_conditioning_not_lab_LOLO/${lab}"

  mkdir -p "$OUTPUT_DIR"

  bsub <<EOF
#!/bin/bash
#BSUB -J lab_cond_lolo_subj_only_${lab}
#BSUB -q ${QUEUE}
#BSUB -W ${WALLTIME}
#BSUB -n ${CPUS}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=${MEM}]"
#BSUB -o ${OUTPUT_DIR}/%J.out
#BSUB -e ${OUTPUT_DIR}/%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

cd /work3/s204070/SPA || exit 1
module load cuda/12.8.1
source .venv/bin/activate

python3 main.py --method train_vae --config_path ${CONFIG}
python3 main.py --method validate_cvae_gmm --config_path ${CONFIG}
EOF

  echo "Submitted: ${lab}"
done

echo "Done. Monitor with: bjobs"
