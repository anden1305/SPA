#!/bin/bash
### Decoder-only cGMVAE subjectwise experiments
### Submits one job per subject config. Each job runs:
### 1) train_vae
### 2) validate_cvae_gmm

set -euo pipefail

SUBJECTS=("sub038" "sub039" "sub041" "sub043" "sub048" "sub054" "sub056" "sub059" "sub060" "sub069")
QUEUE="gpuv100"
WALLTIME="8:00"
MEM="6GB"
CPUS="4"

echo "Submitting decoder-only subjectwise jobs for ${#SUBJECTS[@]} subjects..."

for subject in "${SUBJECTS[@]}"; do
  CONFIG="src/config/run/cvaeprior/decoder_only/subjectwise/${subject}_cgmvae_decoder_only_mssv_frequency.yaml"
  OUTPUT_DIR="hpc/output/decoder_only/subjectwise/${subject}"

  mkdir -p "$OUTPUT_DIR"

  bsub <<EOF
#!/bin/bash
#BSUB -J dec_only_sw_${subject}
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

  echo "Submitted: ${subject}"
done

echo "Done. Monitor with: bjobs"
