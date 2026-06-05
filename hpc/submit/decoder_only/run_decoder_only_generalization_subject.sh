#!/bin/bash
### Decoder-only cGMVAE generalization_subject experiments
### One job per held-out subject config.

set -euo pipefail

SPA_ROOT="/work3/s204070/SPA"
SUBJECTS=("sub039" "sub041" "sub048")
QUEUE="gpuv100"
WALLTIME="8:00"
MEM="6GB"
CPUS="4"

echo "Submitting decoder-only generalization_subject jobs..."

for subject in "${SUBJECTS[@]}"; do
  CONFIG="src/config/run/cvaeprior/decoder_only/generalization_subject/generalization_subject_cgmvae_decoder_only_subject_conditioning_${subject}.yaml"
  OUTPUT_DIR="hpc/output/decoder_only/generalization_subject/${subject}"

  mkdir -p "$OUTPUT_DIR"

  bsub <<EOF
#!/bin/bash
#BSUB -J dec_only_gs_${subject}
#BSUB -q ${QUEUE}
#BSUB -W ${WALLTIME}
#BSUB -n ${CPUS}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=${MEM}]"
#BSUB -o ${OUTPUT_DIR}/%J.out
#BSUB -e ${OUTPUT_DIR}/%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

module load cuda/12.8.1
cd ${SPA_ROOT}
source .venv/bin/activate

python3 main.py --method train_vae --config_path ${CONFIG}
python3 main.py --method validate_cvae_gmm --config_path ${CONFIG}
EOF

  echo "Submitted: ${subject}"
done

echo "Done. Monitor with: bjobs"
