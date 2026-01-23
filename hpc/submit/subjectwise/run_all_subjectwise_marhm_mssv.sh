#!/bin/bash
### Subject-wise MARHMM Experiments
### Launches MARHMM training for each subject (train/val on all runs for that subject)
### Creates one job per subject

set -euo pipefail

SUBJECTS=("sub038" "sub039" "sub041" "sub043" "sub048" "sub054" "sub056" "sub059" "sub060" "sub069")
QUEUE="gpuv100"
WALLTIME="3:00"
MEM="5GB"
CPUS="4"

echo "Launching subject-wise MARHMM experiments for ${#SUBJECTS[@]} subjects..."

for subject in "${SUBJECTS[@]}"; do
    CONFIG="src/config/run/marhmm/subjectwise/${subject}_marhmm_mssv_features.yaml"
    OUTPUT_DIR="hpc/output/subjectwise/marhmm/${subject}"
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Submit job
    bsub <<EOF
#!/bin/bash
#BSUB -J marhmm_${subject}
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

    echo "Submitted: MARHMM for ${subject}"
done

echo ""
echo "All jobs submitted. Monitor with: bjobs"
echo "Output will be in: hpc/output/subjectwise/marhmm/"
echo "Results will be in: results/subjectwise/marhmm/"

# Usage:
# bash hpc/submit/subjectwise/run_all_subjectwise_marhm_mssv.sh
