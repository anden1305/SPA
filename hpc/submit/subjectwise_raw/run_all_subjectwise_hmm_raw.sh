#!/bin/bash
### Subject-wise HMM Raw Experiments
### Launches HMM training with raw data for each subject
### Creates one job per subject

set -euo pipefail

# Assign subjects to different queues
# gpuv100: 6 subjects, gpul40s: 2 subjects, gpua100: 2 subjects
declare -A SUBJECT_QUEUES=(
    ["sub038"]="gpuv100"
    ["sub039"]="gpuv100"
    ["sub041"]="gpuv100"
    ["sub043"]="gpuv100"
    ["sub048"]="gpuv100"
    ["sub054"]="gpuv100"
    ["sub056"]="gpua100"
    ["sub059"]="gpul40s"
    ["sub060"]="gpul40s"
    ["sub069"]="gpua100"
)

SUBJECTS=("sub038" "sub039" "sub041" "sub043" "sub048" "sub054" "sub056" "sub059" "sub060" "sub069")
WALLTIME="6:00"
MEM="5GB"
CPUS="4"

echo "Launching subject-wise HMM raw experiments for ${#SUBJECTS[@]} subjects..."

for subject in "${SUBJECTS[@]}"; do
    QUEUE="${SUBJECT_QUEUES[$subject]}"
    CONFIG="src/config/run/hmm/subjectwise_raw/${subject}_hmm_mssv_raw.yaml"
    OUTPUT_DIR="hpc/output/subjectwise_raw/hmm/${subject}"
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Submit job
    bsub <<EOF
#!/bin/bash
#BSUB -J hmm_raw_${subject}
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

    echo "Submitted: HMM raw for ${subject} (queue: ${QUEUE})"
done

echo ""
echo "All jobs submitted. Monitor with: bjobs"
echo "Output will be in: hpc/output/subjectwise_raw/hmm/"
echo "Results will be in: results/subjectwise_raw/hmm/"

# Usage: ./hpc/submit/subjectwise_raw/run_all_subjectwise_hmm_raw.sh
