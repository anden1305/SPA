#!/bin/bash
# Internal helper: bsub one train_vae job with explicit queue.
# Sourced by resubmit scripts — do not set -u at file scope.

_bsub_train_vae() {
QUEUE="$1"
WALL="$2"
MEM="$3"
TAG="$4"
CONFIG="$5"
LOG_DIR="$6"
LOG_STEM="$7"
CPUS="${CPUS:-4}"

mkdir -p "${LOG_DIR}"

bsub <<EOF
#!/bin/bash
#BSUB -J ${TAG}
#BSUB -q ${QUEUE}
#BSUB -W ${WALL}
#BSUB -n ${CPUS}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=${MEM}]"
#BSUB -o ${LOG_DIR}/${LOG_STEM}_%J.out
#BSUB -e ${LOG_DIR}/${LOG_STEM}_%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
cd /work3/s204070/SPA
module load cuda/12.8.1
source .venv/bin/activate
export PYTHONPATH=/work3/s204070/SPA
python3 main.py --method train_vae --config_path ${CONFIG}
EOF

echo "Submitted: ${TAG} (${QUEUE}, ${WALL})"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  set -euo pipefail
  _bsub_train_vae "$@"
fi
