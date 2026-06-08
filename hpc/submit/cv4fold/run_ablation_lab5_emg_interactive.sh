#!/bin/bash
# Run lab_5 EMG ablations on an *interactive* GPU shell (voltash / sxm2sh / a100sh).
# NOT for login node. Kill matching batch jobs first if still PEND:
#   bkill <JOBID>
#
# Usage (on compute, one job per shell — or set CUDA_VISIBLE_DEVICES):
#   bash hpc/submit/cv4fold/run_ablation_lab5_emg_interactive.sh wide_mlp_notch50
#   bash hpc/submit/cv4fold/run_ablation_lab5_emg_interactive.sh wide_mlp_emg_wide
#   bash hpc/submit/cv4fold/run_ablation_lab5_emg_interactive.sh wide_mlp_emg_wide_notch50
#
# Logs: hpc/output/cv4fold/ablation_emg/lab_5/<variant>_interactive.log

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "${SPA_ROOT}"

VARIANT="${1:-}"
case "${VARIANT}" in
  wide_mlp_notch50)
    CONFIG="src/config/run/cvaemarhmm/cv4fold/ablation_emg/lab_5/wide_mlp_notch50.yaml"
    ;;
  wide_mlp_emg_wide)
    CONFIG="src/config/run/cvaemarhmm/cv4fold/ablation_emg/lab_5/wide_mlp_emg_wide.yaml"
    ;;
  wide_mlp_emg_wide_notch50)
    CONFIG="src/config/run/cvaemarhmm/cv4fold/ablation_emg/lab_5/wide_mlp_emg_wide_notch50.yaml"
    ;;
  *)
    echo "Usage: $0 {wide_mlp_notch50|wide_mlp_emg_wide|wide_mlp_emg_wide_notch50}" >&2
    exit 1
    ;;
esac

if [[ "$(hostname)" == hpclogin* ]]; then
  echo "Refusing to run on login node. Use voltash / sxm2sh / a100sh first." >&2
  exit 1
fi

module load cuda/12.8.1
source .venv/bin/activate

LOG="hpc/output/cv4fold/ablation_emg/lab_5/lab_5_${VARIANT}_interactive.log"
mkdir -p "$(dirname "${LOG}")"

echo "GPU: ${CUDA_VISIBLE_DEVICES:-all}  host: $(hostname)  config: ${CONFIG}"
echo "Log: ${LOG}"
python3 main.py --method train_vae --config_path "${CONFIG}" 2>&1 | tee "${LOG}"
