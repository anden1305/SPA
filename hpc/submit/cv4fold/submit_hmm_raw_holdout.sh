#!/bin/bash
# HMM on raw EEG/EMG — cv4fold holdout (1 run per fold).
# Joint folds 1–4 + within-lab (3 labs × 4 folds) = 16 jobs.
# Skips joint fold 4 if results already exist (see results/cv4fold/hmm_raw/joint/fold_4/).
#
# Generate:
#   PYTHONPATH=. python3 scripts/cv4fold/generate_hmm_raw_holdout.py
# Submit:
#   bash hpc/submit/cv4fold/submit_hmm_raw_holdout.sh

set -euo pipefail
SPA_ROOT="/work3/s204070/SPA"
cd "${SPA_ROOT}"
# shellcheck source=_queue_mix.sh
source "${SPA_ROOT}/hpc/submit/cv4fold/_queue_mix.sh"

JOB_IDX="${JOB_IDX:-0}"
LABS=(lab_2 lab_3 lab_5)
MEM="8GB"
CPUS="4"
WALLTIME="12:00"

mkdir -p hpc/output/cv4fold/hmm_raw

_has_joint_result() {
  local fold="$1"
  local d="${SPA_ROOT}/results/cv4fold/hmm_raw/joint/fold_${fold}"
  # Skip when validation finished or fold accepted via partial log NMI (see hmm_raw_joint_partial.json).
  find "${d}" -name validations.json 2>/dev/null | grep -q . && return 0
  python3 - <<PY
import json, sys
from pathlib import Path
p = Path("${SPA_ROOT}/paper/overleaf/tables/hmm_raw_joint_partial.json")
fold = "${fold}"
if p.is_file() and fold in json.loads(p.read_text()).get("folds", {}):
    sys.exit(0)
sys.exit(1)
PY
}

_submit_joint() {
  local fold="$1"
  if _has_joint_result "${fold}"; then
    echo "Skip joint fold ${fold} (results exist under results/cv4fold/hmm_raw/joint/fold_${fold})"
    return 0
  fi
  _pick_queue
  local queue="${PICKED_QUEUE}"
  local config="src/config/run/hmm/cv4fold/joint_holdout/fold_${fold}/hmm_raw.yaml"
  if [[ ! -f "${config}" ]]; then
    echo "Missing ${config} — run generate_hmm_raw_holdout.py" >&2
    exit 1
  fi
  local tag="cv4_hmm_raw_joint_f${fold}"

  bsub <<EOF
#!/bin/bash
#BSUB -J ${tag}
#BSUB -q ${queue}
#BSUB -W ${WALLTIME}
#BSUB -n ${CPUS}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=${MEM}]"
#BSUB -o hpc/output/cv4fold/hmm_raw/joint_f${fold}_%J.out
#BSUB -e hpc/output/cv4fold/hmm_raw/joint_f${fold}_%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
cd /work3/s204070/SPA
module load cuda/12.8.1
source .venv/bin/activate
python3 main.py --method train --config_path ${config}
EOF

  echo "Submitted: ${tag} (${queue}, ${WALLTIME})"
}

_submit_within() {
  local fold="$1"
  local lab="$2"
  _pick_queue
  local queue="${PICKED_QUEUE}"
  local config="src/config/run/hmm/cv4fold/unified_holdout/${lab}/fold_${fold}/hmm_raw.yaml"
  if [[ ! -f "${config}" ]]; then
    echo "Missing ${config}" >&2
    exit 1
  fi
  local tag="cv4_hmm_raw_f${fold}_${lab}"

  bsub <<EOF
#!/bin/bash
#BSUB -J ${tag}
#BSUB -q ${queue}
#BSUB -W ${WALLTIME}
#BSUB -n ${CPUS}
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=${MEM}]"
#BSUB -o hpc/output/cv4fold/hmm_raw/f${fold}_${lab}_%J.out
#BSUB -e hpc/output/cv4fold/hmm_raw/f${fold}_${lab}_%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
cd /work3/s204070/SPA
module load cuda/12.8.1
source .venv/bin/activate
python3 main.py --method train --config_path ${config}
EOF

  echo "Submitted: ${tag} (${queue}, ${WALLTIME})"
}

echo "=== Joint HMM raw (folds 1–4) ==="
for fold in 1 2 3 4; do
  _submit_joint "${fold}"
done

echo "=== Within-lab HMM raw ==="
for fold in 1 2 3 4; do
  for lab in "${LABS[@]}"; do
    _submit_within "${fold}" "${lab}"
  done
done

echo ""
echo "Done. Monitor: bjobs -u \$USER | grep cv4_hmm_raw"
echo "Logs: hpc/output/cv4fold/hmm_raw/*.out"
