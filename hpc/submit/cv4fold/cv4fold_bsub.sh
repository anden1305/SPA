#!/bin/bash
# bsub via heredoc so #BSUB directives are applied (bsub script.sh arg does NOT parse them).
# Source from submit_smoke_80.sh / submit_all_cv4fold.sh.

SPA_ROOT="/work3/s204070/SPA"

cv4fold_bsub_vae() {
  local cfg="$1"
  local queue="$2"
  local walltime="$3"
  local tag
  tag=$(basename "$cfg" .yaml)
  local mem="${CV4FOLD_VAE_MEM:-8GB}"

  mkdir -p "${SPA_ROOT}/hpc/output/cv4fold/vae"

  bsub <<EOF
#!/bin/bash
#BSUB -J cv4fold_vae_${tag}
#BSUB -q ${queue}
#BSUB -W ${walltime}
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=${mem}]"
#BSUB -o hpc/output/cv4fold/vae/%J.out
#BSUB -e hpc/output/cv4fold/vae/%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
cd ${SPA_ROOT}
module load cuda/12.8.1
source .venv/bin/activate

CONFIG="${cfg}"
python3 main.py --method train_vae --config_path "\${CONFIG}"

RUN_PREFIX=\$(python3 -c "import yaml; print(yaml.safe_load(open('\${CONFIG}'))['run_name'])")
RESULTS_BASE=\$(python3 -c "import yaml; print(yaml.safe_load(open('\${CONFIG}'))['results_dir'])")

LATEST=\$(ls -td "\${RESULTS_BASE}/\${RUN_PREFIX}"_* 2>/dev/null | head -1 || true)
if [[ -n "\${LATEST}" && -f "\${LATEST}/config.json" ]]; then
  python3 -m scripts.cv4fold.postprocess_fold --result-root "\${LATEST}"
  python3 -m scripts.cv4fold.select_best_vae_run --fold-dir "\${LATEST}" --key "\${LATEST}"
fi
EOF
}

cv4fold_bsub_hmm_raw() {
  local cfg="$1"
  local queue="$2"
  local walltime="$3"
  local tag
  tag=$(basename "$cfg" .yaml)
  local mem="${CV4FOLD_HMM_MEM:-12GB}"

  mkdir -p "${SPA_ROOT}/hpc/output/cv4fold/hmm_raw"

  bsub <<EOF
#!/bin/bash
#BSUB -J cv4fold_hmm_${tag}
#BSUB -q ${queue}
#BSUB -W ${walltime}
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=${mem}]"
#BSUB -o hpc/output/cv4fold/hmm_raw/%J.out
#BSUB -e hpc/output/cv4fold/hmm_raw/%J.err
#BSUB -gpu "num=1:mode=exclusive_process"

set -euo pipefail
cd ${SPA_ROOT}
module load cuda/12.8.1
source .venv/bin/activate

CONFIG="${cfg}"
python3 main.py --method train --config_path "\${CONFIG}"

RUN_PREFIX=\$(python3 -c "import yaml; print(yaml.safe_load(open('\${CONFIG}'))['run_name'])")
RESULTS_BASE=\$(python3 -c "import yaml; print(yaml.safe_load(open('\${CONFIG}'))['results_dir'])")

LATEST=\$(ls -td "\${RESULTS_BASE}/\${RUN_PREFIX}"_* 2>/dev/null | head -1 || true)
if [[ -n "\${LATEST}" && -f "\${LATEST}/config.json" ]]; then
  python3 -m scripts.cv4fold.postprocess_fold --result-root "\${LATEST}"
fi
EOF
}
