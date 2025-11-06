#!/bin/bash
### to make sure that all errors are caught
set -euo pipefail

### guard to check for minimum arguments
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <sweep_yaml> <num_agents> [trials_per_agent] [walltime] [queue]"
  exit 1
fi

SWEEP_YAML=$1
NUM_AGENTS=$2
### default values for optional arguments. Default is 1 trial per agent
TRIALS_PER_AGENT=${3:-1}
### default is 4 hours per job
WALLTIME=${4:-6:00}
### default is gpuv100, but options are [gpuv100, gpua100, gpua10, gpul40s]
### bqueues | grep -i gpu
QUEUE=${5:-gpul40s}
### bjobs -p

ABS_SWEEP_YAML=$(realpath "$SWEEP_YAML")

echo "Creating sweep from $ABS_SWEEP_YAML..."

# If a .env file exists, export its variables (API key, project, entity)
if [ -f .env ]; then
  echo "Loading environment from .env"
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Disable any job email notifications from LSF (empty string disables mail)
export LSB_MAILTO=""

# Minimal preflight checks
if [ ! -f .venv/bin/activate ]; then
  echo "ERROR: Python virtual env not found at .venv/bin/activate. Create it and try again."
  exit 3
fi
if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "ERROR: WANDB_API_KEY not set. Add it to .env or export it in your shell, then retry."
  exit 4
fi

# Capture only the last line (our printed id) and strip whitespace to avoid
# picking up any extra wandb stdout like "Create sweep with ID: ..."
SWEEP_ID=$(source .venv/bin/activate && python3 -m src.training.wandb_sweep_runner "$ABS_SWEEP_YAML" --create-only | tail -n 1 | tr -d '[:space:]')
echo "Created sweep: $SWEEP_ID"

if [ -z "$SWEEP_ID" ]; then
  echo "ERROR: Failed to obtain a W&B sweep id. Ensure WANDB_API_KEY is set and you have network access. Aborting."
  exit 2
fi

OUTPUT_DIR=$(pwd)/hpc/output/sweep_${SWEEP_ID}
mkdir -p "$OUTPUT_DIR"

echo "Submitting $NUM_AGENTS agent jobs as an LSF array (each agent runs $TRIALS_PER_AGENT trials)..."
### NOTE:
### Avoid single quotes around shell variables inside the command string passed to bsub.
### Single quotes prevent expansion in this shell, so the remote shell would see an empty --sweep-id.
### Use escaped double quotes instead to both expand here and preserve spaces.
bsub -J "wandb_sweep_agent[1-${NUM_AGENTS}]" \
  -q "$QUEUE" \
  -o "$OUTPUT_DIR/sweep_${SWEEP_ID}_job_%J_agent_%I.out" \
  -e "$OUTPUT_DIR/sweep_${SWEEP_ID}_job_%J_agent_%I.err" \
  -n 4 \
  -R "rusage[mem=4GB]" \
  -R "span[hosts=1]" \
  -W "$WALLTIME" \
  -gpu "num=1" \
  "bash -lc \"# Minimal, quiet init to avoid 'Modules Release ... Usage' banner
  source /etc/profile.d/modules.sh >/dev/null 2>&1 || true; \
  module --silent try-load cuda/12.8.1 >/dev/null 2>&1 || true; \
  source .venv/bin/activate; \
  # Prevent core dumps if the job is killed or crashes
  ulimit -c 0; \
  echo Running agent for sweep: \"$SWEEP_ID\"; \
  python3 -m src.training.wandb_sweep_runner \"$ABS_SWEEP_YAML\" --agent-only --sweep-id \"$SWEEP_ID\" --trials-per-agent \"$TRIALS_PER_AGENT\"\""

echo "Submitted array job. Monitor with bjobs and check hpc/output/sweep_${SWEEP_ID}/ for logs."

### Example usage:

### Synthetic
### bash hpc/submit/launch_wandb_sweep.sh src/config/sweep/hmm/hmm_synth_features.yaml 15 1
### bash hpc/submit/launch_wandb_sweep.sh src/config/sweep/marhmm/marhmm_synth_features.yaml 15 1
### bash hpc/submit/launch_wandb_sweep.sh src/config/sweep/hmm/hmm_synth_raw.yaml 15 1
### bash hpc/submit/launch_wandb_sweep.sh src/config/sweep/marhmm/marhmm_synth_raw.yaml 15 1

### MSSV
### bash hpc/submit/launch_wandb_sweep.sh src/config/sweep/hmm/hmm_mssv_features.yaml 15 1
### bash hpc/submit/launch_wandb_sweep.sh src/config/sweep/marhmm/marhmm_mssv_features.yaml 15 1


