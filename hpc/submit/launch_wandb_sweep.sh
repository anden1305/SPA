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
WALLTIME=${4:-01:30}
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

# Capture only the last line (our printed id) and strip whitespace to avoid
# picking up any extra wandb stdout like "Create sweep with ID: ..."
SWEEP_ID=$(source .venv/bin/activate && python3 -m src.training.wandb_sweep_runner "$ABS_SWEEP_YAML" --create-only | tail -n 1 | tr -d '[:space:]')
echo "Created sweep: $SWEEP_ID"

if [ -z "$SWEEP_ID" ]; then
  echo "ERROR: Failed to obtain a W&B sweep id. Ensure WANDB_API_KEY is set and you have network access. Aborting."
  exit 2
fi

OUTPUT_DIR=$(pwd)/hpc/output
mkdir -p "$OUTPUT_DIR"

echo "Submitting $NUM_AGENTS agent jobs as an LSF array (each agent runs $TRIALS_PER_AGENT trials)..."
### NOTE:
### Avoid single quotes around shell variables inside the command string passed to bsub.
### Single quotes prevent expansion in this shell, so the remote shell would see an empty --sweep-id.
### Use escaped double quotes instead to both expand here and preserve spaces.
bsub -J "wandb_sweep_agent[1-${NUM_AGENTS}]" \
  -q "$QUEUE" \
  -o "$OUTPUT_DIR/wandb_sweep_%J_%I.out" \
  -e "$OUTPUT_DIR/wandb_sweep_%J_%I.err" \
  -n 4 \
  -R "rusage[mem=4GB]" \
  -R "span[hosts=1]" \
  -W "$WALLTIME" \
  -gpu "num=1" \
  "bash -lc \"module load cuda/12.8.1; source .venv/bin/activate; \
  # Load .env inside the remote job if present (API key, project, entity)
  if [ -f .env ]; then set -a; source .env; set +a; fi; \
  # Propagate W&B routing and API key from submitter/session if set
  export WANDB_API_KEY=\${WANDB_API_KEY:-}; export WANDB_PROJECT=\${WANDB_PROJECT:-}; export WANDB_ENTITY=\${WANDB_ENTITY:-}; \
  echo Running agent for sweep: \"$SWEEP_ID\"; \
  python3 -m src.training.wandb_sweep_runner \"$ABS_SWEEP_YAML\" --agent-only --sweep-id \"$SWEEP_ID\" --trials-per-agent \"$TRIALS_PER_AGENT\"\""

echo "Submitted array job. Monitor with bjobs and check hpc/output/ for logs."

### bash hpc/submit/launch_wandb_sweep.sh src/config/sweep/hmm_mssv_features.yaml 15 1
### bash hpc/submit/launch_wandb_sweep.sh src/config/sweep/marhmm_mssv_features.yaml 15 1
