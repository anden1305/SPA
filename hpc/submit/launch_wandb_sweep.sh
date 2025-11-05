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
WALLTIME=${4:-04:00}
### default is gpuv100
QUEUE=${5:-gpuv100}

ABS_SWEEP_YAML=$(realpath "$SWEEP_YAML")

echo "Creating sweep from $ABS_SWEEP_YAML..."
# Read 'count' from the sweep YAML (optional). We use a tiny python snippet to parse safely.
COUNT=$(python3 - <<PY
import sys, yaml
try:
    d = yaml.safe_load(open(sys.argv[1]))
    c = d.get('count') if isinstance(d, dict) else None
    print('' if c is None else int(c))
except Exception:
    print('')
PY
"$ABS_SWEEP_YAML")

if [ -n "$COUNT" ]; then
  echo "Sweep YAML requests count=$COUNT trials in total. Enforcing Option B: total_trials = num_agents * trials_per_agent."
  # If user didn't pass TRIALS_PER_AGENT explicitly (i.e. only 2 args), compute it if divisible
  if [ "$#" -lt 3 ]; then
    if [ $((COUNT % NUM_AGENTS)) -ne 0 ]; then
      echo "ERROR: count ($COUNT) is not divisible by num_agents ($NUM_AGENTS). Provide a trials_per_agent or choose a different num_agents." >&2
      exit 1
    fi
    TRIALS_PER_AGENT=$((COUNT / NUM_AGENTS))
    echo "Auto-set TRIALS_PER_AGENT=$TRIALS_PER_AGENT to evenly split $COUNT trials across $NUM_AGENTS agents."
  else
    # Validate provided TRIALS_PER_AGENT yields exact count
    PRODUCT=$((NUM_AGENTS * TRIALS_PER_AGENT))
    if [ "$PRODUCT" -ne "$COUNT" ]; then
      echo "ERROR: NUM_AGENTS * TRIALS_PER_AGENT = $PRODUCT, but sweep YAML count = $COUNT. Adjust arguments so product equals count." >&2
      exit 1
    fi
  fi
fi

SWEEP_ID=$(python3 src/training/wandb_sweep_runner.py "$ABS_SWEEP_YAML" --create-only)
echo "Created sweep: $SWEEP_ID"

OUTPUT_DIR=$(pwd)/hpc/output
mkdir -p "$OUTPUT_DIR"

echo "Submitting $NUM_AGENTS agent jobs as an LSF array (each agent runs $TRIALS_PER_AGENT trials)..."
bsub -J "wandb_sweep_agent[1-${NUM_AGENTS}]" \
  -q "$QUEUE" \
  -o "$OUTPUT_DIR/wandb_sweep_%J_%I.out" \
  -e "$OUTPUT_DIR/wandb_sweep_%J_%I.err" \
  -n 4 \
  -R "span[hosts=1]" \
  -R "rusage[mem=4GB]" \
  -W "$WALLTIME" \
  -gpu "num=1:mode=exclusive_process" \
  "bash -lc \"module load cuda/12.8.1; source .venv/bin/activate; python3 src/training/wandb_sweep_runner.py '$ABS_SWEEP_YAML' --agent-only --sweep-id '$SWEEP_ID' --trials-per-agent $TRIALS_PER_AGENT\""

echo "Submitted array job. Monitor with bjobs and check hpc/output/ for logs."
