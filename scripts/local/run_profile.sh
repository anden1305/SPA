#!/usr/bin/env bash
# Local cGMVAE training profile (laptop GPU). Does not use HPC/bsub.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${1:-src/config/run/cvaemarhmm/local/profile_cvae_lab2_smoke.yaml}"

export PYTHONPATH=.
export WANDB_PROJECT="${WANDB_PROJECT:-SPA-local}"
# Optional: export WANDB_MODE=offline
# Windows: export PYTHONIOENCODING=utf-8 (avoids emoji UnicodeEncodeError in data_loader logs)
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"

echo "Config: $CONFIG"
echo "WANDB_PROJECT=$WANDB_PROJECT"
echo "Running train_vae with --profile ..."

python main.py --method train_vae --profile -c "$CONFIG"

RUN_DIR="results/local_profile/$(python - <<'PY' 2>/dev/null || true
import yaml, sys
from pathlib import Path
cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
print(f"{cfg['run_name']}/1")
PY
"$CONFIG")"

# Fallback if python one-liner fails
if [[ ! -d "$RUN_DIR" ]]; then
  RUN_NAME="$(basename "$CONFIG" .yaml)"
  RUN_DIR="results/local_profile/${RUN_NAME}/1"
fi

echo ""
echo "Profile artifacts:"
echo "  $RUN_DIR/train.prof"
echo "  $RUN_DIR/train_cprofile_cumulative.txt"
echo "  $RUN_DIR/train_cprofile_tottime.txt"

if command -v snakeviz >/dev/null 2>&1; then
  echo ""
  echo "Interactive flame graph:"
  echo "  snakeviz $RUN_DIR/train.prof"
else
  echo ""
  echo "Install snakeviz for interactive viewing: pip install snakeviz"
  echo "  snakeviz $RUN_DIR/train.prof"
fi
