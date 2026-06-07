#!/usr/bin/env bash
# Open the latest local profile .prof in snakeviz (browser UI).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${1:-src/config/run/cvaemarhmm/local/profile_cvae_lab2_smoke.yaml}"
RUN_NAME="$(python -c "import yaml; from pathlib import Path; print(yaml.safe_load(Path('$CONFIG').read_text())['run_name'])")"
PROF="results/local_profile/${RUN_NAME}/1/train.prof"

if [[ ! -f "$PROF" ]]; then
  echo "Profile not found: $PROF" >&2
  echo "Run scripts/local/run_profile.sh first." >&2
  exit 1
fi

if ! command -v snakeviz >/dev/null 2>&1; then
  echo "snakeviz not installed. Run: pip install snakeviz" >&2
  exit 1
fi

exec snakeviz "$PROF"
