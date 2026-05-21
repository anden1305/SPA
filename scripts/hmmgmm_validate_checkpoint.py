#!/usr/bin/env python3
"""Re-run post-train CVAE prior validation from a saved checkpoint.

Run on a compute node (``bsub`` job or after ``linuxsh``), not on the DTU login node.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a trained CVAE / cHMMGMVAE checkpoint.")
    parser.add_argument("-c", "--config_path", required=True, help="Training YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to cvae_final_model_run*.pth")
    parser.add_argument(
        "--run_name",
        default=None,
        help="Override run_name for plots/metrics (default: parent folder of checkpoint).",
    )
    args = parser.parse_args()

    from src.orchestrator.orchestrator import Orchestrator

    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    orchestrator = Orchestrator(args.config_path)
    orchestrator.global_config.cvae.model_checkpoint_path = str(checkpoint)
    orchestrator.global_config.run_name = args.run_name or checkpoint.parent.name
    print(f"Validating checkpoint: {checkpoint}")
    print(f"Results / plots -> {orchestrator.global_config.results_dir}/{orchestrator.global_config.run_name}/plots")
    orchestrator.predict_cvae()


if __name__ == "__main__":
    main()
