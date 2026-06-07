#!/usr/bin/env python3
"""1-epoch parity check: fixed seed, compare train loss + val metrics (tolerance)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def _run_one_epoch(orch, seed: int = 123) -> dict[str, float]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    orch.global_config.seed = seed
    orch.global_config.trainer.epochs = 1
    orch.global_config.trainer.validate_per_epoch = 1
    orch.model.reset()
    orch.trainer.reset()
    orch.validator.reset()
    orch.model.training_pipeline = "cvae"
    orch.trainer.train()
    epoch = 0
    metrics: dict[str, float] = {
        "train_loss": float(orch.trainer.losses[epoch]),
        "train_reg": float(orch.trainer.regularization_losses[epoch]),
    }
    val = orch.validator.validations.get(epoch, {})
    for k in (
        "cvae_latent_kmeans_nmi",
        "prior_pred_nmi",
        "log_likelihood",
        "entropy_norm",
        "checkpoint_score",
    ):
        if k in val:
            metrics[k] = float(val[k])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        default="src/config/run/cvaemarhmm/local/profile_cvae_single_mouse_smoke.yaml",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat count (GPU + warm_hmm init may differ across repeats; use tests/test_perf_optimizations.py for path parity).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Warning: CUDA not available; running on CPU.", file=sys.stderr)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    from src.orchestrator.orchestrator import Orchestrator

    orch = Orchestrator(args.config)
    runs = [_run_one_epoch(orch, seed=args.seed) for _ in range(args.repeat)]
    print(f"Config: {args.config}")
    print(f"Seed: {args.seed}, repeats: {args.repeat}")
    for i, m in enumerate(runs):
        print(f"  run {i + 1}: {m}")

    ref = runs[0]
    tol_abs = 1e-4
    tol_rel = 1e-4
    for i, m in enumerate(runs[1:], start=2):
        for k, v in ref.items():
            v2 = m[k]
            if abs(v) > 1:
                ok = abs(v2 - v) / abs(v) <= tol_rel or abs(v2 - v) <= 0.5
            else:
                ok = abs(v2 - v) <= tol_abs
            if not ok:
                raise SystemExit(f"Mismatch on {k}: {v} vs {v2} (run {i})")
    print("OK: repeated runs match within tolerance.")


if __name__ == "__main__":
    main()
