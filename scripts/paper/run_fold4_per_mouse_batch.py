#!/usr/bin/env python3
"""Split fold-4 joint holdout best runs into per-mouse metrics for paper figures."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

RUNS = (
    (
        "chmmgmvae_locked",
        REPO / "results/cv4fold/joint_holdout/fold_4/chmmgmvae_locked/joint_ho_f4_chmmgmvae_locked_20260609-042207",
        1,
    ),
    (
        "cgmvae_locked",
        REPO / "results/cv4fold/joint_holdout/fold_4/cgmvae_locked/joint_ho_f4_cgmvae_locked_20260609-040621",
        1,
    ),
)


def main() -> int:
    for model, root, run_num in RUNS:
        out = REPO / f"results/cv4fold/paper_figures/per_mouse/fold_4/{model}/seed_{run_num}"
        cmd = [
            sys.executable,
            str(REPO / "scripts/cv4fold/split_per_mouse.py"),
            "--result-root", str(root),
            "--run-num", str(run_num),
            "--out-dir", str(out),
        ]
        print("+", " ".join(cmd))
        import os
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO)
        subprocess.run(cmd, check=True, cwd=REPO, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
