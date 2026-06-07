#!/usr/bin/env python3
"""Generate per-lab ``no_beta_epochs`` ablation configs (KL delay vs KL from epoch 0).

Each lab uses its **locked best-NMI incohort recipe**; only ``model.params.no_beta_epochs``
changes. Control ``10`` matches all prior ablations; ``0`` applies beta/KL from epoch 1.

| Lab | Base config | Control best NMI (no_beta_epochs=10) |
|-----|-------------|--------------------------------------|
| lab_2 | ``rem_emg_wide_eeg4`` | 0.593 (job 28607461) |
| lab_3 | ``ablation_arch/.../wide_mlp`` | 0.737 |
| lab_5 | ``ablation_arch/.../wide_mlp`` | 0.534 |

See docs/cv4fold/ablations/ablation_no_beta_epochs.md.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "src/config/run/cvaemarhmm/cv4fold"
OUT_ROOT = CONFIG_ROOT / "ablation_beta"

# Locked winner YAML per lab (best-of-3 incohort, no_beta_epochs=10 in base).
LAB_WINNERS: dict[str, dict[str, Any]] = {
    "lab_2": {
        "slug": "rem_emg_wide_eeg4",
        "base": CONFIG_ROOT / "ablation_rem/lab_2/rem_emg_wide_eeg4.yaml",
        "control_nmi": 0.593,
    },
    "lab_3": {
        "slug": "wide_mlp",
        "base": CONFIG_ROOT / "ablation_arch/lab_3/wide_mlp.yaml",
        "control_nmi": 0.737,
    },
    "lab_5": {
        "slug": "wide_mlp",
        "base": CONFIG_ROOT / "ablation_arch/lab_5/wide_mlp.yaml",
        "control_nmi": 0.534,
    },
}


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write(base: dict, lab: str, slug: str, no_beta_epochs: int) -> Path:
    cfg = copy.deepcopy(base)
    cfg["model"]["params"]["no_beta_epochs"] = int(no_beta_epochs)
    cfg["results_dir"] = f"results/cv4fold/ablation_beta/{lab}"
    cfg["run_name"] = f"abl_{lab}_no_beta_epochs_{no_beta_epochs}_{slug}"
    out_dir = OUT_ROOT / lab
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"no_beta_epochs_{no_beta_epochs}_{slug}.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Wrote {out_path}")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labs",
        nargs="+",
        choices=tuple(LAB_WINNERS),
        default=list(LAB_WINNERS),
        help="Labs to generate (default: all).",
    )
    parser.add_argument(
        "--values",
        type=int,
        nargs="+",
        default=[0],
        help="no_beta_epochs values (default: 0 only — use existing winner runs as control 10).",
    )
    args = parser.parse_args()
    for lab in args.labs:
        meta = LAB_WINNERS[lab]
        base_path = meta["base"]
        if not base_path.is_file():
            raise SystemExit(f"Missing base config for {lab}: {base_path}")
        base = _load(base_path)
        for value in args.values:
            if value < 0:
                raise SystemExit(f"no_beta_epochs must be >= 0, got {value}")
            _write(base, lab, meta["slug"], value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
