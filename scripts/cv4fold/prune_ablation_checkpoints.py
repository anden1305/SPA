#!/usr/bin/env python3
"""Delete checkpoint .pth files from worst ablation variants (keep plots/metrics).

Groups by ablation campaign + lab, ranks run dirs by best-of-3 prior NMI from
metrics.txt, keeps top ~25% + force-protected locked winners. Only removes *.pth.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from scripts.cv4fold.metrics_paths import metrics_path
from scripts.cv4fold.select_best_vae_run import parse_nmi

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results/cv4fold"

# Never prune anything under these trees.
PROTECTED_ROOTS = (
    "per_lab_incohort",
    "per_lab_holdout",
    "selection",
)

# Ablation campaigns eligible for pruning.
ABLATION_ROOTS = (
    "ablation_prepro",
    "ablation_arch",
    "ablation_rem",
    "ablation_signals",
    "ablation_beta",
)

# Run dir name substrings → always keep all checkpoints (locked / thesis path).
FORCE_KEEP_SUBSTRINGS: tuple[tuple[str, str], ...] = (
    ("ablation_rem", "eeg4"),
    ("ablation_rem", "rem_emg_wide_eeg4"),
    ("ablation_prepro", "abl_lab_3_baseline_long"),
    ("ablation_prepro", "abl_lab_5_long_"),
    ("ablation_arch", "wide_mlp"),
    ("ablation_signals", "eeg1_eeg4"),
    ("ablation_prepro", "abl_lab_2_no_postnorm_widebp"),
)

NMI_RE = re.compile(r"NMI:\s*([0-9.eE+-]+)")


@dataclass
class RunDir:
    path: Path
    campaign: str
    lab: str
    variant_key: str
    best_nmi: float | None
    force_keep: bool = False
    pth_files: list[Path] = field(default_factory=list)
    pth_bytes: int = 0


def _best_nmi_for_run(run_dir: Path) -> float | None:
    nmis: list[float] = []
    for seed in (1, 2, 3):
        mfile = metrics_path(run_dir, seed)
        if mfile is None:
            continue
        nmi = parse_nmi(mfile)
        if nmi is not None:
            nmis.append(nmi)
    if not nmis:
        for mfile in run_dir.rglob("metrics.txt"):
            nmi = parse_nmi(mfile)
            if nmi is not None:
                nmis.append(nmi)
    return max(nmis) if nmis else None


def _variant_key(name: str) -> str:
    # abl_lab_2_paper_robust_20260607-025004 → paper_robust
    for prefix in ("abl_", "arch_", "rem_", "sig_"):
        if name.startswith(prefix):
            rest = name[len(prefix) :]
            parts = rest.split("_")
            if parts and parts[0].startswith("lab") and len(parts) >= 2:
                return "_".join(parts[2:-1]) if len(parts) > 3 else parts[1]
            break
    return name.rsplit("_", 1)[0]


def _force_keep(campaign: str, run_name: str) -> bool:
    rel = f"{campaign}/{run_name}"
    for camp, sub in FORCE_KEEP_SUBSTRINGS:
        if camp == campaign and sub in run_name:
            return True
    return False


def collect_runs() -> list[RunDir]:
    runs: list[RunDir] = []
    for campaign in ABLATION_ROOTS:
        camp_root = RESULTS / campaign
        if not camp_root.is_dir():
            continue
        for lab_dir in sorted(camp_root.iterdir()):
            if not lab_dir.is_dir() or not lab_dir.name.startswith("lab_"):
                continue
            for run_dir in sorted(lab_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                pths = list(run_dir.rglob("*.pth"))
                if not pths:
                    continue
                size = sum(p.stat().st_size for p in pths)
                runs.append(
                    RunDir(
                        path=run_dir,
                        campaign=campaign,
                        lab=lab_dir.name,
                        variant_key=_variant_key(run_dir.name),
                        best_nmi=_best_nmi_for_run(run_dir),
                        force_keep=_force_keep(campaign, run_dir.name),
                        pth_files=pths,
                        pth_bytes=size,
                    )
                )
    return runs


def plan_deletions(runs: list[RunDir], keep_fraction: float = 0.25) -> tuple[list[Path], list[RunDir], list[RunDir]]:
    to_delete: list[Path] = []
    kept: list[RunDir] = []
    pruned: list[RunDir] = []

    groups: dict[tuple[str, str], list[RunDir]] = {}
    for r in runs:
        groups.setdefault((r.campaign, r.lab), []).append(r)

    for key, group in sorted(groups.items()):
        ranked = sorted(
            group,
            key=lambda r: (r.force_keep, r.best_nmi if r.best_nmi is not None else -1.0),
            reverse=True,
        )
        n = len(ranked)
        keep_n = max(1, math.ceil(n * keep_fraction))
        for i, r in enumerate(ranked):
            if r.force_keep or i < keep_n:
                kept.append(r)
            else:
                pruned.append(r)
                to_delete.extend(r.pth_files)

    return to_delete, kept, pruned


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep-fraction", type=float, default=0.25)
    parser.add_argument("--execute", action="store_true", help="Actually delete files")
    args = parser.parse_args()

    runs = collect_runs()
    if not runs:
        print("No ablation checkpoint dirs found.")
        return 0

    to_delete, kept, pruned = plan_deletions(runs, args.keep_fraction)
    del_bytes = sum(p.stat().st_size for p in to_delete)

    print(f"Runs with checkpoints: {len(runs)}")
    print(f"Keep: {len(kept)} run dirs  |  Prune checkpoints: {len(pruned)} run dirs")
    print(f"Checkpoint files to delete: {len(to_delete)}  ({del_bytes / 1e9:.2f} GB)\n")

    print("=== KEPT (checkpoints preserved) ===")
    for r in sorted(kept, key=lambda x: (x.campaign, x.lab, -(x.best_nmi or -1))):
        tag = " [FORCE]" if r.force_keep else ""
        nmi = f"{r.best_nmi:.3f}" if r.best_nmi is not None else "?"
        print(f"  {r.campaign}/{r.lab}/{r.path.name}  best={nmi}{tag}  ({r.pth_bytes/1e6:.0f} MB)")

    print("\n=== PRUNE checkpoints only (plots/metrics kept) ===")
    for r in sorted(pruned, key=lambda x: (x.campaign, x.lab, x.best_nmi or 0)):
        nmi = f"{r.best_nmi:.3f}" if r.best_nmi is not None else "?"
        print(f"  {r.campaign}/{r.lab}/{r.path.name}  best={nmi}  ({r.pth_bytes/1e6:.0f} MB)")

    if args.execute:
        n = 0
        for p in to_delete:
            p.unlink(missing_ok=True)
            n += 1
        print(f"\nDeleted {n} checkpoint files ({del_bytes / 1e9:.2f} GB).")
    else:
        print("\nDry run — pass --execute to delete.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
