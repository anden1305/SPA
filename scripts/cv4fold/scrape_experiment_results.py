#!/usr/bin/env python3
"""Scrape GMM NMI and diagnostic plot paths from cv4fold experiment logs/results."""

from __future__ import annotations

import glob
import re
from pathlib import Path

NMI_RE = re.compile(r"^GMM NMI: ([\d.]+)", re.MULTILINE)
LOG_DIRS = (
    "hpc/output/cv4fold/ablation_prepro",
    "hpc/output/cv4fold/ablation_arch",
    "hpc/output/cv4fold/ablation_rem",
    "hpc/output/cv4fold/ablation_signals",
)
PLOT_NAMES = (
    "feature_amplitude_per_state.png",
    "feature_variance_per_state.png",
    "hmm_tripanel_pc1_pc2.png",
    "input_channel_total_power_per_state.png",
    "input_emg_band_power_per_state.png",
    "input_eeg_band_power_per_state.png",
)


def _status(text: str) -> str:
    if "Successfully completed" in text:
        return "DONE"
    if "Exited with exit code" in text:
        return "FAILED"
    return "running"


def _find_plots(run_name_prefix: str) -> list[str]:
    hits: list[str] = []
    for root in Path("results/cv4fold").rglob(run_name_prefix + "*"):
        if not root.is_dir():
            continue
        plots = root / "plots"
        if not plots.is_dir():
            continue
        for name in PLOT_NAMES:
            for p in plots.rglob(name):
                hits.append(str(p))
        if hits:
            break
    return sorted(set(hits))[:5]


def main() -> int:
    rows: list[tuple] = []
    for log_dir in LOG_DIRS:
        for out in sorted(glob.glob(f"{log_dir}/*.out")):
            path = Path(out)
            text = path.read_text(errors="ignore")
            nmis = [float(x) for x in NMI_RE.findall(text)]
            status = _status(text)
            best = max(nmis) if nmis else None
            # infer run_name from config path in log if present
            m = re.search(r"config_path src/config/run/cvaemarhmm/[^\s]+/([^\s/]+)/([^\s]+)\.yaml", text)
            prefix = f"abl_{m.group(1)}_{m.group(2)}" if m and "ablation_prepro" in str(path) else ""
            if not prefix and "ablation_arch" in str(path):
                m2 = re.search(r"ablation_arch/([^\s/]+)/([^\s]+)\.yaml", text)
                prefix = f"arch_{m2.group(1)}_{m2.group(2)}" if m2 else ""
            plots = _find_plots(prefix) if prefix and status == "DONE" else []
            rows.append((path.name, status, nmis, best, plots))

    w = max((len(r[0]) for r in rows), default=20)
    for name, status, nmis, best, plots in rows:
        seeds = ", ".join(f"{x:.3f}" for x in nmis) if nmis else "-"
        bs = f"{best:.3f}" if best is not None else "-"
        print(f"{name:<{w}}  {status:<8}  best={bs:<6}  seeds=[{seeds}]")
        for p in plots:
            print(f"    plot: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
