#!/usr/bin/env python3
"""Scrape per-seed GMM NMI from ablation LSF logs for quick monitoring.

Usage: python3 scripts/cv4fold/scrape_ablation_nmi.py
Prints one row per variant log found under hpc/output/cv4fold/ablation_prepro/.
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

LOG_DIR = Path("hpc/output/cv4fold/ablation_prepro")
NMI_RE = re.compile(r"^GMM NMI: ([\d.]+)", re.MULTILINE)
DONE_RE = re.compile(r"Successfully completed|Exited with exit code")


def main() -> int:
    rows = []
    for out in sorted(glob.glob(str(LOG_DIR / "*.out"))):
        text = Path(out).read_text(errors="ignore")
        nmis = [float(x) for x in NMI_RE.findall(text)]
        m = re.search(r"(\w+)_(\d+)\.out$", Path(out).name)
        status = "running"
        if "Successfully completed" in text:
            status = "DONE"
        elif "Exited with exit code" in text:
            status = "FAILED"
        best = max(nmis) if nmis else None
        rows.append((Path(out).name, status, nmis, best))

    w = max((len(r[0]) for r in rows), default=10)
    for name, status, nmis, best in rows:
        seeds = ", ".join(f"{x:.3f}" for x in nmis) if nmis else "-"
        bs = f"{best:.3f}" if best is not None else "-"
        print(f"{name:<{w}}  {status:<8}  best={bs:<6}  seeds=[{seeds}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
