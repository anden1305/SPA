#!/usr/bin/env python3
"""Deprecated wrapper — use generate_ablation_no_beta_epochs.py (--labs lab_2)."""

from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "generate_ablation_no_beta_epochs.py"
    runpy.run_path(str(target), run_name="__main__")
