#!/usr/bin/env python3
"""Cross-lab preprocessing audit entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.preprocessing_audit.config_utils import load_global_config, snapshot_baseline_config
from scripts.preprocessing_audit.manifest import DEFAULT_MANIFEST, export_cohort_inventory, load_manifest
from scripts.preprocessing_audit.phase1_raw import run_phase1
from scripts.preprocessing_audit.phase2_model_input import run_phase2
from scripts.preprocessing_audit.phase3_ablations import run_phase3
from scripts.preprocessing_audit.phase4_linkage import run_phase4
from scripts.preprocessing_audit.phase5_gates import run_phase5

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "src/config/run/cvaeprior/cv4fold/templates/cgmvae_base.yaml"
DEFAULT_OUT = REPO_ROOT / "results/preprocessing_audit"


def _write_readme(out_dir: Path) -> None:
    text = """# Preprocessing audit (cv4fold cohort)

Regenerate:

```bash
cd /work3/s204070/SPA
PYTHONPATH=. python -m scripts.preprocessing_audit.run \\
  --manifest data/manifests/cv_quality_cohort_v1.yaml \\
  --config src/config/run/cvaeprior/cv4fold/templates/cgmvae_base.yaml \\
  --out results/preprocessing_audit
```

HPC (heavier cohort):

```bash
bsub < hpc/submit/preprocessing_audit/run_cohort.sh
```

See `00_INDEX.md` for go/no-go and decision gates.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-lab preprocessing audit")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-sequences", type=int, default=400)
    parser.add_argument(
        "--phases",
        nargs="+",
        default=["0", "1", "2", "3", "4", "5"],
        help="Phases to run: 0 1 2 3 4 5",
    )
    args = parser.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest)

    cfg = None
    per_mouse_df = None
    feature_df = None
    ablation_df = None
    linkage_df = None

    if "0" in args.phases:
        print("Phase 0: scaffold")
        snapshot_baseline_config(args.config, out_dir / "config" / "cv4fold_fft_baseline.yaml")
        export_cohort_inventory(manifest, out_dir / "tables" / "cohort_inventory.csv")
        cfg = load_global_config(args.config)
        _write_readme(out_dir)

    if "1" in args.phases:
        print("Phase 1: raw audit")
        per_mouse_df = run_phase1(manifest, out_dir)

    if "2" in args.phases or "3" in args.phases:
        if cfg is None:
            cfg = load_global_config(args.config)

    if "2" in args.phases:
        print("Phase 2: model-input audit")
        feature_df = run_phase2(cfg, manifest, out_dir, max_sequences=args.max_sequences)

    if "3" in args.phases:
        print("Phase 3: ablations")
        ablation_df = run_phase3(cfg, manifest, out_dir, max_sequences=min(300, args.max_sequences))

    if "4" in args.phases:
        print("Phase 4: linkage")
        feat_path = out_dir / "tables" / "feature_stats_by_lab.csv"
        linkage_df = run_phase4(feat_path, out_dir)

    if "5" in args.phases:
        print("Phase 5: decision gates")
        if per_mouse_df is None and (out_dir / "tables" / "per_mouse_summary.csv").exists():
            import pandas as pd

            per_mouse_df = pd.read_csv(out_dir / "tables" / "per_mouse_summary.csv")
        if feature_df is None and (out_dir / "tables" / "feature_stats_by_lab.csv").exists():
            import pandas as pd

            feature_df = pd.read_csv(out_dir / "tables" / "feature_stats_by_lab.csv")
        if ablation_df is None and (out_dir / "tables" / "feature_stats_ablation.csv").exists():
            import pandas as pd

            ablation_df = pd.read_csv(out_dir / "tables" / "feature_stats_ablation.csv")
        if linkage_df is None and (out_dir / "linkage" / "nmi_vs_feature_separability.csv").exists():
            import pandas as pd

            linkage_df = pd.read_csv(out_dir / "linkage" / "nmi_vs_feature_separability.csv")
        run_phase5(out_dir, per_mouse_df, feature_df, ablation_df, linkage_df)

    print(f"Done. Results: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
