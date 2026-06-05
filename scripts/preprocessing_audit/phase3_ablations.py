"""Phase 3: preprocessing ablation grid."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.preprocessing_audit.config_utils import ABLATION_VARIANTS, config_for_variant, load_global_config
from scripts.preprocessing_audit.data_loading import load_mssv_array, run_vae_preprocessing, subsample_sequences
from scripts.preprocessing_audit.manifest import LABS, iter_runs, load_manifest, mice_by_lab
from scripts.preprocessing_audit.metrics import compute_metrics

# Subset for speed: 2 mice per lab, run 1
def _ablation_subset(manifest: dict) -> list:
    records = []
    by_lab = mice_by_lab(manifest)
    for lab in LABS:
        for mid in by_lab[lab][:2]:
            info = manifest["inventory"][mid]
            records.append(
                type("R", (), {
                    "participant_id": mid,
                    "lab": lab,
                    "run": 1,
                    "signals": list(info["signals"]),
                    "hours": float(info.get("hours", 0)),
                })()
            )
    return records


def run_phase3(base_cfg, manifest: dict, out_dir: Path, max_sequences: int = 300) -> pd.DataFrame:
    deltas_dir = out_dir / "deltas"
    ab_dir = deltas_dir / "preprocessing_step_ablation"
    ab_dir.mkdir(parents=True, exist_ok=True)

    variants = list(ABLATION_VARIANTS.keys()) + ["artifact_harmonized"]
    rows = []
    subset = _ablation_subset(manifest)

    for variant in variants:
        cfg = config_for_variant(base_cfg, variant)
        var_dir = ab_dir / variant
        var_dir.mkdir(parents=True, exist_ok=True)
        for rec in subset:
            try:
                remove_art = True
                if variant == "artifact_harmonized" and rec.lab == "lab_2":
                    remove_art = False
                from scripts.preprocessing_audit.manifest import RunRecord

                r = RunRecord(rec.participant_id, rec.lab, rec.run, rec.signals, rec.hours)
                x_raw, y_raw, meta = load_mssv_array(r, remove_artifact=remove_art)
                x_feat, y_feat = run_vae_preprocessing(cfg, x_raw, y_raw)
                if x_feat.shape[0] == 0:
                    continue
                x_sub, y_sub = subsample_sequences(x_feat, y_feat, max_sequences)
                mets = compute_metrics(x_sub, y_sub)
                mets.update(
                    variant=variant,
                    participant_id=rec.participant_id,
                    lab=rec.lab,
                    run=rec.run,
                )
                rows.append(mets)
            except Exception as exc:
                print(f"ablation {variant} {rec.participant_id}: {exc}")

    df = pd.DataFrame(rows)
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    out_csv = tables / "feature_stats_ablation.csv"
    df.to_csv(out_csv, index=False)

    # pairwise effect sizes between labs (baseline only)
    base = df[df["variant"] == "baseline"]
    pairs = []
    for metric in ("silhouette_k3", "mean_mahalanobis_between_stages"):
        for lab_a in LABS:
            for lab_b in LABS:
                if lab_a >= lab_b:
                    continue
                a = base[base["lab"] == lab_a][metric].dropna()
                b = base[base["lab"] == lab_b][metric].dropna()
                if len(a) and len(b):
                    pairs.append(
                        {
                            "lab_a": lab_a,
                            "lab_b": lab_b,
                            "metric": metric,
                            "mean_a": float(a.mean()),
                            "mean_b": float(b.mean()),
                            "diff": float(a.mean() - b.mean()),
                        }
                    )
    pd.DataFrame(pairs).to_csv(deltas_dir / "lab_pairwise_effect_sizes.csv", index=False)

    _write_summary_deltas(df, deltas_dir)
    return df


def _write_summary_deltas(df: pd.DataFrame, deltas_dir: Path) -> None:
    lines = ["# Preprocessing ablation summary\n\n", "## Decision matrix (median silhouette)\n\n"]
    lines.append("| variant | lab_2 | lab_3 | lab_5 |\n|---------|-------|-------|-------|\n")
    for variant in df["variant"].unique():
        row = [variant]
        for lab in LABS:
            sub = df[(df["variant"] == variant) & (df["lab"] == lab)]["silhouette_k3"]
            row.append(f"{sub.median():.3f}" if len(sub) else "n/a")
        lines.append("| " + " | ".join(row) + " |\n")
    best = []
    for lab in LABS:
        sub = df[df["lab"] == lab]
        if sub.empty:
            continue
        med = sub.groupby("variant")["silhouette_k3"].median()
        if len(med):
            best.append((lab, med.idxmax(), float(med.max())))
    lines.append("\n## Best variant per lab (silhouette)\n\n")
    for lab, var, val in best:
        lines.append(f"- {lab}: `{var}` ({val:.3f})\n")
    (deltas_dir / "SUMMARY_deltas.md").write_text("".join(lines), encoding="utf-8")
