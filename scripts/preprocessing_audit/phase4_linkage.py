"""Phase 4: link audit metrics to cv4fold validation NMI."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CV4FOLD_ROOT = REPO_ROOT / "results/cv4fold"


def _find_metrics_json() -> list[dict]:
    found = []
    for path in CV4FOLD_ROOT.rglob("metrics.json"):
        if "per_mouse" not in str(path):
            continue
        try:
            with path.open(encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        parts = path.parts
        try:
            pm_idx = parts.index("per_mouse")
            run_num = parts[pm_idx + 1]
            mouse = parts[pm_idx + 2]
        except ValueError:
            mouse = path.parent.name
            run_num = ""
        found.append(
            {
                "path": str(path),
                "participant_id": data.get("participant_id", mouse),
                "nmi": data.get("nmi") or data.get("cvae_latent_kmeans_nmi"),
                "accuracy": data.get("accuracy"),
                "run_folder": run_num,
            }
        )
    return found


def _find_validations_best_nmi() -> pd.DataFrame:
    rows = []
    for path in CV4FOLD_ROOT.rglob("validations.json"):
        if "per_mouse" in str(path):
            continue
        try:
            with path.open(encoding="utf-8") as f:
                vals = json.load(f)
        except Exception:
            continue
        best_nmi = None
        for epoch_metrics in vals.values():
            if isinstance(epoch_metrics, dict):
                v = epoch_metrics.get("cvae_latent_kmeans_nmi")
                if v is not None:
                    best_nmi = max(best_nmi or v, v)
        cfg_path = path.parent / "config.json"
        scope = ""
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                scope = cfg.get("results_dir", path.parent.name)
            except Exception:
                pass
        rows.append({"validations_path": str(path), "best_cvae_nmi": best_nmi, "scope": scope})
    return pd.DataFrame(rows)


def run_phase4(feature_stats_path: Path, out_dir: Path) -> pd.DataFrame:
    link_dir = out_dir / "linkage"
    link_dir.mkdir(parents=True, exist_ok=True)

    if not feature_stats_path.exists():
        pd.DataFrame().to_csv(link_dir / "nmi_vs_feature_separability.csv", index=False)
        (link_dir / "SUMMARY_linkage.md").write_text(
            "# Linkage summary\n\nNo feature_stats_by_lab.csv — run phase 2 first.\n",
            encoding="utf-8",
        )
        return pd.DataFrame()

    feat = pd.read_csv(feature_stats_path)
    feat_mouse = feat.groupby("participant_id", as_index=False).agg(
        silhouette_k3=("silhouette_k3", "median"),
        lab=("lab", "first"),
    )

    nmi_rows = _find_metrics_json()
    nmi_df = pd.DataFrame(nmi_rows) if nmi_rows else pd.DataFrame()
    if not nmi_df.empty:
        nmi_mouse = nmi_df.groupby("participant_id", as_index=False)["nmi"].median()
    else:
        nmi_mouse = pd.DataFrame(columns=["participant_id", "nmi"])

    merged = feat_mouse.merge(nmi_mouse, on="participant_id", how="outer")
    merged.to_csv(link_dir / "nmi_vs_feature_separability.csv", index=False)

    val_df = _find_validations_best_nmi()
    if not val_df.empty:
        val_df.to_csv(link_dir / "cv4fold_validations_summary.csv", index=False)

    lines = ["# Linkage: feature separability vs validation NMI\n\n"]
    if merged["nmi"].notna().sum() >= 3 and merged["silhouette_k3"].notna().sum() >= 3:
        sub = merged.dropna(subset=["nmi", "silhouette_k3"])
        fig, ax = plt.subplots(figsize=(7, 5))
        for lab in sub["lab"].dropna().unique():
            m = sub["lab"] == lab
            ax.scatter(sub.loc[m, "silhouette_k3"], sub.loc[m, "nmi"], label=lab)
        ax.set_xlabel("Pre-VAE silhouette (median per mouse)")
        ax.set_ylabel("cv4fold per-mouse NMI")
        ax.legend()
        ax.set_title("Separability vs NMI")
        plt.tight_layout()
        plt.savefig(link_dir / "nmi_vs_silhouette_scatter.png", dpi=150)
        plt.close()
        lines.append("![scatter](nmi_vs_silhouette_scatter.png)\n\n")
        if len(sub) >= 5:
            r = np.corrcoef(sub["silhouette_k3"], sub["nmi"])[0, 1]
            lines.append(f"- Pearson r (silhouette vs NMI): {r:.3f}\n")
        high_sil_low_nmi = sub[(sub["silhouette_k3"] > 0.15) & (sub["nmi"] < sub["nmi"].median())]
        if len(high_sil_low_nmi):
            lines.append(
                "- Some mice have decent separability but low NMI → suspect **model/conditioning**, not preprocessing alone.\n"
            )
        low_both = sub[(sub["silhouette_k3"] < 0.1) & (sub["nmi"] < sub["nmi"].median())]
        if len(low_both):
            lines.append("- Low silhouette and low NMI → prioritize **preprocessing** changes (Phase 3 winners).\n")
    else:
        lines.append(
            "- Few or no cv4fold `per_mouse/metrics.json` files found under `results/cv4fold/`. "
            "Re-run linkage after training, or check paths.\n"
        )
        lines.append(
            "- Feature separability by lab (from audit):\n"
        )
        for lab in feat_mouse["lab"].dropna().unique():
            med = feat_mouse[feat_mouse["lab"] == lab]["silhouette_k3"].median()
            lines.append(f"  - {lab}: median silhouette {med:.3f}\n")

    lines.append("\n## Recommended next steps\n\n")
    lines.append("1. If lab_2/5 silhouette << lab_3: apply Phase 3 winning preprocessing variant.\n")
    lines.append("2. If silhouettes comparable but NMI skewed: try `chmmgmvae` with `conditioning_source: subject_lab`.\n")
    lines.append("3. Smoke80 on worst lab: `bash hpc/submit/cv4fold/submit_smoke_80.sh`\n")

    (link_dir / "SUMMARY_linkage.md").write_text("".join(lines), encoding="utf-8")
    return merged
