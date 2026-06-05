"""Phase 5: decision gates and index summary."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.preprocessing_audit.manifest import LABS

NORMALIZATION_SCOPE_TABLE = """
| Step | Scope (cv4fold baseline) |
|------|---------------------------|
| Windowing, FFT, band-pass | **Per mouse × run** |
| `pre_normalize` | **Per mouse × run** (if enabled) |
| `post_normalize` | **Per mouse × run** (all sequences in recording) |
| `normalize_global` | **Train cohort** (disabled in cv4fold) |
"""


def run_phase5(
    out_dir: Path,
    per_mouse_df: pd.DataFrame | None,
    feature_df: pd.DataFrame | None,
    ablation_df: pd.DataFrame | None,
    linkage_df: pd.DataFrame | None,
) -> None:
    gates = []

    # G1: raw vs fft separability — use per_mouse wake variance vs silhouette
    if per_mouse_df is not None and feature_df is not None:
        for lab in LABS:
            sil = feature_df[feature_df["lab"] == lab]["silhouette_k3"].median()
            if not np.isnan(sil) and sil < 0.1:
                gates.append((f"G1_{lab}", "FFT separability weak", "Tune FFT/BP/post_norm (Phase 3)"))

    # G2 lab_2 artifact proxy
    if per_mouse_df is not None:
        l2 = per_mouse_df[per_mouse_df["lab"] == "lab_2"]
        l3 = per_mouse_df[per_mouse_df["lab"] == "lab_3"]
        if len(l2) and len(l3):
            if l2["wake_variance_proxy"].median() > l3["wake_variance_proxy"].median() * 1.5:
                gates.append(("G2", "lab_2 high wake variance proxy", "Harmonize artifact policy"))

    # G3 placement-aware ablation
    if ablation_df is not None and "bp_placement_aware" in ablation_df["variant"].values:
        base = ablation_df[ablation_df["variant"] == "baseline"].groupby("lab")["silhouette_k3"].median()
        place = ablation_df[ablation_df["variant"] == "bp_placement_aware"].groupby("lab")["silhouette_k3"].median()
        for lab in ("lab_2", "lab_5"):
            if lab in base.index and lab in place.index and place[lab] > base[lab] + 0.02:
                gates.append(("G3", f"{lab} gains from placement BP", "Update band_pass_freqs in template"))

    # G4 linkage
    if linkage_df is not None and linkage_df["nmi"].notna().sum() >= 3:
        for lab in LABS:
            sub = linkage_df[linkage_df["lab"] == lab]
            if len(sub) and sub["silhouette_k3"].median() > 0.12:
                if sub["nmi"].notna().any() and sub["nmi"].median() < linkage_df["nmi"].median():
                    gates.append(("G4", f"{lab} ok features, low NMI", "Try lab/subject_lab conditioning"))

    # Success per lab
    go_table = []
    for lab in LABS:
        if feature_df is None:
            go_table.append((lab, "unknown", "Run phase 2"))
            continue
        sil = feature_df[feature_df["lab"] == lab]["silhouette_k3"].median()
        ok = not np.isnan(sil) and sil >= 0.15
        go_table.append((lab, "go" if ok else "no-go", f"silhouette={sil:.3f}" if not np.isnan(sil) else "n/a"))

    lines = [
        "# Preprocessing audit — executive index\n\n",
        "## Partner note: normalization scope\n\n",
        "Preprocessing is **run-level** (per mouse × run). `post_normalize` is **not lab-level**. ",
        "See `tables/normalization_scope.csv` and `after/normalization_scope/` for counterfactual lab pooling.\n\n",
        NORMALIZATION_SCOPE_TABLE,
        "\n## Go / no-go (pre-VAE silhouette ≥ 0.15)\n\n",
        "| lab | status | detail |\n|-----|--------|--------|\n",
    ]
    for lab, status, detail in go_table:
        lines.append(f"| {lab} | {status} | {detail} |\n")

    lines.append("\n## Decision gates triggered\n\n")
    if gates:
        for gid, evidence, action in gates:
            lines.append(f"- **{gid}**: {evidence} → {action}\n")
    else:
        lines.append("- None triggered automatically; review SUMMARY_*.md files.\n")

    lines.append("\n## Hypothesis checklist (Phase 5)\n\n")
    lines.append("| Gate | Check |\n|------|-------|\n")
    lines.append("| G1 | Raw OK but FFT separability poor? See raw/ vs after/ summaries |\n")
    lines.append("| G2 | lab_2 artifact proxy elevated? |\n")
    lines.append("| G3 | placement-aware BP helps lab_2/5 only? See deltas/ |\n")
    lines.append("| G4 | Features OK but NMI low? See linkage/ |\n")
    lines.append("| G5 | per_lab OK, joint fails? (check after joint cv4fold runs) |\n")

    lines.append("\n## Next training steps (Phase 6 — manual)\n\n")
    lines.append("1. Encode winning preprocessing in `cv_quality_cohort_v1.yaml` or cgmvae template if Phase 3 shows a clear winner.\n")
    lines.append("2. `PYTHONPATH=. python -m scripts.cv4fold.generate_configs --phase full`\n")
    lines.append("3. `bash hpc/submit/cv4fold/submit_smoke_80.sh` on worst lab before full grid.\n")
    lines.append("4. Track B: `chmmgmvae` + `conditioning_source: subject_lab` if G4 fires.\n")

    (out_dir / "00_INDEX.md").write_text("".join(lines), encoding="utf-8")

    rec_path = out_dir / "DECISIONS.md"
    rec_path.write_text(
        "# Audit decisions (fill after review)\n\n"
        "- [ ] Preprocessing changes to apply: _none / list variants_\n"
        "- [ ] Manifest/template YAML updated: yes/no\n"
        "- [ ] Smoke80 submitted for lab: _\n",
        encoding="utf-8",
    )
