# Phase 2 — cross-lab cv4fold

Added 2026-06-06. **Phase 1 (lab_3 proof):** [decoder_only_lab3_chmm_experiments.md](../decoder_only_lab3_chmm_experiments.md). **Roadmap:** [decoder_conditioning_roadmap.md](../decoder_conditioning_roadmap.md).

---

## Why this step exists

Phase 1 showed **~0.63–0.65 prior NMI** on lab_3 with hotstart and decoder-only conditioning.

Phase 2 asks: **does that recipe work on new mice from other labs?**  
No label loss — still **unsupervised GMM/HMM prior** vs sleep stages. Harder because:

- Population **joint** training (many mice, mixed labs)
- **Held-out** validation mice per fold
- Often **no baseline hotstart** (`model_checkpoint_path: null`)
- Hyperparams tuned in-sample may not transfer

---

## Setup

| Item | Value |
|------|--------|
| Folds | 4 (rotate which mice are val) |
| Labs | lab_2, lab_3, lab_5 |
| Models | `cgmvae` (GMM prior), `chmmgmvae` (HMM-GMM prior) |
| Training | `training_pipeline: cvae`, decoder-only conditioning |
| Seeds | 3 runs per fold (`runs: 3`) |
| Primary metric | Pooled **prior NMI** on val fold → `val_nmi_summary.json`, `plots/metrics.txt` |

**Tune winners:** hyperparams (LR, epochs, β schedule, etc.) chosen from an earlier in-sample cv4fold sweep — then applied in batches below.

---

## Experiment families (under `results/cv4fold/`)

| Batch | Conditioning | `sequence_length` | Role |
|-------|--------------|-------------------|------|
| `subject_tune_winners` | `subject` | 64 | Main line — tune winners + subject only |
| `subject_tune_winners_seq1` | `subject` | 1 | Ablate seq length (lab_3 used seq1) |
| `subject_lab_tune_winners` | `subject_lab` | 64 | Add lab embedding — **failed** on holdout |
| `subject_lab_tune_winners_seq1` | `subject_lab` | 1 | subject_lab + seq1 ablation |
| `decoder_only_fold4` | (pilot) | 1 vs 64 | Fold-4-only recipe comparison |

Same fold splits across batches; compare **conditioning** and **seq length** on the **same holdout mice**.

---

## Fold 4 pilot — what we learned

Auto table: `results/cv4fold/decoder_only_fold4/comparison.md`. Snapshot:

| Step | Recipe | Pooled NMI | Notes |
|------|--------|------------|-------|
| **B** | seq64 + `subject` | **0.31** | Best so far; lab_2/lab_5 ~0.42–0.47, **lab_3 ~0.08** |
| **A** | seq1 + `subject` | 0.07 | seq1 recipe may be lab_3-specific |
| **C** | seq64 + `subject_lab` | **0.002** | Prior collapse — do not scale until re-tuned |

**cHMM vs cGMVAE** (subject_tune, fold 4): cGMVAE best run **0.31**, cHMM **~0.22** — both above zero but far below lab_3.

**Seed issue:** many YAMLs give **NMI ≈ 0 on runs 2–3**; only run 1 (seed 123) succeeds for subject_tune seq64.

**Decision (provisional):** use **seq64 + subject** for the next full-cv4fold line; pause **subject_lab** until fold-4 re-tune. See `results/cv4fold/decoder_only_fold4/decision.md`.

---

## How this differs from Phase 1

| | Phase 1 (lab_3) | Phase 2 (cv4fold) |
|--|-----------------|-------------------|
| Mice | Same lab, familiar val set | Held-out mice, mixed labs |
| Init | Hotstart from baseline `.pth` | Usually scratch |
| NMI | ~0.63–0.72 | ~0–0.31 (fold 4) |
| Goal | Ablate model & conditioning | Test **generalization** |

Low cv4fold NMI does **not** overturn Phase 1 — it shows **generalization** is the hard part.

---

## Reading results

Per run directory:

```
results/cv4fold/<batch>/<model>/joint/fold_<k>/<run_name>/
  val_nmi_summary.json      # pooled + per-run NMI
  val_nmi_by_lab.csv        # lab_2 / lab_3 / lab_5 breakdown
  per_mouse/run_<n>/        # per-subject NMI
  <n>/plots/metrics.txt     # prior NMI + likelihood (best ckpt)
  <n>/checkpoints/validation_checkpoint.txt
```

Batch analysis example: `results/cv4fold/subject_lab_tune_winners/analysis_report.md`.

---

## Next steps

1. **Lock recipe:** seq64 + `subject` (not subject_lab).
2. **Run remaining folds** (1–3) with that recipe for cGMVAE and cHMM.
3. **Stabilize seeds** — investigate runs 2–3 collapse (GMM init, checkpoint score, more seeds).
4. **Optional:** fold-4 hotstart from lab_3 baseline — does in-lab checkpoint help holdout?
5. **Re-tune subject_lab** on fold 4 only before any full subject_lab sweep.

---

## Related

- [cvae_checkpointing.md](../cvae_checkpointing.md) — `model_checkpoint_path`, best prior ckpt
- [decoder_only_checkpoint_tracing.md](../decoder_only_checkpoint_tracing.md)
