# Incohort follow-up ablations (analysis matrix)

**Added 2026-06-08.** Four cGMVAE incohort jobs from [cgmvae_incohort_analysis_20260608.md](cgmvae_incohort_analysis_20260608.md) minimal matrix. Scratch only (`model_checkpoint_path: null`), **3 seeds**, best-of-3 NMI for selection.

## Variants

| Job | Lab | Config | Epochs | Change vs locked |
|-----|-----|--------|--------|------------------|
| P1 | lab_5 | `ablation_followup/lab_5/emg_wide_notch50.yaml` | 300 | EMG 3–100 Hz + notch50 on `wide_mlp` |
| P2 | lab_2 | `ablation_followup/lab_2/wide_mlp.yaml` | 200 | `rem_emg_wide_eeg4` + wide MLP |
| P3 | lab_2 | `ablation_followup/lab_2/epochs300.yaml` | 300 | locked recipe, longer train |
| P4 | lab_2 | `ablation_followup/lab_2/rem_recall_ckpt.yaml` | 200 | locked recipe; **`validation_checkpoint: rem_recall`** |

P4 uses new trainer knob `trainer.validation_checkpoint: rem_recall` — post-train validation loads `cvae_best_rem_recall.pth` instead of best prior NMI. Training still logs both metrics each epoch.

## Generate & submit

```bash
source .venv/bin/activate
PYTHONPATH=. python3 scripts/cv4fold/generate_ablation_incohort_followup.py
bash hpc/submit/cv4fold/submit_ablation_incohort_followup.sh
```

| Field | Value |
|-------|-------|
| Queue / walltime | `gpuv100`; lab_5 + lab_2 ep300 → **6:00**; lab_2 ep200 → **4:00** |
| LSF logs | `hpc/output/cv4fold/ablation_followup/<lab>/followup_*_%J.{out,err}` |
| Results | `results/cv4fold/ablation_followup/<lab>/abl_followup_*` |

## Decision gates

| Lab | Control | Adopt if |
|-----|---------|----------|
| lab_5 | `wide_mlp` **0.534** | P1 best-of-3 > **0.540**, ≥2/3 healthy seeds |
| lab_2 | `rem_emg_wide_eeg4` **0.593** | P2/P3 best-of-3 > **0.593** (or P3 > **0.60**) |
| lab_2 P4 | same NMI as locked | REM recall ↑ in confusion matrix at comparable NMI |

Related: [`generate_ablation_incohort_followup.py`](../../../scripts/cv4fold/generate_ablation_incohort_followup.py), [cvae_checkpointing.md](../../training/cvae_checkpointing.md).
