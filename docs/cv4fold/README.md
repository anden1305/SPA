# cv4fold experiments

Cross-lab validation for cGMVAE / cHMM-GMVAE (Phase 2 of the [decoder conditioning roadmap](../decoder_conditioning_roadmap.md)).

**HQ cohort only:** 20 mice in [`data/manifests/cv_quality_cohort_v1.yaml`](../../data/manifests/cv_quality_cohort_v1.yaml) (labs 2, 3, 5). All configs are generated from this manifest.

## Selection metric (final cv4fold)

Each YAML runs **`runs: 3`** (three seeds). For **reporting and recipe lock**, use the **best GMM prior NMI** among the three seeds (`plots/<seed>/metrics.txt`), not the mean.

| Context | Metric |
|---------|--------|
| **Final cv4fold / thesis tables** | **Best-of-3** prior NMI per YAML × fold |
| **Ablation tie-break / stability check** | Also note all three seeds; discard configs with **≥2 collapsed seeds** (~0.0–0.1 NMI) even if one seed peaks |
| **Reliability SEM (10 seeds)** | Mean ± SEM on scratch runs — separate from cv4fold best-of-3 ([cvae_checkpointing.md](../training/cvae_checkpointing.md)) |

Scrape helper (shows best + all seeds): `python3 scripts/cv4fold/scrape_experiment_results.py`

**After cv4fold / holdout jobs finish:** run postprocess — [postprocess.md](postprocess.md) and [per_lab_holdout_pilot.md § When jobs finish](per_lab_holdout_pilot.md#when-jobs-finish-postprocess).

**Status (2026-06-08):** Incohort locks — **lab_2 cGMVAE** 0.593; **lab_3 cHMM warm** 0.734 T=64; **lab_5 cHMM** seq32 0.640 (1/3 seeds — reliability before thesis lock). Enc+dec worse on lab_2. **Next:** [per_lab_holdout_pilot.md](per_lab_holdout_pilot.md) fold 4. Full write-up: [ablation_findings_20260608.md](ablations/ablation_findings_20260608.md).

| Doc | Contents |
|-----|----------|
| [cross_lab_cv4fold.md](cross_lab_cv4fold.md) | Story, folds, conditioning ablations, fold-4 results, next steps |
| [incohort_to_cross_lab_strategy.md](incohort_to_cross_lab_strategy.md) | **How incohort locks → holdout → joint cv4fold** (strategies & pitfalls) |
| [lab_2_locked_montage.md](lab_2_locked_montage.md) | **Locked EEG1+EEG4+EMG** + `rem_emg_wide_eeg4` recipe (0.593) |
| [per_lab_holdout_pilot.md](per_lab_holdout_pilot.md) | **Within-lab holdout** (fold 4): cgmvae vs chmmgmvae, locked recipes |
| [unified_holdout_paper_line.md](unified_holdout_paper_line.md) | **Paper main line** — 3 models × 2 scopes, unified recipes + per-lab prepro |
| [postprocess.md](postprocess.md) | **Lean postprocess** — best-of-3 JSON/CSV; disk-safe defaults |
| [lab_preprocessing_review_20260607.md](lab_preprocessing_review_20260607.md) | **Data-driven lab review**, HQ plots, notch/EMG ablation backlog |
| [latent_separability_guide.md](latent_separability_guide.md) | Which plots/metrics matter; input vs latent; what to skip |
| [overnight_experiments_20260606.md](overnight_experiments_20260606.md) | **Overnight plan:** arch sweep + job tracker |
| Phase 1 (lab_3) | [decoder_only_lab3_chmm_experiments.md](../decoder_only/decoder_only_lab3_chmm_experiments.md) |

### Ablations ([ablations/](ablations/))

| Doc | Contents |
|-----|----------|
| [ablation_prepro_lab2_lab5.md](ablations/ablation_prepro_lab2_lab5.md) | Preprocessing ablations (rounds 1–4, incl. lab_3/5 EEG 0.3–25 Hz) |
| [ablation_lab2_rem.md](ablations/ablation_lab2_rem.md) | lab_2 REM input ablations + input-space EEG/EMG diagnostics |
| [ablation_lab2_signals.md](ablations/ablation_lab2_signals.md) | lab_2 EEG1+EEG4 vs EEG1+EEG3 montage (winner prepro) |
| [ablation_lab2_findings_20260607.md](ablations/ablation_lab2_findings_20260607.md) | **Interim findings**, curve interpretation, prioritized next ablations |
| [ablation_no_beta_epochs.md](ablations/ablation_no_beta_epochs.md) | `no_beta_epochs` 0 vs 10 on **all labs'** locked best-NMI recipes |
| [ablation_lab2_beta.md](ablations/ablation_lab2_beta.md) | Pointer to `no_beta_epochs` ablation |
| [ablation_chmm_incohort.md](ablations/ablation_chmm_incohort.md) | **cHMM-GMVAE incohort** — `hmm_gmm` vs `warm_hmm_gmm`, scratch only |
| [cgmvae_incohort_analysis_20260608.md](ablations/cgmvae_incohort_analysis_20260608.md) | **cGMVAE incohort synthesis** — separability, curves, prioritized next ablations |
| [ablation_incohort_followup.md](ablations/ablation_incohort_followup.md) | **Follow-up matrix** (wide_mlp, 300ep, EMG/notch, REM-recall ckpt) |
| [ablation_enc_dec_conditioning.md](ablations/ablation_enc_dec_conditioning.md) | **cGMVAE enc+dec conditioning** vs locked decoder-only winners |
| [ablation_findings_20260608.md](ablations/ablation_findings_20260608.md) | **Overnight synthesis** — cHMM locks, seq T, Feature 7/REM, plot pipeline, holdout plan |
| [ablation_chmm_lab2_seq64.md](ablations/ablation_chmm_lab2_seq64.md) | **lab_2 cHMM T=64 stability** — warm, notch, 300ep, REM-recall ckpt |
| [ablation_joint_chmm_fold4.md](ablations/ablation_joint_chmm_fold4.md) | **Joint fold-4 cHMM ablations** — 9 deltas toward NMI ≥ 0.52 |

## Per-lab in-cohort (baby step)

```bash
cd /work3/s204070/SPA && source .venv/bin/activate
PYTHONPATH=. python3 scripts/cv4fold/generate_configs.py --phase per_lab_incohort --labs lab_5 lab_2
bash hpc/submit/cv4fold/submit_per_lab_incohort.sh
```

LSF logs: `hpc/output/cv4fold/per_lab_incohort/{lab_5,lab_2}_<JOBID>.{out,err}`  
Results: `results/cv4fold/per_lab_incohort/<lab>/`

**Results:** `results/cv4fold/`  
**Fold-4 pilot table:** `results/cv4fold/decoder_only_fold4/comparison.md`
