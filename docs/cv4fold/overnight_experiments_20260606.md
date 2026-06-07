# Overnight experiment plan — 2026-06-06

**Goal:** Per-lab in-cohort cGMVAE that clears **GMM NMI > 0.45** (stretch: **lab_3 ~0.72** with hotstart — see reference ladder below) using structured ablations on preprocessing, capacity, and latent dimension.

**Window:** ~5 h GPU queue time from ~23:15; jobs submitted without further approval.

Related: [`ablation_prepro_lab2_lab5.md`](ablation_prepro_lab2_lab5.md), [`lab_measurement_preprocessing_2_3_5.md`](../lab_measurement_preprocessing_2_3_5.md), [`cross_lab_cv4fold.md`](cross_lab_cv4fold.md).

---

## Experiment matrix (all submitted)

| Phase | What | Jobs | Submit script | Status |
|-------|------|------|---------------|--------|
| R1 | Prepro + epochs (lab_2 grid, lab_5 long) | 28605764–68 | `submit_ablation_prepro.sh` | **DONE** |
| R2 | `no_postnorm` on lab_3/5 | 28606252–56 | `submit_ablation_prepro_round2.sh` | RUN/PEND |
| R3 | Paper robust front-end (all labs) | 28606501–03 | `submit_ablation_paper_robust.sh` | PEND |
| **R4** | **Architecture sweep (latent + MLP)** | **28606509–20** | `submit_ablation_arch_overnight.sh` | **PEND** |

**Queue at submit (~23:15):** 4 RUN (R2) + 16 PEND (R3 + R4) ≈ 20 jobs.

---

## Round 4 — architecture ablation (NEW)

**Hypothesis:** After fixing preprocessing, remaining gap vs lab_3 may need **more latent capacity** or **wider MLP/CNN** (lab_2 Bern montage + 1 s scoring is harder).

Generator:

```bash
PYTHONPATH=. python3 scripts/cv4fold/generate_ablation_arch.py
```

Configs: `src/config/run/cvaemarhmm/cv4fold/ablation_arch/<lab>/{lat4,lat16,wide_mlp,wide_lat16}.yaml`

Fixed per run: best prepro from R1–2, `epochs: 200`, `runs: 3`, from scratch, decoder-only cGMVAE, GMM prior.

### lab_3 — which number is the reference?

| Scope | Setup | Typical GMM prior NMI | Source |
|-------|--------|----------------------|--------|
| **Thesis / reliability (canonical)** | lab_3, **hotstart** from `cvae_decoder_only_model.pth` | **~0.72** | [cvae_checkpointing.md](../cvae_checkpointing.md), W&B runs yklyqlsh / cme0uc0e |
| Phase 1 decoder-only | lab_3 HQ, hotstart, decoder-only | **~0.65–0.67** | job `28605085`, [decoder_only_lab3_chmm_experiments.md](../decoder_only_lab3_chmm_experiments.md) |
| Encoder+decoder | lab_3, hotstart | **~0.74** | job D in same doc |
| **This ablation (R2/R4)** | 10 HQ mice, **from scratch**, per-lab incohort | **~0.59** interim (R2 s1) | not comparable to ~0.72 without hotstart + full recipe |

**Architecture sweep (R4)** compares variants **within the same from-scratch HQ setup**. The **north star** for lab_3 remains **~0.72** (hotstart reliability); closing that gap likely needs hotstart + full lab_3 mouse set after we lock prepro/arch.

| Lab | Prepro locked | Within-sweep ref (from-scratch HQ) | Canonical target |
|-----|---------------|-------------------------------------|------------------|
| lab_2 | no postnorm, EEG 0–30 | **0.568** (`no_postnorm_widebp`) | gate >0.45 ✓ |
| lab_3 | postnorm, EEG 0–20 | ~0.59 interim (`baseline_long`, R2) | **~0.72** hotstart |
| lab_5 | postnorm, EEG 0–20 | **0.508** (`long`) | TBD |

| Variant | `latent_dim` | MLP enc / dec | JOBIDs |
|---------|-------------|---------------|--------|
| `lat4` | 4 | 256,128 / 128,256 | lab_2 **28606509**, lab_3 **28606513**, lab_5 **28606517** |
| `lat16` | 16 | 256,128 / 128,256 | lab_2 **28606510**, lab_3 **28606514**, lab_5 **28606518** |
| `wide_mlp` | 8 | 512,256,128 / 128,256,512 | lab_2 **28606511**, lab_3 **28606515**, lab_5 **28606519** |
| `wide_lat16` | 16 | 512,256,128 / 128,256,512 | lab_2 **28606512**, lab_3 **28606516**, lab_5 **28606520** |

- Queue: `gpuv100`, walltime **6:00**, mem **8GB**
- Logs: `hpc/output/cv4fold/ablation_arch/<lab>_<variant>_%J.{out,err}`
- Results: `results/cv4fold/ablation_arch/<lab>/arch_<lab>_<variant>_<timestamp>/`

**Decision rule after R4:** pick variant with highest **mean** GMM NMI across 3 seeds per lab; if `wide_lat16` ≈ `lat16`, prefer smaller model.

---

## Monitoring (copy-paste)

```bash
bjobs -u $USER
source .venv/bin/activate && python3 scripts/cv4fold/scrape_experiment_results.py
tail -f hpc/output/cv4fold/ablation_arch/lab_2_wide_lat16_*.out
```

---

## Results tracker

_Update this table as jobs finish (`scrape_experiment_results.py`)._

### Round 4 architecture (GMM NMI, **best-of-3**) — complete 2026-06-07

| Lab | lat4 | lat16 | wide_mlp | wide_lat16 | **Winner** |
|-----|------|-------|----------|------------|------------|
| lab_2 | 0.541 | 0.486† | 0.530 | 0.564 | **prepro 0.568** |
| lab_3 | 0.680 | 0.662 | **0.737** | 0.736 | **`wide_mlp` 0.737** |
| lab_5 | 0.524 | 0.477 | **0.534** | 0.476 | **`wide_mlp` 0.534** (vs prepro long 0.508) |

†lab_2 lat16 seeds 2–3 ~0.0 — unstable; would fail **≥2/3 healthy** rule despite best 0.486.

### Round 2 / 3 / 4 prepro — complete (bp25 partial)

See [ablation_prepro_lab2_lab5.md](ablation_prepro_lab2_lab5.md) for full tables.

**Locked (best-of-3):** lab_2 `no_postnorm_widebp` **0.568**; lab_3 `baseline_long` + **`wide_mlp` → 0.737**; lab_5 `long` + **`wide_mlp` → 0.534**.

### lab_2 REM + montage (2026-06-07)

See [ablation_lab2_findings_20260607.md](ablation_lab2_findings_20260607.md). Leading candidate: **`rem_emg_wide_eeg4` best 0.593** (1/3 seeds at scrape).

---

## Diagnostic plots — what we learned so far

Always check after a run completes:

```
results/cv4fold/<experiment>/<lab>/<run_name>/plots/
  input_channel_total_power_per_state.png  # pre-VAE EEG/EMG (all runs w/ summary_statistics)
  input_emg_band_power_per_state.png       # REM atonia cue before encoder
  input_eeg_band_power_per_state.png
  feature_amplitude_per_state.png          # post-encoder latent (not raw EMG)
  feature_variance_per_state.png
  <seed>/hmm_tripanel_pc1_pc2.png
  <seed>/results.npz
```

**lab_2 REM round:** [ablation_lab2_rem.md](ablation_lab2_rem.md) — 8 variants, VAE encoder unchanged.

| Run | Path (representative) | Observation |
|-----|----------------------|-------------|
| lab_2 baseline 80ep | `per_lab_incohort/lab_2/...162334/plots/` | True classes overlap in latent PC space; Awake↔REM confusion; **fix prepro not arch first** |
| lab_5 baseline 80ep | `per_lab_incohort/lab_5/...161924/plots/` | Better True separation; undertrained at ep80 |
| lab_2 no_postnorm_widebp | `ablation_prepro/lab_2/abl_lab_2_no_postnorm_widebp_*/plots/` | Tripanel True panel less smeared; NMI 0.56 |
| lab_2 no_postnorm | `ablation_prepro/lab_2/abl_lab_2_no_postnorm_*/plots/1/hmm_tripanel_pc1_pc2.png` | Compare side-by-side with baseline_long |

**Interpretation guide:**

- **True panel messy, KMeans ceiling low** → preprocessing / input features (not prior or epoch count alone).
- **True panel OK, HMM panel wrong** → capacity (latent dim, MLP width) or more epochs.
- **feature_amplitude: REM vs Awake EMG not separated** → bandpass or normalization killing EMG tone cue (lab_2).

---

## Next steps after overnight (for thesis path)

1. Lock per-lab template in `generate_configs.py` (prepro + best arch from R4; lab_2 montage/EMG when `rem_emg_wide_eeg4` confirms).
2. **Per-lab holdout folds** — best-of-3 on held-out mice ([cross_lab_cv4fold.md](cross_lab_cv4fold.md)).
3. Joint cross-lab cv4fold with locked templates — **best-of-3 per YAML per fold** for thesis tables.

---

## Changelog

- **2026-06-06 23:15** — Submitted R4 arch sweep (28606509–28606520). R2 running, R3 PEND.
- **2026-06-07 ~11:00** — R1–R3 + lab_2/lab_3 arch DONE. lab_3 **wide_mlp 0.737**.
- **2026-06-07 (pm)** — R4 all labs DONE; prepro/bp25/RAM/montage results; **best-of-3** selection metric documented in [README.md](README.md).
