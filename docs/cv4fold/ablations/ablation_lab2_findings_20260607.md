# lab_2 ablation findings & next steps

**Added 2026-06-07.** Updated **2026-06-07 (pm)** with full scrape, **best-of-3**
selection metric, and cross-lab readiness.

Related: [ablation_lab2_signals.md](ablation_lab2_signals.md),
[ablation_lab2_rem.md](ablation_lab2_rem.md),
[ablation_prepro_lab2_lab5.md](ablation_prepro_lab2_lab5.md),
[latent_separability_guide.md](../latent_separability_guide.md),
[README.md](README.md#selection-metric-final-cv4fold).

---

## Selection metric

Each ablation YAML: **`runs: 3`**. **Lock recipes and report cv4fold numbers using the best prior NMI** among seeds (`plots/<seed>/metrics.txt`).

Also record all three seeds — **reject** configs with **≥2 collapsed seeds** (~0.0–0.1) even if one seed peaks (e.g. `rem_emg_wide` EEG3). Mean NMI is for stability notes only, not thesis selection.

---

## Locked recipes (best-of-3, per-lab incohort scratch)

| Lab | Prepro | Arch | Montage / input | Best NMI |
|-----|--------|------|-----------------|----------|
| **lab_2** | no postnorm, EEG 0–30 Hz | default | **EEG1+EEG4+EMG**, EMG **3–100 Hz** | **0.593** ✅ |
| **lab_3** | postnorm, EEG 0–20 Hz, 200 ep | **`wide_mlp`** | EEG1+EEG2+EMG | **0.737** |
| **lab_5** | postnorm, EEG 0–20 Hz, 200 ep | **`wide_mlp`** | EEG1+EEG2+EMG | **0.534** |

**lab_2 locked (28607461):** best **0.593**, seeds **[0.593, 0.578, 0.541]** — all healthy. +0.018 vs `rem_winner`, +0.025 vs prepro-only.

**Cross-lab cv4fold:** **not yet** — per-lab **holdout folds** next ([cross_lab_cv4fold.md](../cross_lab_cv4fold.md)).

---

## Reference numbers (GMM prior NMI, 200 ep, scratch)

### Preprocessing (lab_2)

| Run | Best | seeds |
|-----|------|-------|
| `no_postnorm_widebp` | **0.568** | [0.567, 0.541, 0.568] |

### Signal montage ([signals doc](ablation_lab2_signals.md))

| Run | Best | seeds |
|-----|------|-------|
| `baseline_eeg1_eeg3` | 0.517 | [0.515, 0.370, 0.517] |
| `eeg1_eeg4` | 0.551 | [0.537, 0.538, 0.551] |
| `rem_winner` (EEG3) | **0.575** | [0.575, 0.539, 0.522] |

### REM input ([REM doc](ablation_lab2_rem.md))

| Run | Best | seeds | Verdict |
|-----|------|-------|---------|
| `rem_emg_wide` EEG3 | 0.549 | [0.523, 0.549, **0.097**] | Unstable — discard |
| `rem_emg_low` EEG3 | 0.523 | [0.520, 0.476, 0.523] | No gain |
| `rem_emg_low_eeg4` | 0.566 | [0.566, 0.555, 0.565] | Stable, ≈ prepro |
| **`rem_emg_wide_eeg4`** | **0.593** | [0.593, 0.578, 0.541] | **LOCK** — lab_2 winner |

### Architecture (lab_2 — no beat prepro)

| Variant | Best | seeds |
|---------|------|-------|
| `wide_lat16` | 0.564 | [0.352, 0.564, 0.547] |
| prepro winner | **0.568** | — |

---

## Training-curve findings (W&B)

Both montages show **expected** shapes on locked recipe (`no_beta_epochs: 10`, early stopping off):

| Pattern | Interpretation | Change training? |
|---------|----------------|------------------|
| Flat `prior_pred_nmi` until epoch 10 | KL off (`no_beta_epochs`) | **No** |
| `log_likelihood` plateau ~epoch 20 | Normal | **No** |
| `n_unique_pred_states` → 2 after flicker | ~2 GMM states on val | **No** (montage-independent) |

**Montage-specific (fold 1, seed 124):** EEG4 still climbing at ep 200 (**0.584**); EEG3 peaked ep 71 (**0.574**) then drifted. **Do not early-stop on prior NMI** or shorten 200 epochs.

**Do not retune LR (`3e-4`), beta, or `no_beta_epochs`** from these curves alone.

---

## What we learned (by pillar)

| Pillar | Status |
|--------|--------|
| **Data / cohort** | HQ manifest OK; per-lab incohort works |
| **Preprocessing** | **Per-lab required** — lab_2 no postnorm; lab_3/5 postnorm; paper_robust rejected; bp25 trending worse |
| **Montage (lab_2)** | **Locked:** EEG1+EEG4+EMG + wide EMG **0.593** |
| **REM / latent separability** | Best incohort NMI so far on lab_2; holdout still required |
| **Architecture** | **lab_3 `wide_mlp` essential** (0.737); lab_2/lab_5 marginal vs prepro |
| **LR / warmup** | **Settled** — no change |

---

## Prioritized next steps

### P0 — In flight

| Job | Status | Action |
|-----|--------|--------|
| ~~`rem_emg_wide_eeg4` (28607461)~~ | **DONE** | **0.593** [0.593, 0.578, 0.541] — lab_2 locked |
| lab_3/5 `no_postnorm_bp25` | RUN/PEND | Archive when done; unlikely to win |

### P1b — `no_beta_epochs` ablation (all labs)

Test **`no_beta_epochs: 0`** on each lab's locked winner; control **10** = existing best runs:

| Lab | Control (10) | Test config |
|-----|--------------|-------------|
| lab_2 | `rem_emg_wide_eeg4` **0.593** | `ablation_beta/lab_2/no_beta_epochs_0_*` |
| lab_3 | `wide_mlp` **0.737** | `ablation_beta/lab_3/no_beta_epochs_0_*` |
| lab_5 | `wide_mlp` **0.534** | `ablation_beta/lab_5/no_beta_epochs_0_*` |

```bash
PYTHONPATH=. python3 scripts/cv4fold/generate_ablation_no_beta_epochs.py
bash hpc/submit/cv4fold/submit_ablation_no_beta_epochs.sh
```

See [ablation_no_beta_epochs.md](ablation_no_beta_epochs.md).

### P1 — lab_2 recipe ✅ LOCKED

- Montage: **EEG1, EEG4, EMG**
- EMG band: **3–100 Hz**
- Prepro: **no postnorm**, EEG **0–30 Hz**
- Next: wire into `generate_configs.py` / manifest, then **holdout folds**

### P2 — lab_5

Keep **`long` + postnorm + `wide_mlp`** (best **0.534**). Weakest lab — accept or add lab-specific input ablations later; do not block lab_2/lab_3 holdout on lab_5 perfection.

### P3 — Per-lab holdout folds (seq64, scratch, best-of-3)

Before joint cross-lab cv4fold:

1. Lock templates in `generate_configs.py` / `locked_recipes.py` (prepro + arch + lab_2 montage/EMG).
2. **Pilot:** fold 4 within-lab holdout — see [per_lab_holdout_pilot.md](../per_lab_holdout_pilot.md).
3. Run folds 1–3 with same recipe; gate on **best-of-3 holdout NMI**.

### P4 — Cross-lab cv4fold (Phase 2)

Only after P3: **seq64 + `subject`**, not `subject_lab` (fold-4 collapse). Apply **best-of-3 per YAML per fold** for thesis tables.

### Do not do yet

- LR / β sweeps
- `subject_lab` joint training
- Full 20-mouse joint before per-lab holdout pass

---

## Live queue (2026-06-07)

| JOBID | Name | STAT |
|-------|------|------|
| ~~28607461~~ | `rem_emg_wide_eeg4` | **DONE** best=0.593 |
| 28607237/38 | lab_3/5 `no_postnorm_bp25` | RUN/PEND |

---

## Quick result paths

| Experiment | Results |
|------------|---------|
| Prepro | `results/cv4fold/ablation_prepro/lab_*/abl_lab_2_*` |
| Signal montage | `results/cv4fold/ablation_signals/lab_2/abl_lab_2_*` |
| REM input | `results/cv4fold/ablation_rem/lab_2/abl_lab_2_*` |
| Architecture | `results/cv4fold/ablation_arch/lab_*/arch_*` |
| Per-seed NMI | `plots/<seed>/metrics.txt` |

---

## Changelog

- **2026-06-07 (eve)** — **`rem_emg_wide_eeg4` DONE** (28607461): best **0.593**, seeds [0.593, 0.578, 0.541]; lab_2 locked.
- **2026-06-07** — Initial interim synthesis from W&B curves and partial REM/signal runs.
