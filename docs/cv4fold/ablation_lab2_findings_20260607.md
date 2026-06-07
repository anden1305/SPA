# lab_2 ablation findings & next steps

**Added 2026-06-07.** Updated **2026-06-07 (pm)** with full scrape, **best-of-3**
selection metric, and cross-lab readiness.

Related: [ablation_lab2_signals.md](ablation_lab2_signals.md),
[ablation_lab2_rem.md](ablation_lab2_rem.md),
[ablation_prepro_lab2_lab5.md](ablation_prepro_lab2_lab5.md),
[latent_separability_guide.md](latent_separability_guide.md),
[README.md](README.md#selection-metric-final-cv4fold).

---

## Selection metric

Each ablation YAML: **`runs: 3`**. **Lock recipes and report cv4fold numbers using the best prior NMI** among seeds (`plots/<seed>/metrics.txt`).

Also record all three seeds — **reject** configs with **≥2 collapsed seeds** (~0.0–0.1) even if one seed peaks (e.g. `rem_emg_wide` EEG3). Mean NMI is for stability notes only, not thesis selection.

---

## Locked recipes (best-of-3, per-lab incohort scratch)

| Lab | Prepro | Arch | Montage / input | Best NMI |
|-----|--------|------|-----------------|----------|
| **lab_2** | no postnorm, EEG 0–30 Hz | default (arch ≈ tie) | EEG3 **0.575** (`rem_winner`); EEG4 **`rem_emg_wide` 0.593*** | 0.568 prepro-only |
| **lab_3** | postnorm, EEG 0–20 Hz, 200 ep | **`wide_mlp`** | EEG1+EEG2+EMG | **0.737** |
| **lab_5** | postnorm, EEG 0–20 Hz, 200 ep | **`wide_mlp`** | EEG1+EEG2+EMG | **0.534** |

\*`rem_emg_wide_eeg4` — **1/3 seeds done** at scrape time; confirm before manifest lock.

**Cross-lab cv4fold:** **not yet** — finish EEG4 REM seeds + lab_5 confirmation, then **per-lab holdout folds** before 20-mouse joint training ([cross_lab_cv4fold.md](cross_lab_cv4fold.md)).

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
| **`rem_emg_wide_eeg4`** | **0.593** | [0.593, —, —] | **RUN** — best candidate |

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
| **Montage (lab_2)** | EEG4 stable vs EEG3; **`rem_emg_wide_eeg4`** may be new lab_2 best if 3 seeds hold |
| **REM / latent separability** | Still hardest on lab_2; arch did not fix; wide EMG + EEG4 most promising |
| **Architecture** | **lab_3 `wide_mlp` essential** (0.737); lab_2/lab_5 marginal vs prepro |
| **LR / warmup** | **Settled** — no change |

---

## Prioritized next steps

### P0 — In flight

| Job | Status | Action |
|-----|--------|--------|
| `rem_emg_wide_eeg4` (28607461) | RUN | Wait for seeds 2–3 |
| lab_3/5 `no_postnorm_bp25` | RUN | Archive when done; unlikely to win |

```bash
bjobs -u $USER | grep -E 'eeg4|bp25'
source .venv/bin/activate && PYTHONPATH=. python3 scripts/cv4fold/scrape_experiment_results.py | tail -20
```

### P1 — Lock lab_2 recipe (after P0)

If **`rem_emg_wide_eeg4` best-of-3 ≥ 0.568** and **≥2/3 seeds healthy**:

- Montage: **EEG1, EEG4, EMG**
- EMG band: **3–100 Hz** (`rem_emg_wide`)
- Regenerate manifest / `generate_configs.py` lab_2 `signals`

Else: keep **EEG1+EEG3** + `no_postnorm_widebp` (**0.575** best on `rem_winner`).

### P2 — lab_5

Keep **`long` + postnorm + `wide_mlp`** (best **0.534**). Weakest lab — accept or add lab-specific input ablations later; do not block lab_2/lab_3 holdout on lab_5 perfection.

### P3 — Per-lab holdout folds (seq64, scratch, best-of-3)

Before joint cross-lab cv4fold:

1. Lock templates in `generate_configs.py` (prepro + arch + lab_2 montage/EMG).
2. Run **holdout folds 1–3 per lab** with locked recipe.
3. Gate: best-of-3 on **held-out mice only** (not incohort pooled).

### P4 — Cross-lab cv4fold (Phase 2)

Only after P3: **seq64 + `subject`**, not `subject_lab` (fold-4 collapse). Apply **best-of-3 per YAML per fold** for thesis tables.

### Do not do yet

- LR / β sweeps
- `subject_lab` joint training
- Full 20-mouse joint before per-lab holdout pass

---

## Live queue (2026-06-07 pm)

| JOBID | Name | STAT |
|-------|------|------|
| 28607461 | `rem_emg_wide_eeg4` | RUN |
| 28607237 | lab_3 `no_postnorm_bp25` | RUN |
| 28607238 | lab_5 `no_postnorm_bp25` | RUN |

**Done since AM:** `rem_emg_low_eeg4` (best **0.566**, 3/3 stable); all signal montage + most REM EEG3 variants; R4 arch all labs.

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

- **2026-06-07 (pm)** — Full scrape; best-of-3 metric; locked per-lab table; cross-lab gate; EEG4 REM leading.
- **2026-06-07** — Initial interim synthesis from W&B curves and partial REM/signal runs.
