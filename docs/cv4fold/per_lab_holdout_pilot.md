# Per-lab within-lab holdout (cv4fold pilot)

**Added 2026-06-07.** Taste-test **held-out mice within each lab** before joint cross-lab cv4fold. Aligns with Morten’s plan (per-lab analysis first, 4 folds, subject conditioning only, best-of-3 runs).

**Not in scope here:** joint all-lab training, HMM raw baseline, substages sweep, lab conditioning, global Bayes tune winners — see [cross_lab_cv4fold.md](cross_lab_cv4fold.md) and deferred section below.

---

## Morten plan → this pilot

| Recommendation | This pilot |
|----------------|------------|
| Frequency (FFT) input + mouse conditioning | ✓ decoder-only subject embedding |
| Run each config **3 times**, take **best** | ✓ `runs: 3`; report best prior NMI per YAML |
| Analyse **each lab separately** (2, 3, 5) | ✓ one job per lab × model |
| **4-fold** CV (lab_5 has 4 mice) | ✓ manifest `splits.fold_1…4`; **pilot starts at fold 4** |
| Subject conditioning only (drop lab cond.) | ✓ no `conditioning_source: lab` |
| cGMVAE vs cHMM-GMVAE | ✓ `prior: gmm` vs `warm_hmm_gmm` |
| Scratch / no hotstart for fair generalisation | ✓ `model_checkpoint_path: null` |
| β ends at 1 (`max_beta: 1`, warmup ladder OK) | ✓ same as ablation recipes (`no_beta_epochs: 10`) |

---

## Selection metric

**Best-of-3** GMM prior NMI per YAML (`plots/<seed>/metrics.txt`). See [README.md](README.md#selection-metric-final-cv4fold).

Holdout NMI is computed on **val mice only** (manifest fold holdout for that lab). Do not compare to incohort pooled NMI.

---

## Fold splits (manifest)

From [`data/manifests/cv_quality_cohort_v1.yaml`](../../data/manifests/cv_quality_cohort_v1.yaml):

| Fold | lab_2 holdout | lab_3 holdout | lab_5 holdout |
|------|---------------|---------------|---------------|
| 1 | sub-071 | sub-038, sub-039 | sub-087 |
| 2 | sub-072, sub-076 | sub-041, sub-048, sub-069 | sub-088 |
| 3 | sub-077 | sub-043, sub-054 | sub-089 |
| **4** | **sub-080, sub-081** | **sub-056, sub-059, sub-060** | **sub-092** |

Within-lab: **train** = all other HQ mice in that lab; **val** = holdout mice only.

---

## Locked recipes (from ablation winners)

Source: [`scripts/cv4fold/locked_recipes.py`](../../scripts/cv4fold/locked_recipes.py)

| Lab | Prepro | Arch | Notes |
|-----|--------|------|-------|
| lab_2 | `rem_emg_wide_eeg4` | default | no postnorm, EEG 0–30 Hz, EMG 3–100 Hz, EEG1+EEG4 |
| lab_3 | `baseline_long` | `wide_mlp` | postnorm, EEG 0–20 Hz, 200 ep |
| lab_5 | `long` | `wide_mlp` | postnorm, EEG 0–20 Hz, 200 ep |

| Model | Prior | Extra |
|-------|-------|-------|
| **cgmvae** | `gmm` | static GMM; `sequence_length: 1` |
| **chmmgmvae** | per-lab (`simple` lab_2/5, `warm` lab_3) | **`sequence_length` locked:** lab_2 **64**, lab_3 **64**, lab_5 **32** (T>1 only — temporal HMM) |

Shared: LR `3e-4`, scratch, `decoder_only_conditioning: true`, `save_results_npz: true` on holdout.

Regenerate after lock changes:

```bash
PYTHONPATH=. python3 scripts/cv4fold/generate_configs.py --phase per_lab_holdout --fold 4
```

**lab_2 montage locked 2026-06-07:** manifest + all new configs use **EEG1+EEG4+EMG** via [`rem_emg_wide_eeg4`](../../src/config/run/cvaemarhmm/cv4fold/ablation_rem/lab_2/rem_emg_wide_eeg4.yaml) (best **0.593**, seeds [0.593, 0.578, 0.541]).

---

## Generated configs (fold 4 pilot)

```
src/config/run/cvaemarhmm/cv4fold/per_lab_holdout/
  lab_2/fold_4/{cgmvae,chmmgmvae}.yaml   # train 4 mice, holdout 080+081
  lab_3/fold_4/{cgmvae,chmmgmvae}.yaml   # train 7 mice, holdout 056+059+060
  lab_5/fold_4/{cgmvae,chmmgmvae}.yaml   # train 3 mice, holdout 092
```

Results:

```
results/cv4fold/per_lab_holdout/<lab>/fold_4/<model>/<run_name_timestamp>/
  plots/1/metrics.txt          # post-train prior NMI, seed 1 (best-of-3 picks here)
  plots/2/metrics.txt
  plots/3/metrics.txt
  plots/1/*.png                # tripanel, confusion matrix, … (always saved)
  1/checkpoints/               # largest on disk
  2/checkpoints/
  3/checkpoints/
  config.json
```

**Not saved on pilot configs:** `plots/{1,2,3}/results.npz` (`save_results_npz: false` — saves quota). **Plots are still saved.** For final thesis runs set `save_results_npz: true` — see [postprocess.md](postprocess.md).

---

## Generate configs

```bash
source .venv/bin/activate
PYTHONPATH=. python3 scripts/cv4fold/generate_configs.py \
  --phase per_lab_holdout --fold 4 \
  --labs lab_2 lab_3 lab_5 \
  --models cgmvae chmmgmvae
```

Other folds:

```bash
PYTHONPATH=. python3 scripts/cv4fold/generate_configs.py \
  --phase per_lab_holdout --fold 1 --labs lab_2 lab_3 lab_5
```

Single lab / model:

```bash
PYTHONPATH=. python3 scripts/cv4fold/generate_configs.py \
  --phase per_lab_holdout --fold 4 --labs lab_2 --models cgmvae
```

---

## Submit (when ready — not auto-submitted)

**6 jobs** for fold-4 pilot (3 labs × 2 models):

```bash
bash hpc/submit/cv4fold/submit_per_lab_holdout.sh --fold 4
```

Logs: `hpc/output/cv4fold/per_lab_holdout/f4_<lab>_<model>_%J.{out,err}`

Monitor:

```bash
bjobs -u $USER | grep cv4_ho
tail -f hpc/output/cv4fold/per_lab_holdout/f4_lab_2_cgmvae_<JOBID>.out
```

---

## When jobs finish (postprocess)

**Do this once** after all 6 jobs exit successfully (login or `linuxsh`). Full detail: [postprocess.md](postprocess.md).

```bash
source .venv/bin/activate
cd /work3/s204070/SPA

FOLD=4
for lab in lab_2 lab_3 lab_5; do
  for model in cgmvae chmmgmvae; do
    base="results/cv4fold/per_lab_holdout/${lab}/fold_${FOLD}/${model}"
    for run_dir in "${base}"/*/; do
      [[ -f "${run_dir}config.json" ]] || continue
      echo "postprocess: ${run_dir}"
      PYTHONPATH=. python3 scripts/cv4fold/postprocess_fold.py --result-root "${run_dir}"
    done
    if [[ -d "${base}" ]]; then
      PYTHONPATH=. python3 scripts/cv4fold/summarize_experiment.py \
        --search-root "${base}" --out "${base}/summary.csv"
    fi
  done
done
```

Read **`results/cv4fold/per_lab_holdout/<lab>/fold_4/<model>/summary.csv`** — columns `best_nmi`, `run_1`…`run_3`. Compare **cgmvae vs chmmgmvae per lab** on holdout mice only.

**Pilot:** `save_results_npz: false` — no validation npz; postprocess uses `metrics.txt` only (~KB written). **Final runs:** enable `save_results_npz: true` before submit if you need prediction arrays — see [postprocess.md § Final runs](postprocess.md#final-runs-predictions--per-mouse-nmi).

---

## Interpretation vs incohort ablations

| Setting | Incohort ablations | This holdout pilot |
|---------|-------------------|-------------------|
| Train mice | All HQ in lab | All except fold holdout |
| Val mice | Same as train (pooled) | **Holdout only** |
| Expected NMI | lab_2 ~0.57, lab_3 ~0.74 | **Lower** — true generalisation test |
| Purpose | Lock prepro/arch | Taste before full 4-fold + joint cv4fold |

If holdout NMI stays reasonable (e.g. lab_2 >0.45, lab_3 >0.55), proceed to folds 1–3 then joint line in [cross_lab_cv4fold.md](cross_lab_cv4fold.md).

---

## Deferred (final thesis experiments)

- **Joint** 20-mouse training with same fold IDs across labs
- **HMM raw** on identical folds (baseline)
- **HMMGMVAE** (no conditioning) vs **cHMMGMVAE** on raw — separate model family
- **Substages** 3–15 for cHMM only
- Global hyperparameter tune / error bars across many seeds
- Lab conditioning (`subject_lab`) — dropped per Morten

---

## Code map

| Piece | Path |
|-------|------|
| Manifest splits | `data/manifests/cv_quality_cohort_v1.yaml` |
| Locked recipes | `scripts/cv4fold/locked_recipes.py` |
| Config generator | `scripts/cv4fold/generate_configs.py --phase per_lab_holdout` |
| Submit | `hpc/submit/cv4fold/submit_per_lab_holdout.sh` |
| Postprocess | `scripts/cv4fold/postprocess_fold.py` — [postprocess.md](postprocess.md) |
| Train/val split helpers | `scripts/cv4fold/manifest_utils.py` |

---

## Changelog

- **2026-06-07** — Fold-4 pilot setup; cgmvae + chmmgmvae; locked ablation recipes; best-of-3 metric documented.
- **2026-06-07** — Postprocess checklist + result layout; `save_results_npz: false` on pilot configs.
