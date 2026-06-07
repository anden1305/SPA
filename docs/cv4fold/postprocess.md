# cv4fold postprocess (lean, disk-safe)

**Added 2026-06-07.** Cherry-picked from `chmmgmm_fold` with **storage-first defaults** — you are on a **200 GiB** work3 quota.

## What gets saved during training

| Artifact | Pilot / holdout (default) | Final thesis runs | Typical size |
|----------|---------------------------|-------------------|--------------|
| **PNG plots** (tripanel, confusion matrix, losses, …) | ✓ always (`visualizer.*: true`) | ✓ always | Small (MB per seed) |
| **`plots/{1,2,3}/metrics.txt`** | ✓ always | ✓ always | ~50 B |
| **`plots/{1,2,3}/results.npz`** (predictions + latents) | ✗ **`save_results_npz: false`** | ✓ set **`save_results_npz: true`** | **Large** (100s MB–GB) |
| **Checkpoints** (`{1,2,3}/checkpoints/*.pth`) | ✓ always | ✓ always | **Largest** (GB per run) |

**`save_results_npz: false` does not disable plots** — only skips the fat validation npz. Keep all visualizer flags on; PNGs are fine for quota.

**Final experiments:** you will want **`save_results_npz: true`** on the YAML (or override in `locked_recipes.py` / regenerate configs) so post-train validation writes one npz per seed. That is enough for thesis figures and per-mouse analysis; you do not need a separate predictions export unless you want split files.

---

## When holdout jobs finish (copy-paste)

Run on **login or `linuxsh`** after all 6 fold-4 jobs complete. Finds every timestamped result dir automatically — no need to type `<run_name_timestamp>`.

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
      echo "Wrote ${base}/summary.csv"
    fi
  done
done
```

**Outputs:** per run → `best_run.json`, `val_nmi_by_lab.csv`, `val_nmi_summary.json`. Per lab/model → **`summary.csv`** (best-of-3 NMI for every timestamped run in that folder).

Step-by-step holdout workflow: [per_lab_holdout_pilot.md § When jobs finish](per_lab_holdout_pilot.md#when-jobs-finish-postprocess).

---

## Default postprocess (one run dir)

```bash
source .venv/bin/activate
cd /work3/s204070/SPA
PYTHONPATH=. python3 scripts/cv4fold/postprocess_fold.py \
  --result-root results/cv4fold/per_lab_holdout/lab_2/fold_4/cgmvae/<run_name_timestamp>/
```

| File | Size | Contents |
|------|------|----------|
| `best_run.json` | ~200 B | Best-of-3 seed + NMI |
| `val_nmi_by_lab.csv` | ~300 B | Pooled holdout NMI per seed |
| `val_nmi_summary.json` | ~500 B | All seeds + best run pointer |

---

## Final runs: predictions & per-mouse NMI

1. **Before submit** — set on the YAML (or in `locked_recipes.py` for regenerated configs):

```yaml
visualizer:
  save_results_npz: true   # default false on pilot configs
```

2. **After jobs finish** — same loop as above, plus optional split (reads npz once; metrics-only by default):

```bash
PYTHONPATH=. python3 scripts/cv4fold/postprocess_fold.py \
  --result-root results/.../<run_name_timestamp>/ \
  --per-mouse-metrics
```

| Flag | When | Disk |
|------|------|------|
| (default) | Tables / best-of-3 | KB |
| `--per-mouse-metrics` | Per-mouse NMI in `per_mouse/run_*/sub-XXX/metrics.json` | ~100 B/mouse |
| `--save-predictions` | Also `predictions.npz` per mouse | Large — thesis archive only |
| `--save-plots` | Minimal `pca_pc1_pc2.png` per mouse | Moderate — usually skip (tripanel PNGs already in `plots/`) |

For thesis you typically keep **`plots/{1,2,3}/results.npz`** from training and run **`--per-mouse-metrics`** only if you need per-mouse NMI in CSV form.

---

## Disk tips

1. **Pilot/holdout:** keep `save_results_npz: false`; plots + `metrics.txt` + postprocess CSVs are enough.
2. **Delete old ablation trees** after NMI is in findings docs (`scrape_ablation_nmi.py`).
3. **Checkpoints** dominate quota — delete failed runs and non-best seeds after postprocess picks best-of-3.
4. Check quota: `getquota_work3.sh`

---

## Scripts

| Script | Purpose |
|--------|---------|
| [`postprocess_fold.py`](../../scripts/cv4fold/postprocess_fold.py) | One result dir → JSON/CSV |
| [`select_best_vae_run.py`](../../scripts/cv4fold/select_best_vae_run.py) | Best-of-3 only |
| [`aggregate_val_nmi_by_lab.py`](../../scripts/cv4fold/aggregate_val_nmi_by_lab.py) | CSV/JSON aggregates |
| [`split_per_mouse.py`](../../scripts/cv4fold/split_per_mouse.py) | Optional per-mouse metrics |
| [`summarize_experiment.py`](../../scripts/cv4fold/summarize_experiment.py) | Batch CSV over many runs |
| [`metrics_paths.py`](../../scripts/cv4fold/metrics_paths.py) | Path helpers |

## Metrics path layout

Post-train validation writes:

```
<run_name_timestamp>/plots/{1,2,3}/metrics.txt
<run_name_timestamp>/plots/{1,2,3}/results.npz   # only if save_results_npz: true
```

Training plots also under `<run_name_timestamp>/{1,2,3}/plots/` and `<run_name_timestamp>/plots/` (seed 1 latent diagnostics).

## Related

- Holdout pilot: [per_lab_holdout_pilot.md](per_lab_holdout_pilot.md)
- Selection metric: [README.md](README.md#selection-metric-final-cv4fold)
- Branch review: [../training/chmmgmm_fold_branch_review.md](../training/chmmgmm_fold_branch_review.md)
