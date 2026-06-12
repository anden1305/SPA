# Paper figure outputs (cv4fold holdout)

**Added 2026-06-09.** Isolated from thesis `results/substages/` and `results/substages_analysis/`.

## Curated figures (main + SI)

**Canonical build:** `docs/paper/figures/curated/` via:

```bash
PYTHONPATH=. python3 scripts/paper/build_curated_figures.py
```

Copies to `docs/paper/figures/main/` and `figures/supplementary/` (see [`figures/README.md`](figures/README.md)).

| Main fig | Content | Script |
|----------|---------|--------|
| Fig1 | Protocol + ladder schematic | `plot_fig1_schematic.py` |
| Fig2 | Joint holdout NMI | `plot_ladder_figure.py` |
| Fig3 | Within-lab heatmap | `plot_within_lab_heatmap.py` |
| Fig4 | 2×2 substage biology | `plot_fig4_substage_compact.py` |
| Fig5 | Transitions + hypnogram | `plot_figure27_compact.py` |

## Output root (NPZ / analysis)

`results/cv4fold/paper_figures/`

| Subfolder | Contents |
|-----------|----------|
| `k_sweep/` | Thesis Fig 23-style dual axis: mean NMI (blue) + mean log p(z₁:T) (orange) vs K; PDF + PNG + JSON |
| `substages/K{n}_seed{s}/analysis/` | PCA, transitions, frequency grid, label distribution (300 dpi PNG) |
| `substages/K{n}_seed{s}/publication/` | Fig 3 panels + Fig 4 combined/hypnogram (PDF) |
| `manifest.json` | Paths and chosen K |

## Biology meeting pack (all K, thesis-style)

```bash
PYTHONPATH=. python3 scripts/paper/run_biology_meeting_pack.py
```

→ `results/cv4fold/paper_figures/biology_meeting/` (+ sync to `docs/paper/figures/professor_meeting/`)

## Single-K / paper staging

```bash
PYTHONPATH=. python3 scripts/paper/run_paper_substage_figures.py \
  --npz results/cv4fold/paper_k_sweep/fold_4/K4/<run>/plots/3/results.npz \
  --copy-to-paper
```

```bash
bsub < hpc/submit/paper/run_substage_figures.sh
```

Log: `hpc/output/paper/substage_figures_%J.out`

## Thesis Fig 30 taxonomy schematic

**Added 2026-06-10.** Node layout = latent PCA centroids; edges = change-only transition matrix (p tiers like thesis).

Requires a **higher-K** cHMM-GMVAE holdout run with `results.npz` (locked K=3 is macro-only; use K-sweep).

```bash
# 1) Generate fold configs (once)
PYTHONPATH=. python3 scripts/cv4fold/generate_chmm_k_sweep.py --fold 3 --k 7 13

# 2) Train (bsub) — see hpc/submit/cv4fold/submit_chmm_k_sweep.sh with FOLD=3

# 3) Plot (pick best-seed npz under results/cv4fold/paper_k_sweep/fold_3/K13/.../plots/<seed>/results.npz)
PYTHONPATH=. python3 scripts/paper/plot_taxonomy_schematic.py \
  --npz results/cv4fold/paper_k_sweep/fold_3/K13/<run>/plots/<seed>/results.npz \
  --out results/cv4fold/paper_figures/fold_3_K13_taxonomy.pdf
```

Optional manual labels after expert review: `--taxonomy-json path/to/overrides.json` (`{"5": {"role": "transition", "exclude": false}, ...}`).

## Scripts

- [`scripts/paper/run_paper_substage_figures.py`](../../scripts/paper/run_paper_substage_figures.py) — orchestrator
- [`scripts/paper/plot_k_sweep_dual_axis.py`](../../scripts/paper/plot_k_sweep_dual_axis.py) — K-sweep (thesis Fig 23 style)
- [`scripts/paper/plot_figure27_compact.py`](../../scripts/paper/plot_figure27_compact.py) — publication Fig 3–4
- [`scripts/paper/plot_taxonomy_schematic.py`](../../scripts/paper/plot_taxonomy_schematic.py) — thesis Fig 30-style taxonomy
- [`scripts/substage_analysis/frequency_plot.py`](../../scripts/substage_analysis/frequency_plot.py) — thesis Fig 27 grid (`subject_cmap=twilight_shifted`, `font_scale=1.45`)

## Staging

With `--copy-to-paper`, key PDFs also land in `docs/paper/figures/` and `paper/overleaf/figures/`.

Default winner: **K=4, seed 3** (best prior NMI 0.643 on fold-4 holdout).
