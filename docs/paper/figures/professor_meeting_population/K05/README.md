# K = 5

- **Best seed:** 5
- **Prior NMI:** 0.4844
- **Run:** `population_k_sweep_K5_20260613-124308`

## Show Birgitte (in order)

1. `frequency_plot_gmm_predicted.png` — thesis Fig 27 physiology grid (all K rows)
2. `pca_comparison_true_vs_predicted.png` — expert vs cHMM–GMVAE substages in latent space
3. `tsne_scatter_true.png` / `tsne_scatter_predicted.png` — t-SNE of same latent subsample
4. `transition_matrix_predicted.png` — switching dynamics
5. `label_distribution.png` — occupancy / rare states
6. `biology_compact.pdf` — compact summary panel
7. `transition_hypnogram.pdf` — 2 h excerpt vs expert macro

## Full stack

| File | Content |
|------|---------|
| `pca_scatter_*.png` | PCA true / predicted / kmeans |
| `tsne_scatter_*.png` | t-SNE true / predicted (shared embedding) |
| `transition_matrix_*.png` | Transition matrices |
| `latent_feature_boxplots.png` | Latent dim separation |
| `substage_panels.pdf` | Publication physiology grid |

**Ask:** stable substages vs transitions? Name each row? Keep this K for the paper?
