# K = 8

- **Best seed:** 2
- **Prior NMI:** 0.5290
- **Run:** `joint_k_sweep_f4_K8_20260609-232933`

## Show Birgitte (in order)

1. `frequency_plot_gmm_predicted.png` — thesis Fig 27 physiology grid (all K rows)
2. `pca_comparison_true_vs_predicted.png` — expert vs GM substages in latent space
3. `transition_matrix_predicted.png` — switching dynamics
4. `label_distribution.png` — occupancy / rare states
5. `biology_compact.pdf` — compact summary panel
6. `transition_hypnogram.pdf` — 2 h excerpt vs expert macro

## Full stack

| File | Content |
|------|---------|
| `pca_scatter_*.png` | PCA true / predicted / kmeans |
| `transition_matrix_*.png` | Transition matrices |
| `latent_feature_boxplots.png` | Latent dim separation |
| `substage_panels.pdf` | Publication physiology grid |

**Ask:** stable substages vs transitions? Name each row? Keep this K for the paper?
