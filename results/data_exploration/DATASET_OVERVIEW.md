# Mouse Sleep Staging Validation Dataset (MSSV) – Guided Tour and Technical Overview

> Version: ds006366 (OpenNeuro)  
> DOI: 10.18112/openneuro.ds006366.v1.0.0  
> Local extraction date: (fill in)

## 1. Start Simple (Explain Like I'm 12)
Mice sleep in different *modes* (wake, deep-ish sleep, dream-like sleep). Scientists record tiny electrical signals from the brain (EEG) and muscles (EMG) to see which mode the mouse is in every few seconds. This dataset is a big collection of those recordings from many labs. We use it to teach computers how to tell the difference between the sleep modes automatically.

## 2. Plain Language Summary
This dataset contains 92 curated polysomnographic recordings (EEG + EMG) from healthy wild-type mice collected across five independent laboratories. All recordings are standardized to:
- Sampling rate: 128 Hz
- Epoch resolution: 4-second annotated windows
- BIDS-compliant structure (Brain Imaging Data Structure)

Each mouse has a single labeled recording with per-epoch sleep stage labels. Stage labels differ subtly across labs historically, but have been harmonized to a coarse taxonomy suitable for automated sleep staging transfer learning and benchmarking.

## 3. Scientific Context
Rodent sleep studies are essential for modeling sleep disorders, pharmacological interventions, and translational neurobiology (e.g., narcolepsy, arousal regulation, network oscillations). This dataset underpins a published pipeline adapting the U-Sleep architecture—originally trained on human PSG—to mouse EEG/EMG for automated sleep staging and phenotype feature extraction (see Rose et al. 2025, Sleep Advances). The pipeline uses per-epoch predictions to derive phenotypic features (bout statistics, spectral power distributions, fragmentation metrics) relevant to disease modeling.

## 4. Data Provenance & Cohort Structure
Although the original full study spans multiple cohorts (including diseased genotypes), this subset (MSSV) includes **healthy mice only** from 5 labs:
- lab_1: 10 subjects, minimal channel montage (EEG1 + EMG)
- lab_4: 27 subjects
- lab_3: 32 subjects (some dual EEG channels)
- lab_2: 17 subjects (multi-EEG montage, up to 4 EEG channels + EMG)
- lab_5: 6 subjects

Channel availability per lab is given in `labs.tsv` (stored as Python-list-like strings). Consistent presence of one EMG channel enables downstream EMG amplitude and RMS-based arousal proxies.

## 5. Files & Modalities
Key file types per subject (example: `sub-001`):
- `*_eeg.edf`: Continuous multi-channel biosignal (EEG/EMG)
- `*_events.tsv`: Epoch-wise staging (columns: onset [s], duration [s], stage [int])
- `*_channels.tsv`: Channel names and types

Global metadata:
- `participants.tsv`: Subject-to-lab mapping
- `labs.tsv`: Channel inventory per lab
- `dataset_description.json`: BIDS metadata & authorship
- `README.md`: High-level narrative

## 6. Sleep Stage Encoding & Harmonization
The raw numeric `stage` codes in `*_events.tsv` are **lab-specific encodings** (not explicitly documented inside the file). Our exploration script infers a mapping to a canonical triad: Wake / NREM / REM using heuristics:
1. Longest cumulative duration → assumed NREM
2. Smallest cumulative duration → assumed REM
3. Remaining class with longest mean bout length → assumed Wake
Any surplus classes retain generic names (StageX). This heuristic is logged in `results/data_exploration/README.txt` and should be replaced with authoritative mapping if available. The 4-second epoch resolution aligns with downsampled consensus across labs.

## 7. Feature Domains of Interest
Downstream analyses typically derive:
- Time allocation: proportion of time in Wake / NREM / REM
- Fragmentation: bouts per hour and mean bout length per stage
- Bout size distribution: proportion of episodes in micro (4 s), short (4–32 s), intermediate (32–60 s), medium (1–5 min), long (>5 min) bins
- Transition dynamics: stage-to-stage transition probability matrix
- (Potential future) Spectral metrics: relative band powers (delta/theta/alpha/beta sub-bands) via Welch estimates

## 8. Generated Outputs (this repository)
The script `notebooks/data_exploration.py` produces:
- `stage_summary_stats.csv`: Counts, total seconds, hours, and proportion per inferred stage label
- `recording_meta.csv`: Per-subject duration and row counts
- `bout_stats.csv`: Bout length and fragmentation per subject & stage
- Plots (PNG):
  - `participants_per_lab.png`
  - `stage_distribution_overall.png`
  - `stage_distribution_per_lab.png`
  - `mean_bout_length_per_stage.png`
  - `bouts_per_hour_per_stage.png`
  - `bout_size_distribution.png`
  - `episode_duration_boxplot.png` / `episode_duration_violin.png`
  - `hypnogram_<subject>.png` (30 s aggregated example)
  - `transition_matrix.png`
  - `recording_duration_hist.png`

## 9. Methodological Caveats
- Heuristic staging: Without explicit original codebook, numeric→semantic mapping may misassign Wake vs NREM if distributional assumptions fail (e.g., unusually long REM in some manipulations—unlikely for healthy WT).
- Inter-lab variability: Different electrode placements and scoring conventions can shift spectral characteristics and fragmentation patterns; no normalization beyond epoch granularity performed here.
- Single recording per subject: Limits intra-individual variability modeling.
- No artifact rejection: Potential noise segments could bias bout boundary detection.

## 10. Recommended Extensions
1. Incorporate authoritative stage label mapping (if accessible from original pipeline repository or supplemental docs). 
2. Compute spectral feature panel (delta subdivided, theta bands, alpha, beta) per stage and per lab → cross-lab comparability.
3. Add circadian segmentation (light vs dark phase) if clock time metadata becomes available.
4. Introduce artifact detection (e.g., amplitude IQR clipping, z-score thresholding, or ICA-based EMG contamination removal where multi-EEG).
5. Provide reproducible notebook (Jupyter) with interactive exploration and QC metrics.

## 11. Quality Control Suggestions
| Aspect | Simple Check | Enhancement |
|--------|--------------|-------------|
| Epoch continuity | Verify onset increments of 4 s | Flag gaps/overlaps |
| Duration sanity | Total duration vs EDF length | Cross-validate with raw signal length |
| Channel integrity | Compare channels.tsv to EDF header | Auto-fix naming mismatches |
| Stage balance | Report per-lab class imbalance | Weighted sampling in downstream models |
| Noise detection | RMS / variance outlier detection | Adaptive notch & band filtering |

## 12. Reproducibility
Re-run exploration:
```
uv run python notebooks/data_exploration.py
```
(Add `--force` option later if you implement caching.) Ensure dependencies (`pandas numpy matplotlib seaborn`) are installed via `pyproject.toml`.

## 13. Ethical & Licensing Notes
- License: CC0 (dataset), enabling unrestricted reuse with attribution courteous but not legally required.
- Animal ethics approvals specified in the metadata; secondary analysis should still cite the originating paper (Rose et al. 2025) and dataset DOI.

## 14. Citations
If you use this dataset & scripts, cite:
- Rose et al. (2025) Sleep Advances. DOI: 10.1093/sleepadvances/zpaf025
- OpenNeuro Dataset: doi:10.18112/openneuro.ds006366.v1.0.0
- U-Sleep adaptation (Perslev et al. NPJ Digit Med., 2021) for methodological lineage.

## 15. Glossary
- **Epoch**: A fixed-length time window (4 s here) labeled with a sleep stage.
- **Bout**: Consecutive epochs of the same stage.
- **Fragmentation**: Increased number of short bouts and transitions.
- **Hypnogram**: Timeline plot of sleep stage vs time.

## 16. Quick Data Access Patterns (Python)
```python
import pandas as pd, pathlib as p
root = p.Path('data/ds006366')
participants = pd.read_csv(root/'participants.tsv', sep='\t')
first_events = pd.read_csv(next((root/'sub-001'/'eeg').glob('*_events.tsv')), sep='\t')
```

## 17. Limitations of Current Script
- No parallelization (could be slow if extended to multi-day data)
- In-memory aggregation; refactor to generator + incremental CSV for scalability
- Lacks integrated spectral analysis – planned future enhancement

## 18. Data Model (Simplified)
```
Subject -> Recording (.edf)
         -> Events (onset, duration, stage)
Aggregate -> Stage summary / Bout statistics / Transition probabilities
```

## 19. High-Level Takeaways
- Dataset breadth (multi-lab) supports generalization studies for mouse sleep staging.
- Class balance skew (NREM dominant, REM sparse) demands thoughtful modeling strategies (sampling, loss weighting) when training classifiers.
- Fragmentation & transition matrices provide early signals for phenotype differentiation in disease model extensions (not part of healthy subset but structurally relevant).

---
Prepared as part of the SPA project data familiarization. Update sections (e.g. mapping) once authoritative stage documentation is integrated.
