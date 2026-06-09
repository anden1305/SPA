# ChatGPT Deep Research prompt

Copy the block below into ChatGPT Deep Research (or similar). Merge output into [`references.bib`](references.bib) and [`related_work_novelty.md`](related_work_novelty.md).

---

```
ROLE: You are a biomedical ML literature reviewer preparing a Related Work section for a short journal paper (PLOS Computational Biology or Sleep).

PAPER WE ARE WRITING:
Title (working): "Decoder-only conditional latent models with temporal mixture priors for unsupervised cross-lab mouse sleep staging and substage discovery"

METHOD IN ONE PARAGRAPH:
We study unsupervised sleep staging on the Mouse Sleep Staging Validation dataset (MSSV, OpenNeuro ds006366, 92 mice, multi-lab EEG+EMG, 4s epochs). We compare a controlled model ladder on the same features and evaluation protocol:
(1) HMM / MAR-HMM on raw or engineered spectral features (thesis baseline),
(2) HMMGMVAE — Gaussian mixture VAE with HMM prior over latent windows, no subject conditioning,
(3) cGMVAE — subject-conditioned decoder only (encoder sees signal only),
(4) cHMMGMVAE — cGMVAE + warm sticky HMM-GMM prior (T≈32–64 windows).
Primary metric: prior-prediction NMI vs expert Wake/NREM/REM under leave-mice-out / cv4fold holdout across labs 2, 3, 5. Secondary: substage resolution sweep (K=3–15 mixture components), cross-seed stability, transition matrices, and physiological validation panels (PSD, theta/delta, EMG power, bout lengths, macro-label composition, subject mixing) analogous to Katsageorgiou et al. 2018 PLOS Biology Figure-style summaries. Biological interpretation will be done with a sleep neuroscientist (Kornum lab).

WHAT COUNTS AS "SIMILAR":
- Unsupervised or weakly supervised mouse/rat sleep staging from EEG+EMG
- HMM, HSMM, MAR-HMM, GMM-HMM, VAE, GMVAE, VAE-HMM, RBM/mcRBM on sleep data
- Substage / microstate discovery in rodents
- Cross-subject, cross-lab, or domain-generalization sleep staging
- MSSV dataset users or methods explicitly evaluated on multi-lab mouse data

EXCLUDE unless highly relevant:
- Pure human clinical PSG papers without rodent parallel
- Supervised-only CNN papers with no unsupervised discovery angle (mention only as performance ceiling)

DELIVERABLES (structured report):
1. EXECUTIVE SUMMARY (10 bullets): biggest gaps our paper could fill vs state of the art.
2. TIERED BIBLIOGRAPHY (15–25 papers):
   - Tier A: nearly identical problem (unsupervised rodent staging + dynamics/substages)
   - Tier B: method analog (VAE-HMM, GMVAE, conditional VAE, HMM on features)
   - Tier C: dataset/benchmark/supervised ceiling (SPINDLE, AccuSleep, SlumberNet, MLS-Net)
   For each paper: citation, 2-sentence summary, what they report (NMI/accuracy/cross-subject?), what they do NOT do that we do.
3. COMPARISON TABLE: columns = {Paper, Species, Supervision, Representation, Temporal model, Cross-subject?, Substates?, Dataset, Metric, Main limitation relative to us}.
4. VENUE FIT: rank PLOS Computational Biology, Sleep, Journal of Sleep Research, PLOS Biology, Bioinformatics, Frontiers in Neuroscience for THIS story (5-page main + heavy supplement). One paragraph each with precedent papers.
5. NOVELTY STRESS TEST: list 5 reviewer objections ("incremental over mcRBM", "only NMI", "lab-specific tuning", etc.) with suggested rebuttals grounded in citations.
6. MUST-CITE LIST: minimum set for introduction (≤12 refs).
7. RECENT (2023–2026): flag any papers we likely missed.

MUST INCLUDE (same-author network as our supervisors/data):
- Rose et al. 2026 bioRxiv DOI 10.64898/2026.03.27.714381 — "Lack of Consensus for Manual Mouse Sleep Scoring Limits Implementation of Automatic Deep Learning Models" (Kornum, Mørup, Zahid, García Ciudad et al.). Frame how our unsupervised cross-lab holdout relates to their finding that DL models fail across labs when manual labels disagree.

SEARCH KEYWORDS TO USE:
mouse sleep staging unsupervised HMM; Gaussian mixture VAE sleep; conditional VAE EEG subject invariant; mcRBM sleep substages; FASTER2 Gaussian HMM; MSSV ds006366; Rose 2026 manual consensus mouse sleep scoring; VAE-HMM time series; decoder-only conditioning VAE; cross-lab sleep staging rodents; Somnotate vigilance transitions; SPINDLE cross-lab mouse sleep.

CONSTRAINTS:
- Prefer peer-reviewed journals; include arXiv only if highly cited or direct architectural match.
- Distinguish macro staging (3 states) vs substage discovery.
- Note when papers use population training vs transductive / subject-specific fitting.
```
