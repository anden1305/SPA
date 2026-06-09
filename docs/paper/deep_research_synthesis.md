# Deep-research synthesis (ChatGPT + Gemini)

**Added 2026-06-09.** One-page merge for Morten/Birgitte. Sources: [`chatgpt_deep_research.md`](../google_sheets/chatgpt_deep_research.md), [`gemini_deep_research.md`](../google_sheets/gemini_deep_research.md).

---

## Problem we own

**Rose et al. 2026 (bioRxiv):** manual mouse scoring lacks cross-lab consensus; published supervised DL fails across MSSV sites until retrained on multi-lab labels. Our paper complements this with **unsupervised** population learning, a **controlled model ladder**, and **zero-shot holdout inference** (no per-mouse calibration).

---

## What we do that prior work does not combine

| Gap | Prior art | Our answer |
|-----|-----------|------------|
| Cross-lab unsupervised holdout | mcRBM pooled; SegWay per-mouse; WUCSS per-session | cv4fold leave-mice-out on MSSV labs 2/3/5 |
| Temporal structure | FASTER flat clustering; mcRBM no HMM | Sticky HMM–GMM prior on latent windows |
| Subject invariance | AccuSleep mixture z-score (10-min manual baseline); encoder-conditioned cVAE (thesis) | Decoder-only conditioning **at training**; **no subject ID at holdout staging** |
| Substates + transitions | mcRBM (~190 states); Grieger supervised pre-REM | K sweep + physiology panels + HMM transition matrix |
| Fair ablation | Papers compare different protocols | Locked ladder: HMM features → HMMGMVAE → cGMVAE → cHMMGMVAE |

---

## Contrast triad (for intro/discussion)

1. **Supervised ceiling** (SPINDLE, REST, Somnotate): high accuracy, needs labels, inherits rater noise.
2. **Transductive unsupervised** (AccuSleep, SegWay): per-subject calibration or per-animal fit.
3. **Static unsupervised** (FASTER/FASTER2, mcRBM, WUCSS): no deep latent + sticky HMM + MSSV holdout protocol.

---

## Key citations to add (main vs supplement)

| Paper | Role |
|-------|------|
| Rose 2026 | Lead framing |
| AccuSleep 2019 | Calibration contrast |
| SegWay 2016 | HMM temporal smoothing precedent |
| WUCSS 2024 | Recent unsupervised substates |
| Johnson SVAE 2016 | VAE+HMM lineage (Methods) |
| REST 2026 | Supervised temporal ceiling (intro clause; supplement table) |
| Grieger 2021 | Transition-state biology (Discussion) |
| Özdenizci A-cVAE 2020 | Subject-invariance contrast (supplement only) |

---

## Reviewer stress-test (condensed)

| Objection | Rebuttal |
|-----------|----------|
| Incremental over mcRBM | Holdout + temporal prior + decoder-only zero-shot staging |
| NMI vs 97% accuracy | Rose 2026 label noise; discovery not mimicry of single rater |
| Needs AccuSleep calibration | No subject embedding at eval; no 10-min manual baseline |
| K=15 meaningless | K sweep + PSD/bout panels + expert exclusion |
| VAE+HMM not novel | **Decoder-only + warm sticky HMM-GMM on MSSV ladder** |

---

## Venue (both agents agree)

**Primary:** PLOS Computational Biology (Somnotate precedent).  
**If Fig 3 biology strong:** Sleep / JSR.  
**5-page skeleton** expands via supplement for journal submission.

---

## Code pointers

- Holdout protocol: [`holdout_experiments.md`](holdout_experiments.md)  
- Zero-shot inference: `src/models/vae.py` (`decoder_only_conditioning`, `predict_gmm_labels`)  
- Locked recipes: `scripts/cv4fold/locked_recipes.py`
