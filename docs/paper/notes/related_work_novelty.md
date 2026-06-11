# Related work and novelty positioning

**Added 2026-06-09.** Literature map and gap analysis for the 5-page paper (*Decoder-only conditional latent models with temporal mixture priors for unsupervised cross-lab mouse sleep staging and substage discovery*). BibTeX: [`references.bib`](references.bib). Deep-research merge: [`deep_research_synthesis.md`](deep_research_synthesis.md).

---

## Executive summary (gaps we fill)

1. **Controlled model ladder** on MSSV with fixed evaluation: feature HMM → HMMGMVAE → cGMVAE → cHMMGMVAE (same features, holdout protocol).
2. **Zero-shot holdout inference** — population-trained; held-out mice scored via encoder + mixture prior **without subject embeddings** (contrast AccuSleep per-mouse calibration).
3. **Decoder-only conditioning (training-only)** — subject embeddings in decoder reconstruction only; encoder learns subject-invariant latents.
4. **Joint VAE + sticky HMM-GMM prior** for sleep (architecture cousins: SVAE, VAME, HMM-VAE; none on multi-lab mouse MSSV holdout).
5. **Substage resolution sweep** with physiology panels (Katsageorgiou-style) plus **HMM transition matrix** from the learned prior.
6. **Cross-lab cv4fold holdout** (labs 2/3/5) — leave-mice-out, not per-animal SegWay/WUCSS fitting.
7. **When temporal priors help** — lab-heterogeneous wins (cHMM lab_3/5; cGMVAE lab_2) as an honest scientific result.
8. **Rose 2026 framing** — manual consensus limits cross-lab supervised DL; we complement with unsupervised discovery.
9. **Biological readout** — expert-guided substage naming (Birgitte); transition substates at macro boundaries.
10. **Reproducible pipeline** — locked recipes in `scripts/cv4fold/locked_recipes.py`, config generators, postprocess.

---

## Tier 1 — cite prominently

1. **Rose et al. 2026** — [bioRxiv 10.64898/2026.03.27.714381](https://www.biorxiv.org/content/10.64898/2026.03.27.714381v1): **Primary framing** — manual scoring lacks cross-lab consensus; SOTA DL fails across labs.
2. **Rose et al. 2025 (MSSV)** — OpenNeuro ds006366.
3. **Katsageorgiou et al. 2018** — mcRBM substages (*PLOS Biology*).
4. **Yamada 2024 (FASTER2)** — unsupervised Gaussian-HMM baseline.
5. **Somnotate 2024** — transition / intermediate states (*PLOS Comp Biol*).
6. **Barger 2019 (AccuSleep)** — transductive mixture z-score contrast.
7. **Yaghouby & Sunderam 2016 (SegWay)** — k-means + HMM temporal smoothing.
8. **Cusinato et al. 2024 (WUCSS)** — recent unsupervised light/deep NREM substates.

---

## Comparison table

| Paper | Species | Supervision | Representation | Temporal | Cross-subject | Substates | Dataset | Metric | Limitation vs us |
|-------|---------|-------------|----------------|----------|---------------|-----------|---------|--------|------------------|
| Rose 2026 bioRxiv | Mouse | Supervised DL eval | Published DL | Varies | **Fails cross-lab** | No | Multi-site | Accuracy | Labels + DL; not unsupervised substages |
| AccuSleep 2019 | Mouse | Supervised | CNN | None | Cross-mouse w/ calibration | No | Multi | ~97% acc | 10-min manual baseline per mouse; not zero-shot |
| SegWay 2016 | Mouse | Unsupervised | k-means features | HMM | Per-mouse fit | No | In-house | ~94% agree | No population holdout; macro only |
| Katsageorgiou 2018 | Mouse | Unsupervised | mcRBM | No | Pooled | Yes | In-house | MI | No VAE, no cross-lab holdout, no HMM prior |
| WUCSS 2024 | Mouse | Unsupervised | Clustering | No | Per-session | Light/deep NREM | In-house | vs expert | No VAE; no population latent training |
| Yamada 2024 FASTER2 | Mouse | Unsupervised | 3 hand-crafted stats | GHMM | Within study | No | Multi | Accuracy | No learned latent; macro only |
| REST 2026 | Mouse | Supervised | Transformer | Long context | Cross-strain | No | Multi | κ≈0.87 | Labels required; supervised ceiling |
| Miladinovic 2019 SPINDLE | Mouse | Supervised CNN | End-to-end | CNN | Cross-lab | No | Multi-lab | Accuracy | Labels required |
| **This work** | Mouse | Unsupervised | cVAE latent + spectral | HMM-GMM prior | **Zero-shot cv4fold holdout** | Yes (K sweep) | MSSV | Prior NMI | Modest absolute holdout NMI |

---

## Novelty stress test (reviewer objections)

| Objection | Rebuttal |
|-----------|----------|
| Incremental over mcRBM | Learned embeddings, **zero-shot holdout**, temporal HMM prior, physiology panels |
| Only NMI vs expert macro labels | Rose 2026 label noise; **substage biology** and **transitions** are the payoff |
| Needs AccuSleep-style calibration | Holdout scored with **no subject embedding**, no 10-min manual baseline |
| Lab-specific tuning | Per-lab best model reported transparently; `cvae_overrides` in supplement |
| Thesis duplicate | Main = decoder-only + cHMM + multi-lab holdout; thesis = supplement |
| Holdout NMI modest | **Relative lift** over ladder; unsupervised cross-lab harder than thesis LOSO |
| Rose 2026 already showed DL fails | We **complement** with unsupervised discovery + substages |
| VAE+HMM not novel | SVAE is generic; **decoder-only + warm sticky HMM-GMM on MSSV ladder** is the fusion |

---

## Must-cite (introduction, ≤12)

1. Rose et al. 2026 (consensus / cross-lab DL)
2. Rose et al. 2025 (MSSV dataset)
3. Katsageorgiou et al. 2018 (substages)
4. Yamada et al. 2024 (FASTER2)
5. Miladinovic 2019 (SPINDLE ceiling)
6. Barger 2019 (AccuSleep calibration contrast)
7. Gelegen 2024 / Somnotate (transitions)
8. Cusinato et al. 2024 (WUCSS)
9. Johnson et al. 2016 (SVAE)
10. Dilokthanakul 2016 / Sohn 2015 (GMVAE / cVAE)
11. Sunagawa 2013 (FASTER)
12. Elmgreen & Bigom 2026 (thesis baseline)

*Supplement:* Rayan 2024, Biedebach 2025, Özdenizci 2020, Grieger 2021, REST 2026.

---

## Venue fit (short)

| Venue | Fit | Precedent |
|-------|-----|-----------|
| **PLOS Computational Biology** | **Best default** | Somnotate; methods + transitions |
| Sleep / J Sleep Research | Strong if biology leads | Substages, bout/transition focus |
| PLOS Biology | High bar | Katsageorgiou mcRBM |
| Bioinformatics | Methods fallback | Pipeline novelty |

See [`google_sheets/publication_place_email.md`](google_sheets/publication_place_email.md).
