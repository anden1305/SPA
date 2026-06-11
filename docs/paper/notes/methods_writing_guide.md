# Methods writing guide (paper adaptation)

**Added 2026-06-09.** Adapts the thesis professor prompt for `plos_my_paper.tex` Materials and methods.

## Mode for this paper

**Writing only** — Methods should be stable before K-sweep results land. Results change; protocol does not.

## Sources (read in this order)

1. **Code truth:** `src/models/vae.py` (`decoder_only_conditioning`, `predict_gmm_labels`, `predict_hmm_labels`)
2. **Locked numbers:** `scripts/cv4fold/locked_recipes.py`, joint YAML `chmmgmvae_locked.yaml`
3. **Protocol:** `docs/paper/notes/holdout_experiments.md`, `docs/cv4fold/unified_holdout_paper_line.md`
4. **Thesis baseline:** `Master Thesis.pdf` / Ch. 2.2–2.4, 3.4–3.5, 5.6 (via `scripts/paper/extract_source_materials.py`)
5. **Framing doc:** `docs/google_sheets/Special Course - Project framing.docx` — **stale**; use only for CNN/end-to-end future work, not current spectral pipeline

## Paper vs thesis (do not blur)

| Topic | Thesis | This paper |
|-------|--------|------------|
| Conditioning | Encoder + decoder | **Decoder-only** at train |
| Holdout eval | Subject ID in encoder | **No subject ID** at staging |
| Cohort | 10 mice, lab 3 subset | **92 mice, labs 2/3/5**, cv4fold |
| HMM input | Expert features | **Spectral CNN** ladder |
| cHMM temporal prior | Often two-step MAR-HMM | **End-to-end warm HMM–GMM** in VAE |
| LOSO NMI ~0.70 | Primary result | **Supplement contrast** only |
| Holdout NMI ~0.56 | Not reported | **Primary result** |

## Style rules (from thesis prompt, adapted)

- British English, impersonal voice in Methods
- No invented claims — cite thesis or point to code/config
- ML terminology (distribution shift, label noise, nuisance factors)
- Minimal equations — NMI \eqref{eq:nmi} is sufficient in Methods
- Cross-reference SI for locks, ablations, synthetic validation
- Avoid repeating Intro claims; Methods = reproducible procedure

## Subsection checklist

- [x] Animals and dataset
- [x] Epoch construction and preprocessing
- [x] Input representation
- [x] Cross-validation scopes
- [x] Model ladder
- [x] Encoder–decoder architecture
- [x] Decoder-only conditioning
- [x] Latent priors and assignment
- [x] Holdout inference
- [x] Training and checkpoints
- [x] NMI metric
- [x] K sweep
- [x] Physiology and transitions
- [ ] **After K-sweep:** one sentence naming chosen K criterion result (Results only)

## Style reference (QSLP-AE colleague paper)

Ciudad et al., same MSSV + Morten/Birgitte — [OpenReview K3Z4jVHUnf](https://openreview.net/pdf?id=K3Z4jVHUnf). Full extract: [`source_extraction/qslp_ae_reference.txt`](source_extraction/qslp_ae_reference.txt). Cursor rule: [`.cursor/rules/report-writing-qslp-style.mdc`](../../.cursor/rules/report-writing-qslp-style.mdc).

## Rebuild report PDF

```bash
bash docs/paper/build/compile_report.sh
```
