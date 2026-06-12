# Specialkursus — opfyldelse af læringsmål

**Projekt:** Håndtering af individuel variabilitet og dynamikker i søvnstadier baseret på variational autoencoders  
**Periode:** 16-03-2026 – 08-06-2026 · **10 ECTS** · **Vurdering:** rapport + mundtlig eksamen  
**Vejledere:** Morten Mørup (DTU), Birgitte Rahbek Kornum (KU)  
**Vurderet:** 2026-06-11 (repo + [`docs/paper/plos_my_paper_report.pdf`](paper/plos_my_paper_report.pdf))

## Samlet vurdering

| Læringsmål | Opfyldelse | Kort konklusion |
|------------|------------|-----------------|
| 1. Conditioning + zero-shot VAE | **~90%** | Implementeret, ableret og evalueret på holdout; generalisering begrænset ved lab-heterogenitet |
| 2. GMVAE → HMM-dynamik | **~95%** | cHMM–GMVAE med sticky HMM–GMM prior i produktion; slår cGMVAE på joint holdout |
| 3. Anvendelse + kvantitativ evaluering | **~90%** | Fuld MSSV-pipeline, NMI/accuracy/per-mouse/lab; enkelte baselines ufuldstændige |
| 4. Artikeludkast | **~85%** | PLOS-klar manuscript + figurer; endnu ikke indsendt, få elementer under polish |

**Overordnet:** Alle fire læringsmål er **substansielt opfyldt** for bestået på et 10-ECTS specialkursus. Resten er polish og sekundære eksperimenter (HMM raw-grid, endelig K-sweep), ikke manglende kernekompetence.

---

## 1. Conditioning i VAE + zero-shot latente repræsentationer

**Mål (eng.):** Understand and apply conditioning in a VAE context accommodating zero-shot learning of latent representations.

**Status: Largely fulfilled**

**Hvad der er gjort**

- **Subject conditioning** i `src/models/vae.py`: embeddings i encoder og/eller decoder; `decoder_only_conditioning` så identitet kun på decoder under træning.
- **Zero-shot holdout:** ved validering bruges encoder + mixture-prior **uden** subject-ID (`holdout_experiments.md`, Methods i manuscript).
- **Systematisk ablation:** decoder-only vs encoder+decoder (`docs/cv4fold/ablations/ablation_enc_dec_conditioning.md`); `subject` vs `subject_lab` (`cross_lab_cv4fold.md`).
- **Fase 1 (lab_3):** decoder-only hotstart ~0.63–0.65 prior NMI; encoder+decoder ~0.74 (`decoder_only_lab3_chmm_experiments.md`).
- **Fase 2 (cv4fold):** joint 3-lab holdout med leave-mice-out; subject + seq64 som låst opskrift.

**Evidens**

- Manuscript: decoder-only conditioning som hovedbidrag; zero-shot cross-lab staging.
- Holdout ladder: cHMM–GMVAE joint mean prior NMI **0.58** vs cGMVAE **0.28** (4 folds, `paper/overleaf/tables/holdout_ladder.csv`).

**Huller / nuancer (egnet til mundtlig diskussion)**

- Zero-shot **virker**, men ikke lige godt på tværs af lab (lab 2/5 svagere end lab 3; REM recall lav på nogle folds).
- `subject_lab`-conditioning kollapsede på holdout — lab og mouse er ikke fuldt adskilt.
- Conditioning løser ikke al cross-lab variabilitet alene (montage, kohortestørrelse, scoring).

---

## 2. GMVAE udvidet med HMM til state dynamics

**Mål (eng.):** Advance existing GMM-based VAEs to account for state dynamics using hidden Markov modeling procedures.

**Status: Fulfilled**

**Hvad der er gjort**

- **cGMVAE:** statisk GMM-prior på latente means (`T=1`).
- **cHMM–GMVAE:** warm sticky **HMM–GMM** prior (`T=64`, κ≈0.92), K-means warm-start, Viterbi-dekodning (`src/models/hmm_gmm_prior.py`, `vae.py`).
- **Fair sammenligning:** samme spektrale front-end, folds, hyperparam-locking; cHMM vs cGMVAE vs HMMGMVAE (uden subject).
- **Temporal readouts:** bout-længder, transition-matricer, K-sweep substage-fysiologi (Fig 3–4, S1–S3).

**Evidens**

- cHMM slår cGMVAE på **alle fire** joint holdout-folds (fx fold 3: 0.68 vs 0.31 NMI).
- Lab_3 proof: cHMM marginalt bedre end cGMVAE ved samme checkpoint (Phase 1).
- Omfattende incohort-ablations (seq length, warm vs hmm_gmm, joint fold-4 deltas) dokumenteret under `docs/cv4fold/ablations/`.

**Huller**

- I nogle tidlige fold-4 piloter var fordelen mindre end forventet (generalisering + seed-fragilitet).
- Kompleksitet og tuning (warm-start schedule, sticky κ) er høj — del af læringsudbyttet, men ikke “plug-and-play”.

---

## 3. Anvendelse på søvndata + kvantitativ evaluering (annotation + variabilitet)

**Mål (eng.):** Apply tools on sleep recordings; evaluate correspondence to mouse sleep annotation and influence of variability across mice.

**Status: Largely fulfilled**

**Hvad der er gjort**

- **Data:** MSSV (`ds006366`), labs 2/3/5, spektral pipeline, 4-fold CV med fold-splits dokumenteret (S8/S9 tables).
- **Annotation:** expert Wake/NREM/REM; prior NMI, macro-aligned accuracy, 3×3 confusion (S3 Fig), per-mouse NMI (S2 Fig).
- **Variabilitet på tværs af mus/lab:** per-lab holdout vs joint; per-mouse barplots; diskussion af kohortestørrelse, timer/mus, udstyr (`plos_my_paper.tex` Results).
- **Reliability:** 10-seed decoder-only grids på lab 2/3/5 (`results/decoder_only/reliability/`).
- **Substages:** K-sweep (K=3–15), fysiologi-grids, population + holdout packs.

**Evidens**

- 20 HQ-mus, ~92k val-epoker fold 4; confusion viser Wake/NREM høj recall, REM svag (~3% på exemplar-seed).
- Within-lab træning > joint på samme holdout-mus (Table 2) — direkte mål på cross-mouse/lab variabilitet.

**Huller**

- **HMM (raw)** baseline: ~7/16 holdout-celler færdige (`holdout_results_status.md`) — sekundær baseline, ikke kerneclaim.
- Nogle **3. seed-reruns** og **K-sweep jobs** stadig i kø (`FRIDAY_CHECKLIST.md`).
- Evaluering er primært mod **makro**-labels; substage “sandhed” er uovervåget discovery (som mcRBM-traditionen).

---

## 4. Formidling som artikeludkast

**Mål (eng.):** Disseminate modeling procedures and results as an article draft suitable for submission to a relevant journal.

**Status: Largely fulfilled (draft ready; submission pending)**

**Hvad der er gjort**

- **Manuscript:** [`docs/paper/plos_my_paper.tex`](paper/plos_my_paper.tex) — Abstract, Introduction, Data, Methods, Results/Discussion, Conclusion, Appendix (S1–S9).
- **Mål-journal:** PLOS Computational Biology (template, `plos2025.bst`, `references.bib` inkl. Stevner, Katsageorgiou, Somnotate).
- **Figurer/tabeller:** Fig 1–4 + S1–S3; holdout ladder; within-lab table; CV split tables; report-PDF med indlejrede figurer.
- **Reproducerbarhed:** YAML configs, HPC submit scripts, postprocess docs (`docs/cv4fold/postprocess.md`), plot scripts under `scripts/paper/`.

**Huller før indsendelse**

- Endelig **K**-valg og taxonomy-panel (Fig 3) kan opdateres når K-sweep færdig.
- Sidste **typos/word budget** pass (`notes/word_budget.md`, ~3–3.5k ord).
- **Ikke indsendt** endnu — udkastet er “suitable for submission”, ikke submitted.

---

## Styrker til mundtlig eksamen (1–2 min per mål)

1. **Conditioning:** Forklar decoder-only vs encoder+decoder; vis at inference ikke bruger subject-ID; nævn holdout-NMI som zero-shot metric.
2. **HMM:** Skitser warm GMM → sticky HMM–GMM → Viterbi; cHMM vs cGMVAE som controlled ablation.
3. **Evaluering:** 4-fold cross-lab, per-mouse/lab breakdown, limitations (REM, lab 5, seed fragility).
4. **Artikel:** PLOS-struktur, novelty vs supervised CNNs og mcRBM; appendix som reproducibility layer.

## Referencer i repo

| Emne | Sti |
|------|-----|
| Roadmap | [`docs/decoder_conditioning_roadmap.md`](decoder_conditioning_roadmap.md) |
| Phase 1 | [`docs/decoder_only/decoder_only_lab3_chmm_experiments.md`](decoder_only/decoder_only_lab3_chmm_experiments.md) |
| Phase 2 / cv4fold | [`docs/cv4fold/cross_lab_cv4fold.md`](cv4fold/cross_lab_cv4fold.md) |
| Manuscript | [`docs/paper/plos_my_paper.tex`](paper/plos_my_paper.tex) |
| Hand-in checklist | [`docs/paper/notes/FRIDAY_CHECKLIST.md`](paper/notes/FRIDAY_CHECKLIST.md) |
