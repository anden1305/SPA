# Structured interview guide — Birgitte Rahbek Kornum

**Added 2026-06-09.** Use after K substage sweep is locked and Fig-27-style panels are generated. Duration: 30–45 min. Bring: printed panel figure, transition matrix, hypnogram excerpt, list of excluded rare substages.

---

## Goals

1. Assign **biological names** to stable substages (active awake, quiet awake, NREM substages, REM, transitions).
2. Confirm which substages are **transition-like** vs **stable** (bout length + mixed PSD/EMG).
3. Flag **confounded/rare** substates to exclude from main text (thesis: substages 12–13).
4. Validate **subject mixing** — substages not driven by one genotype/lab.
5. Capture 2–3 **quotable sentences** for Discussion.

---

## Materials to share beforehand

- `paper/overleaf/figures/figure3_substage_panels.pdf` (or PNG)
- `paper/overleaf/figures/figure4_transition_hypnogram.pdf`
- Table: substage ID → prevalence (%) → dominant macro label (%)
- Model: cHMMGMVAE, K=___ , joint holdout fold ___

---

## Question script

### A. Macro alignment (5 min)

1. For each substage row in the panel, does the **dominant macro label mix** (Wake/NREM/REM) match your expectation?
2. Are any substages clearly **artifact or low-quality EEG** rather than physiology?

### B. Stable substages (10 min)

For rows with **long bout lengths** and consistent PSD:

3. Which rows correspond to **active awake** (high EMG, theta/delta ≈ 1)?
4. Which to **quiet / NREM-like** (low EMG, theta/delta < 1)?
5. Which to **REM-like** (low EMG, theta/delta > 1)?
6. Do you see **multiple NREM substages** as in mcRBM (Katsageorgiou 2018), or finer wake substages?

### C. Transition substages (10 min)

7. Rows with **short bouts** and mixed EEG/EMG — do these sit at **Wake↔NREM**, **NREM↔REM**, or **REM↔Wake** boundaries? (cf. thesis substages 5,7,8,10,11)
8. Does the **transition matrix** match known switching asymmetries (e.g. harder Wake→REM)?

### D. Exclusions (5 min)

9. Which substages should we **exclude** from interpretation (low N, confounded, artifact)?
10. Minimum prevalence threshold for main-text mention? (suggest: <1% or uneven subject mix)

### E. Cross-lab / methods (5 min)

11. Are spectral differences across labs **visible in substage panels**? Should we stratify by lab in supplement?
12. Is **decoder-only conditioning** biologically plausible (shared sleep states, subject-specific gain in decoder only)?

### F. Paper framing (5 min)

13. One sentence: why do **substages** matter for mouse sleep research vs macro Wake/NREM/REM only?
14. Preferred venue emphasis: **computational** (PLOS Comp Biol) vs **sleep biology** (Sleep / JSR)?

---

## Recording template (fill during meeting)

| Substage ID | Proposed name | Category (Awake/NREM/REM/Transition/Exclude) | Notes |
|-------------|---------------|---------------------------------------------|-------|
| 1 | | | |
| … | | | |

**Quotes for Discussion:**

- 
- 

**Excluded from main figure:**

- 

**Sign-off:** Agreed K for main figure = ___ ; max rows in Fig 3 = ___
