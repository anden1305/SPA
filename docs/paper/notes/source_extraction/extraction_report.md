# Source material extraction report

Generated: 2026-06-09T15:02:15.066330+00:00

## Honest answer: was the thesis PDF read for `plos_my_paper.tex`?

The first manuscript draft was built from **holdout CSV/JSON results**, 
`docs/paper/*.md`, cursor plans, and **cited thesis numbers** (HMM NMI ≈ 0.51, 
cGMVAE ≈ 0.70). It did **not** run multi-engine PDF extraction or a 
section-by-section thesis→paper mapping. Use this report to close that gap.

## Thesis PDF cross-check

**File:** `Master Thesis.pdf`

**Status:** PASS

| Engine | Pages | Chars | Words |
|--------|-------|-------|-------|
| pymupdf | 117 | 303901 | 45190 |
| pdfplumber | 117 | 308515 | 49005 |
| pypdf | 117 | 304162 | 45434 |

- Page counts agree: **True**
- Char spread: **1.5%**
- Similarity to `Master Thesis.txt`: **0.7548**

### Anchors missing from all engines
- decoder-only

### Anchors found by only some engines (OCR/layout drift)
- population NMI of about 0.70 (2 engines)
- conditional Gaussian mixture variational autoencoder (2 engines)

## Paper claims vs thesis text

- `thesis_hmm_nmi`: yes
- `thesis_cgmvae_nmi`: yes
- `decoder_only`: **NO**
- `zero_shot`: yes
- `mssv`: yes

## Special Course framing (docx)

**Stale warning:** Take with a grain of salt — early special-course framing; superseded by holdout ladder results and docs/paper/*.md.

- Paragraphs: 849, words: 9813
- Extract: `docs/paper/source_extraction/special_course_framing.txt`

### Preview (first ~500 chars)
```
Special Course - Project framing

Special Course - Project framing

Towards subject-invariant and temporally coherent end-to-end latent models for unsupervised mouse sleep staging

Main aim
Build an end-to-end version of your thesis pipeline that:

replaces FFT-based inputs with CNN feature learning

keeps conditioning only in the decoder

replaces the static GM prior with a temporally dependent latent prior

tests zero-shot transfer from decoder-only conditioning

ends in a 5–10 page paper draf
```

## Cursor plans loaded

- `.cursor/plans/5_page_paper_plan.md` (1719 words)
- `.cursor/plans/plos_manuscript_draft_f5977e3b.plan.md` (2129 words)

## Re-run

```bash
pip install -r scripts/paper/requirements-extract.txt
PYTHONPATH=. python3 scripts/paper/extract_source_materials.py
```
