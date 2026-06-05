# Decoder-only cGMVAE — fold 4 roadmap (KISS)

**Added:** 2026-06-04

## Story (3 lines)

1. **Thesis:** decoder-only cGMVAE on lab_3 + subject conditioning works well (NMI ~0.7+ in your reliability table).
2. **cv4fold:** harder — train on 17 mice / 3 labs, predict **fold-4 holdout** (labs 2, 3, 5).
3. **This roadmap:** three fold-4 jobs, same tune hyperparams, only **seq length** and **conditioning** change.

## Fold 4 holdout mice

From [`data/manifests/cv_quality_cohort_v1.yaml`](../../data/manifests/cv_quality_cohort_v1.yaml):

| Lab | Mice |
|-----|------|
| lab_2 | sub-080, sub-081 |
| lab_3 | sub-056, sub-059, sub-060 |
| lab_5 | sub-092 |

## Steps A / B / C

| Step | `sequence_length` | `conditioning_source` | Config phase | Results |
|------|-------------------|----------------------|--------------|---------|
| **A** | 1 | subject | `subject_tune_winners_seq1` | `results/cv4fold/subject_tune_winners_seq1/cgmvae/joint/fold_4/` |
| **B** | 64 | subject | `subject_tune_winners` | `results/cv4fold/subject_tune_winners/cgmvae/joint/fold_4/` |
| **C** | 64 | subject_lab | `subject_lab_tune_winners` | `results/cv4fold/subject_lab_tune_winners/cgmvae/joint/fold_4/` |

All use **cgmvae** + **`decoder_only_conditioning: true`** (thesis “cGMVAE Prior Decoder” style).

## Generate + submit

```bash
cd /work3/s204070/SPA
source .venv/bin/activate
export PYTHONPATH=.

bash hpc/submit/cv4fold/submit_decoder_only_fold4_cgmvae.sh
```

**Logs:** `hpc/output/cv4fold/vae/%J.out`  
**Walltime:** 4h per job

Monitor:

```bash
bjobs -u $USER
tail -f hpc/output/cv4fold/vae/<JOBID>.out
```

## After jobs finish

Postprocess + comparison table:

```bash
export PYTHONPATH=.
python3 scripts/cv4fold/summarize_decoder_only_fold4.py --postprocess
```

Writes:

- `results/cv4fold/decoder_only_fold4/comparison.csv`
- `results/cv4fold/decoder_only_fold4/comparison.md`

**Primary metric:** NMI in `plots/metrics.txt` (prior predictions) — not latent KMeans alone. See [cvae_wandb_checkpoint_metrics.md](../cvae_wandb_checkpoint_metrics.md).

## Comparison table

Auto-updated: `results/cv4fold/decoder_only_fold4/comparison.md`

| Step | Pooled val NMI | lab_2 | lab_3 | lab_5 | Notes |
|------|----------------|-------|-------|-------|-------|
| A seq1 subject | (pending) | | | | Job after submit |
| B seq64 subject | 0.31 | 0.47 | 0.08 | 0.42 | Prior run; refresh after job 28600451 |
| C seq64 subject_lab | ~0 | ~0 | ~0 | 0 | Collapse — re-tune before full grid |

**How to read:**

- **A vs B:** seq1 (thesis-like) vs seq64 on the same holdout.
- **B vs C:** subject vs subject+lab on seq64.
- If B is OK and C ≈ 0 → re-tune `subject_lab`, do not change preprocessing first.

## Done when

- [ ] All three jobs finished without error in LSF logs.
- [ ] `comparison.md` filled (best run per step).
- [ ] One sentence: which setting goes to full 4-fold / thesis cross-lab text.

## Lab_3 thesis reliability (cHMMGMVAE)

Same protocol as [`decoder_only/reliability`](../../src/config/run/cvaeprior/decoder_only/reliability/README.md):

```bash
bsub < hpc/submit/decoder_only/run_decoder_only_reliability_chmmgmvae_lab3_thesis.sh
```

Config: `reliability_chmmgmvae_decoder_only_lab3_thesis_style.yaml`

## Do not mix in

- `cgmvae_cvae_final` — encoder+decoder conditioning, not decoder-only.
- Preprocessing audit silhouette — not the same as holdout NMI.
