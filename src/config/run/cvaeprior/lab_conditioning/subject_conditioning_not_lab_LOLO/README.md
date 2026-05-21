# Subject conditioning, leave-one-lab-out (LOLO)

Same lab holdout splits as `lab_conditioning_not_subject_LOLO/` and
`lab_and_subject_conditioning_LOLO/`, but:

- **`conditioning_source: subject`** (not lab)
- **`decoder_only_conditioning: true`** (subject embedding in the decoder only)

## Three-way LOLO comparison

| Folder | Holdout | Conditioning |
|--------|---------|--------------|
| `lab_conditioning_not_subject_LOLO/` | lab 2, 3, or 5 | `lab` |
| `subject_conditioning_not_lab_LOLO/` | lab 2, 3, or 5 | `subject` |
| `lab_and_subject_conditioning_LOLO/` | lab 2, 3, or 5 | `subject_lab` |

Configs: `generalization_lab_holdout_lab{2,3,5}.yaml`

Cohort IDs and `signals` per lab are the same as in
`lab_conditioning_not_subject_LOLO/` (aligned with
`results/decoder_only/reliability/` mice).

Results: `results/lab_conditioning/subject_conditioning_not_lab_LOLO/holdout_lab{N}/`

## Submit

```bash
bash hpc/submit/lab_conditioning/run_lab_conditioning_subject_only_LOLO.sh
```
