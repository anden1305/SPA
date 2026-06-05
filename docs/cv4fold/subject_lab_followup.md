# subject_lab batch — post-run analysis and follow-up

After `bash hpc/submit/cv4fold/submit_subject_lab_tune_winners.sh` finishes (or partially), use this workflow.

## Automated analysis

```bash
cd /work3/s204070/SPA
source .venv/bin/activate
export PYTHONPATH=.

bash hpc/submit/cv4fold/run_after_subject_lab_batch.sh
```

Writes:

- `results/cv4fold/subject_lab_tune_winners/summary.csv`
- `results/cv4fold/subject_lab_tune_winners/analysis_report.md`

## Optional: submit follow-up (within reason)

If completed **cgmvae** folds show mean best prior NMI &lt; 0.05, the script recommends a **subject-only** ablation with the **same tune hyperparams** (how winners were tuned), on **fold 4 only** (2 GPU jobs):

```bash
bash hpc/submit/cv4fold/run_after_subject_lab_batch.sh --submit-followup
```

Or manually (fold 4 A/B, **4h** walltime):

```bash
PYTHONPATH=. python3 scripts/cv4fold/generate_configs.py --phase subject_tune_winners --folds 4
bash hpc/submit/cv4fold/submit_subject_tune_winners_fold4.sh

PYTHONPATH=. python3 scripts/cv4fold/generate_configs.py --phase subject_lab_tune_winners --folds 4
bash hpc/submit/cv4fold/submit_subject_lab_tune_winners_fold4.sh
```

Results: `results/cv4fold/subject_tune_winners/{model}/joint/fold_4/`

## Canceling remaining subject_lab jobs

If early folds show **NMI ≈ 0**, remaining PEND/RUN jobs with the same YAML are unlikely to help:

```bash
bjobs -u $USER
bkill <JOBID> ...   # subject_lab cgmvae fold 3/4 and/or all chmm if desired
```

Only keep chmm PEND if you still want subject_lab on HMM despite cgmvae failure.

## Agent / Cursor

The agent **cannot** watch the queue overnight. When jobs finish, send a message like “subject_lab batch done — analyze and follow up” or run the script above yourself.

**Added:** 2026-06-03
