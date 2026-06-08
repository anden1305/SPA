# cGMVAE encoder+decoder conditioning ablation

**Added 2026-06-08.** Compare **full subject conditioning** (encoder + decoder) vs locked **decoder-only** incohort winners.

| Setting | `decoder_only_conditioning` | Subject emb in |
|---------|----------------------------|----------------|
| Locked baseline | `true` | decoder only |
| This ablation | `false` | encoder + decoder |

Same locked prepro/arch per lab as [locked_recipes.py](../../../scripts/cv4fold/locked_recipes.py). Scratch, `prior: gmm`, `runs: 3`, seq=1.

**Generate:**

```bash
PYTHONPATH=. python3 scripts/cv4fold/generate_ablation_enc_dec_conditioning.py
```

**Submit:**

```bash
bash hpc/submit/cv4fold/submit_ablation_enc_dec_conditioning.sh
```

Logs: `hpc/output/cv4fold/ablation_conditioning/{lab}_enc_dec_%J.out`  
Results: `results/cv4fold/ablation_conditioning/{lab}/abl_cgmvae_{lab}_enc_dec_locked_*`

Compare best-of-3 GMM prior NMI to decoder-only locked NMI (lab_2 0.593, lab_3 0.737, lab_5 0.534).

## Results (2026-06-08)

| Lab | Dec-only ref | enc+dec best [s1,s2,s3] | Verdict |
|-----|--------------|---------------------------|---------|
| lab_2 | **0.593** | 0.578 [0.578, 0.532, 0.513] | **Keep decoder-only** |
| lab_3 | **0.737** | 0.621 [0.621, —, —] | Incomplete; trending down |
| lab_5 | **0.534** | 0.430 [0.418, 0.430, —] | Incomplete; trending down |

Synthesis: [ablation_findings_20260608.md](ablation_findings_20260608.md).
