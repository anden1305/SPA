# Decoder-only reliability configs

Thesis-style **lab_3** cohort (same mice as `reliability_cgmvae_decoder_only_subject_conditioning.yaml`).

| Config | Model | Prior | Epochs | Compare to table |
|--------|-------|-------|--------|------------------|
| `reliability_cgmvae_decoder_only_subject_conditioning.yaml` | cGMVAE | GMM | 80 | **cGMVAE Prior Decoder** |
| `reliability_chmmgmvae_decoder_only_lab3_thesis_style.yaml` | cHMMGMVAE | warm_hmm_gmm | 80 | New row (same protocol) |

Both: `sequence_length: 1`, `subject`, `decoder_only_conditioning: true`, `runs: 10`, train=val mouse list.

## Pretrained weights (required for thesis-style stability)

Reliability YAMLs load **`results/decoder_only/cvae_final_decoder_only/cvae_decoder_only_model.pth`** at the start of each `train_vae` run.

Install from your latest `cvae_baseline` training:

```bash
cd /work3/s204070/SPA
bash hpc/submit/decoder_only/setup_pretrained_from_cvae_baseline.sh
```

Source used today: `results/decoder_only/cvae_baseline_20260604-145955/1/checkpoints/cvae_final_model.pth`

Log should show `Loading CVAE model from checkpoint: ...` (not "checkpoint not found").

## Submit (manual)

```bash
cd /work3/s204070/SPA

# cGMVAE (existing)
bsub < hpc/submit/decoder_only/run_decoder_only_reliability.sh

# cHMMGMVAE lab_3 thesis-style
bsub < hpc/submit/decoder_only/run_decoder_only_reliability_chmmgmvae_lab3_thesis.sh
```

## Results

- cGMVAE: `results/decoder_only/reliability/cgmvae_decoder_only/reliability_cgmvae_decoder_only_mssv_frequency_<ts>/{1..10}/plots/metrics.txt`
- cHMM: `results/decoder_only/reliability/chmmgmvae_decoder_only_lab3/reliability_chmmgmvae_decoder_only_lab3_thesis_style_<ts>/{1..10}/plots/metrics.txt`

Report **mean ± SEM** of NMI in `metrics.txt` over 10 runs (Pred. NMI for thesis table).

## Monitor

```bash
bjobs -u $USER
tail -f hpc/output/decoder_only/reliability/chmmgmvae_lab3/<JOBID>.out
```

**Added:** 2026-06-04
