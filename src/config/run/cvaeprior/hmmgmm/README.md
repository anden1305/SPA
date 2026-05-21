# cHMMGMVAE experiments

Temporal HMM-GMM latent prior for the conditional VAE. The encoder, decoder, and
conditioning machinery in `ConditionalVAE` are unchanged; only the prior term in
the ELBO is replaced.

## Lab 3 reliability (vs decoder-only baseline)

| Config | Prior | Epochs | Compare to |
|--------|-------|--------|------------|
| `reliability/lab3_chmmgmvae_decoder_only_subject_conditioning.yaml` | `warm_hmm_gmm`, seq 64 | 120 | `decoder_only/reliability/reliability_cgmvae_decoder_only_subject_conditioning.yaml` |

Same lab-3 subjects/runs, `runs: 10`, same pretrained checkpoint. Baseline uses `prior: gmm`, `sequence_length: 1`, 80 epochs.

```bash
bash hpc/submit/hmmgmm/submit_reliability_lab3_compare.sh
# or: bsub < hpc/submit/hmmgmm/run_reliability_lab3_chmmgmvae.sh
```

## Fair comparison (recommended)

Paired POC configs on sub-039, `sequence_length: 64`, **same architecture and
beta schedule** as
`decoder_only/generalization/generalization_cgmvae_decoder_only_subject_conditioning_sub039.yaml`
(`max_beta: 1.0`, `no_beta_epochs: 10`, `prior: gmm` with `gmm_warmup_epochs: 0`).

| Config | `prior` | Role |
|--------|---------|------|
| `poc/sub039_cgmvae_gmm_seq64_baseline.yaml` | `gmm` | Static mixture; i.i.d. over time (flattened B×T KL) |
| `poc/sub039_chmmgmvae_poc.yaml` | `warm_hmm_gmm` | Same beta; warm schedule before sequence HMM KL |
| `poc/sub039_cgmvae_warm_gmm_seq64.yaml` | `warm_gmm` | Optional if plain `gmm` at seq64 is unstable |
| `ablations/sub039_chmmgmvae_poc_stable.yaml` | `warm_hmm_gmm` | Conservative schedule if the pair diverges |

```bash
# GMM vs cHMMGMVAE only
bash hpc/submit/hmmgmm/submit_poc_compare.sh

# Smoke + pair + stable ablation
bash hpc/submit/hmmgmm/submit_poc_verify.sh
```

## Other layout

- `poc/sub039_chmmgmvae_smoke.yaml` — 3-epoch smoke crossing HMM phase boundaries.
- `lolo_subject_lab/generalization_lab_holdout_lab{2,3,5}.yaml` — LOLO with
  `conditioning_source: subject_lab` (not part of the sub-039 fair pair).

## Run

```bash
python3 main.py --method train_vae \
  -c src/config/run/cvaeprior/hmmgmm/poc/sub039_chmmgmvae_smoke.yaml

bash hpc/submit/hmmgmm/submit_poc_compare.sh
bash hpc/submit/hmmgmm/submit_poc_verify.sh
bsub < hpc/submit/hmmgmm/run_poc_sub039_warm_gmm.sh   # optional
bsub < hpc/submit/hmmgmm/run_poc_sub039_stable.sh
```

**Do not** run `train_vae` or `pytest` on the login node. Use `bsub` or `linuxsh`.

## Tests (compute node only)

```bash
# From login node — runs on compute via LSF
bsub < hpc/submit/run_pytest.sh

# Or subset (set before bsub):
PYTEST_ARGS="tests/test_hmm_gmm_prior.py tests/test_vae_hmmgmm_integration.py -q" \
  bsub < hpc/submit/run_pytest.sh

# Interactive compute shell
linuxsh
cd /work3/s204070/SPA && source .venv/bin/activate
pytest tests/test_hmm_gmm_prior.py -q
```

## Prior parameters (`model.params`)

| Key | Meaning |
|-----|---------|
| `prior: gmm` | static mixture; per-step KL (B×T flattened at seq>1) |
| `prior: warm_gmm` | standard KL → KMeans → `gmm` |
| `prior: warm_hmm_gmm` | standard → KMeans → GMM-seq → HMM-GMM |
| `num_gmm_states` | K (default **3** with `remove_artifact: true`) |
| `gmm_warmup_epochs` | epochs of N(0,I) KL before KMeans (warm priors only) |
| `hmm_warmup_epochs` | epoch when HMM transitions turn on |
| `hmm_transition_ramp_epochs` | blend GMM-seq into HMM KL |
| `hmm_sticky_kappa` | sticky self-transition for init |
| `hmm_estimate_transitions` | estimate π, A from argmax paths at HMM phase |

`training_pipeline: cvae` keeps the MAR-HMM stage disabled; temporal structure
lives in the VAE prior only.
