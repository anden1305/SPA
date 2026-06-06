# CVAE checkpointing (simple)

Added 2026-06-05. Branch: `vae_decoder_chmm` (decoder-only + cHMM reliability).

## One knob

In your YAML:

```yaml
cvae:
  model_checkpoint_path: results/decoder_only/cvae_final_decoder_only/cvae_decoder_only_model.pth
  # or null to train from scratch
```

That is the **only** path the orchestrator uses to load/save pretrained CVAE weights.

**Both load and save** when the path is set: load at the **start** of each run (if the file exists), overwrite at the **end** of each run. It is not “load only”.

**Legacy subject embedding:** checkpoints must have `subject_emb.weight` shape **`[92, emb_dim]`** (sub-NNN → row N). Incompatible shapes (e.g. compact `[10, 4]` from another branch) are **skipped** with a log line; training continues from scratch and overwrites the path at run end.

## Should you train from scratch?

| Stage | YAML | `model_checkpoint_path` | What you want |
|-------|------|-------------------------|---------------|
| **Baseline** (once) | `cvae_final_decoder_only.yaml` | `null` | Train **from scratch** on lab_3 mice |
| **Baseline** (encoder+decoder) | `cvae_final.yaml` | points at `cvae_baseline_model.pth` | First run: scratch (file missing). Later runs: finetune from last baseline |
| **Reliability / cHMM** | `reliability_*_mssv_frequency.yaml` | points at baseline `.pth` | **Finetune** from baseline — this is what gave ~0.72 NMI in April |

You do **not** always train from scratch. For reliability you almost always want **finetune from a baseline** trained on the same branch (`vae_decoder`, 92-row `subject_emb`). Training reliability from scratch (`null`) worked but was ~0.68–0.70 vs ~0.72 with the checkpoint.

**Rule of thumb:** `null` = scratch. Path to an existing baseline `.pth` = finetune (hotstart).

**Hotstart vs scratch:**

| Question | Config | Checkpoint |
|----------|--------|------------|
| cHMMGMVAE **beats** cGMVAE Prior (fair) | `reliability_chmmgmvae_decoder_only_mssv_frequency.yaml` + `reliability_cgmvae_decoder_only_mssv_frequency.yaml` | Same `cvae_decoder_only_model.pth`, `runs: 10` |
| cHMMGMVAE **can learn** lab_3 end-to-end | `reliability_chmmgmvae_decoder_only_mssv_frequency_scratch.yaml` | `null`, `runs: 10` |

Hotstart matches thesis ~0.72; scratch ablation often ~0.68–0.70 for cGMVAE GMM.

Lab_3 experiment matrix (submit scripts, mermaid): [decoder_only_lab3_chmm_experiments.md](decoder_only_lab3_chmm_experiments.md).

## What happens on `train_vae`

1. **Start of each seed/run** — if `model_checkpoint_path` is set and the file exists, load **CVAE weights only** (`cvae.*` keys, `strict=False`). Log line: `Loading CVAE model from checkpoint: ...`
2. **Train** — finetune from those weights (or from scratch if path is `null` or missing). During training, when `validate_per_epoch > 0`:
   - **`{run_name}/{N}/checkpoints/cvae_best_prior_pred_nmi.pth`** — updated when **prior-prediction** NMI improves (GMM marginal or HMM Viterbi; not latent KMeans).
   - **`cvae_best_checkpoint_score.pth`** — when `trainer.checkpoint_score.enabled: true` and thesis score `S(e)` improves (after warmup).
3. **End of each run** — save final weights:
   - `{results_dir}/{run_name}/{N}/checkpoints/cvae_final_model.pth`
   - legacy `{results_dir}/{run_name}/cvae_final_model_run{N}.pth`
4. **End of each run** — if `save_pretrained_checkpoint: true` (default) and `model_checkpoint_path` is set, overwrite that path with **final** epoch weights. Set `save_pretrained_checkpoint: false` on multi-seed reliability so the shared baseline path is not overwritten each seed.
5. **Post-train validation** — reload **best** checkpoint (score → prior-pred NMI → final), write `validation_checkpoint.txt`, run prior validation → plots + `validation_info.json` under `{run_name}/{N}/plots/`.

Latent KMeans NMI is still logged each validation epoch (`cvae_latent_kmeans_nmi`) but is **not** used for checkpoint selection — it can disagree with prior-pred NMI (e.g. collapse with high KMeans).

### Optional: checkpoint score (cv4fold-style)

```yaml
trainer:
  checkpoint_score:
    enabled: true
    beta: 0.9
    warmup_frac: 0.05
```

When enabled, post-train validation prefers `cvae_best_checkpoint_score.pth` over prior-pred-best.

HMM-GMM prior parameters are **not** loaded from an old checkpoint in a special way; they warm up during training (KMeans → transitions).

## Canonical baseline checkpoints (decoder-only lab_3)

| Variant | Set in YAML | Written when |
|---------|-------------|--------------|
| Decoder-only baseline | `cvae_final_decoder_only.yaml` → path is `null` today | Only `cvae_final_model_run1.pth` under timestamped run dir unless you set `model_checkpoint_path` |
| Encoder+decoder baseline | `cvae_final.yaml` → `results/decoder_only/cvae_baseline/cvae_baseline_model.pth` | Overwritten each time that config trains |

Reliability / cHMM YAMLs point at the fixed paths above so finetuning starts from a pretrained baseline.

## How to check what a run actually used

See [decoder_only_checkpoint_tracing.md](decoder_only_checkpoint_tracing.md) (WandB, HPC log, `config.json`, `validation_info.json`, `.pth` shape check).

## Code

- Load/save: `src/orchestrator/orchestrator.py` → `_load_cvae_checkpoint_if_available()`, `_save_cvae_checkpoint_to_config_path()`
- Validation audit: `_write_validation_info()` → `validation_info.json`
