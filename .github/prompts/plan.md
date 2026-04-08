## Plan: Lab-Conditioned Decoder

Add laboratory conditioning to the decoder-only CVAE path while preserving existing training/data interfaces. The safest route is to reuse current subject ID flow and derive lab IDs internally from dataset metadata, so we avoid breaking tuple contracts used across trainer, validator, visuals, and non-CVAE models.

**Steps**
1. Phase 1 - Metadata and Mapping Foundation
2. In [src/data/data_loader_collection.py](src/data/data_loader_collection.py), add stable mappings built at init:
3. `lab_map`: reserve `0` for `unknown`, then map observed labs (e.g., `lab_1..lab_5`) to `1..N`.
4. `subject_to_lab_id`: map each encoded subject ID (the same IDs currently used by `sub_ids`) to its lab ID.
5. Add a helper method for CVAE use, e.g., `map_subject_ids_to_lab_ids(subject_ids: torch.Tensor) -> torch.Tensor`, preserving shape `(B,S)` or `(B,)` and device.
6. Keep `get_all_data()` and `__next__()` return signatures unchanged (strict compatibility requirement).

7. Phase 2 - Decoder Conditioning Extension
8. In [src/models/vae.py](src/models/vae.py), add optional decoder lab-conditioning params read from `model.params`:
9. `condition_on_lab_decoder` (default `False`), `lab_emb_dim` (default `0`), optional `n_labs` override (otherwise infer from data loader map).
10. Create `self.lab_emb = nn.Embedding(num_labs_with_unknown, lab_emb_dim)` only when enabled.
11. Update decoder input dimensionality so decoder MLP receives `latent_dim + subject_emb_dim (+ lab_emb_dim when enabled)`.
12. Keep encoder unchanged (decoder-only scope).
13. In `decode()`, derive `lab_ids` from incoming `subject_ids` via the new data loader helper, fetch lab embeddings, and concatenate to decoder input.
14. Ensure behavior is identical to current implementation when `condition_on_lab_decoder=false`.

15. Phase 3 - CVAE Wrapper and Call-Site Compatibility Check
16. In [src/models/cvae_mar_hmm.py](src/models/cvae_mar_hmm.py), keep signatures unchanged (`forward(x, subject_ids, epoch)` etc.); no extra args should be required because lab IDs are derived inside CVAE.
17. Confirm no trainer/validator API changes are needed in [src/training/trainer.py](src/training/trainer.py) and dependent consumers.

18. Phase 4 - Config Wiring for Experiment
19. In [src/config/run/cvaemarhmm/final/cvae_final_decoder_only.yaml](src/config/run/cvaemarhmm/final/cvae_final_decoder_only.yaml), add:
20. `condition_on_lab_decoder: true`
21. `lab_emb_dim: <small value, recommended 2 or 4>`
22. Keep `decoder_only_conditioning: true` and current subject embedding settings.
23. Leave all other configs unchanged initially; optional follow-up ablations can tune subject/lab embedding dimensions.

24. Phase 5 - Robustness and Backward Compatibility Guards
25. Ensure non-MSSV or missing-lab datasets map to `unknown` (ID `0`) without exceptions.
26. Add explicit validation/warnings in data loader if subject→lab mapping is missing for any subject ID encountered.
27. Confirm code path works when lab conditioning is disabled (default), preserving prior behavior for existing experiments.

**Relevant files**
- [src/data/data_loader_collection.py](src/data/data_loader_collection.py) - add lab mapping and subject-to-lab lookup helper without changing iterator/get_all_data outputs.
- [src/models/vae.py](src/models/vae.py) - add optional lab embedding and decoder concat logic.
- [src/models/cvae_mar_hmm.py](src/models/cvae_mar_hmm.py) - verify wrapper signatures remain unchanged and compatible.
- [src/config/run/cvaemarhmm/final/cvae_final_decoder_only.yaml](src/config/run/cvaemarhmm/final/cvae_final_decoder_only.yaml) - enable lab-conditioned decoder experiment.
- [hpc/submit/run_vae_decoder_only.sh](hpc/submit/run_vae_decoder_only.sh) - no code change expected; reuse same launch path after config update.

**Verification**
1. Static check: instantiate dataset + [src/data/data_loader_collection.py](src/data/data_loader_collection.py) and assert lab map contains `unknown` + observed labs, and subject-to-lab lookup covers all subjects in the run.
2. Model shape check: create a small batch and run `ConditionalVAE.decode()` with lab conditioning enabled; verify decoder MLP input size and output shape remain `(B,S,C,F)`.
3. Smoke training run: execute one short CVAE run with [src/config/run/cvaemarhmm/final/cvae_final_decoder_only.yaml](src/config/run/cvaemarhmm/final/cvae_final_decoder_only.yaml) and confirm no interface regressions in trainer/validator.
4. Regression check: run same path with `condition_on_lab_decoder:false`; ensure metrics/logs and behavior match pre-change baseline within expected stochastic variance.
5. Compatibility check: run a non-MSSV/synthetic config to confirm unknown-lab fallback does not break training.

**Decisions**
- Included: laboratory conditioning only, decoder-only injection, strict backward compatibility.
- Excluded: montage conditioning, per-channel montage IDs, changing dataloader tuple signatures, encoder lab conditioning.
- Architecture decision: derive lab from subject IDs inside CVAE to minimize churn across [src/training/trainer.py](src/training/trainer.py), [src/validation/validator.py](src/validation/validator.py), and [src/visuals/visualizer.py](src/visuals/visualizer.py).

**Further Considerations**
1. Embedding size recommendation: start with `lab_emb_dim=2` to avoid overpowering latent/subject channels; only increase if reconstruction error remains lab-biased.
2. Optional ablation matrix after base implementation: subject-only vs lab-only vs subject+lab in decoder under identical seeds to quantify nuisance absorption.
3. Optional regularization follow-up: add mild embedding norm penalty if lab embedding dominates reconstructions.