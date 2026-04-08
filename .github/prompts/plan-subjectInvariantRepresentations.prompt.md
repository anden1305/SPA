## Plan: Subject-Invariant Representation Ablations

Scope is strictly Section 5.9.1: isolate subject-invariant representation effects by moving conditioning to decoder-only and running controlled normalization/representation ablations while keeping optimizer, training protocol, and downstream temporal model fixed.

**Steps**
1. Phase 1 — Freeze protocol and fixed components (blocks all later steps)
1.1 Lock one baseline config as the protocol anchor (same seeds policy, optimizer, scheduler, epochs, early stopping, batching, MAR/HMM downstream settings, validator toggles).
1.2 Define immutable evaluation outputs to match thesis Table 11 / Section 5.4.2 style: global metrics, per-subject metrics, cross-subject stability.
1.3 Freeze train/val subject partitions and LOSO fold definitions in one manifest file to prevent split drift.
2. Phase 2 — Implement decoder-only conditioning (depends on 1)
2.1 Add explicit conditioning-mode switch in model config: encoder+decoder (reference), encoder-only (legacy), decoder-only (target), none.
2.2 Refactor VAE/CVAE path so encoder can run without subject embeddings while decoder receives subject embeddings in decoder-only mode.
2.3 Add shape/assert checks for subject_ids flow through train, infer, and predict to avoid silent misalignment.
2.4 Keep downstream temporal model unchanged; verify no changes to transition/prior/hyperparameters in this project scope.
3. Phase 3 — Add subject-invariance ablation knobs (depends on 1; parallel with 2.3)
3.1 Normalization ablations: per-loader normalization, global-train normalization, and no normalization under fixed protocol.
3.2 Representation ablations within current feature family: spectral-only reference vs hybrid feature set variants (without introducing new temporal prior/CNN backbone in this scope).
3.3 Feature-selection ablation criteria: add stability-driven selection objective across LOSO folds (same model/training settings).
4. Phase 4 — Evaluation instrumentation for Section 5.9.1 claims (depends on 2 and 3)
4.1 Add per-subject reporting for NMI/accuracy/state distinctness and aggregate with mean±std across subjects and seeds.
4.2 Add cross-subject latent separability/stability analysis matching Section 5.4.2 logic (including cross-NMI and subject-transfer diagnostics).
4.3 Add zero-shot evaluation path for held-out subjects without any held-out-subject representation training usage.
5. Phase 5 — Controlled experiment matrix (depends on 4)
5.1 Minimal matrix (must-have):
- Reference conditioned model (current best protocol)
- Decoder-only conditioning (target)
- Unconditioned model (sanity floor)
- Decoder-only + each normalization setting
- Decoder-only + representation/hybrid variants
5.2 Run 3 seeds for pilot; expand to 5 seeds only for finalists.
5.3 Keep run budget strict: gate progression by predefined success criteria (below).
6. Phase 6 — Analysis and paper-ready outputs (depends on 5)
6.1 Produce one final result table aligned to thesis metrics (direct comparability).
6.2 Produce three core figures: latent separability/stability, cross-subject transfer, failure-case panel.
6.3 Write 5–10 page draft focused only on Section 5.9.1 contribution and limitations.

**Execution timeline (2 days/week, now to hand-in)**
1. Week 1: Freeze protocol, split manifest, experiment matrix skeleton, success criteria.
2. Week 2: Implement conditioning-mode switch + decoder-only path and regression checks.
3. Week 3: Implement normalization ablations and global-train stats handling.
4. Week 4: Implement representation/hybrid and stability-driven feature-selection ablations.
5. Week 5: Add per-subject/cross-subject instrumentation and zero-shot evaluation scripts.
6. Week 6: Pilot runs (3 seeds), prune matrix, lock final experiment list.
7. Week 7-8: Main runs (3-5 seeds finalists), collect final tables/figures.
8. Week 9: Draft intro/method/experimental setup + ablation protocol section.
9. Week 10: Draft results/discussion centered on invariance and transfer claims.
10. Week 11: Final polish, supervisor feedback integration, oral prep slides.

**Relevant files**
- /work3/s204070/SPA/src/models/vae.py — encoder/decoder subject embedding flow and conditioning injection points.
- /work3/s204070/SPA/src/models/cvae_mar_hmm.py — forward/predict path carrying subject_ids into VAE + temporal model.
- /work3/s204070/SPA/src/config/config.py — add conditioning-mode enum/fields and ablation config schema.
- /work3/s204070/SPA/src/data/data_loader.py — normalization switch behavior and any subject-invariant normalization hooks.
- /work3/s204070/SPA/src/data/data_loader_collection.py — subject_map integrity and split consistency across datasets.
- /work3/s204070/SPA/src/validation/validator.py — per-subject/cross-subject metric reporting and Section 5.4.2-style analyses.
- /work3/s204070/SPA/src/orchestrator/orchestrator.py — model/dataset assembly and ablation config dispatch.
- /work3/s204070/SPA/src/config/run/** — immutable baseline and ablation YAMLs for reproducible matrix execution.

**Verification**
1. Configuration integrity: load all new YAMLs through Pydantic validation without defaults silently changing baseline behavior.
2. Conditioning-path correctness: unit/integration checks that encoder receives no subject embedding in decoder-only mode while decoder does.
3. Protocol lock: automated diff check that optimizer/scheduler/temporal-model params match baseline across ablations.
4. Split integrity: assert no held-out subject leakage into representation training for zero-shot runs.
5. Metric parity: confirm output metrics include thesis-comparable Table 11 set and Section 5.4.2 analyses.
6. Reproducibility: rerun one selected configuration with same seed and confirm metric tolerance window.

**Decisions**
- Included: decoder-only conditioning, subject-invariant normalization and representation/feature-selection ablations, controlled evaluations under fixed temporal model/protocol.
- Excluded: end-to-end CNN backbone replacement and temporal latent-prior redesign (out of Section 5.9.1 scope).
- Baseline comparability is a hard requirement: all claimed gains must come only from representation/conditioning choices.

**Further Considerations**
1. If compute is constrained, prioritize: decoder-only vs reference + normalization ablations first; run representation variants only if pilot variance permits.
2. If LOSO cost is too high, run full LOSO on finalists and reduced-fold proxy during development, then disclose this clearly in methods.
3. Thesis PDF context: current tooling could not read the repository PDF due size sync limits; if needed, provide key pages as text excerpts to tighten metric/section wording alignment.