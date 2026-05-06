## Plan: Lab-Conditioned Decoder-Only Ablations

Scope is strictly the decoder-side lab-conditioning experiments for `src/config/run/cvaemarhmm/decoder_only`. Start with Lab 5 (E) as the target conditioning label, keep the montage fixed to the same 1 parietal + 1 frontal electrode pair, and compare against Lab 3 under an otherwise identical protocol.

**Steps**
1. Phase 1 - Freeze protocol and lab comparison anchor (blocks all later steps)
1.1 Lock one Lab 3 decoder-only config as the protocol anchor (same seeds policy, optimizer, scheduler, epochs, early stopping, batching, downstream MAR/HMM settings, validator toggles).
1.2 Define immutable evaluation outputs for direct Lab 3 vs Lab 5 comparison: global metrics, per-lab metrics, cross-lab stability.
1.3 Freeze train/val subject partitions, LOSO fold definitions, and lab membership mapping in one manifest file.
1.4 Fix the montage and preprocessing stack so Lab 3 and Lab 5 use the same 1 parietal + 1 frontal electrode inputs and differ only in the decoder lab signal.
2. Phase 2 - Implement decoder-only lab conditioning (depends on 1)
2.1 Add explicit conditioning-mode switch in model config: encoder+decoder (reference), decoder-only lab conditioning (target), none, with the encoder remaining unconditioned in decoder-only mode.
2.2 Refactor the CVAE path so the decoder receives lab embeddings while the encoder runs without lab embeddings in decoder-only mode.
2.3 Add shape/assert checks for lab_ids flow through train, infer, and predict to avoid silent mismatch between Lab 3 and Lab 5 runs.
2.4 Keep downstream temporal model unchanged; verify no changes to transition/prior/hyperparameters in this project scope.
3. Phase 3 - Add lab-conditioning ablation knobs (depends on 1; parallel with 2.3)
3.1 Build the minimal comparison matrix first: Lab 3 reference, Lab 5 target, and unconditioned sanity floor.
3.2 Add a toggle for decoder lab-conditioning strength or embedding injection point only if the first comparison shows it is needed; otherwise keep the experiment space minimal.
3.3 Keep representation, optimizer, scheduler, and temporal-model settings fixed so any gain is attributable to lab conditioning alone.
3.4 If later expansion is needed, add only lab-specific normalization or representation variants after the Lab 3 vs Lab 5 baseline comparison is locked.
4. Phase 4 - Evaluation instrumentation for the lab comparison (depends on 2 and 3)
4.1 Add per-lab reporting for NMI/accuracy/state distinctness and aggregate with mean±std across subjects and seeds.
4.2 Add cross-lab latent separability/stability analysis to compare Lab 3 and Lab 5 representations under the same montage.
4.3 Add zero-shot evaluation path for held-out subjects if the lab-conditioning claim needs it, but keep the first pass focused on the direct Lab 3 vs Lab 5 comparison.
5. Phase 5 - Controlled experiment matrix (depends on 4)
5.1 Minimal matrix (must-have):
- Lab 3 decoder-conditioned reference
- Lab 5 decoder-conditioned target
- Unconditioned model sanity floor
- Optional: Lab 5 with any fixed montage-preserving preprocessing variant, only if needed to isolate lab conditioning
5.2 Run 3 seeds for the pilot comparison; expand only the winner to 5 seeds if the effect is stable.
5.3 Keep the run budget strict: gate progression by predefined success criteria.
6. Phase 6 - Analysis and paper-ready outputs (depends on 5)
6.1 Produce one final result table aligned to thesis metrics with explicit Lab 3 vs Lab 5 columns.
6.2 Produce core figures: latent separability/stability, cross-lab transfer, failure-case panel.
6.3 Write a short draft focused on the lab-conditioning contribution, with the montage constraint called out explicitly.

**Execution timeline (2 days/week, now to hand-in)**
1. Week 1: Freeze protocol, lab mapping, split manifest, experiment matrix skeleton, success criteria.
2. Week 2: Implement decoder-only lab-conditioning path and regression checks.
3. Week 3: Add per-lab reporting and cross-lab stability analysis.
4. Week 4: Run the pilot Lab 3 vs Lab 5 comparison, then lock the final experiment list.
5. Week 5-6: Main runs for finalists, collect final tables/figures.
6. Week 7: Draft method/results/discussion around lab conditioning and montage control.

**Relevant files**
- /work3/s204070/SPA/src/models/vae.py — encoder/decoder conditioning injection points.
- /work3/s204070/SPA/src/models/cvae_mar_hmm.py — forward/predict path carrying lab_ids or equivalent conditioning into VAE + temporal model.
- /work3/s204070/SPA/src/config/config.py — add conditioning-mode enum/fields and ablation config schema.
- /work3/s204070/SPA/src/data/data_loader.py — montage-preserving preprocessing and any lab-conditioned normalization hooks.
- /work3/s204070/SPA/src/data/data_loader_collection.py — lab/subject map integrity and split consistency across datasets.
- /work3/s204070/SPA/src/validation/validator.py — per-lab/cross-lab metric reporting and stability analyses.
- /work3/s204070/SPA/src/orchestrator/orchestrator.py — model/dataset assembly and ablation config dispatch.
- /work3/s204070/SPA/src/config/run/cvaemarhmm/decoder_only/** — immutable baseline and lab-conditioned YAMLs for reproducible matrix execution.

**Verification**
1. Configuration integrity: load all new YAMLs through Pydantic validation without silently changing baseline behavior.
2. Conditioning-path correctness: unit/integration checks that the encoder receives no lab embedding in decoder-only mode while the decoder does.
3. Protocol lock: automated diff check that optimizer/scheduler/temporal-model params match baseline across lab ablations.
4. Montage integrity: assert the Lab 3 and Lab 5 configs use the same 1 parietal + 1 frontal electrode montage.
5. Metric parity: confirm output metrics include the thesis-comparable set and the new per-lab analyses.
6. Reproducibility: rerun one selected configuration with same seed and confirm metric tolerance window.

**Decisions**
- Included: decoder-only lab conditioning, Lab 3 vs Lab 5 comparison, montage-fixed controlled evaluations.
- Excluded: end-to-end CNN backbone replacement and temporal latent-prior redesign.
- Baseline comparability is a hard requirement: any claimed gain must come only from lab conditioning choices.

**Further Considerations**
1. If compute is constrained, prioritize: Lab 3 vs Lab 5 decoder-only comparison first, then the unconditioned baseline.
2. If the effect is ambiguous, keep the montage fixed and add only one additional lab-conditioned variant at a time.
3. If the lab labels are not exposed cleanly in the current data path, add a dedicated lab_id plumbing layer before expanding the matrix.