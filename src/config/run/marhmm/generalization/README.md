# Generalization Experiments - Between-Mouse Testing

This directory contains configuration files for testing model generalization across different mice from the same laboratory.

## Purpose

Generalization experiments assess how well models trained on one set of mice can predict sleep stages for completely unseen mice:
1. **Test robustness**: Can the model generalize beyond the training subjects?
2. **Evaluate overfitting**: High train performance but poor generalization indicates overfitting
3. **Compare models**: Which architecture (HMM vs MAR-HMM) generalizes better?
4. **Assess data requirements**: How much training data from how many mice is needed?

## Experimental Design

### Methodology
- **Train**: All clean Lab 3 mice EXCEPT one held-out target mouse
- **Validate**: The held-out target mouse (all 3 runs)
- **Repeat**: For each target mouse (sub-039, sub-041, sub-048)

This leave-one-mouse-out cross-validation tests true generalization to novel subjects.

## Files

### MAR-HMM Configs (lr=0.0001, init=random_uniform)
- `generalization_marhmm_mssv_features_test_sub039.yaml` - Train on all except sub-039, test on sub-039
- `generalization_marhmm_mssv_features_test_sub041.yaml` - Train on all except sub-041, test on sub-041
- `generalization_marhmm_mssv_features_test_sub048.yaml` - Train on all except sub-048, test on sub-048

## Data Split Examples

### Example: Testing on sub-039

**Training mice (15 recordings):**
- sub-038: runs 1,2
- sub-041: runs 1,2,3
- sub-043: run 1
- sub-048: runs 1,2,3
- sub-054: runs 1,2
- sub-056: run 1
- sub-059: run 1
- sub-060: run 1
- sub-069: run 1

**Validation mouse (3 recordings):**
- sub-039: runs 1,2,3 ✓ (held-out, unseen during training)

## How to Run

```bash
# Test MAR-HMM generalization to sub-039
uv run main.py --method train --config_path src/config/run/marhmm/generalization/generalization_marhmm_mssv_features_test_sub039.yaml

# Test MAR-HMM generalization to sub-041
uv run main.py --method train --config_path src/config/run/marhmm/generalization/generalization_marhmm_mssv_features_test_sub041.yaml

# Run all MAR-HMM generalization tests
for mouse in sub039 sub041 sub048; do
    uv run main.py --method train --config_path src/config/run/marhmm/generalization/generalization_marhmm_mssv_features_test_${mouse}.yaml
done
```

## Key Metrics

### Primary Metrics

1. **Validation Predictive Log-Likelihood**
   - Higher is better (less negative)
   - Measures how well the model predicts held-out data
   - **Key comparison**: Compare val likelihood vs train likelihood
     - Similar values = good generalization
     - Much lower val = overfitting to training mice

2. **NMI (Normalized Mutual Information)**
   - Range: 0 (random) to 1 (perfect alignment)
   - Measures alignment with human-labeled sleep stages
   - **Key comparison**: Compare NMI on held-out mouse vs training mice NMI
     - Similar NMI = model generalizes well
     - Much lower NMI = model doesn't capture general sleep patterns

3. **Accuracy (via Hungarian Alignment)**
   - Percentage of correctly classified time windows
   - Complements NMI with absolute performance metric

### Secondary Metrics

4. **State Distinctness**
   - Measures separation between predicted states
   - Low distinctness on held-out data may indicate confusion

5. **Confusion Matrix**
   - Shows which states are confused on held-out data
   - Helps identify systematic errors (e.g., REM/Wake confusion)

## Results Interpretation

### Scenarios and Interpretations

| Train Loss | Val Loss | Train NMI | Val NMI | Interpretation |
|-----------|----------|-----------|---------|----------------|
| Low | Low | High | High | ✅ **Excellent**: Model generalizes well |
| Low | High | High | Low | ⚠️ **Overfitting**: Memorized training mice |
| High | High | Low | Low | ❌ **Underfitting**: Model too simple or bad hyperparams |
| Low | Medium | High | Medium | ✔️ **Good**: Some generalization, expected performance drop |

### What to Compare

1. **Across held-out mice**:
   - Is performance consistent across sub-039, sub-041, sub-048?
   - If one mouse performs much worse, investigate why (data quality, biological differences)

2. **MAR-HMM vs HMM**:
   - Compare generalization with HMM results (in `../hmm/generalization/`)
   - Which model achieves better generalization (higher val likelihood, higher val NMI)?
   - **These ARE comparable** - same data, same validation sets
   - MAR-HMM's autoregressive structure may capture temporal dynamics better

3. **Against reliability experiments**:
   - Compare generalization NMI vs reliability NMI (same-distribution testing)
   - Gap indicates how much performance degrades on new subjects

## Expected Outcomes

### Realistic Expectations
- **Train NMI**: 0.6-0.8 (good alignment with labels on training mice)
- **Val NMI**: 0.4-0.6 (expected drop due to individual differences)
- **NMI drop**: 10-30% is reasonable for between-subject generalization

### MAR-HMM Specific Considerations
- **Autoregressive advantage**: May better capture sleep stage transitions
- **Risk**: Could overfit temporal patterns specific to training mice
- **Expectation**: Should match or exceed HMM generalization

### Red Flags
- Val NMI < 0.3: Model fails to capture general sleep patterns
- Val NMI > Train NMI: Something is wrong (data leakage or very lucky)
- Huge variance across held-out mice: Model may be unstable

## Biological Considerations

### Why Generalization is Challenging
1. **Individual differences**: Each mouse has unique sleep patterns
2. **Circadian variations**: Time of day affects sleep architecture
3. **Environmental factors**: Subtle differences in recording conditions
4. **Labeller variability**: Different human annotators may have labeled different mice

### What Good Generalization Means
- Model learned general sleep stage features (theta rhythms, EMG activity, transitions)
- Not just memorizing individual mouse quirks
- MAR-HMM captures universal temporal structure of sleep (e.g., NREM→REM cycles)
- Potential for application to new subjects without retraining

## Comparison with Other Experiment Types

| Experiment | Train Set | Val Set | Purpose | Metric Comparison |
|-----------|-----------|---------|---------|-------------------|
| Hyperparameter Sweep | All mice | All mice | Find best params | Loss only (not comparable) |
| Reliability | All mice | All mice | Test consistency | Cross-NMI (comparable) |
| Substages | All except sub-039 | sub-039 | Find optimal states | NMI (comparable) |
| **Generalization** | **All except target** | **Target mouse** | **Test generalization** | **NMI, Likelihood (comparable)** |

## Next Steps

After running all generalization experiments:

1. **Aggregate results**: Average NMI and likelihood across all three held-out mice
2. **Create summary table**: Compare MAR-HMM vs HMM generalization performance
3. **Analyze failure cases**: Which states are most often misclassified on new mice?
4. **Examine transitions**: Does MAR-HMM better predict state transitions on new mice?
5. **Consider improvements**:
   - More diverse training data (more mice, more labs)
   - Data augmentation techniques
   - Fine-tuning on small amount of target mouse data
6. **Report findings**: Include generalization metrics in paper/thesis

## Related Experiments

- **Hyperparameter Sweeps** (Experiments 1-4): Found optimal lr and init_strategy
- **Reliability** (Experiment 5): Tests consistency on same-distribution data
- **Substages** (Experiment 6): Finds optimal number of states with held-out validation
- **HMM Generalization**: Compare with `../hmm/generalization/` for model comparison
