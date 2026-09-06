# SIH 26170 — Final Backend Release

This release is designed for a leakage-safe, auditable final benchmark. It does not hard-code or fabricate a desired recall, FPR, escape recall, MAE, or lead-time result.

## Runtime

Use Python 3.11 and the locked package versions enforced by `retrain_models.sh`.

## Retraining

```bash
./retrain_models.sh
```

The script trains the anomaly model, the 168 h forecast model, a **train-only OOD reference**, and a safety policy selected using validation data only.

## Final benchmark

Recommended first validation run:

```bash
PYTHONPATH=. python -m src.evaluation benchmark \
  data/processed/module_A_dataset.csv \
  --n-seeds 1 \
  --target-horizon 168 \
  --max-reject-rate 0.25
```

Final campaign:

```bash
PYTHONPATH=. python -m src.evaluation benchmark \
  data/processed/module_A_dataset.csv \
  --n-seeds 5 \
  --target-horizon 168 \
  --max-reject-rate 0.25
```

`--max-reject-rate` is an operational constraint, not a desired FPR. The benchmark reports the resulting FPR, escalation rate, automatic reject burden, confusion counts, latent-escape recall, mechanism results, progressive lead time, robustness, ablations and forecast baselines.

## Artifacts

The benchmark writes:

- `benchmark_results.json`
- `benchmark_table.md`
- `system_metrics_mean_std.csv`
- `latent_escape_mean_std.csv`
- `mechanism_metrics_mean.csv`
- `forecast_metrics_mean.csv`
- `progressive_metrics_mean.csv`
- `robustness_metrics_mean.csv`
- `ablation_metrics_mean.csv`
- per-seed `test_screening_final_policy.csv`
- per-seed `selection.json`

## Counterfactual

```bash
python scripts/demo_latent_escape.py \
  --input data/processed/module_A_dataset.csv \
  --part-id MMIC_ANALOG_008_00010 \
  --parameter quiescent_current
```

The demonstration is an empirical one-variable sensitivity analysis. It reports the first observed decision transition if one occurs, otherwise it reports the channel that remains dominant. It does not claim causal proof.

## Attribution

SHAP is supplementary, not a required runtime dependency:

```bash
python scripts/generate_attributions.py <model.joblib> <features.csv> <output.csv>
```

## Scientific release rules

1. Fit models on development/training data.
2. Select operating policy on validation only.
3. Build OOD reference from training/reference data only.
4. Never calibrate thresholds on final test data.
5. Never claim perfect recall without the associated FPR, burden and confusion counts.
6. Never label a conformal interval by its nominal coverage unless empirical coverage is reported.
7. Never use latent truth or future labels as inference features.
8. Treat unavailable temporal features as not-applicable, not fabricated zeros.
