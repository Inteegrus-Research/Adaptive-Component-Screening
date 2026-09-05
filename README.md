# Phase 4 — Safety Intelligence

Phase 4 turns model outputs into a conservative, auditable component-screening decision. The implementation keeps **model evidence separate from decision policy**.

## Safety architecture

```text
Module A evidence ─┐
Module B evidence ─┤
OOD evidence ──────┤→ Safety policy → SAFE / REVIEW / REJECT / UNKNOWN
Data quality ──────┤
Engineering limits ┘
                         ↓
                    Explanation
```

### Design rules

- An authoritative absolute-limit violation is a non-overridable `REJECT`.
- OOD means domain novelty/uncertainty, not defect. Severe OOD routes to `REVIEW`.
- Unsupported parameter semantics route to `UNKNOWN`, never to a guessed model decision.
- High uncertainty routes to `REVIEW` when evidence exists but the prediction is not reliable enough.
- Insufficient evidence routes to `UNKNOWN`.
- The policy penalizes false negatives more strongly than false positives while explicitly forbidding the trivial “reject everything” solution.
- Raw anomaly evidence remains auditable; calibrated risk is not allowed to overwrite the underlying evidence.
- Every output retains a machine-readable decision trace and a human-readable explanation.

## OOD profile

Build the OOD reference artifact from a representative training/reference domain:

```bash
python -m src.safety fit-ood data/processed/module_A_dataset.csv models/calibration/ood_profile.joblib
```

For very large references, the profile uses a deterministic bounded sample for numerical OOD calibration; categorical domain vocabularies are preserved.

## Screening

```bash
python -m src.safety screen ANOMALY.csv INPUT.csv OUTPUT.csv \
  --forecast FORECAST.csv \
  --ood-artifact models/calibration/ood_profile.joblib
```

Then render QA explanations:

```bash
python -m src.explain OUTPUT.csv reports/explanations.csv
```

## API

The core interface is:

```python
from src.safety import assess_screening
from src.explain import explain_frame

screening = assess_screening(anomaly_df, forecast_df, input_df, ood_profile)
report = explain_frame(screening)
```

No application/UI logic belongs in this module.
