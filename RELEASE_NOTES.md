# SIH 26170 Backend Release Notes

## Implemented hardening

- Validation-selected operating policy is the same policy family reapplied to blind test decisions.
- The benchmark separates escalation (`REVIEW`/`REJECT`) from automatic rejection (`REJECT`).
- Final system metrics report TP/TN/FP/FN, recall, FNR, FPR, specificity, NPV, precision, SAFE/REVIEW/REJECT/UNKNOWN rates, escalation burden, false-review and false-reject rates, and cost-weighted loss.
- Latent escape is evaluated separately from overall future-defect detection.
- Mechanism metrics include latent change-point, latent accelerating, latent abrupt, hard early failure, and normal aging.
- High-but-safe and measurement-only anomaly cases are reported as hard negatives.
- OOD calibration uses a training-only reference in retraining and benchmark runs.
- OOD schemas do not require future labels or future readpoints as deployment inputs.
- Sparse temporal features are treated as unavailable until their minimum history exists; the pipeline can expose a feature availability manifest.
- Forecast training uses observed `target_<h>h` only and learns a relative target against `value_0h`, then rescales to physical units.
- Forecast benchmark includes persistence, linear, ridge, gradient boosting, and selected model comparisons where available.
- Conformal coverage is measured empirically and reported with its nominal target.
- Progressive screening evaluates 12/24/48/72/96/120/144/168 h origins.
- Counterfactual demonstration is a one-variable empirical sensitivity study and reports the first actual decision transition, rather than asserting causal proof.
- SHAP attribution is supplementary and optional; engineering evidence remains the primary explanation layer.
- Artifact manifests and hashes remain part of the release workflow.
- A final backend integrity gate was added; it never asserts that a performance target has been achieved.

## What this release does not do

It does not fabricate a target recall, escape recall, FPR, MAE, lead time, or conformal coverage. Those are empirical properties of the dataset and frozen artifacts and must be measured in the user's local environment.
