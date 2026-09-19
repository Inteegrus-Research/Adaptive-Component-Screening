
# ACS — Adaptive Component Screening
**Arch Rival**
**Smart India Hackathon 2026 — PS 2026SIH170**

**ACS is an auditable, physics-grounded intelligence pipeline for aerospace component burn-in and screening. It reduces required chamber time by a theoretical 85.7% while maintaining a deterministic, zero-escape safety boundary.**

> **IMPORTANT FOR EVALUATORS:** 
> This README is a high-level architectural overview. For the complete mathematical formulation, conformal uncertainty proofs, and rigorous benchmark metrics, please read **`Project Documentation.pdf`** included in this repository.

---

## 1. The Core Philosophy: Decoupled Authority

Conventional AI systems act as black boxes. In aerospace, this is unacceptable. ACS solves this by enforcing strict operational decoupling:
* **AI generates statistical evidence and future forecasts.**
* **Engineering policy evaluates that evidence.**
* **Authoritative physical limits remain non-overridable.**

If a component's telemetry violates a known physical limit, the system executes a **Hard Override to REJECT**, regardless of the AI's confidence. 

---

## 2. System Architecture & The 9-Stage Trace

To guarantee explainability, ACS processes telemetry through an immutable 9-stage pipeline. Every decision leaves a deterministic JSON trace for QA inspectors.

```text
[01] RAW TELEMETRY       (12h, 24h, 48h... measurements)
          │
[02] VALIDATION          (Schema, unit, and missingness audit)
          │
[03] FEATURE ENGINE      (Lot-relative robust statistics & temporal dynamics)
          │
          ├──────────────────────────────┐
          ▼                              ▼
[04] MODULE A                    [05] MODULE B
     Dynamic Anomaly                  168h Drift Forecast
     (Isolation Forest, MAD)          (Quantile Loss Gradient Boosting)
          │                              │
          └──────────────┬───────────────┘
          ▼
[06] OOD TRUST GATE      (Mahalanobis distance: novel domain ≠ defect)
          │
[07] SAFETY POLICY       (Validation-calibrated failure risk assessment)
          │
[08] HARD OVERRIDE       (Absolute engineering-limit enforcement)
          │
          ▼
[09] FINAL DECISION      (SAFE / REVIEW / REJECT / UNKNOWN)

```

---

## 3. Key Technical Innovations

* **Module A (Lot-Relative Anomaly):** Detects sub-threshold latent defects by comparing components against their specific manufacturing lot using Robust Part Average Testing (MAD) and Isolation Forests.
* **Module B (Trajectory Forecasting):** Projects early (24h) telemetry to the 168h boundary using `HistGradientBoostingRegressor`.
* **Conformal Uncertainty:** Wraps forecasts in empirical 95% confidence intervals to measure limit-crossing risk.
* **Out-of-Distribution (OOD) Gating:** Measures epistemic uncertainty. If a component is operating in an unknown regime, it is flagged as `UNKNOWN` or `REVIEW`—it is never hallucinated as `SAFE`.
* **Edge-Native & Air-Gapped:** Operates entirely on CPU without external cloud APIs, ensuring ITAR/ISRO data compliance.

---

## 4. Technology Stack

**Backend & ML Engine (Air-Gapped / CPU-Optimized):**

* **Python 3.11+**
* **Scikit-Learn:** Core algorithmic engine (Isolation Forests, HistGradientBoosting).
* **NumPy / SciPy:** Robust statistics, Mahalanobis covariance, and signal processing.
* **FastAPI:** High-performance, asynchronous REST API serving JSON decision traces.

**Frontend (Mission Intelligence Center):**

* **Vanilla JS / React / HTML5:** High-fidelity, dependency-light visual client.
* **Chart.js:** Telemetry rendering, conformal bounding boxes, and scatter plot anomaly boundaries.

---

## 5. One-Command Reproduction

The system includes a physics-grounded synthetic telemetry generator to evaluate pipeline integrity without violating flight-data restrictions.

To execute the full benchmark, generate the data, and run the safety gate:

```bash
# 1. Setup Environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Run the full evaluation suite
python scripts/run_final_submission.py

```

The actual exact terminal output.
```bash
 Fri 18 Sep - 21:14  ~/Projects/Sih/project   main 17☀ 19● 1‒ 
 @keerthi_kumar  python scripts/run_final_submission.py

$ /home/keerthi_kumar/Projects/Sih/project/venv/bin/python /home/keerthi_kumar/Projects/Sih/project/data/reference/sih26170_generator_gen2_final.py --profile final --parts 400 --lots 80 --seed 26170
{
  "seed": 26170,
  "parts": 400,
  "lots": 80,
  "families": {
    "CAPACITOR": 67,
    "DIGITAL_LOGIC": 67,
    "MMIC_ANALOG": 67,
    "POWER_REG": 67,
    "POWER_SEMICONDUCTOR": 66,
    "SENSOR_IF": 66
  },
  "defect_base_rate": 0.25,
  "negative_control": "NORMAL_AGING",
  "controlled_fixture": true,
  "operational_base_rate_claim": false,
  "split_strategy": "lot_grouped",
  "parts_per_lot": 5,
  "correlation_target_quiescent_input_leakage": 0.4,
  "missingness_random_rate": 0.05,
  "missingness_block_part_rate": 0.1,
  "accelerating_coefficient": 0.002,
  "change_point_onset_h": 12.0,
  "change_point_denominator": 250,
  "description": "Physics-grounded controlled burn-in fixture; not realistic field base rate."
}

$ /home/keerthi_kumar/Projects/Sih/project/venv/bin/python scripts/run_benchmark.py --input /home/keerthi_kumar/Projects/Sih/project/data/processed/module_A_dataset.csv --output-dir reports/final_submission --config /home/keerthi_kumar/Projects/Sih/project/configs/benchmark.yaml --seeds 20260831 20260832 20260833 --origins 12 24 48 72 96 120 144 168 --bootstrap 1000
configuration_sha256: 49cb487e62166a1dc858a33d05c9a3a632bf1b36af1bffe40a6a043249b22448
=== INGESTION AUDIT ===
{
  "input_rows": 9765,
  "canonicalized_rows": 9765,
  "ingestion_mode": "LONG",
  "unknown_parameters_quarantined": 0,
  "unit_conversions_applied": 1651,
  "irregular_timestamp_groups": 0,
  "duplicate_timestamp_rows_removed": 0,
  "resolved_parameters": [
    "applied_current",
    "burnin_temperature",
    "capacitance",
    "drain_source_leakage",
    "drain_source_on_resistance",
    "esr",
    "input_leakage_current",
    "leakage_current",
    "propagation_delay",
    "quiescent_current",
    "supply_operating_current",
    "threshold_voltage"
  ],
  "schema_profile": {
    "unique_parameters": [
      "applied_current",
      "burnin_temperature",
      "capacitance",
      "drain_source_leakage",
      "drain_source_on_resistance",
      "esr",
      "input_leakage_current",
      "leakage_current",
      "propagation_delay",
      "quiescent_current",
      "supply_operating_current",
      "threshold_voltage"
    ],
    "unique_families": [
      "CAPACITOR",
      "DIGITAL_LOGIC",
      "MMIC_ANALOG",
      "POWER_REG",
      "POWER_SEMICONDUCTOR",
      "SENSOR_IF"
    ],
    "time_min": 0.0,
    "time_max": 168.0,
    "rows": 9765,
    "source": "/home/keerthi_kumar/Projects/Sih/project/data/processed/module_A_dataset.csv"
  }
}

========== SEED 20260831 ==========
[features] origin=12 train=240 val=80 test=80
[anomaly] origin=12 calibrated
[forecast] origin=12 models=18
[failure-risk] origin=12 calibrated=True
[features] origin=24 train=240 val=80 test=80
[anomaly] origin=24 calibrated
[forecast] origin=24 models=18
[failure-risk] origin=24 calibrated=True
[features] origin=48 train=240 val=80 test=80
[anomaly] origin=48 calibrated
[forecast] origin=48 models=18
[failure-risk] origin=48 calibrated=False
[features] origin=72 train=240 val=80 test=80
[anomaly] origin=72 calibrated
[forecast] origin=72 models=18
[failure-risk] origin=72 calibrated=False
[features] origin=96 train=240 val=80 test=80
[anomaly] origin=96 calibrated
[forecast] origin=96 models=18
[failure-risk] origin=96 calibrated=False
[features] origin=120 train=240 val=80 test=80
[anomaly] origin=120 calibrated
[forecast] origin=120 models=18
[failure-risk] origin=120 calibrated=False
[features] origin=144 train=240 val=80 test=80
[anomaly] origin=144 calibrated
[forecast] origin=144 models=18
[failure-risk] origin=144 calibrated=False
[features] origin=168 train=240 val=80 test=80
[anomaly] origin=168 calibrated
[failure-risk] origin=168 calibrated=False

=== INGESTION AUDIT ===
input_rows: 9765
canonicalized_rows: 9765
ingestion_mode: LONG
unknown_parameters_quarantined: 0
unit_conversions_applied: 1651
irregular_timestamp_groups: 0
resolved_parameters: ['applied_current', 'burnin_temperature', 'capacitance', 'drain_source_leakage', 'drain_source_on_resistance', 'esr', 'input_leakage_current', 'leakage_current', 'propagation_delay', 'quiescent_current', 'supply_operating_current', 'threshold_voltage']

=== FEATURE AUDIT ===
parts_processed: 80
parameter_pairs: 80
rows_with_INSUFFICIENT_reference: 0
rows_with_<8_reference_points: 0
mean_reference_mad: 0.8549807106289726

=== MODULE A AUDIT ===
mean_anomaly_score_healthy: 0.11567507563983308
mean_anomaly_score_defective: 0.5833598931004103
separation_ratio: 5.043090656078451
threshold_anomaly_review: 0.995
threshold_anomaly_reject: 0.999
early_slope_reject_threshold: 0.0019068436803862953
review_rate_at_val: 0.0

=== MODULE B AUDIT ===
families_fitted: [('CAPACITOR', 'capacitance'), ('CAPACITOR', 'esr'), ('CAPACITOR', 'leakage_current'), ('DIGITAL_LOGIC', 'input_leakage_current'), ('DIGITAL_LOGIC', 'propagation_delay'), ('DIGITAL_LOGIC', 'quiescent_current'), ('MMIC_ANALOG', 'input_leakage_current'), ('MMIC_ANALOG', 'quiescent_current'), ('MMIC_ANALOG', 'supply_operating_current'), ('POWER_REG', 'applied_current'), ('POWER_REG', 'burnin_temperature'), ('POWER_REG', 'supply_operating_current'), ('POWER_SEMICONDUCTOR', 'drain_source_leakage'), ('POWER_SEMICONDUCTOR', 'drain_source_on_resistance'), ('POWER_SEMICONDUCTOR', 'threshold_voltage'), ('SENSOR_IF', 'input_leakage_current'), ('SENSOR_IF', 'propagation_delay'), ('SENSOR_IF', 'supply_operating_current')]
families_fallback_cold_start: []
mean_conformal_radius: 101.78526026797205
coverage_at_95_nominal: 1.0
join_overlap: 231

=== SAFETY POLICY AUDIT ===
ood_threshold_per_family: {"CAPACITOR": 4.660228883544385, "DIGITAL_LOGIC": 3.7600758988714564, "MMIC_ANALOG": 4.7987897945489895, "POWER_REG": 3.846951759573624, "POWER_SEMICONDUCTOR": 3.530038019723286, "SENSOR_IF": 4.369987415558782}
hard_override_fires: 2
ood_escalations: 6
policy_review_rate: 0.0
policy_safe_rate: 0.8125
policy_reject_rate: 0.1875
policy_unknown_rate: 0.0

=== TEST SET METRICS ===
recall: 1.0
fnr: 0.0
fpr: 0.015151515151515152
precision: 0.9333333333333333
pr_auc: 0.7591123530752014
review_burden: 0.0
critical_escapes: 0

========== SEED 20260832 ==========
[features] origin=12 train=240 val=80 test=80
[anomaly] origin=12 calibrated
[forecast] origin=12 models=18
[failure-risk] origin=12 calibrated=True
[features] origin=24 train=240 val=80 test=80
[anomaly] origin=24 calibrated
[forecast] origin=24 models=18
[failure-risk] origin=24 calibrated=True
[features] origin=48 train=240 val=80 test=80
[anomaly] origin=48 calibrated
[forecast] origin=48 models=18
[failure-risk] origin=48 calibrated=False
[features] origin=72 train=240 val=80 test=80
[anomaly] origin=72 calibrated
[forecast] origin=72 models=18
[failure-risk] origin=72 calibrated=False
[features] origin=96 train=240 val=80 test=80
[anomaly] origin=96 calibrated
[forecast] origin=96 models=18
[failure-risk] origin=96 calibrated=False
[features] origin=120 train=240 val=80 test=80
[anomaly] origin=120 calibrated
[forecast] origin=120 models=18
[failure-risk] origin=120 calibrated=False
[features] origin=144 train=240 val=80 test=80
[anomaly] origin=144 calibrated
[forecast] origin=144 models=18
[failure-risk] origin=144 calibrated=False
[features] origin=168 train=240 val=80 test=80
[anomaly] origin=168 calibrated
[failure-risk] origin=168 calibrated=False

=== INGESTION AUDIT ===
input_rows: 9765
canonicalized_rows: 9765
ingestion_mode: LONG
unknown_parameters_quarantined: 0
unit_conversions_applied: 1651
irregular_timestamp_groups: 0
resolved_parameters: ['applied_current', 'burnin_temperature', 'capacitance', 'drain_source_leakage', 'drain_source_on_resistance', 'esr', 'input_leakage_current', 'leakage_current', 'propagation_delay', 'quiescent_current', 'supply_operating_current', 'threshold_voltage']

=== FEATURE AUDIT ===
parts_processed: 80
parameter_pairs: 80
rows_with_INSUFFICIENT_reference: 0
rows_with_<8_reference_points: 0
mean_reference_mad: 0.8549807106289726

=== MODULE A AUDIT ===
mean_anomaly_score_healthy: 0.12740680831004922
mean_anomaly_score_defective: 0.5742587531770117
separation_ratio: 4.507284663936731
threshold_anomaly_review: 0.995
threshold_anomaly_reject: 0.999
early_slope_reject_threshold: 0.0019068436803862953
review_rate_at_val: 0.0

=== MODULE B AUDIT ===
families_fitted: [('CAPACITOR', 'capacitance'), ('CAPACITOR', 'esr'), ('CAPACITOR', 'leakage_current'), ('DIGITAL_LOGIC', 'input_leakage_current'), ('DIGITAL_LOGIC', 'propagation_delay'), ('DIGITAL_LOGIC', 'quiescent_current'), ('MMIC_ANALOG', 'input_leakage_current'), ('MMIC_ANALOG', 'quiescent_current'), ('MMIC_ANALOG', 'supply_operating_current'), ('POWER_REG', 'applied_current'), ('POWER_REG', 'burnin_temperature'), ('POWER_REG', 'supply_operating_current'), ('POWER_SEMICONDUCTOR', 'drain_source_leakage'), ('POWER_SEMICONDUCTOR', 'drain_source_on_resistance'), ('POWER_SEMICONDUCTOR', 'threshold_voltage'), ('SENSOR_IF', 'input_leakage_current'), ('SENSOR_IF', 'propagation_delay'), ('SENSOR_IF', 'supply_operating_current')]
families_fallback_cold_start: []
mean_conformal_radius: 101.78526026797205
coverage_at_95_nominal: 1.0
join_overlap: 231

=== SAFETY POLICY AUDIT ===
ood_threshold_per_family: {"CAPACITOR": 4.660228883544385, "DIGITAL_LOGIC": 3.7600758988714564, "MMIC_ANALOG": 4.7987897945489895, "POWER_REG": 3.846951759573624, "POWER_SEMICONDUCTOR": 3.530038019723286, "SENSOR_IF": 4.369987415558782}
hard_override_fires: 2
ood_escalations: 6
policy_review_rate: 0.0
policy_safe_rate: 0.8125
policy_reject_rate: 0.1875
policy_unknown_rate: 0.0

=== TEST SET METRICS ===
recall: 1.0
fnr: 0.0
fpr: 0.015151515151515152
precision: 0.9333333333333333
pr_auc: 0.7493085153799439
review_burden: 0.0
critical_escapes: 0

========== SEED 20260833 ==========
[features] origin=12 train=240 val=80 test=80
[anomaly] origin=12 calibrated
[forecast] origin=12 models=18
[failure-risk] origin=12 calibrated=True
[features] origin=24 train=240 val=80 test=80
[anomaly] origin=24 calibrated
[forecast] origin=24 models=18
[failure-risk] origin=24 calibrated=True
[features] origin=48 train=240 val=80 test=80
[anomaly] origin=48 calibrated
[forecast] origin=48 models=18
[failure-risk] origin=48 calibrated=False
[features] origin=72 train=240 val=80 test=80
[anomaly] origin=72 calibrated
[forecast] origin=72 models=18
[failure-risk] origin=72 calibrated=False
[features] origin=96 train=240 val=80 test=80
[anomaly] origin=96 calibrated
[forecast] origin=96 models=18
[failure-risk] origin=96 calibrated=False
[features] origin=120 train=240 val=80 test=80
[anomaly] origin=120 calibrated
[forecast] origin=120 models=18
[failure-risk] origin=120 calibrated=False
[features] origin=144 train=240 val=80 test=80
[anomaly] origin=144 calibrated
[forecast] origin=144 models=18
[failure-risk] origin=144 calibrated=False
[features] origin=168 train=240 val=80 test=80
[anomaly] origin=168 calibrated
[failure-risk] origin=168 calibrated=False

=== INGESTION AUDIT ===
input_rows: 9765
canonicalized_rows: 9765
ingestion_mode: LONG
unknown_parameters_quarantined: 0
unit_conversions_applied: 1651
irregular_timestamp_groups: 0
resolved_parameters: ['applied_current', 'burnin_temperature', 'capacitance', 'drain_source_leakage', 'drain_source_on_resistance', 'esr', 'input_leakage_current', 'leakage_current', 'propagation_delay', 'quiescent_current', 'supply_operating_current', 'threshold_voltage']

=== FEATURE AUDIT ===
parts_processed: 80
parameter_pairs: 80
rows_with_INSUFFICIENT_reference: 0
rows_with_<8_reference_points: 0
mean_reference_mad: 0.8549807106289726

=== MODULE A AUDIT ===
mean_anomaly_score_healthy: 0.11981756147698079
mean_anomaly_score_defective: 0.5596471023851979
separation_ratio: 4.670827009717741
threshold_anomaly_review: 0.995
threshold_anomaly_reject: 0.999
early_slope_reject_threshold: 0.0019068436803862953
review_rate_at_val: 0.0

=== MODULE B AUDIT ===
families_fitted: [('CAPACITOR', 'capacitance'), ('CAPACITOR', 'esr'), ('CAPACITOR', 'leakage_current'), ('DIGITAL_LOGIC', 'input_leakage_current'), ('DIGITAL_LOGIC', 'propagation_delay'), ('DIGITAL_LOGIC', 'quiescent_current'), ('MMIC_ANALOG', 'input_leakage_current'), ('MMIC_ANALOG', 'quiescent_current'), ('MMIC_ANALOG', 'supply_operating_current'), ('POWER_REG', 'applied_current'), ('POWER_REG', 'burnin_temperature'), ('POWER_REG', 'supply_operating_current'), ('POWER_SEMICONDUCTOR', 'drain_source_leakage'), ('POWER_SEMICONDUCTOR', 'drain_source_on_resistance'), ('POWER_SEMICONDUCTOR', 'threshold_voltage'), ('SENSOR_IF', 'input_leakage_current'), ('SENSOR_IF', 'propagation_delay'), ('SENSOR_IF', 'supply_operating_current')]
families_fallback_cold_start: []
mean_conformal_radius: 101.78526026797205
coverage_at_95_nominal: 1.0
join_overlap: 231

=== SAFETY POLICY AUDIT ===
ood_threshold_per_family: {"CAPACITOR": 4.660228883544385, "DIGITAL_LOGIC": 3.7600758988714564, "MMIC_ANALOG": 4.7987897945489895, "POWER_REG": 3.846951759573624, "POWER_SEMICONDUCTOR": 3.530038019723286, "SENSOR_IF": 4.369987415558782}
hard_override_fires: 2
ood_escalations: 6
policy_review_rate: 0.0
policy_safe_rate: 0.8125
policy_reject_rate: 0.1875
policy_unknown_rate: 0.0

=== TEST SET METRICS ===
recall: 1.0
fnr: 0.0
fpr: 0.015151515151515152
precision: 0.9333333333333333
pr_auc: 0.7496809857012634
review_burden: 0.0
critical_escapes: 0
{
  "mean": {
    "recall": 1.0,
    "fnr": 0.0,
    "fpr": 0.015151515151515152,
    "precision": 0.9333333333333332,
    "pr_auc": 0.7527006180521362,
    "review_burden": 0.0,
    "safe_rate": 0.8125,
    "reject_rate": 0.1875,
    "unknown_rate": 0.0,
    "critical_escapes": 0.0,
    "test_parts": 80.0,
    "val_parts": 80.0,
    "policy_review_threshold": 0.995,
    "policy_risk_reject_threshold": 0.999,
    "policy_early_slope_reject_threshold": 0.0019068436803862953,
    "coverage_95": 1.0,
    "ood_severe_rate": 0.075
  },
  "std": {
    "recall": 0.0,
    "fnr": 0.0,
    "fpr": 0.0,
    "precision": 1.3597399555105182e-16,
    "pr_auc": 0.005555847643660359,
    "review_burden": 0.0,
    "safe_rate": 0.0,
    "reject_rate": 0.0,
    "unknown_rate": 0.0,
    "critical_escapes": 0.0,
    "test_parts": 0.0,
    "val_parts": 0.0,
    "policy_review_threshold": 0.0,
    "policy_risk_reject_threshold": 0.0,
    "policy_early_slope_reject_threshold": 0.0,
    "coverage_95": 0.0,
    "ood_severe_rate": 0.0
  }
}
FINAL BENCHMARK COMPLETE: reports/final_submission

$ /home/keerthi_kumar/Projects/Sih/project/venv/bin/python scripts/diagnose_scores.py --run-dir reports/final_submission

=== SEED 20260831 / TEST DIAGNOSTICS ===
rows=80 positives=14 negatives=66
evidence_score           PR-AUC=0.7688 ROC-AUC=0.8550 healthy_mean=0.3950 defective_mean=0.6333 q95=0.7603 q99=0.7866
robust_population        PR-AUC=0.6529 ROC-AUC=0.6705 healthy_mean=0.4627 defective_mean=0.6384 q95=0.9446 q99=0.9764
isolation_forest         PR-AUC=0.6963 ROC-AUC=0.8198 healthy_mean=0.4365 defective_mean=0.7619 q95=0.9527 q99=0.9798
temporal                 PR-AUC=0.3826 ROC-AUC=0.3258 healthy_mean=0.5220 defective_mean=0.3875 q95=0.9025 q99=0.9765
multivariate             PR-AUC=0.6034 ROC-AUC=0.7067 healthy_mean=0.4656 defective_mean=0.6790 q95=0.9698 q99=0.9900
anomaly_score            PR-AUC=0.5493 ROC-AUC=0.7424 healthy_mean=0.1157 defective_mean=0.5834 q95=1.0000 q99=1.0000
failure_risk             PR-AUC=0.9787 ROC-AUC=0.9957 healthy_mean=0.0839 defective_mean=0.7633 q95=0.9207 q99=0.9207
risk unique values: 61 of 80
corr(anomaly_score, failure_risk): 0.6708026575223223
decisions: {'SAFE': 65, 'REJECT': 15}

=== SEED 20260832 / TEST DIAGNOSTICS ===
rows=80 positives=14 negatives=66
evidence_score           PR-AUC=0.7504 ROC-AUC=0.8377 healthy_mean=0.3959 defective_mean=0.6201 q95=0.7657 q99=0.7880
robust_population        PR-AUC=0.6529 ROC-AUC=0.6705 healthy_mean=0.4627 defective_mean=0.6384 q95=0.9446 q99=0.9764
isolation_forest         PR-AUC=0.6990 ROC-AUC=0.8014 healthy_mean=0.4347 defective_mean=0.7429 q95=0.9648 q99=0.9797
temporal                 PR-AUC=0.3826 ROC-AUC=0.3258 healthy_mean=0.5220 defective_mean=0.3875 q95=0.9025 q99=0.9765
multivariate             PR-AUC=0.6034 ROC-AUC=0.7067 healthy_mean=0.4656 defective_mean=0.6790 q95=0.9698 q99=0.9900
anomaly_score            PR-AUC=0.5804 ROC-AUC=0.7159 healthy_mean=0.1274 defective_mean=0.5743 q95=1.0000 q99=1.0000
failure_risk             PR-AUC=0.9589 ROC-AUC=0.9913 healthy_mean=0.0827 defective_mean=0.6986 q95=0.9207 q99=0.9207
risk unique values: 68 of 80
corr(anomaly_score, failure_risk): 0.6885380232881099
decisions: {'SAFE': 65, 'REJECT': 15}

=== SEED 20260833 / TEST DIAGNOSTICS ===
rows=80 positives=14 negatives=66
evidence_score           PR-AUC=0.7683 ROC-AUC=0.8431 healthy_mean=0.3995 defective_mean=0.6229 q95=0.7543 q99=0.7791
robust_population        PR-AUC=0.6529 ROC-AUC=0.6705 healthy_mean=0.4627 defective_mean=0.6384 q95=0.9446 q99=0.9764
isolation_forest         PR-AUC=0.6962 ROC-AUC=0.8063 healthy_mean=0.4445 defective_mean=0.7536 q95=0.9398 q99=0.9699
temporal                 PR-AUC=0.3826 ROC-AUC=0.3258 healthy_mean=0.5220 defective_mean=0.3875 q95=0.9025 q99=0.9765
multivariate             PR-AUC=0.6034 ROC-AUC=0.7067 healthy_mean=0.4656 defective_mean=0.6790 q95=0.9698 q99=0.9900
anomaly_score            PR-AUC=0.5767 ROC-AUC=0.7018 healthy_mean=0.1198 defective_mean=0.5596 q95=1.0000 q99=1.0000
failure_risk             PR-AUC=0.9604 ROC-AUC=0.9913 healthy_mean=0.0891 defective_mean=0.7009 q95=0.9190 q99=0.9190
risk unique values: 70 of 80
corr(anomaly_score, failure_risk): 0.6887621103997169
decisions: {'SAFE': 65, 'REJECT': 15}

WROTE: reports/final_submission/ml_diagnostics.csv
Test diagnostics are descriptive only and were not used for tuning.

$ /home/keerthi_kumar/Projects/Sih/project/venv/bin/python scripts/inspect_critical_escapes.py --run-dir reports/final_submission
seed_20260831: no defective components classified SAFE
seed_20260832: no defective components classified SAFE
seed_20260833: no defective components classified SAFE

$ /home/keerthi_kumar/Projects/Sih/project/venv/bin/python scripts/final_success_gate.py --run-dir reports/final_submission
=== FINAL SUBMISSION GATE ===
recall: 1.0  target >= 0.9  [PASS]
fpr: 0.015151515151515152  target <= 0.1  [PASS]
review_burden: 0.0  target <= 0.2  [PASS]
critical_escapes: 0.0  target == 0.0  [PASS]
coverage_95: 1.0  target >= 0.9  [PASS]
```

The machine-readable results will be exported to `reports/final_submission/benchmark_results.json`. In the locked benchmark evaluation, ACS V4.0 achieved **100% defect recall**, **0 critical escapes**, and an **FPR of 1.52%**.

***Please refer to `Project Documentation.pdf` for the complete evidence dossier, ablation studies, and operational migration paths.***

