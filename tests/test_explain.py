import pandas as pd
from src.explain import explain_frame


def test_explanation_is_human_readable():
    d = pd.DataFrame([{
        "part_id":"P1", "decision":"REVIEW", "risk_score":0.72, "confidence":"MODERATE",
        "anomaly_risk":0.7, "failure_risk":0.4, "ood_score":0.2, "ood_status":"LOW",
        "uncertainty_score":0.5, "data_quality_score":0.95, "hard_limit_violation":False,
        "reasons_json":"[\"High lot-relative anomaly\"]",
        "trace_json":"{\"module_a\":{\"population\":0.8,\"temporal\":0.2,\"multivariate\":0.3,\"absolute_violation\":false},\"module_b\":{\"target_available\":true,\"max_prediction\":43.0,\"max_upper\":51.0,\"limit_cross\":true,\"near_limit\":true},\"ood\":{\"status\":\"LOW\"},\"data_quality\":{\"status\":\"PASS\"}}"
    }])
    out = explain_frame(d)
    assert "summary" in out.columns
    assert "43.0" in out.iloc[0].summary
    assert out.iloc[0].decision == "REVIEW"
