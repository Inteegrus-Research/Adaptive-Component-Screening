"""Safety intelligence and policy engine.

Pruned clean of legacy duplicate function blocks. Implements calibrated OODProfileV4
with the 10% Physical, 20% Contextual, and 70% Population evidence schema.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from src.utils import (
    PROJECT_ROOT,
    load_yaml,
    seed_everything,
    stable_config_hash as _stable_hash,
    write_artifact_manifest,
    validate_artifact_compatibility,
)

DECISIONS = ("SAFE", "REVIEW", "REJECT", "UNKNOWN")


@dataclass
class OODProfileV4:
    profile_version: str
    config_hash: str
    expected_columns: list[str]
    known_families: list[str]
    known_parameters: list[str]
    known_semantic_types: list[str]
    known_test_methods: list[str]
    known_stress_modes: list[str]
    reference_rows: int
    reference_parts: int
    max_reference_rows: int
    conditional_groups: dict[str, dict[str, dict[str, float]]]
    trajectory_groups: dict[str, dict[str, dict[str, float]]]
    moderate_threshold: float = 0.40
    severe_threshold: float = 0.75


# Backward-compatible aliases
OODProfile = OODProfileV4
OODProfileV3 = OODProfileV4


@dataclass
class ScreeningAssessment:
    part_id: str
    decision: str
    risk_score: float
    confidence: str
    failure_risk: float
    anomaly_risk: float
    ood_score: float
    ood_status: str
    uncertainty_score: float
    data_quality_score: float
    evidence_state: str
    hard_limit_violation: bool
    near_limit: bool
    supporting_evidence_count: int
    high_but_safe: bool = False
    measurement_only_anomaly: bool = False
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)


def _read_any(path: str | Path, table: str | None = None) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
        with sqlite3.connect(p) as con:
            if table is not None:
                return pd.read_sql_query(f"SELECT * FROM {table}", con)
            for candidate in ("module_A_dataset", "module_B_train", "component_parameter_state", "measurements"):
                try:
                    return pd.read_sql_query(f"SELECT * FROM {candidate}", con)
                except Exception:
                    continue
        raise ValueError("No supported table found in database")
    raise ValueError(f"Unsupported input format: {p.suffix}")


def _config_hash(cfg: Mapping[str, Any]) -> str:
    return _stable_hash(cfg)


def _canonical_category(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([None] * len(df), index=df.index, dtype="object")
    return df[col].astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})


def _aggregate_unique(values: pd.Series) -> set[str]:
    return {str(v) for v in values.dropna().astype(str).unique()}


def _operational_schema(columns: Sequence[str] | list[str]) -> list[str]:
    """Return schema fields that are legitimately observable at inference.

    Evaluation labels, split metadata, latent truth and future targets are not schema
    requirements for a deployment upload. Future value readpoints are dynamic and are
    therefore treated separately from static schema requirements.
    """
    excluded = {
        "split", "defect_state", "latent_defect_label", "failure_mode", "primary_failure_mode",
        "high_but_safe", "measurement_only_anomaly", "scenario_id", "catastrophic_false_negative_target",
        "drift_failure_count",
    }
    out=[]
    for c in map(str, columns):
        if c in excluded or c.startswith("target_") or c.startswith("future_defective_") or c.startswith("latent_") or c.startswith("absolute_fail_") or c.startswith("drift_failure_") or re.fullmatch(r"value_\d+(?:\.\d+)?h", c):
            continue
        out.append(c)
    return sorted(set(out))


def _canonical_parameter_series(s: pd.Series) -> pd.Series:
    try:
        from src.ingest import ParameterOntology
        ontology = ParameterOntology(load_yaml(PROJECT_ROOT / "configs" / "parameters.yaml"))
        def one(x):
            if pd.isna(x):
                return x
            spec, _, _ = ontology.resolve(x)
            return spec.name if spec is not None else str(x).strip()
        return s.map(one)
    except Exception:
        return s.map(lambda x: x if pd.isna(x) else str(x).strip())


def _quality_from_input(input_df: pd.DataFrame, policy: Mapping[str, Any]) -> dict[str, Any]:
    n = len(input_df)
    if n == 0:
        return {"status": "FAIL", "score": 0.0, "reasons": ["No observations supplied."], "missing_fraction": 1.0}
    parts = int(input_df["part_id"].nunique()) if "part_id" in input_df.columns else 0
    missing_cells = float(input_df.isna().mean().mean())
    duplicate_fraction = 0.0
    if "observation_id" in input_df.columns:
        duplicate_fraction = float(input_df["observation_id"].duplicated().mean()) if n else 0.0
    req = policy.get("data_quality", {})
    reasons: list[str] = []
    pass_flag = True
    if req.get("require_part_identifier", True) and "part_id" not in input_df.columns:
        pass_flag = False
        reasons.append("Part identifier is missing.")
    if req.get("require_parameter_identifier", True) and "parameter" not in input_df.columns:
        pass_flag = False
        reasons.append("Parameter identifier is missing.")
    if missing_cells > float(req.get("maximum_missing_fraction", 0.40)):
        pass_flag = False
        reasons.append(f"Overall missing-cell fraction {missing_cells:.3f} exceeds policy limit.")
    if duplicate_fraction > float(req.get("maximum_duplicate_fraction", 0.01)):
        pass_flag = False
        reasons.append(f"Duplicate observation fraction {duplicate_fraction:.3f} exceeds policy limit.")
    if parts < int(req.get("minimum_parts", 1)):
        pass_flag = False
        reasons.append("Insufficient unique parts.")
    score = max(0.0, 1.0 - missing_cells / max(float(req.get("maximum_missing_fraction", 0.40)), 1e-9))
    if duplicate_fraction > 0:
        score *= max(0.0, 1.0 - duplicate_fraction / max(float(req.get("maximum_duplicate_fraction", 0.01)), 1e-9))
    return {
        "status": "PASS" if pass_flag else "FAIL",
        "score": float(np.clip(score, 0, 1)),
        "reasons": reasons,
        "rows": n,
        "parts": parts,
        "missing_fraction": missing_cells,
        "duplicate_fraction": duplicate_fraction,
    }


def _part_quality(input_df: pd.DataFrame, part_id: str, policy: Mapping[str, Any]) -> dict[str, Any]:
    if "part_id" not in input_df.columns:
        return {"status": "FAIL", "score": 0.0, "missing_fraction": 1.0, "reasons": ["part_id unavailable"]}
    d = input_df[input_df["part_id"].astype(str).eq(str(part_id))].copy()
    if d.empty:
        return {"status": "FAIL", "score": 0.0, "missing_fraction": 1.0, "reasons": ["No observations for part"]}
    q = _quality_from_input(d, policy)
    if "quality_flag" in d.columns:
        bad = d["quality_flag"].astype(str).str.upper().isin({"ERROR", "INVALID", "BAD"})
        q["bad_quality_fraction"] = float(bad.mean())
        if bad.any():
            q["status"] = "FAIL"
            q["score"] *= float(1.0 - bad.mean())
            q["reasons"].append(f"{int(bad.sum())} observation(s) carry bad quality flags.")
    if "measurement_status" in d.columns:
        invalid = d["measurement_status"].astype(str).str.upper().isin({"ERROR", "INVALID"})
        if invalid.any():
            q["status"] = "FAIL"
            q["score"] *= float(1.0 - invalid.mean())
            q["reasons"].append(f"{int(invalid.sum())} invalid measurement-status row(s).")
    return q


def _v3_key(family: Any, parameter: Any) -> str:
    f = "<MISSING>" if family is None or (isinstance(family, float) and pd.isna(family)) else str(family)
    p = "<MISSING>" if parameter is None or (isinstance(parameter, float) and pd.isna(parameter)) else str(parameter)
    return f"{f}||{p}"


def _v3_robust_stats(s: pd.Series) -> dict[str, float] | None:
    x = pd.to_numeric(s, errors="coerce").dropna().to_numpy(float)
    if x.size < 8:
        return None
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    q01, q99 = np.quantile(x, [0.01, 0.99])
    return {
        "median": med,
        "scale": max(1.4826 * mad, float((q99 - q01) / 5.15), 1e-12),
        "q01": float(q01), "q99": float(q99), "n": float(x.size),
    }


def _v3_from_wide(d: pd.DataFrame) -> pd.DataFrame:
    required = {"part_id", "parameter"}
    if not required.issubset(d.columns):
        return d.copy()
    time_cols = []
    import re
    for c in d.columns:
        m = re.search(r"(?:value|val)[_@]?(\d+(?:\.\d+)?)h$", str(c), flags=re.I)
        if m:
            time_cols.append((c, float(m.group(1))))
    if not time_cols:
        return d.copy()
    id_cols = [c for c in d.columns if c not in {c for c, _ in time_cols}]
    rows = []
    for c, t in time_cols:
        tmp = d[id_cols].copy()
        tmp["time_h"] = t
        tmp["value_asof"] = pd.to_numeric(d[c], errors="coerce")
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True)


def _v3_prepare(d: pd.DataFrame, as_of_h: float | None = None) -> pd.DataFrame:
    x = _v3_from_wide(d)
    if "part_id" not in x.columns:
        x["part_id"] = "<UNKNOWN>"
    if "parameter" not in x.columns:
        x["parameter"] = "<UNKNOWN>"
    x["part_id"] = x["part_id"].astype(str)
    x["parameter"] = _canonical_parameter_series(x["parameter"])
    if "component_family" not in x.columns:
        x["component_family"] = "<MISSING>"
    if "time_h" not in x.columns:
        x["time_h"] = np.nan
    x["time_h"] = pd.to_numeric(x["time_h"], errors="coerce")
    x["value_num"] = pd.to_numeric(x.get("value_asof", x.get("value", np.nan)), errors="coerce")
    if as_of_h is not None and x["time_h"].notna().any():
        x = x[x["time_h"] <= float(as_of_h)].copy()
    return x


def _ood_v4_status(score: float, profile: OODProfileV4) -> str:
    if score >= profile.severe_threshold:
        return "SEVERE"
    if score >= profile.moderate_threshold:
        return "MODERATE"
    return "LOW"


def _ood_v4_score_part(
    profile: OODProfileV4,
    part: pd.DataFrame,
    global_schema: set[str],
) -> dict[str, Any]:
    """Score one part against the conditional OOD reference profile.

    Design rules:
      1. Numeric comparisons are conditioned on family + parameter where available.
      2. Different physical quantities are never pooled into one numeric space.
      3. Context novelty is reported separately from physical distribution shift.
      4. Missing optional schema degrades completeness but does not fabricate OOD.
      5. Robust scaling prevents ordinary reference-tail observations becoming OOD.
      6. reference_level is explicit audit metadata.
    """
    reasons: list[str] = []
    components: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Semantic identity
    # ------------------------------------------------------------------
    family = (
        str(part["component_family"].dropna().iloc[0])
        if (
            "component_family" in part.columns
            and part["component_family"].notna().any()
        )
        else "<MISSING>"
    )

    params = (
        sorted({str(v) for v in part["parameter"].dropna().unique()})
        if "parameter" in part.columns
        else []
    )

    family_status = (
        "MISSING"
        if family == "<MISSING>"
        else (
            "KNOWN"
            if family in profile.known_families
            else "NOVEL"
        )
    )

    components["family_status"] = family_status
    components["family_novelty"] = float(family_status == "NOVEL")

    if family_status == "NOVEL":
        reasons.append(
            f"Component family '{family}' is outside the reference domain."
        )

    unknown_parameters = [
        p for p in params
        if p not in profile.known_parameters
    ]

    if not params:
        parameter_status = "MISSING"
    elif unknown_parameters:
        parameter_status = "NOVEL"
    else:
        parameter_status = "KNOWN"

    components["parameter_status"] = parameter_status
    components["parameter_novelty"] = float(bool(unknown_parameters))

    if unknown_parameters:
        reasons.append(
            "Parameter semantics are unseen: "
            + ", ".join(unknown_parameters[:5])
        )

    # ------------------------------------------------------------------
    # Context semantics
    # ------------------------------------------------------------------
    for col, known_values, key in (
        ("test_method", profile.known_test_methods, "test_method"),
        ("stress_mode", profile.known_stress_modes, "stress"),
    ):
        if col not in part.columns or not part[col].notna().any():
            components[f"{key}_status"] = "MISSING"
            components[f"{key}_novelty"] = 0.0
            continue

        values = {
            str(v)
            for v in part[col].dropna().unique()
        }

        if not known_values:
            # The reference simply had no observations of this field.
            # It is not legitimate to call that "novel".
            components[f"{key}_status"] = "UNOBSERVED_IN_REFERENCE"
            components[f"{key}_novelty"] = 0.0
        else:
            novel_values = values - set(known_values)

            components[f"{key}_status"] = (
                "NOVEL" if novel_values else "KNOWN"
            )
            components[f"{key}_novelty"] = float(bool(novel_values))

            if novel_values:
                reasons.append(
                    f"{col} values are outside the reference domain: "
                    + ", ".join(sorted(novel_values)[:5])
                )

    # ------------------------------------------------------------------
    # Schema completeness
    # ------------------------------------------------------------------
    expected_schema = set(
        _operational_schema(profile.expected_columns)
    )
    observed_schema = set(
        _operational_schema(global_schema)
    )

    missing_fields = sorted(
        expected_schema - observed_schema
    )

    components["schema_missing_fraction"] = (
        len(missing_fields) / max(len(expected_schema), 1)
    )

    components["schema_status"] = (
        "DEGRADED" if missing_fields else "COMPLETE"
    )

    # IMPORTANT:
    # Missing optional fields affect completeness only.
    # They do NOT become numeric OOD evidence.

    # ------------------------------------------------------------------
    # Population-relative and trajectory-relative evidence
    # ------------------------------------------------------------------
    x = _v3_prepare(part)

    measurement_scores: list[float] = []
    trajectory_scores: list[float] = []

    for (fam, param), group in x.groupby(
        ["component_family", "parameter"],
        dropna=False,
        sort=True,
    ):
        family_parameter_key = _v3_key(fam, param)
        parameter_only_key = _v3_key("*", param)

        reference = (
            profile.conditional_groups.get(family_parameter_key)
            or profile.conditional_groups.get(parameter_only_key)
        )

        # --------------------------------------------------------------
        # Population / measurement deviation
        # --------------------------------------------------------------
        if reference and reference.get("value"):
            stats = reference["value"]

            values = (
                pd.to_numeric(
                    group["value_num"],
                    errors="coerce",
                )
                .dropna()
                .to_numpy(float)
            )

            if values.size:
                z = np.abs(
                    (values - stats["median"])
                    / max(stats["scale"], 1e-12)
                )

                tail_fraction = float(
                    np.mean(
                        (values < stats["q01"])
                        | (values > stats["q99"])
                    )
                )

                # Robust calibration.
                #
                # Previous /2 scaling was too aggressive:
                # ordinary reference-tail variation could saturate.
                #
                # /6 retains strong sensitivity to real distribution
                # displacement while preventing normal 2-3 sigma
                # observations from automatically becoming HIGH OOD.
                measurement_score = float(
                    np.clip(
                        np.median(z) / 6.0
                        + 0.20 * tail_fraction,
                        0.0,
                        1.0,
                    )
                )

                measurement_scores.append(measurement_score)

                if measurement_score >= 0.50:
                    reasons.append(
                        f"{param}: measurement distribution shifted"
                    )

        # --------------------------------------------------------------
        # Trajectory / temporal deviation
        # --------------------------------------------------------------
        trajectory_reference = (
            profile.trajectory_groups.get(family_parameter_key)
            or profile.trajectory_groups.get(parameter_only_key)
        )

        if trajectory_reference and trajectory_reference.get("slope"):
            slope_stats = trajectory_reference["slope"]

            for _, part_group in group.groupby(
                "part_id",
                sort=False,
            ):
                part_group = (
                    part_group
                    .dropna(subset=["time_h", "value_num"])
                    .sort_values("time_h")
                )

                if len(part_group) < 2:
                    continue

                delta_t = float(
                    part_group["time_h"].iloc[-1]
                    - part_group["time_h"].iloc[0]
                )

                if delta_t <= 0:
                    continue

                slope = float(
                    (
                        part_group["value_num"].iloc[-1]
                        - part_group["value_num"].iloc[0]
                    )
                    / delta_t
                )

                trajectory_score = float(
                    np.clip(
                        abs(
                            slope - slope_stats["median"]
                        )
                        / max(
                            6.0 * slope_stats["scale"],
                            1e-12,
                        ),
                        0.0,
                        1.0,
                    )
                )

                trajectory_scores.append(trajectory_score)

                if trajectory_score >= 0.50:
                    reasons.append(
                        f"{param}: trajectory differs from reference"
                    )

    components["measurement_shift"] = (
        float(np.mean(measurement_scores))
        if measurement_scores
        else 0.0
    )

    components["trajectory_shift"] = (
        float(np.mean(trajectory_scores))
        if trajectory_scores
        else 0.0
    )

    # ------------------------------------------------------------------
    # Explicit evidence axes
    # ------------------------------------------------------------------
    physical = max(
        components["family_novelty"],
        components["parameter_novelty"],
    )

    context = max(
        components.get("test_method_novelty", 0.0),
        components.get("stress_novelty", 0.0),
    )

    population = max(
        components["measurement_shift"],
        components["trajectory_shift"],
    )

    components["physical_score"] = float(physical)
    components["context_score"] = float(context)
    components["population_score"] = float(population)

    # OOD score is domain novelty, NOT defect probability.
    score = float(
        np.clip(
            0.10 * physical
            + 0.20 * context
            + 0.70 * population,
            0.0,
            1.0,
        )
    )

    # ------------------------------------------------------------------
    # Status semantics
    # ------------------------------------------------------------------
    if physical >= 1.0:
        # Completely unseen physical semantic domain.
        status = "SEVERE"

    elif context >= 1.0:
        # Unseen method/stress context is meaningful, but does not
        # automatically mean an unknown physical quantity.
        status = "MODERATE"

    else:
        status = _ood_v4_status(
            score,
            profile,
        )

    if components["measurement_shift"] >= 0.50:
        reasons.append(
            "Population-relative measurement shift is elevated."
        )

    if components["trajectory_shift"] >= 0.50:
        reasons.append(
            "Population-relative trajectory shift is elevated."
        )

    # ------------------------------------------------------------------
    # Reference provenance / auditability
    # ------------------------------------------------------------------
    if (
        family_status != "KNOWN"
        or parameter_status != "KNOWN"
        or not params
    ):
        reference_level = "none"

    elif (
        len(params) == 1
        and _v3_key(
            family,
            params[0],
        ) in profile.conditional_groups
    ):
        reference_level = "family+parameter"

    elif (
        len(params) >= 1
        and _v3_key(
            "*",
            params[0],
        ) in profile.conditional_groups
    ):
        reference_level = "parameter"

    else:
        reference_level = "none"

    components["reference_level"] = reference_level

    return {
        "part_id": str(part["part_id"].iloc[0]),
        "score": score,
        "status": status,
        "reasons": list(dict.fromkeys(reasons)),
        "components": components,
        "schema_missing_fields": missing_fields,
        "semantic_status": family_status,
        "parameter_status": parameter_status,
    }

def fit_ood_profile(
    reference: pd.DataFrame | str | Path,
    output_path: str | Path | None = None,
    config_path: str | Path | None = None
) -> tuple[OODProfileV4, dict[str, Any]]:
    cfg = load_yaml(config_path or (PROJECT_ROOT / "configs" / "models.yaml"))
    d = _read_any(reference) if isinstance(reference, (str, Path)) else reference.copy()
    original_rows = len(d)
    max_rows = int(cfg.get("ood", {}).get("max_reference_rows", 50000))
    if len(d) > max_rows:
        d = d.sample(n=max_rows, random_state=int(cfg.get("runtime", {}).get("random_seed", 20260831)))
    expected = _operational_schema(list(d.columns))
    x = _v3_prepare(d)
    fam = sorted(_aggregate_unique(_canonical_category(d, "component_family")))
    params = sorted(_aggregate_unique(_canonical_category(x, "parameter")))
    sem = sorted(_aggregate_unique(_canonical_category(d, "semantic_type")))
    methods = sorted(_aggregate_unique(_canonical_category(d, "test_method")))
    stress = sorted(_aggregate_unique(_canonical_category(d, "stress_mode")))
    cond = {}
    traj = {}
    for (family, param), g in x.groupby(["component_family", "parameter"], dropna=False, sort=True):
        key = _v3_key(family, param)
        st = _v3_robust_stats(g["value_num"])
        if st:
            cond[key] = {"value": st}
        ss = []
        for _, pg in g.groupby("part_id", sort=False):
            pg = pg.dropna(subset=["time_h", "value_num"]).sort_values("time_h")
            if len(pg) >= 2:
                dt = float(pg["time_h"].iloc[-1] - pg["time_h"].iloc[0])
                if dt > 0:
                    ss.append(float((pg["value_num"].iloc[-1] - pg["value_num"].iloc[0]) / dt))
        rst = _v3_robust_stats(pd.Series(ss))
        if rst:
            traj[key] = {"slope": rst}
    for param, g in x.groupby("parameter", dropna=False, sort=True):
        key = _v3_key("*", param)
        if key not in cond:
            st = _v3_robust_stats(g["value_num"])
            if st:
                cond[key] = {"value": st}
        if key not in traj:
            ss = []
            for _, pg in g.groupby("part_id", sort=False):
                pg = pg.dropna(subset=["time_h", "value_num"]).sort_values("time_h")
                if len(pg) >= 2:
                    dt = float(pg["time_h"].iloc[-1] - pg["time_h"].iloc[0])
                    if dt > 0:
                        ss.append(float((pg["value_num"].iloc[-1] - pg["value_num"].iloc[0]) / dt))
            rst = _v3_robust_stats(pd.Series(ss))
            if rst:
                traj[key] = {"slope": rst}
    provisional = OODProfileV4(
        "ood_profile_v4", _config_hash(cfg), expected, fam, params, sem, methods, stress,
        int(len(d)), int(d["part_id"].nunique()) if "part_id" in d else 0, max_rows, cond, traj, 0.40, 0.75
    )
    wide = _v3_prepare(d)
    scores = []
    if "part_id" in wide.columns:
        for _, pg in wide.groupby(wide["part_id"].astype(str), sort=False):
            scores.append(_ood_v4_score_part(provisional, pg, set(expected))["score"])
    arr = np.asarray(scores, dtype=float)
    moderate = max(0.40, float(np.quantile(arr, 0.95)) if arr.size else 0.40)
    severe = max(0.75, float(np.quantile(arr, 0.99)) if arr.size else 0.75)
    if severe <= moderate:
        severe = min(1.0, moderate + 0.05)
    profile = OODProfileV4(
        "ood_profile_v4", _config_hash(cfg), expected, fam, params, sem, methods, stress,
        int(len(d)), int(d["part_id"].nunique()) if "part_id" in d else 0, max_rows, cond, traj, moderate, severe
    )
    metrics = {
        "profile_version": profile.profile_version,
        "reference_rows": profile.reference_rows,
        "original_rows": original_rows,
        "reference_parts": profile.reference_parts,
        "moderate_threshold": moderate,
        "severe_threshold": severe
    }
    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"format": "ood_profile_v4", "profile": asdict(profile)}, p, compress=3)
        write_artifact_manifest(
            p, artifact_kind="ood_profile", config_hash=profile.config_hash,
            training_data=reference if isinstance(reference, (str, Path)) else None,
            extra={"format": "ood_profile_v4", "moderate_threshold": moderate, "severe_threshold": severe}
        )
    return profile, metrics


def load_ood_profile(path: str | Path, *, require_compatibility: bool = True) -> OODProfileV4:
    p = Path(path)
    if require_compatibility:
        validate_artifact_compatibility(p, artifact_kind="ood_profile", require_manifest=True)
    payload = joblib.load(p)
    if isinstance(payload, OODProfileV4):
        return payload
    if isinstance(payload, dict) and payload.get("format") == "ood_profile_v4":
        return OODProfileV4(**payload["profile"])
    if isinstance(payload, dict) and payload.get("format") in {"ood_profile_v2", "ood_profile_v3", "ood_profile_v3_1"}:
        prof = payload.get("profile", payload)
        prof = dict(prof)
        prof.setdefault("moderate_threshold", 0.40)
        prof.setdefault("severe_threshold", 0.75)
        return OODProfileV4(**prof)
    raise ValueError("Unsupported OOD profile artifact; rebuild with ood_profile_v4.")


def assess_data_ood(
    profile: OODProfileV4,
    input_df: pd.DataFrame,
    as_of_h: float | None = None,
) -> dict[str, Any]:
    """Assess OOD at part and dataset level.

    Dataset-level status is deliberately robust to isolated tails.

    A single unusual component must not cause an otherwise familiar
    production batch to be labelled globally OOD.
    """

    d = _v3_prepare(
        input_df,
        as_of_h=as_of_h,
    )

    schema = set(
        map(str, input_df.columns)
    )

    if "part_id" not in d.columns:
        return {
            "score": 1.0,
            "status": "SEVERE",
            "parts": [],
            "profile_version": profile.profile_version,
            "schema": {
                "status": "DEGRADED",
                "missing_fields": ["part_id"],
            },
        }

    parts = [
        _ood_v4_score_part(
            profile,
            group,
            schema,
        )
        for _, group in d.groupby(
            d["part_id"].astype(str),
            sort=False,
        )
    ]

    part_scores = np.asarray(
        [
            p["score"]
            for p in parts
        ],
        dtype=float,
    )

    overall_score = float(
        np.max(part_scores)
        if part_scores.size
        else 0.0
    )

    # ------------------------------------------------------------------
    # Schema status is independent of OOD status.
    # ------------------------------------------------------------------
    missing_schema = sorted(
        set(
            _operational_schema(
                profile.expected_columns
            )
        )
        -
        set(
            _operational_schema(
                schema
            )
        )
    )

    schema_status = (
        "COMPLETE"
        if not missing_schema
        else "DEGRADED"
    )

    # ------------------------------------------------------------------
    # Distribution statistics
    # ------------------------------------------------------------------
    q75 = float(
        np.quantile(
            part_scores,
            0.75,
        )
        if part_scores.size
        else 0.0
    )

    q95 = float(
        np.quantile(
            part_scores,
            0.95,
        )
        if part_scores.size
        else 0.0
    )

    severe_threshold = float(
        profile.severe_threshold
    )

    moderate_threshold = float(
        profile.moderate_threshold
    )

    severe_fraction = float(
        np.mean(
            part_scores >= severe_threshold
        )
        if part_scores.size
        else 0.0
    )

    moderate_fraction = float(
        np.mean(
            part_scores >= moderate_threshold
        )
        if part_scores.size
        else 0.0
    )

    # ------------------------------------------------------------------
    # Semantic novelty
    # ------------------------------------------------------------------
    semantic_novel = any(
        (
            p.get("semantic_status") == "NOVEL"
            or p.get("parameter_status") == "NOVEL"
        )
        for p in parts
    )

    # ------------------------------------------------------------------
    # Context novelty prevalence
    # ------------------------------------------------------------------
    novel_context_parts = sum(
        1
        for p in parts
        if (
            p.get(
                "components",
                {},
            ).get(
                "test_method_novelty",
                0.0,
            ) >= 1.0
            or
            p.get(
                "components",
                {},
            ).get(
                "stress_novelty",
                0.0,
            ) >= 1.0
        )
    )

    policy = load_yaml(
        PROJECT_ROOT
        / "configs"
        / "policy.yaml"
    )

    dataset_policy = (
        policy
        .get("ood", {})
        .get("dataset_status", {})
    )

    severe_fraction_min = float(
        dataset_policy.get(
            "severe_part_fraction_min",
            0.05,
        )
    )

    moderate_fraction_min = float(
        dataset_policy.get(
            "moderate_part_fraction_min",
            0.10,
        )
    )

    # ------------------------------------------------------------------
    # Dataset-level OOD decision
    # ------------------------------------------------------------------
    if semantic_novel:
        dataset_status = "SEVERE"

    elif (
        q95 >= severe_threshold
        and severe_fraction >= severe_fraction_min
    ):
        dataset_status = "SEVERE"

    elif (
        q75 >= moderate_threshold
        and moderate_fraction >= moderate_fraction_min
    ):
        dataset_status = "MODERATE"

    elif (
        novel_context_parts
        >= max(
            3,
            int(
                np.ceil(
                    0.10
                    * max(
                        len(parts),
                        1,
                    )
                )
            ),
        )
    ):
        dataset_status = "MODERATE"

    else:
        # Isolated tails do not label the entire population as OOD.
        dataset_status = "LOW"

    return {
        "score": overall_score,
        "status": dataset_status,
        "parts": parts,
        "profile_version": profile.profile_version,
        "schema_expected": profile.expected_columns,
        "schema": {
            "status": schema_status,
            "missing_fields": missing_schema,
        },
        "distribution": {
            "q75": q75,
            "q95": q95,
            "severe_fraction": severe_fraction,
            "moderate_fraction": moderate_fraction,
            "novel_context_parts": novel_context_parts,
        },
    }


def _extract_forecast_evidence(forecast: pd.DataFrame, part_id: str) -> dict[str, Any]:
    d = forecast[forecast.get("part_id", pd.Series(index=forecast.index, dtype="object")).astype(str).eq(str(part_id))].copy()
    if d.empty:
        return {"failure_risk": 0.0, "uncertainty": 1.0, "near_limit": False, "limit_cross": False, "target_available": False, "evidence_count": 0, "rows": 0}
    pred = pd.to_numeric(d.get("prediction_168h"), errors="coerce") if "prediction_168h" in d else pd.Series(np.nan, index=d.index)
    lo = pd.to_numeric(d.get("prediction_lower"), errors="coerce") if "prediction_lower" in d else pd.Series(np.nan, index=d.index)
    hi = pd.to_numeric(d.get("prediction_upper"), errors="coerce") if "prediction_upper" in d else pd.Series(np.nan, index=d.index)
    p_ex = pd.to_numeric(d.get("limit_exceedance_probability_proxy"), errors="coerce") if "limit_exceedance_probability_proxy" in d else pd.Series(np.nan, index=d.index)
    upper_limit = None
    for c in ("absolute_limit_upper", "engineering_limit_upper", "upper_limit", "spec_upper"):
        if c in d.columns:
            upper_limit = pd.to_numeric(d[c], errors="coerce")
            break
    valid_pred = pred.notna()
    failure_risk = float(p_ex[valid_pred].max()) if valid_pred.any() and p_ex[valid_pred].notna().any() else 0.0
    limit_cross = bool(((hi > upper_limit) & upper_limit.notna()).any()) if upper_limit is not None else False
    near_limit = False
    if upper_limit is not None:
        ratio = (pred / upper_limit.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        near_limit = bool((ratio >= 0.95).fillna(False).any())
    widths = (hi - lo).abs() if len(lo) else pd.Series(dtype=float)
    rel_width = widths / pred.abs().replace(0, np.nan) if len(pred) else pd.Series(dtype=float)
    uncertainty = float(np.clip(rel_width.median(skipna=True) if rel_width.notna().any() else 1.0, 0, 1))
    risk_thr = load_yaml(PROJECT_ROOT / "configs" / "policy.yaml").get("risk_score", {}).get("evidence_thresholds", {})
    high_channel_min = float(risk_thr.get("high_channel_min", 0.60))
    support = int(limit_cross) + int(near_limit) + int(failure_risk >= high_channel_min) + int("safety_slope_flag" in d.columns and pd.to_numeric(d["safety_slope_flag"], errors="coerce").fillna(0).max() > 0)
    return {
        "failure_risk": failure_risk,
        "uncertainty": uncertainty,
        "near_limit": near_limit,
        "limit_cross": limit_cross,
        "target_available": valid_pred.any(),
        "evidence_count": support,
        "rows": len(d),
        "max_prediction": float(pred.max()) if valid_pred.any() else None,
        "max_upper": float(hi.max()) if hi.notna().any() else None
    }


def _extract_anomaly_evidence(anomaly: pd.DataFrame, part_id: str) -> dict[str, Any]:
    d = anomaly[anomaly.get("part_id", pd.Series(index=anomaly.index, dtype="object")).astype(str).eq(str(part_id))].copy()
    if d.empty:
        return {"anomaly_risk": 0.0, "absolute_violation": False, "support": 0, "population": 0.0, "temporal": 0.0, "multivariate": 0.0}
    raw = pd.to_numeric(d.get("anomaly_score"), errors="coerce") if "anomaly_score" in d else pd.Series(0.0, index=d.index)
    cal = pd.to_numeric(d.get("calibrated_risk_score"), errors="coerce") if "calibrated_risk_score" in d else pd.Series(np.nan, index=d.index)
    risk = cal.max(skipna=True) if cal.notna().any() else raw.max(skipna=True)
    abs_v = bool(pd.to_numeric(d.get("absolute_violation"), errors="coerce").fillna(0).max() >= 1) if "absolute_violation" in d else False
    support = int(pd.to_numeric(d.get("anomaly_support_count"), errors="coerce").fillna(0).max()) if "anomaly_support_count" in d else 0
    return {
        "anomaly_risk": float(np.clip(risk if pd.notna(risk) else 0, 0, 1)),
        "raw_anomaly": float(np.clip(raw.max(skipna=True) if raw.notna().any() else 0, 0, 1)),
        "absolute_violation": abs_v,
        "support": support,
        "population": float(pd.to_numeric(d.get("population_evidence"), errors="coerce").fillna(0).max()) if "population_evidence" in d else 0.0,
        "temporal": float(pd.to_numeric(d.get("temporal_evidence"), errors="coerce").fillna(0).max()) if "temporal_evidence" in d else 0.0,
        "multivariate": float(pd.to_numeric(d.get("multivariate_component_score"), errors="coerce").fillna(0).max()) if "multivariate_component_score" in d else 0.0,
        "high_but_safe": bool(pd.to_numeric(d.get("high_but_safe", 0), errors="coerce").fillna(0).max() >= 1) if "high_but_safe" in d else False,
        "measurement_only_anomaly": bool(pd.to_numeric(d.get("measurement_only_anomaly", 0), errors="coerce").fillna(0).max() >= 1) if "measurement_only_anomaly" in d else False
    }


def _combine_risk(a: dict[str, Any], f: dict[str, Any], ood: dict[str, Any], quality: dict[str, Any]) -> tuple[float, float, float, int]:
    anomaly = float(a["anomaly_risk"])
    failure = float(f["failure_risk"])
    combined = max(anomaly, failure)
    joint_thr = float(load_yaml(PROJECT_ROOT / "configs" / "policy.yaml").get("risk_score", {}).get("evidence_thresholds", {}).get("joint_high_min", 0.60))
    if anomaly >= joint_thr and failure >= joint_thr:
        combined = min(1.0, combined + 0.12)
    if a["absolute_violation"]:
        combined = 1.0
    uncertainty = float(np.clip(max(f.get("uncertainty", 1.0 if not f.get("target_available") else 0.0), ood.get("score", 0.0), 1.0 - quality.get("score", 0.0)), 0, 1))
    support = int(a.get("support", 0) >= 2) + int(f.get("evidence_count", 0) >= 1) + int(a.get("population", 0) >= 0.65) + int(a.get("temporal", 0) >= 0.65) + int(a.get("multivariate", 0) >= 0.65)
    return float(np.clip(combined, 0, 1)), float(failure), uncertainty, support


def _human_confidence(risk: float, uncertainty: float, ood_score: float, quality: float) -> str:
    cfg = load_yaml(PROJECT_ROOT / "configs" / "policy.yaml")
    ct = cfg.get("confidence_thresholds", {}) if isinstance(cfg, dict) else {}
    low_quality_min = float(ct.get("low_quality_min", 0.60))
    high_u = float(ct.get("high_uncertainty_max", 0.30))
    moderate_u = float(ct.get("moderate_uncertainty_max", 0.70))
    if quality < low_quality_min or uncertainty >= moderate_u or ood_score >= 0.75:
        return "LOW"
    if uncertainty >= high_u or ood_score >= 0.40:
        return "MODERATE"
    if risk >= 0.80 or risk <= 0.20:
        return "HIGH"
    return "MODERATE"


def _load_runtime_policy(policy_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML policy, then overlay the frozen validation-calibrated runtime thresholds.

    The calibrated artifact is deliberately separate from the normative YAML so that
    engineering defaults are not silently mistaken for measured operating thresholds.
    """
    base = load_yaml(policy_path or (PROJECT_ROOT / "configs" / "policy.yaml"))
    if policy_path is not None:
        return base
    cal_path = PROJECT_ROOT / "models" / "calibration" / "safety_policy.json"
    if not cal_path.exists():
        return base
    try:
        payload = json.loads(cal_path.read_text(encoding="utf-8"))
        thr = payload.get("thresholds", {})
        if "risk_score" not in base:
            base["risk_score"] = {"thresholds": {}}
        base["risk_score"].setdefault("thresholds", {}).update({
            k: float(v) for k, v in thr.items()
            if k in {"safe_max", "review_max", "reject_min"} and v is not None
        })
        base.setdefault("calibration", {})
        base["calibration"].update({
            "artifact": str(cal_path),
            "selection_split": payload.get("selection_split", "validation"),
            "optimization": payload.get("optimization", {}),
        })
    except Exception:
        # A malformed optional calibration must never make the core YAML policy unreadable.
        pass
    return base


def _decision_from_trace_dict(trace: Mapping[str, Any], policy: Mapping[str, Any], part_id: str,
                              high_but_safe: bool = False, measurement_only_anomaly: bool = False) -> ScreeningAssessment:
    """Re-apply only the decision policy to an already-computed evidence trace."""
    a = dict(trace.get("module_a", {}) or {})
    f = dict(trace.get("module_b", {}) or {})
    o = dict(trace.get("ood", {}) or {})
    q = dict(trace.get("data_quality", {}) or {})
    return _decision_for_case(part_id, {**a, "high_but_safe": high_but_safe, "measurement_only_anomaly": measurement_only_anomaly}, f, o, q, policy)


def redecide_screening(screening: pd.DataFrame, *, safe_max: float, reject_min: float,
                       policy_path: str | Path | None = None) -> pd.DataFrame:
    """Re-apply a candidate risk operating point without recomputing model evidence.

    This is used exclusively by validation-time threshold search. It avoids accidentally
    recalibrating OOD/model evidence on the test set while making the deployed policy
    exactly reproducible.
    """
    policy = _load_runtime_policy(policy_path)
    policy = json.loads(json.dumps(policy))
    policy.setdefault("risk_score", {}).setdefault("thresholds", {})
    policy["risk_score"]["thresholds"].update({"safe_max": float(safe_max), "reject_min": float(reject_min), "review_max": float(reject_min)})
    rows = []
    for _, row in screening.iterrows():
        trace = row.get("trace_json", {})
        if isinstance(trace, str):
            try:
                trace = json.loads(trace)
            except Exception:
                trace = {}
        assessment = _decision_from_trace_dict(trace, policy, str(row.get("part_id")),
                                              bool(row.get("high_but_safe", False)),
                                              bool(row.get("measurement_only_anomaly", False)))
        rows.append(asdict(assessment))
    out = pd.DataFrame(rows)
    if not out.empty:
        out["reasons_json"] = out["reasons"].apply(json.dumps)
        out["warnings_json"] = out["warnings"].apply(json.dumps)
        out["trace_json"] = out["trace"].apply(lambda x: json.dumps(x, default=str))
        out = out.drop(columns=["reasons", "warnings", "trace"])
    return out


def _decision_for_case(part_id: str, anomaly_e: dict[str, Any], forecast_e: dict[str, Any], ood_e: dict[str, Any], quality_e: dict[str, Any], policy: Mapping[str, Any]) -> ScreeningAssessment:
    combined, failure_risk, uncertainty, support = _combine_risk(anomaly_e, forecast_e, ood_e, quality_e)
    rp = policy["risk_score"]["thresholds"]
    safe_max = float(rp["safe_max"])
    reject_min = float(rp["reject_min"])
    hard_limit = bool(anomaly_e["absolute_violation"])
    near = bool(forecast_e["near_limit"])
    reasons: list[str] = []
    warnings: list[str] = list(quality_e.get("reasons", []))

    if hard_limit:
        decision = "REJECT"
        reasons.append("Authoritative absolute-limit violation detected; safety policy mandates REJECT.")
    elif ood_e.get("status") == "SEVERE":
        decision = "REVIEW"
        reasons.extend(ood_e.get("reasons", [])[:3])
    elif quality_e.get("status") != "PASS":
        decision = "UNKNOWN"
        reasons.extend(quality_e.get("reasons", [])[:3])
    elif not anomaly_e.get("support") and not forecast_e.get("target_available"):
        decision = "UNKNOWN"
        reasons.append("Insufficient anomaly/forecast evidence for a defensible safety decision.")
    elif uncertainty >= float(policy["uncertainty"].get("high_uncertainty_threshold", 0.70)):
        decision = "REVIEW"
        reasons.append("Prediction/anomaly evidence is too uncertain for automatic disposition.")
    elif failure_risk >= reject_min and support >= 2:
        decision = "REJECT"
        reasons.append(f"Forecasted/observed failure risk is high ({failure_risk:.2f}) with independent supporting evidence.")
    elif combined >= reject_min and support >= 2:
        decision = "REJECT"
        reasons.append(f"Combined risk is high ({combined:.2f}) with multiple supporting evidence channels.")
    elif near:
        decision = "REVIEW"
        reasons.append("Upper prediction evidence approaches/crosses the engineering limit; human review required.")
    elif ood_e.get("status") == "MODERATE":
        decision = "REVIEW"
        reasons.append("Input lies outside part of the reference operating distribution.")
    elif combined > safe_max:
        decision = "REVIEW"
        reasons.append(f"Risk ({combined:.2f}) is above the automatic-safe region.")
    else:
        decision = "SAFE"
        reasons.append("No hard-limit violation and no material supported anomaly/future-risk evidence was found.")

    if anomaly_e.get("population", 0) >= 0.65:
        reasons.append(f"Lot-relative population evidence is elevated ({anomaly_e['population']:.2f}).")
    if anomaly_e.get("temporal", 0) >= 0.65:
        reasons.append(f"Early temporal-deviation evidence is elevated ({anomaly_e['temporal']:.2f}).")
    if forecast_e.get("limit_cross"):
        reasons.append("Forecast upper bound crosses the available engineering limit.")
    elif forecast_e.get("target_available"):
        reasons.append(f"168 h point prediction is {forecast_e.get('max_prediction')!s}; upper bound {forecast_e.get('max_upper')!s}.")

    high_safe = bool(anomaly_e.get("high_but_safe", False))
    measurement_only = bool(anomaly_e.get("measurement_only_anomaly", False))
    confidence = _human_confidence(combined, uncertainty, ood_e.get("score", 0), quality_e.get("score", 0))
    evidence_state = "SUPPORTED" if support >= 2 else ("LIMITED" if support == 1 else "INSUFFICIENT")
    trace = {
        "module_a": anomaly_e,
        "module_b": forecast_e,
        "ood": {k: v for k, v in ood_e.items() if k != "parts"},
        "data_quality": quality_e,
        "fusion": {"combined_risk": combined, "supporting_evidence_count": support, "uncertainty": uncertainty},
        "policy": {"safe_max": safe_max, "reject_min": reject_min, "hard_limit_override": hard_limit, "calibration": policy.get("calibration", {})},
    }
    return ScreeningAssessment(
        part_id=str(part_id), decision=decision, risk_score=combined, confidence=confidence,
        failure_risk=failure_risk, anomaly_risk=anomaly_e["anomaly_risk"], ood_score=float(ood_e.get("score", 0)),
        ood_status=str(ood_e.get("status", "UNKNOWN")), uncertainty_score=uncertainty,
        data_quality_score=float(quality_e.get("score", 0)), evidence_state=evidence_state,
        hard_limit_violation=hard_limit, near_limit=near, supporting_evidence_count=support,
        high_but_safe=high_safe, measurement_only_anomaly=measurement_only,
        reasons=list(dict.fromkeys(reasons)), warnings=warnings,
        trace=trace,
    )


def assess_screening(
    anomaly: pd.DataFrame,
    forecast: pd.DataFrame | None,
    input_df: pd.DataFrame,
    ood_profile: OODProfileV4 | None = None,
    policy_path: str | Path | None = None
) -> pd.DataFrame:
    policy = _load_runtime_policy(policy_path)
    if "part_id" not in input_df.columns:
        raise ValueError("Safety assessment requires part_id in input data")
    if ood_profile is None:
        ood_rows = {str(p): {"score": 0.0, "status": "UNKNOWN", "reasons": []} for p in input_df.part_id.astype(str).unique()}
    else:
        ood_global = assess_data_ood(ood_profile, input_df)
        ood_rows = {r["part_id"]: r for r in ood_global["parts"]}
    rows = []
    for pid in input_df.part_id.astype(str).unique():
        a = _extract_anomaly_evidence(anomaly, pid)
        f = _extract_forecast_evidence(forecast, pid) if forecast is not None else {"failure_risk": 0.0, "uncertainty": 1.0, "near_limit": False, "limit_cross": False, "target_available": False, "evidence_count": 0}
        o = ood_rows.get(pid, {"score": 0.5, "status": "UNKNOWN", "reasons": ["OOD profile did not cover this part."]})
        q = _part_quality(input_df, pid, policy)
        assessment = _decision_for_case(pid, a, f, o, q, policy)
        rows.append(asdict(assessment))
    out = pd.DataFrame(rows)
    if not out.empty:
        out["reasons_json"] = out["reasons"].apply(json.dumps)
        out["warnings_json"] = out["warnings"].apply(json.dumps)
        out["trace_json"] = out["trace"].apply(lambda x: json.dumps(x, default=str))
        out = out.drop(columns=["reasons", "warnings", "trace"])
    return out


def screen(
    anomaly_path: str | Path,
    forecast_path: str | Path | None,
    input_path: str | Path,
    ood_artifact: str | Path | None = None,
    output_path: str | Path | None = None,
    policy_path: str | Path | None = None
) -> pd.DataFrame:
    anomaly = _read_any(anomaly_path)
    forecast = _read_any(forecast_path) if forecast_path else None
    input_df = _read_any(input_path)
    profile = load_ood_profile(ood_artifact) if ood_artifact else None
    out = assess_screening(anomaly, forecast, input_df, profile, policy_path)
    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(p, index=False)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Safety/OOD decision engine")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit-ood")
    f.add_argument("reference")
    f.add_argument("artifact")
    s = sub.add_parser("screen")
    s.add_argument("anomaly")
    s.add_argument("input")
    s.add_argument("output")
    s.add_argument("--forecast", default=None)
    s.add_argument("--ood-artifact", default=None)
    args = p.parse_args()
    if args.cmd == "fit-ood":
        _, m = fit_ood_profile(args.reference, args.artifact)
        print(json.dumps(m, indent=2))
    else:
        d = screen(args.anomaly, args.forecast, args.input, args.ood_artifact, args.output)
        print(d[["part_id", "decision", "risk_score", "confidence", "ood_status"]].to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()


