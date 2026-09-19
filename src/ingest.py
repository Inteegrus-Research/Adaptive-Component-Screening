"""Final schema-adaptive ingestion layer for ACS benchmark/runtime."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json, re
import numpy as np
import pandas as pd

ALIASES = {
    "part_id": ["part_id", "part", "component_id", "component", "serial_id", "device_id"],
    "lot_id": ["lot_id", "lot", "lot_no", "lot_number", "batch", "batch_id"],
    "component_family": ["component_family", "family", "part_family", "device_family"],
    "parameter": ["parameter", "param", "metric", "test_parameter", "measurement"],
    "raw_parameter": ["raw_parameter", "raw_param", "source_parameter"],
    "time_h": ["time_h", "time_hr", "hours", "burnin_h", "timestamp_h", "time"],
    "value": ["value", "measurement_value", "reading", "measured_value"],
    "split": ["split", "partition", "dataset_split"],
    "future_defective": ["future_defective", "future_defect", "label", "target", "defective_future"],
    "defect_state": ["defect_state", "state", "failure_mode", "mechanism"],
}
KNOWN_FAMILIES = {"DIGITAL_LOGIC","MMIC_ANALOG","POWER_REG","CAPACITOR","SENSOR_IF","POWER_SEMICONDUCTOR"}
KNOWN_PARAMETERS = {
    "quiescent_current","input_leakage_current","supply_operating_current","propagation_delay",
    "threshold_voltage","drain_source_leakage","drain_source_on_resistance","capacitance",
    "leakage_current","esr","burnin_temperature","applied_current",
}
PARAMETER_ALIASES = {
    "IDDQ":"quiescent_current", "quiescent_current":"quiescent_current",
    "pin7_leakage_nA":"input_leakage_current", "input_leakage_current":"input_leakage_current",
    "I_supply_mA":"supply_operating_current", "supply_operating_current":"supply_operating_current",
    "tpd_ns":"propagation_delay", "propagation_delay":"propagation_delay",
    "VTH":"threshold_voltage", "threshold_voltage":"threshold_voltage",
    "I_DS_leak_uA":"drain_source_leakage", "drain_source_leakage":"drain_source_leakage",
    "RDS_on_mOhm":"drain_source_on_resistance", "drain_source_on_resistance":"drain_source_on_resistance",
    "C_nF":"capacitance", "capacitance":"capacitance",
    "cap_leak_uA":"leakage_current", "leakage_current":"leakage_current",
    "ESR_Ohm":"esr", "esr":"esr",
    "BurnInTemp_C":"burnin_temperature", "burnin_temperature":"burnin_temperature",
    "AppliedCurrent_mA":"applied_current", "applied_current":"applied_current",
}
@dataclass
class IngestionAudit:
    input_rows: int
    canonicalized_rows: int
    ingestion_mode: str
    unknown_parameters_quarantined: int
    unit_conversions_applied: int
    irregular_timestamp_groups: int
    duplicate_timestamp_rows_removed: int
    resolved_parameters: list[str]
    schema_profile: dict[str, Any]

def _find(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {str(c).lower(): str(c) for c in df.columns}
    for c in candidates:
        if c in df.columns: return c
        if c.lower() in lookup: return lookup[c.lower()]
    return None

def canonicalize_parameters(values: pd.Series) -> pd.Series:
    def one(v: Any) -> str:
        if pd.isna(v): return "UNKNOWN_PARAMETER"
        s = str(v).strip()
        return PARAMETER_ALIASES.get(s, PARAMETER_ALIASES.get(s.lower(), s if s in KNOWN_PARAMETERS else "UNKNOWN_PARAMETER"))
    return values.map(one)

def ingest_dataframe(df: pd.DataFrame, source_name: str = "<dataframe>") -> tuple[pd.DataFrame, IngestionAudit, pd.DataFrame]:
    raw = df.copy()
    input_rows = len(raw)
    wide = not ({_find(raw, ALIASES["time_h"]), _find(raw, ALIASES["value"]) } <= set(raw.columns))
    if "time_h" not in raw.columns and "value" not in raw.columns:
        # Common wide representation: value_0h / value_24h / ...
        id_cols = [c for c in ["part_id","component_id","lot_id","family","component_family","parameter","raw_parameter","split","defect_state","future_defective","absolute_limit_lower","absolute_limit_upper"] if c in raw.columns]
        value_cols = [c for c in raw.columns if re.match(r"^value_[-+]?\d+(?:\.\d+)?h$", str(c))]
        if value_cols:
            long = raw.melt(id_vars=id_cols, value_vars=value_cols, var_name="_time_name", value_name="value")
            long["time_h"] = long["_time_name"].str.extract(r"([-+]?\d+(?:\.\d+)?)")[0].astype(float)
            raw = long.drop(columns=["_time_name"])
        wide = False
    rename = {}
    for target, cand in ALIASES.items():
        c = _find(raw, cand)
        if c and c != target: rename[c] = target
    raw = raw.rename(columns=rename)
    required = ["part_id","lot_id","component_family","parameter","time_h","value"]
    missing = [c for c in required if c not in raw.columns]
    if missing: raise ValueError(f"Missing required canonical fields: {missing}")
    if "raw_parameter" not in raw.columns: raw["raw_parameter"] = raw["parameter"].astype(str)
    raw["raw_parameter"] = raw["raw_parameter"].astype(str)
    raw["parameter"] = canonicalize_parameters(raw["parameter"])
    if raw["parameter"].eq("UNKNOWN_PARAMETER").all():
        raw["parameter"] = canonicalize_parameters(raw["raw_parameter"])
    raw["part_id"] = raw["part_id"].astype(str)
    raw["lot_id"] = raw["lot_id"].astype(str)
    raw["component_family"] = raw["component_family"].astype(str)
    raw["time_h"] = pd.to_numeric(raw["time_h"], errors="coerce")
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    before_unknown = len(raw)
    unknown_mask = raw["parameter"].eq("UNKNOWN_PARAMETER")
    quarantined = raw.loc[unknown_mask].copy()
    work = raw.loc[~unknown_mask].copy()
    group = ["part_id","parameter"]
    work = work.sort_values(group + ["time_h"]).copy()
    dup_mask = work.duplicated(group + ["time_h"], keep="last")
    duplicates = int(dup_mask.sum())
    work = work.loc[~dup_mask].copy()
    irregular = int((work.groupby(group)["time_h"].nunique().lt(2)).sum())
    unit_conversions = int((work.get("unit", pd.Series(index=work.index)).fillna("").astype(str) != work.get("raw_unit", pd.Series(index=work.index)).fillna("").astype(str)).sum()) if "unit" in work.columns and "raw_unit" in work.columns else 0
    schema_profile = {
        "unique_parameters": sorted(work["parameter"].dropna().astype(str).unique().tolist()),
        "unique_families": sorted(work["component_family"].dropna().astype(str).unique().tolist()),
        "time_min": float(work["time_h"].min()) if len(work) else None,
        "time_max": float(work["time_h"].max()) if len(work) else None,
        "rows": len(work),
        "source": str(source_name),
    }
    audit = IngestionAudit(input_rows=input_rows, canonicalized_rows=len(work), ingestion_mode="WIDE" if wide else "LONG",
                           unknown_parameters_quarantined=len(quarantined), unit_conversions_applied=unit_conversions,
                           irregular_timestamp_groups=irregular, duplicate_timestamp_rows_removed=duplicates,
                           resolved_parameters=sorted(work["parameter"].unique().tolist()), schema_profile=schema_profile)
    return work.reset_index(drop=True), audit, quarantined.reset_index(drop=True)

def ingest_csv(path: str | Path) -> tuple[pd.DataFrame, IngestionAudit, pd.DataFrame]:
    p = Path(path)
    return ingest_dataframe(pd.read_csv(p), str(p))
