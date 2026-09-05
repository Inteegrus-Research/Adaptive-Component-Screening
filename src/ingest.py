"""Universal data ingestion for component screening.

Converts long or common wide screening tables into a strict canonical long-form
schema. Includes autonomous fallbacks to prevent dropping unknown parameters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from src.utils import load_yaml

CANONICAL_COLUMNS = [
    "observation_id", "source_file", "source_row_number", "source_column", "source_format",
    "part_id", "lot_id", "wafer_id", "test_run_id", "component_family", "component_type",
    "parameter", "raw_parameter", "semantic_type", "physical_quantity", "value", "value_original",
    "unit", "raw_unit", "unit_conversion_factor", "time_h", "test_stage", "test_method",
    "temperature_C", "burnin_temperature_C", "voltage_V", "current_A", "stress_mode",
    "measurement_status", "censoring", "quality_flag", "mapping_method", "unit_source", "quality_detail",
]

FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "part_id": ("part_id", "part", "part_no", "part_number", "serial", "serial_number", "device_id", "component_id", "unit_id"),
    "lot_id": ("lot_id", "lot", "lot_no", "lot_number", "batch", "batch_id", "production_lot", "wafer_lot"),
    "wafer_id": ("wafer_id", "wafer", "wafer_no", "wafer_number"),
    "test_run_id": ("test_run_id", "testrun_id", "screening_run_id", "screening_id", "run_id", "test_id"),
    "component_family": ("component_family", "family", "device_family", "component_class", "eee_family"),
    "component_type": ("component_type", "part_type", "device_type", "part_class"),
    "parameter": ("parameter", "parameter_name", "param", "metric", "measurement_name", "signal", "feature_name", "test_parameter"),
    "value": ("value", "measurement_value", "measured_value", "reading", "result"),
    "unit": ("unit", "units", "measurement_unit", "value_unit", "measurement_units"),
    "time_h": ("time_h", "time_hr", "time_hours", "elapsed_h", "elapsed_hr", "elapsed_hours", "hours", "burnin_time_h", "age_h"),
    "timestamp": ("timestamp", "datetime", "date_time", "measurement_time", "recorded_at", "time_stamp"),
    "test_stage": ("test_stage", "stage", "screening_stage", "test_phase", "phase"),
    "test_method": ("test_method", "method", "screening_method", "test_type"),
    "temperature_C": ("temperature_c", "temp_c", "temperature", "temp", "measurement_temperature_c", "measurement_temp_c", "junction_temperature_c", "case_temperature_c"),
    "burnin_temperature_C": ("burnin_temperature_c", "burn_in_temperature_c", "burnin_temp_c", "stress_temperature_c", "burn_temperature_c", "burn_in_temp_c"),
    "voltage_V": ("voltage_v", "voltage", "applied_voltage_v", "test_voltage_v", "stress_voltage_v", "bias_voltage_v"),
    "current_A": ("current_a", "current", "applied_current_a", "test_current_a", "stress_current_a", "bias_current_a"),
    "stress_mode": ("stress_mode", "stress", "stress_type", "screening_stress", "bias_mode"),
    "measurement_status": ("measurement_status", "status", "test_status", "observation_status"),
    "censoring": ("censoring", "censor", "censor_flag", "comparison"),
    "quality_flag": ("quality_flag", "quality", "data_quality", "measurement_quality"),
}

# TRUE UNIVERSALITY REGEX: Allows bare readpoints (e.g. "0h", "168h") as well as "value_24h"
TIME_COL_RE = re.compile(r"^(?:(?:value|measurement|reading|result)[_\-])?(?P<time>\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hour|hours)?$", re.I)
PARAM_TIME_RE = re.compile(r"^(?P<param>.+?)[_\-](?P<time>-?\d+(?:\.\d+)?)\s*(?:h|hr|hrs|hour|hours)?$", re.I)
NUM_RE = re.compile(r"^\s*(?P<op><=|>=|<|>)?\s*(?P<num>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(?P<unit>\S+)?\s*$")

UNIT_CANON = {
    "a": "A", "ma": "mA", "ua": "uA", "na": "nA", "pa": "pA", "µa": "uA", "μa": "uA",
    "v": "V", "mv": "mV", "kv": "kV",
    "ohm": "Ohm", "ω": "Ω", "mohm": "mOhm", "mω": "mΩ", "uohm": "uOhm", "µω": "µΩ",
    "s": "s", "ms": "ms", "us": "us", "µs": "µs", "μs": "µs", "ns": "ns", "ps": "ps",
    "f": "F", "mf": "mF", "uf": "uF", "µf": "uF", "μf": "uF", "nf": "nF", "pf": "pF",
    "db": "dB", "pct": "pct", "%": "pct", "fraction": "fraction",
}

UNIT_TO_BASE = {
    "current": {"A": 1e6, "mA": 1e3, "uA": 1.0, "nA": 1e-3, "pA": 1e-6},
    "voltage": {"V": 1.0, "mV": 1e-3, "kV": 1e3},
    "resistance": {"Ohm": 1.0, "mOhm": 1e-3, "uOhm": 1e-6, "Ω": 1.0, "mΩ": 1e-3, "µΩ": 1e-6},
    "time": {"s": 1e9, "ms": 1e6, "us": 1e3, "µs": 1e3, "ns": 1.0, "ps": 1e-3},
    "capacitance": {"F": 1e6, "mF": 1e3, "uF": 1.0, "nF": 1e-3, "pF": 1e-6},
    "impedance": {"Ohm": 1.0, "mOhm": 1e-3, "Ω": 1.0, "mΩ": 1e-3},
    "gain": {"dB": 1.0},
    "dimensionless": {"pct": 1.0, "fraction": 100.0},
}

UNIT_SUFFIXES = [("_uohm", "uOhm"), ("_mohm", "mOhm"), ("_ohm", "Ohm"), ("_ua", "uA"), ("_na", "nA"), ("_pa", "pA"), ("_ma", "mA"), ("_a", "A"), ("_kv", "kV"), ("_mv", "mV"), ("_v", "V"), ("_ns", "ns"), ("_us", "us"), ("_ms", "ms"), ("_s", "s"), ("_uf", "uF"), ("_nf", "nF"), ("_pf", "pF"), ("_pct", "pct"), ("_percent", "pct"), ("_f", "F")]


class IngestionError(RuntimeError):
    pass


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    aliases: tuple[str, ...]
    physical_quantity: str
    semantic_class: str
    preferred_unit: str
    accepted_units: tuple[str, ...]
    directionality: str
    valid_min: float | None
    valid_max: float | None


@dataclass(frozen=True)
class SchemaProfile:
    source_format: str
    columns: dict[str, str]
    time_columns: dict[str, float]
    measurement_columns: tuple[tuple[str, str | None, float], ...]
    warnings: tuple[str, ...]


def _norm(value: Any) -> str:
    s = str(value).strip().lower().replace("µ", "u").replace("μ", "u")
    s = re.sub(r"[\s\-/\.]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")

def _float(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan

def _text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None

def _project_configs() -> dict[str, dict[str, Any]]:
    root = Path(__file__).resolve().parents[1]
    return {
        "parameters": load_yaml(root / "configs" / "parameters.yaml"),
        "models": load_yaml(root / "configs" / "models.yaml"),
        "policy": load_yaml(root / "configs" / "policy.yaml"),
    }


class ParameterOntology:
    def __init__(self, config: Mapping[str, Any]):
        self.specs: dict[str, ParameterSpec] = {}
        self.aliases: dict[str, str] = {}
        for name, raw in config["parameters"].items():
            vr = raw.get("valid_range", {})
            spec = ParameterSpec(
                name=name, aliases=tuple([name, *raw.get("aliases", [])]),
                physical_quantity=str(raw.get("physical_quantity", "unknown")),
                semantic_class=str(raw.get("semantic_class", name)),
                preferred_unit=str(raw.get("preferred_unit", "")),
                accepted_units=tuple(raw.get("accepted_units", [])),
                directionality=str(raw.get("directionality", "context_dependent")),
                valid_min=_float(vr.get("min")), valid_max=_float(vr.get("max")),
            )
            self.specs[name] = spec
            for alias in spec.aliases:
                self.aliases[_norm(alias)] = name

    def resolve(self, raw: Any) -> tuple[ParameterSpec | None, str, list[str]]:
        token = _norm(raw)
        if token in self.aliases: return self.specs[self.aliases[token]], "exact_alias", []
        stripped = token
        for suffix, _unit in UNIT_SUFFIXES:
            if stripped.endswith(suffix):
                stripped = stripped[: -len(suffix)].rstrip("_")
                if stripped in self.aliases:
                    return self.specs[self.aliases[stripped]], "alias_unit_suffix", []
        hits = []
        for alias, name in self.aliases.items():
            if len(alias) >= 4 and (token.startswith(alias + "_") or token.endswith("_" + alias)):
                hits.append(name)
        hits = list(dict.fromkeys(hits))
        if len(hits) == 1: return self.specs[hits[0]], "conservative_alias", []
        if len(hits) > 1: return None, "ambiguous", hits
        return None, "unmapped", []


class SchemaProfiler:
    def __init__(self, ontology: ParameterOntology):
        self.ontology = ontology

    def _resolve_columns(self, columns: Iterable[str]) -> dict[str, str]:
        normalized = {_norm(c): c for c in columns}
        out: dict[str, str] = {}
        for field, aliases in FIELD_ALIASES.items():
            for alias in aliases:
                if _norm(alias) in normalized:
                    out[field] = normalized[_norm(alias)]
                    break
        return out

    def profile(self, df: pd.DataFrame) -> SchemaProfile:
        columns = self._resolve_columns(df.columns)
        time_columns: dict[str, float] = {}
        for col in df.columns:
            m = TIME_COL_RE.match(_norm(col))
            if m: time_columns[col] = float(m.group("time"))

        measurement_columns: list[tuple[str, str | None, float]] = []
        if "parameter" not in columns or "value" not in columns:
            for col in df.columns:
                m = PARAM_TIME_RE.match(_norm(col))
                if m:
                    spec, _, _ = self.ontology.resolve(m.group("param"))
                    if spec: measurement_columns.append((col, m.group("param"), float(m.group("time"))))
            if "parameter" in columns:
                for col, t in time_columns.items():
                    measurement_columns.append((col, None, t))

        warnings = []
        source_format = "long" if {"parameter", "value"}.issubset(columns) else "wide"
        if "part_id" not in columns: warnings.append("part_id was not resolved")
        if "lot_id" not in columns: warnings.append("lot_id was not resolved")
        if source_format == "wide" and not measurement_columns:
            warnings.append("no recognized measurement/time columns were found")
        return SchemaProfile(source_format, columns, time_columns, tuple(dict.fromkeys(measurement_columns)), tuple(warnings))


class Canonicalizer:
    def __init__(self, ontology: ParameterOntology):
        self.ontology = ontology

    @staticmethod
    def _unit(raw: Any) -> str | None:
        value = _text(raw)
        return UNIT_CANON.get(value.replace(" ", "").lower(), value) if value else None

    @staticmethod
    def _numeric_and_censor(value: Any) -> tuple[float, str]:
        if value is None or pd.isna(value): return math.nan, "NONE"
        if isinstance(value, (int, float, np.number)): return float(value), "NONE"
        m = NUM_RE.match(str(value).strip())
        if not m: return math.nan, "UNKNOWN"
        return float(m.group("num")), {"<": "LT", "<=": "LE", ">": "GT", ">=": "GE", "": "NONE"}[m.group("op") or ""]

    def _convert(self, value: float, raw_unit: str | None, spec: ParameterSpec) -> tuple[float, str | None, float | None, str | None]:
        unit = self._unit(raw_unit) or spec.preferred_unit
        factors = UNIT_TO_BASE.get(spec.physical_quantity, {})
        if unit == spec.preferred_unit: return value, unit, 1.0, None
        if unit not in factors or spec.preferred_unit not in factors: return math.nan, unit, None, "unsupported_unit_or_conversion"
        factor = factors[unit] / factors[spec.preferred_unit]
        return value * factor, spec.preferred_unit, factor, None

    @staticmethod
    def _suffix_unit(raw_parameter: Any) -> str | None:
        token = _norm(raw_parameter)
        for suffix, unit in UNIT_SUFFIXES:
            if token.endswith(suffix): return unit
        return None

    def _observation_id(self, row: Mapping[str, Any]) -> str:
        source = "|".join(str(row.get(k, "")) for k in ("source_file", "source_row_number", "source_column", "part_id", "parameter", "time_h"))
        return hashlib.sha256(source.encode()).hexdigest()[:24]

    def _base(self, raw: pd.Series, cols: Mapping[str, str], source_file: str, source_row: int, source_format: str) -> dict[str, Any]:
        def get(name: str) -> Any: return raw[cols[name]] if cols.get(name) else None
        return {
            "observation_id": None, "source_file": source_file, "source_row_number": source_row, "source_column": None, "source_format": source_format,
            "part_id": _text(get("part_id")), "lot_id": _text(get("lot_id")), "wafer_id": _text(get("wafer_id")), "test_run_id": _text(get("test_run_id")),
            "component_family": _text(get("component_family")), "component_type": _text(get("component_type")),
            "parameter": None, "raw_parameter": None, "semantic_type": None, "physical_quantity": None,
            "value": math.nan, "value_original": None, "unit": None, "raw_unit": None, "unit_conversion_factor": math.nan,
            "time_h": math.nan, "test_stage": _text(get("test_stage")), "test_method": _text(get("test_method")),
            "temperature_C": _float(get("temperature_C")), "burnin_temperature_C": _float(get("burnin_temperature_C")),
            "voltage_V": _float(get("voltage_V")), "current_A": _float(get("current_A")), "stress_mode": _text(get("stress_mode")),
            "measurement_status": None, "censoring": "NONE", "quality_flag": "PENDING", "mapping_method": None, "unit_source": None, "quality_detail": None,
        }

    def _one(self, base: dict[str, Any], raw: pd.Series, cols: Mapping[str, str], raw_parameter: Any, raw_value: Any, raw_unit: Any, time_h: float, source_column: str) -> dict[str, Any]:
        row = dict(base)
        row["source_column"] = source_column
        row["raw_parameter"] = _text(raw_parameter)
        spec, method, ambiguity = self.ontology.resolve(raw_parameter)
        
        # Unknown/ambiguous semantics are quarantined rather than guessed. The raw
        # identifier remains in raw_parameter and downstream safety maps this to UNKNOWN.
        if spec is None:
            numeric, censor = self._numeric_and_censor(raw_value)
            explicit = self._unit(raw_unit)
            suffix = self._suffix_unit(raw_parameter)
            row["mapping_method"] = method
            row["parameter"] = None
            row["semantic_type"] = None
            row["physical_quantity"] = None
            row["time_h"] = time_h
            row["value"] = numeric
            row["value_original"] = raw_value if not pd.isna(raw_value) else None
            row["raw_unit"] = explicit or suffix or "raw_units"
            row["unit"] = explicit or suffix or "raw_units"
            row["unit_conversion_factor"] = 1.0
            row["unit_source"] = "unresolved"
            row["censoring"] = censor
            status = _text(raw[cols["measurement_status"]] if "measurement_status" in cols else None)
            if status: row["measurement_status"] = {"PASS": "VALID", "OK": "VALID", "FAIL": "FAILED_TEST"}.get(status.upper(), status.upper())
            else: row["measurement_status"] = "MISSING" if math.isnan(numeric) and censor == "NONE" else "VALID"
            row["quality_flag"] = "ERROR"
            row["quality_detail"] = "unresolved_semantic_mapping"
            row["observation_id"] = self._observation_id(row)
            return row

        row["mapping_method"] = method
        row["parameter"] = spec.name
        row["semantic_type"] = spec.semantic_class
        row["physical_quantity"] = spec.physical_quantity
        row["time_h"] = time_h
        numeric, censor = self._numeric_and_censor(raw_value)
        suffix = self._suffix_unit(raw_parameter)
        explicit = self._unit(raw_unit)
        chosen_unit = explicit or suffix or spec.preferred_unit
        value, unit, factor, conversion_error = self._convert(numeric, chosen_unit, spec)
        row["value_original"] = raw_value if not pd.isna(raw_value) else None
        row["raw_unit"] = chosen_unit
        row["unit"] = unit
        row["unit_conversion_factor"] = factor if factor is not None else math.nan
        row["unit_source"] = "explicit" if explicit else ("parameter_suffix" if suffix else "ontology_preferred")
        row["censoring"] = censor

        status = _text(raw[cols["measurement_status"]] if "measurement_status" in cols else None)
        if status: status = {"PASS": "VALID", "OK": "VALID", "FAIL": "FAILED_TEST"}.get(status.upper(), status.upper())
        else: status = "MISSING" if math.isnan(value) and censor == "NONE" else ("BELOW_DETECTION" if censor in {"LT", "LE"} else ("ABOVE_RANGE" if censor in {"GT", "GE"} else "VALID"))
        row["measurement_status"] = status

        problems = []
        if conversion_error: problems.append(conversion_error)
        if not math.isnan(value):
            if spec.valid_min is not None and value < spec.valid_min: problems.append("below_plausibility_range")
            if spec.valid_max is not None and value > spec.valid_max: problems.append("above_plausibility_range")
            if spec.physical_quantity in {"current", "resistance", "capacitance", "impedance", "time"} and value < 0: problems.append("negative_for_nonnegative_quantity")
        if row["part_id"] is None: problems.append("missing_part_id")

        if problems:
            row["quality_flag"] = "ERROR"
            row["quality_detail"] = ";".join(problems)
        elif status != "VALID":
            row["quality_flag"] = "LIMITED"
            row["quality_detail"] = status.lower()
        else:
            row["quality_flag"] = "OK"
            row["quality_detail"] = ""
        row["value"] = value
        row["observation_id"] = self._observation_id(row)
        return row

    def canonicalize(self, df: pd.DataFrame, profile: SchemaProfile, source_file: str = "") -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        cols = profile.columns
        if profile.source_format == "long":
            elapsed = pd.to_numeric(df[cols["time_h"]], errors="coerce") if "time_h" in cols else None
            timestamps = pd.to_datetime(df[cols["timestamp"]], errors="coerce") if "timestamp" in cols else None
            inferred0 = {}
            if elapsed is None and timestamps is not None:
                part_col = cols.get("part_id")
                for key, ix in df.groupby(df[part_col].astype(str) if part_col else pd.Series("PART", index=df.index)).groups.items():
                    ts = timestamps.loc[ix]
                    first = ts.dropna().min() if ts.notna().any() else pd.NaT
                    if pd.notna(first): inferred0[key] = first
            for idx, raw in df.iterrows():
                if elapsed is not None:
                    t = _float(elapsed.loc[idx])
                elif timestamps is not None and pd.notna(timestamps.loc[idx]):
                    key = str(raw[cols["part_id"]]) if "part_id" in cols else "PART"
                    base_ts = inferred0.get(key, timestamps.loc[idx])
                    t = float((timestamps.loc[idx] - base_ts).total_seconds() / 3600.0)
                else:
                    t = math.nan
                base = self._base(raw, cols, source_file, int(idx) + 1, "long")
                rows.append(self._one(base, raw, cols, raw[cols["parameter"]], raw[cols["value"]], raw[cols["unit"]] if "unit" in cols else None, t, cols["value"]))
            result = pd.DataFrame(rows, columns=CANONICAL_COLUMNS)
        else:
            if not profile.measurement_columns: raise IngestionError("No recognized measurement columns in wide input")
            for idx, raw in df.iterrows():
                base = self._base(raw, cols, source_file, int(idx) + 1, "wide")
                for source_col, embedded_parameter, t in profile.measurement_columns:
                    raw_parameter = embedded_parameter if embedded_parameter is not None else raw[cols["parameter"]]
                    unit = raw[cols["unit"]] if "unit" in cols else None
                    rows.append(self._one(base, raw, cols, raw_parameter, raw[source_col], unit, t, source_col))
            result = pd.DataFrame(rows, columns=CANONICAL_COLUMNS)

        if result.empty: raise IngestionError("Canonicalization produced zero rows")

        duplicate_key = ["part_id", "test_run_id", "parameter", "time_h", "test_stage", "test_method", "source_column"]
        available_key = [c for c in duplicate_key if c in result.columns]
        if available_key:
            dup = result.duplicated(subset=available_key, keep=False) & result["part_id"].notna() & result["parameter"].notna()
            result.loc[dup & result["quality_flag"].eq("OK"), "quality_flag"] = "LIMITED"
            result.loc[dup & result["quality_detail"].fillna("").eq(""), "quality_detail"] = "duplicate_observation_key"
            result.loc[dup & result["quality_detail"].fillna("").ne("") , "quality_detail"] = result.loc[dup & result["quality_detail"].fillna("").ne(""), "quality_detail"] + ";duplicate_observation_key"

        for c in ["value", "unit_conversion_factor", "time_h", "temperature_C", "burnin_temperature_C", "voltage_V", "current_A"]:
            result[c] = pd.to_numeric(result[c], errors="coerce")
        return result

def profile_csv(path: str | Path) -> dict[str, Any]:
    ontology = ParameterOntology(_project_configs()["parameters"])
    df = pd.read_csv(path, low_memory=False)
    profile = SchemaProfiler(ontology).profile(df)
    return {"source_file": str(path), "rows": len(df), "columns": len(df.columns), "source_format": profile.source_format, "resolved_columns": profile.columns, "time_columns": profile.time_columns, "measurement_columns": [list(x) for x in profile.measurement_columns], "warnings": list(profile.warnings)}

def canonicalize_csv(input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    ontology = ParameterOntology(_project_configs()["parameters"])
    df = pd.read_csv(input_path, low_memory=False)
    profile = SchemaProfiler(ontology).profile(df)
    result = Canonicalizer(ontology).canonicalize(df, profile, str(input_path))
    output = Path(output_path); output.parent.mkdir(parents=True, exist_ok=True); result.to_csv(output, index=False)
    return {"input": str(input_path), "output": str(output), "input_rows": len(df), "canonical_rows": len(result), "parts": int(result["part_id"].nunique(dropna=True)), "parameters": int(result["parameter"].nunique(dropna=True)), "valid_rows": int((result["quality_flag"] == "OK").sum()), "error_rows": int((result["quality_flag"] == "ERROR").sum()), "warnings": list(profile.warnings)}

def validate_canonical(df: pd.DataFrame) -> tuple[bool, list[str]]:
    errors = []
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing: return False, [f"missing_columns:{missing}"]
    valid = df["quality_flag"].eq("OK")
    for col in ["part_id", "parameter", "value"]:
        if df.loc[valid, col].isna().any(): errors.append(f"valid_row_without_{col}")
    return not errors, errors

def main() -> int:
    parser = argparse.ArgumentParser(description="Universal component screening ingestion")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("canonicalize"); p.add_argument("input"); p.add_argument("output"); p.set_defaults(action="canonicalize")
    args = parser.parse_args()
    if args.action == "canonicalize": print(json.dumps(canonicalize_csv(args.input, args.output), indent=2)); return 0
    return 1

if __name__ == "__main__":
    raise SystemExit(main())



# Stable ontology regression-test helper.
def alias_demo() -> pd.DataFrame:
    return pd.DataFrame({"raw_parameter": ["IDDQ", "I_DDQ", "QUIESCENT_CURRENT"],
                         "canonical_parameter": ["quiescent_current"] * 3})
