from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml


CANONICAL_COLUMNS = [
    "component_id",
    "lot_id",
    "batch_id",
    "family",
    "parameter",
    "value",
    "unit",
    "time_h",
    "temperature",
    "source_row",
    "quarantine",
    "quarantine_reason",
]


class IngestionError(ValueError):
    """Raised when an input cannot be safely canonicalized."""


@dataclass(frozen=True)
class IngestionAudit:
    canonicalized_rows: int
    unknown_parameters_quarantined: int
    unit_conversions_applied: int
    irregular_timestamps_detected: int
    input_rows: int
    dropped_invalid_rows: int
    ingestion_mode: str
    unknown_columns: tuple[str, ...]
    resolved_parameters: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        result = yaml.safe_load(handle) or {}
    if not isinstance(result, dict):
        raise IngestionError(f"Invalid YAML root: {path}")
    return result


def load_parameter_config(path: str | Path = "configs/parameters.yaml") -> dict[str, Any]:
    return load_yaml(path)


def normalize_token(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def normalize_unit(value: Any) -> str:
    token = str(value).strip().lower()
    aliases = {
        "µa": "uA",
        "ua": "uA",
        "μa": "uA",
        "na": "nA",
        "ma": "mA",
        "a": "A",
        "v": "V",
        "mv": "mV",
        "uv": "uV",
        "kv": "kV",
        "°c": "C",
        "degc": "C",
        "c": "C",
        "k": "K",
        "ohm": "Ohm",
        "ω": "Ohm",
        "kohm": "kohm",
        "mohm": "Mohm",
        "db": "dB",
    }
    return aliases.get(token, str(value).strip())


def build_parameter_alias_index(config: Mapping[str, Any]) -> dict[str, str]:
    index: dict[str, str] = {}
    for canonical, spec in config.get("parameters", {}).items():
        index[normalize_token(canonical)] = str(canonical)
        for alias in spec.get("aliases", []):
            index[normalize_token(alias)] = str(canonical)
    return index


def build_schema_alias_index(config: Mapping[str, Any]) -> dict[str, str]:
    index: dict[str, str] = {}
    for canonical, aliases in config.get("schema", {}).get("aliases", {}).items():
        index[normalize_token(canonical)] = str(canonical)
        for alias in aliases:
            index[normalize_token(alias)] = str(canonical)
    return index


def _find_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    lookup = {normalize_token(column): column for column in df.columns}
    for alias in aliases:
        hit = lookup.get(normalize_token(alias))
        if hit is not None:
            return hit
    return None


def _canonical_column(
    df: pd.DataFrame,
    schema_index: Mapping[str, str],
    canonical: str,
) -> str | None:
    aliases = [key for key, value in schema_index.items() if value == canonical]
    return _find_column(df, aliases)


def _coerce_time_hours(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().sum() >= max(2, int(len(series) * 0.80)):
        return numeric.astype(float)

    timestamps = pd.to_datetime(series, errors="coerce", utc=True)
    if timestamps.notna().sum() < 2:
        return numeric.astype(float)

    first_valid = timestamps.dropna().iloc[0]
    return (timestamps - first_valid).dt.total_seconds() / 3600.0


def _parameter_spec(
    config: Mapping[str, Any],
    parameter: str,
) -> dict[str, Any]:
    return dict(config.get("parameters", {}).get(parameter, {}))


def _conversion_factor(
    config: Mapping[str, Any],
    parameter: str,
    source_unit: str,
) -> float | None:
    spec = _parameter_spec(config, parameter)
    quantity = str(spec.get("physical_quantity", ""))
    canonical_unit = str(spec.get("canonical_unit", ""))
    units = config.get("units", {}).get(quantity, {})

    source_unit = normalize_unit(source_unit)

    if source_unit == canonical_unit:
        return 1.0

    if source_unit not in units or canonical_unit not in units:
        return None

    return float(units[source_unit]) / float(units[canonical_unit])


def _wide_time_columns(
    df: pd.DataFrame,
    config: Mapping[str, Any],
) -> list[tuple[str, float]]:
    patterns = [
        re.compile(pattern, flags=re.IGNORECASE)
        for pattern in config.get("wide_format", {}).get("time_patterns", [])
    ]

    result: list[tuple[str, float]] = []

    for column in df.columns:
        name = str(column).strip()

        for pattern in patterns:
            match = pattern.match(name)
            if match:
                result.append((column, float(match.group(1))))
                break

    return result


def _is_long_format(
    df: pd.DataFrame,
    schema_index: Mapping[str, str],
) -> bool:
    required = ["component_id", "parameter", "value", "time_h"]

    return all(
        _canonical_column(df, schema_index, field) is not None
        for field in required
    )


def _prepare_base_columns(
    df: pd.DataFrame,
    schema_index: Mapping[str, str],
) -> dict[str, str | None]:
    fields = [
        "component_id",
        "lot_id",
        "batch_id",
        "family",
        "parameter",
        "value",
        "unit",
        "time_h",
        "temperature",
    ]

    return {
        field: _canonical_column(df, schema_index, field)
        for field in fields
    }


def _canonicalize_long(
    df: pd.DataFrame,
    config: Mapping[str, Any],
    parameter_index: Mapping[str, str],
    schema_index: Mapping[str, str],
) -> tuple[pd.DataFrame, int, int, list[str], int]:
    columns = _prepare_base_columns(df, schema_index)

    missing = [
        field
        for field in ["component_id", "parameter", "value", "time_h"]
        if columns[field] is None
    ]

    if missing:
        raise IngestionError(f"Missing required fields: {missing}")

    result = pd.DataFrame(index=df.index)

    for field, source in columns.items():
        if source is None:
            result[field] = np.nan
        else:
            result[field] = df[source]

    result["source_row"] = np.arange(len(result), dtype=int)

    result["time_h"] = _coerce_time_hours(result["time_h"])
    result["value"] = pd.to_numeric(result["value"], errors="coerce")

    parameters: list[str] = []
    quarantined: list[bool] = []
    quarantine_reasons: list[str] = []

    for raw_parameter in result["parameter"]:
        token = normalize_token(raw_parameter)

        if token in parameter_index:
            parameters.append(parameter_index[token])
            quarantined.append(False)
            quarantine_reasons.append("")
        else:
            parameters.append("UNKNOWN_PARAMETER")
            quarantined.append(True)
            quarantine_reasons.append("UNMAPPED_PARAMETER_ALIAS")

    result["parameter"] = parameters
    result["quarantine"] = quarantined
    result["quarantine_reason"] = quarantine_reasons

    conversions = 0
    normalized_units: list[str] = []

    for idx in result.index:
        parameter = str(result.at[idx, "parameter"])
        raw_unit = result.at[idx, "unit"]

        if parameter == "UNKNOWN_PARAMETER":
            normalized_units.append(
                normalize_unit(raw_unit) if pd.notna(raw_unit) else ""
            )
            continue

        spec = _parameter_spec(config, parameter)
        canonical_unit = str(spec.get("canonical_unit", ""))
        source_unit = normalize_unit(raw_unit) if pd.notna(raw_unit) else canonical_unit

        factor = _conversion_factor(config, parameter, source_unit)

        if factor is not None:
            result.at[idx, "value"] = float(result.at[idx, "value"]) * factor

            if factor != 1.0:
                conversions += 1

            normalized_units.append(canonical_unit)
        else:
            normalized_units.append(source_unit)
            result.at[idx, "quarantine"] = True
            result.at[idx, "quarantine_reason"] = "UNSUPPORTED_UNIT"

    result["unit"] = normalized_units

    for field in ["component_id", "lot_id", "batch_id", "family"]:
        result[field] = result[field].fillna("").astype(str)

    result["temperature"] = pd.to_numeric(
        result["temperature"],
        errors="coerce",
    )

    valid = (
        result["component_id"].ne("")
        & result["time_h"].notna()
        & result["value"].notna()
    )

    dropped = int((~valid).sum())

    result = result.loc[valid, CANONICAL_COLUMNS].copy()

    ignored_columns = set(
        column
        for column in columns.values()
        if column is not None
    )

    unknown_columns = sorted(
        str(column)
        for column in df.columns
        if column not in ignored_columns
        and normalize_token(column) not in schema_index
    )

    unknown_parameters = int(
        result["parameter"].eq("UNKNOWN_PARAMETER").sum()
    )

    return result, unknown_parameters, conversions, unknown_columns, dropped


def _canonicalize_wide(
    df: pd.DataFrame,
    config: Mapping[str, Any],
    parameter_index: Mapping[str, str],
    schema_index: Mapping[str, str],
) -> tuple[pd.DataFrame, int, int, list[str], int]:
    time_columns = _wide_time_columns(df, config)

    if not time_columns:
        raise IngestionError(
            "Input is neither recognized long format nor supported wide format."
        )

    identity = _prepare_base_columns(df, schema_index)

    parameter_source = identity["parameter"]

    if parameter_source is None:
        parameter_source = None

    records: list[dict[str, Any]] = []
    unknown_parameters = 0
    conversions = 0

    for source_row, row in df.iterrows():
        raw_parameter = (
            row[parameter_source]
            if parameter_source is not None
            else "UNKNOWN_PARAMETER"
        )

        parameter = parameter_index.get(
            normalize_token(raw_parameter),
            "UNKNOWN_PARAMETER",
        )

        if parameter == "UNKNOWN_PARAMETER":
            unknown_parameters += len(time_columns)

        raw_unit = (
            normalize_unit(row[identity["unit"]])
            if identity["unit"] is not None
            else ""
        )

        spec = _parameter_spec(config, parameter)
        canonical_unit = str(spec.get("canonical_unit", raw_unit))

        factor = (
            _conversion_factor(config, parameter, raw_unit)
            if parameter != "UNKNOWN_PARAMETER" and raw_unit
            else 1.0
        )

        if factor is None:
            factor = 1.0

        for time_column, time_h in time_columns:
            value = pd.to_numeric(
                pd.Series([row[time_column]]),
                errors="coerce",
            ).iloc[0]

            if pd.isna(value):
                continue

            converted = float(value) * factor

            if factor != 1.0:
                conversions += 1

            records.append(
                {
                    "component_id": (
                        row[identity["component_id"]]
                        if identity["component_id"] is not None
                        else ""
                    ),
                    "lot_id": (
                        row[identity["lot_id"]]
                        if identity["lot_id"] is not None
                        else ""
                    ),
                    "batch_id": (
                        row[identity["batch_id"]]
                        if identity["batch_id"] is not None
                        else ""
                    ),
                    "family": (
                        row[identity["family"]]
                        if identity["family"] is not None
                        else ""
                    ),
                    "parameter": parameter,
                    "value": converted,
                    "unit": canonical_unit,
                    "time_h": float(time_h),
                    "temperature": (
                        row[identity["temperature"]]
                        if identity["temperature"] is not None
                        else np.nan
                    ),
                    "source_row": int(source_row),
                    "quarantine": parameter == "UNKNOWN_PARAMETER",
                    "quarantine_reason": (
                        "UNMAPPED_PARAMETER_ALIAS"
                        if parameter == "UNKNOWN_PARAMETER"
                        else ""
                    ),
                }
            )

    result = pd.DataFrame.from_records(
        records,
        columns=CANONICAL_COLUMNS,
    )

    if result.empty:
        raise IngestionError("No valid observations were found.")

    for field in ["component_id", "lot_id", "batch_id", "family"]:
        result[field] = result[field].fillna("").astype(str)

    result["time_h"] = pd.to_numeric(
        result["time_h"],
        errors="coerce",
    )
    result["value"] = pd.to_numeric(
        result["value"],
        errors="coerce",
    )
    result["temperature"] = pd.to_numeric(
        result["temperature"],
        errors="coerce",
    )

    valid = (
        result["component_id"].ne("")
        & result["time_h"].notna()
        & result["value"].notna()
    )

    dropped = int((~valid).sum())

    result = result.loc[valid, CANONICAL_COLUMNS].copy()

    known = {
        value
        for value in identity.values()
        if value is not None
    }

    known.update(
        column
        for column, _ in time_columns
    )

    unknown_columns = sorted(
        str(column)
        for column in df.columns
        if column not in known
    )

    return (
        result,
        unknown_parameters,
        conversions,
        unknown_columns,
        dropped,
    )


def _count_irregular_groups(canonical: pd.DataFrame) -> int:
    irregular = 0

    for _, group in canonical.groupby(
        ["component_id", "parameter"],
        dropna=False,
    ):
        times = np.sort(
            pd.to_numeric(
                group["time_h"],
                errors="coerce",
            ).dropna().unique()
        )

        if len(times) < 3:
            continue

        diffs = np.diff(times)

        if np.std(diffs) > max(
            1e-12,
            0.25 * abs(np.median(diffs)),
        ):
            irregular += 1

    return irregular


def ingest_dataframe(
    data: pd.DataFrame,
    parameters_path: str | Path = "configs/parameters.yaml",
) -> tuple[pd.DataFrame, IngestionAudit]:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    if data.empty:
        empty = pd.DataFrame(columns=CANONICAL_COLUMNS)

        return (
            empty,
            IngestionAudit(
                canonicalized_rows=0,
                unknown_parameters_quarantined=0,
                unit_conversions_applied=0,
                irregular_timestamps_detected=0,
                input_rows=0,
                dropped_invalid_rows=0,
                ingestion_mode="EMPTY",
                unknown_columns=(),
                resolved_parameters=(),
            ),
        )

    config = load_parameter_config(parameters_path)

    parameter_index = build_parameter_alias_index(config)
    schema_index = build_schema_alias_index(config)

    if _is_long_format(data, schema_index):
        (
            canonical,
            unknown_parameters,
            conversions,
            unknown_columns,
            dropped,
        ) = _canonicalize_long(
            data,
            config,
            parameter_index,
            schema_index,
        )

        mode = "LONG"
    else:
        (
            canonical,
            unknown_parameters,
            conversions,
            unknown_columns,
            dropped,
        ) = _canonicalize_wide(
            data,
            config,
            parameter_index,
            schema_index,
        )

        mode = "WIDE"

    canonical = (
        canonical
        .sort_values(
            ["component_id", "parameter", "time_h"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    audit = IngestionAudit(
        canonicalized_rows=len(canonical),
        unknown_parameters_quarantined=unknown_parameters,
        unit_conversions_applied=conversions,
        irregular_timestamps_detected=_count_irregular_groups(
            canonical
        ),
        input_rows=len(data),
        dropped_invalid_rows=dropped,
        ingestion_mode=mode,
        unknown_columns=tuple(unknown_columns),
        resolved_parameters=tuple(
            sorted(canonical["parameter"].astype(str).unique())
        ),
    )

    return canonical, audit


def ingest_csv(
    path: str | Path,
    parameters_path: str | Path = "configs/parameters.yaml",
) -> tuple[pd.DataFrame, IngestionAudit]:
    return ingest_dataframe(
        pd.read_csv(path),
        parameters_path=parameters_path,
    )
