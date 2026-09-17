"""Read-only result service for the V4 consolidated ``triage.csv`` contract."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class ResultsService:
    """Serve persisted V4 screening reports to the application/frontend.

    V4 intentionally consolidates anomaly evidence, forecasts, trust state and
    final disposition into ``triage.csv``. This service never attempts to join
    the old V3 ``screening.csv`` + ``anomaly.csv`` + ``forecast.csv`` contract.
    """

    def __init__(self, reports_root: str | Path = "reports/live_runs") -> None:
        self.reports_root = Path(reports_root)

    @staticmethod
    def _json_safe(value: Any) -> Any:
        if is_dataclass(value):
            return ResultsService._json_safe(asdict(value))
        if isinstance(value, pd.DataFrame):
            return ResultsService._json_safe(value.to_dict(orient="records"))
        if isinstance(value, pd.Series):
            return ResultsService._json_safe(value.to_dict())
        if isinstance(value, dict):
            return {str(k): ResultsService._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [ResultsService._json_safe(v) for v in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            number = float(value)
            return None if not np.isfinite(number) else number
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    @staticmethod
    def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
        d = frame.copy()

        aliases = {
            "component_id": "part_id",
            "family": "component_family",
            "disposition": "decision",
            "failure_risk": "risk_score",
            "anomaly_score": "anomaly_risk",
            "predicted_value_at_horizon": "prediction_168h",
        }

        for target, source in aliases.items():
            if target in d.columns and source not in d.columns:
                d[source] = d[target]

        if "prediction_168h" not in d.columns:
            for column in [
                "predicted_value_at_horizon",
                "prediction_168h",
            ]:
                if column in d.columns:
                    d["prediction_168h"] = d[column]
                    break

        if "risk_score" not in d.columns:
            if "failure_risk" in d.columns:
                d["risk_score"] = d["failure_risk"]
            else:
                d["risk_score"] = np.nan

        if "decision" not in d.columns and "disposition" in d.columns:
            d["decision"] = d["disposition"]

        if "confidence" not in d.columns:
            if "forecast_confidence" in d.columns:
                d["confidence"] = pd.to_numeric(
                    d["forecast_confidence"],
                    errors="coerce",
                )
            else:
                d["confidence"] = np.nan

        return d

    def _run_dirs(self) -> list[Path]:
        if not self.reports_root.exists():
            return []

        return sorted(
            [
                path
                for path in self.reports_root.iterdir()
                if path.is_dir()
                and (path / "triage.csv").exists()
            ],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

    def resolve_run(self, run_id: str | None = None) -> Path:
        if run_id:
            candidate = self.reports_root / str(run_id)
            if not candidate.is_dir():
                raise FileNotFoundError(
                    f"Run {run_id!r} does not exist."
                )
            if not (candidate / "triage.csv").exists():
                raise FileNotFoundError(
                    f"Run {run_id!r} has no triage.csv."
                )
            return candidate

        runs = self._run_dirs()
        if not runs:
            raise FileNotFoundError(
                f"No V4 reports with triage.csv found under {self.reports_root}."
            )
        return runs[0]

    def _load_triage(self, run_id: str | None = None) -> tuple[Path, pd.DataFrame]:
        run_dir = self.resolve_run(run_id)
        frame = pd.read_csv(
            run_dir / "triage.csv",
            low_memory=False,
        )
        return run_dir, self._normalize_columns(frame)

    def _load_canonical(self, run_dir: Path) -> pd.DataFrame:
        path = run_dir / "canonical_telemetry.csv"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path, low_memory=False)

    def _load_features(self, run_dir: Path) -> pd.DataFrame:
        path = run_dir / "features.csv"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path, low_memory=False)

    def _load_explanations(self, run_dir: Path) -> list[dict[str, Any]]:
        path = run_dir / "explanations.json"
        if path.exists():
            try:
                payload = json.loads(
                    path.read_text(encoding="utf-8")
                )
                return payload if isinstance(payload, list) else []
            except json.JSONDecodeError:
                return []

        csv_path = run_dir / "explanations.csv"
        if csv_path.exists():
            return pd.read_csv(csv_path).to_dict(orient="records")

        return []

    def _load_json(self, run_dir: Path, name: str) -> dict[str, Any]:
        path = run_dir / name
        if not path.exists():
            return {}
        try:
            payload = json.loads(
                path.read_text(encoding="utf-8")
            )
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            return {}

    def summary(self, run_id: str | None = None) -> dict[str, Any]:
        run_dir, triage = self._load_triage(run_id)

        decision_counts = (
            triage["decision"]
            .astype(str)
            .value_counts()
            .to_dict()
            if "decision" in triage.columns
            else {}
        )

        return {
            "run_id": run_dir.name,
            "rows": int(len(triage)),
            "components": int(
                triage["component_id"].nunique()
            )
            if "component_id" in triage.columns
            else 0,
            "parameters": int(
                triage["parameter"].nunique()
            )
            if "parameter" in triage.columns
            else 0,
            "decisions": decision_counts,
            "risk_mean": self._numeric_mean(
                triage,
                "risk_score",
            ),
            "ood_status": self._dominant_value(
                triage,
                "ood_status",
            ),
            "capability_manifest": self._load_json(
                run_dir,
                "capability_manifest.json",
            ),
            "ingestion_audit": self._load_json(
                run_dir,
                "ingestion_audit.json",
            ),
        }

    def components(self, run_id: str | None = None) -> list[dict[str, Any]]:
        _, triage = self._load_triage(run_id)
        if triage.empty:
            return []

        group_columns = [
            column
            for column in [
                "component_id",
                "lot_id",
                "batch_id",
                "family",
            ]
            if column in triage.columns
        ]

        if not group_columns:
            return []

        rows: list[dict[str, Any]] = []
        for key, group in triage.groupby(
            group_columns,
            dropna=False,
            sort=False,
        ):
            if not isinstance(key, tuple):
                key = (key,)

            row = {
                column: key[index]
                for index, column in enumerate(group_columns)
            }

            row.update(
                {
                    "parameters": int(
                        group["parameter"].nunique()
                    )
                    if "parameter" in group.columns
                    else 0,
                    "risk_score": self._numeric_max(
                        group,
                        "risk_score",
                    ),
                    "anomaly_risk": self._numeric_max(
                        group,
                        "anomaly_risk",
                    ),
                    "ood_status": self._dominant_value(
                        group,
                        "ood_status",
                    ),
                    "decision": self._worst_decision(
                        group,
                    ),
                    "prediction_168h": self._numeric_max(
                        group,
                        "prediction_168h",
                    ),
                }
            )
            rows.append(row)

        return self._json_safe(rows)

    def intelligence(
        self,
        component_id: str,
        *,
        run_id: str | None = None,
        parameter: str | None = None,
    ) -> dict[str, Any] | None:
        run_dir, triage = self._load_triage(run_id)

        key = str(component_id)
        mask = triage["component_id"].astype(str).eq(key)

        if parameter is not None and "parameter" in triage.columns:
            mask &= triage["parameter"].astype(str).eq(
                str(parameter)
            )

        selected = triage.loc[mask].copy()
        if selected.empty:
            return None

        canonical = self._load_canonical(run_dir)
        if not canonical.empty and "component_id" in canonical.columns:
            history = canonical.loc[
                canonical["component_id"].astype(str).eq(key)
            ].copy()
            if parameter is not None and "parameter" in history.columns:
                history = history.loc[
                    history["parameter"].astype(str).eq(
                        str(parameter)
                    )
                ]
        else:
            history = pd.DataFrame()

        explanations = self._load_explanations(run_dir)
        explanation_index = {
            (
                str(item.get("component_id", "")),
                str(item.get("parameter", "")),
            ): item
            for item in explanations
        }

        current_rows = selected.to_dict(orient="records")
        explanation_rows = []
        for row in current_rows:
            explanation = explanation_index.get(
                (
                    str(row.get("component_id", "")),
                    str(row.get("parameter", "")),
                )
            )
            if explanation is not None:
                explanation_rows.append(explanation)

        return self._json_safe(
            {
                "run_id": run_dir.name,
                "component_id": key,
                "parameters": current_rows,
                "trajectory": history.to_dict(orient="records"),
                "explanations": explanation_rows,
                "capability_manifest": self._load_json(
                    run_dir,
                    "capability_manifest.json",
                ),
                "ingestion_audit": self._load_json(
                    run_dir,
                    "ingestion_audit.json",
                ),
            }
        )

    def explanation(
        self,
        component_id: str,
        *,
        parameter: str | None = None,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        result = self.intelligence(
            component_id,
            parameter=parameter,
            run_id=run_id,
        )
        if result is None:
            return []
        return result.get("explanations", [])

    @staticmethod
    def _numeric_mean(
        frame: pd.DataFrame,
        column: str,
    ) -> float | None:
        if column not in frame.columns:
            return None
        values = pd.to_numeric(
            frame[column],
            errors="coerce",
        ).dropna()
        return float(values.mean()) if not values.empty else None

    @staticmethod
    def _numeric_max(
        frame: pd.DataFrame,
        column: str,
    ) -> float | None:
        if column not in frame.columns:
            return None
        values = pd.to_numeric(
            frame[column],
            errors="coerce",
        ).dropna()
        return float(values.max()) if not values.empty else None

    @staticmethod
    def _dominant_value(
        frame: pd.DataFrame,
        column: str,
    ) -> str | None:
        if column not in frame.columns:
            return None
        values = (
            frame[column]
            .dropna()
            .astype(str)
        )
        if values.empty:
            return None
        return str(values.value_counts().index[0])

    @staticmethod
    def _worst_decision(
        frame: pd.DataFrame,
    ) -> str:
        priority = {
            "REJECT": 4,
            "UNKNOWN": 3,
            "REVIEW": 2,
            "PASS": 1,
            "SAFE": 1,
        }
        if "decision" not in frame.columns:
            return "UNKNOWN"
        values = frame["decision"].astype(str).tolist()
        return max(
            values,
            key=lambda value: priority.get(value, 0),
        ) if values else "UNKNOWN"
