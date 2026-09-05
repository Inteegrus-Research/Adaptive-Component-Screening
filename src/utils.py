"""Shared project utilities and reproducibility/compatibility guards."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore
try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
PARAMETERS_FILE = CONFIG_DIR / "parameters.yaml"
MODELS_FILE = CONFIG_DIR / "models.yaml"
POLICY_FILE = CONFIG_DIR / "policy.yaml"
REQUIRED_FILES = {"parameters": PARAMETERS_FILE, "models": MODELS_FILE, "policy": POLICY_FILE}

class ConfigError(ValueError):
    pass

class ArtifactCompatibilityError(RuntimeError):
    pass

def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path}: expected mapping/object, got {type(value).__name__}")
    return value

def _require_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ConfigError(f"{path}: expected list, got {type(value).__name__}")
    return value

def _require_keys(mapping: Mapping[str, Any], keys: Iterable[str], path: str) -> None:
    missing = [k for k in keys if k not in mapping]
    if missing:
        raise ConfigError(f"{path}: missing required keys: {', '.join(missing)}")

def _require_nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path}: expected a non-empty string")
    return value

def load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise ConfigError("PyYAML is not installed")
    if not path.exists():
        raise ConfigError(f"Missing configuration file: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    return dict(_require_mapping(payload, str(path)))

def load_configs() -> dict[str, dict[str, Any]]:
    return {name: load_yaml(path) for name, path in REQUIRED_FILES.items()}

def _validate_parameters(cfg: Mapping[str, Any]) -> None:
    _require_keys(cfg, ["schema_version", "ontology", "components", "parameters", "context_parameters", "aliases_policy"], "parameters")
    ontology = _require_mapping(cfg["ontology"], "parameters.ontology")
    _require_keys(ontology, ["canonical_fields", "directionality", "source_classes", "measurement_status_values"], "parameters.ontology")
    directionality = set(_require_list(ontology["directionality"]["values"], "parameters.ontology.directionality.values"))
    source_classes = set(_require_list(ontology["source_classes"]["values"], "parameters.ontology.source_classes.values"))
    components = _require_mapping(cfg["components"], "parameters.components")
    families = set(components)
    all_params = {}
    all_params.update(_require_mapping(cfg["parameters"], "parameters.parameters"))
    all_params.update(_require_mapping(cfg["context_parameters"], "parameters.context_parameters"))
    canonical_ids = set(cfg["parameters"])
    owner: dict[str, str] = {}
    for pid, raw in all_params.items():
        path = f"parameters.{('parameters' if pid in canonical_ids else 'context_parameters')}.{pid}"
        spec = _require_mapping(raw, path)
        _require_keys(spec, ["aliases","physical_quantity","semantic_class","unit_family","preferred_unit","accepted_units","valid_range","source_class"], path)
        aliases = _require_list(spec["aliases"], f"{path}.aliases")
        if not aliases: raise ConfigError(f"{path}.aliases: empty")
        for alias in aliases:
            text = _require_nonempty_string(alias, f"{path}.aliases")
            key = text.strip().upper().replace(" ", "")
            prev = owner.get(key)
            if prev and prev != pid:
                raise ConfigError(f"Ambiguous alias '{text}' maps to both '{prev}' and '{pid}'")
            owner[key] = pid
        if spec["source_class"] not in source_classes:
            raise ConfigError(f"{path}.source_class: unsupported value")
        if "directionality" in spec and spec["directionality"] not in directionality:
            raise ConfigError(f"{path}.directionality: unsupported value")
        vr = _require_mapping(spec["valid_range"], f"{path}.valid_range")
        _require_keys(vr, ["min","max","unit"], f"{path}.valid_range")
        if float(vr["min"]) >= float(vr["max"]): raise ConfigError(f"{path}.valid_range: min >= max")
        if "applicable_families" in spec:
            unknown = sorted(set(_require_list(spec["applicable_families"], f"{path}.applicable_families")) - families)
            if unknown: raise ConfigError(f"{path}.applicable_families: unknown families {unknown}")

def _validate_models(cfg: Mapping[str, Any], parameters_cfg: Mapping[str, Any]) -> None:
    _require_keys(cfg, ["schema_version","runtime","common","anomaly","forecast","ood","model_selection","validation"], "models")
    runtime = _require_mapping(cfg["runtime"], "models.runtime")
    if not isinstance(runtime.get("random_seed"), int): raise ConfigError("models.runtime.random_seed must be int")
    if not isinstance(runtime.get("deterministic"), bool): raise ConfigError("models.runtime.deterministic must be bool")
    common = _require_mapping(cfg["common"], "models.common")
    if not bool(common.get("target_leakage_policy")): raise ConfigError("models.common.target_leakage_policy required")
    weights = _require_mapping(cfg["anomaly"]["ensemble"]["fusion"]["weights"], "models.anomaly.ensemble.fusion.weights")
    if abs(sum(float(v) for v in weights.values()) - 1.0) > 1e-8: raise ConfigError("anomaly ensemble weights must sum to 1")
    splits = _require_mapping(cfg["validation"]["splits"], "models.validation.splits")
    if splits.get("strategy") != "group_by_lot" or not bool(splits.get("no_row_level_random_split")):
        raise ConfigError("models.validation.splits must enforce group-by-lot validation")
    ontology = parameters_cfg["ontology"]["canonical_fields"]
    if "component_family" not in ontology or "parameter" not in ontology:
        raise ConfigError("parameters.ontology.canonical_fields must include component_family and parameter")

def _validate_policy(cfg: Mapping[str, Any]) -> None:
    _require_keys(cfg, ["schema_version","decision_states","objective","costs","hard_limit_policy","risk_score","uncertainty","ood","data_quality","minimum_evidence","safety_override","explainability"], "policy")
    states = set(_require_list(cfg["decision_states"]["allowed"], "policy.decision_states.allowed"))
    if states != {"SAFE","REVIEW","REJECT","UNKNOWN"}: raise ConfigError("policy decision states must be SAFE, REVIEW, REJECT, UNKNOWN")
    costs = _require_mapping(cfg["costs"], "policy.costs")
    for key in ["false_negative","false_positive"]:
        if float(costs.get(key, -1)) < 0: raise ConfigError(f"policy.costs.{key} must be non-negative")
    if not cfg["objective"].get("forbid_trivial_all_reject_solution", False): raise ConfigError("trivial all-reject safeguard must remain enabled")
    if cfg["hard_limit_policy"].get("generic_config_fallback_allowed", True): raise ConfigError("generic safety-limit fallback must remain disabled")
    if float(cfg["risk_score"]["thresholds"]["safe_max"]) >= float(cfg["risk_score"]["thresholds"]["review_max"]): raise ConfigError("risk thresholds must be increasing")

def validate_configs(configs: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    c = configs or load_configs()
    _validate_parameters(c["parameters"])
    _validate_models(c["models"], c["parameters"])
    _validate_policy(c["policy"])
    return c

def stable_config_hash(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",",":"), default=str).encode()).hexdigest()

def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256(); p = Path(path)
    with p.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk: break
            h.update(chunk)
    return h.hexdigest()

def runtime_signature() -> dict[str, str]:
    def ver(name: str) -> str:
        try: return importlib.import_module(name).__version__
        except Exception: return "unavailable"
    return {
        "python": platform.python_version(),
        "python_major_minor": ".".join(platform.python_version().split(".")[:2]),
        "numpy": ver("numpy"), "pandas": ver("pandas"), "scipy": ver("scipy"),
        "scikit_learn": ver("sklearn"), "joblib": ver("joblib"), "pyyaml": ver("yaml"),
    }

def artifact_manifest_path(path: str | Path) -> Path:
    p = Path(path)
    return p.with_name(p.name + ".manifest.json")

def write_artifact_manifest(path: str | Path, *, artifact_kind: str, config_hash: str | None = None,
                            training_data: str | Path | None = None, extra: Mapping[str, Any] | None = None) -> Path:
    p = Path(path)
    manifest = {
        "format": "artifact_manifest_v1",
        "artifact_kind": artifact_kind,
        "artifact_file": p.name,
        "artifact_sha256": file_sha256(p),
        "runtime": runtime_signature(),
        "config_hash": config_hash,
    }
    if training_data is not None:
        td = Path(training_data)
        manifest["training_data"] = {"path": str(td), "sha256": file_sha256(td) if td.exists() else None}
    if extra: manifest.update(dict(extra))
    mp = artifact_manifest_path(p)
    mp.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return mp

def validate_artifact_compatibility(path: str | Path, *, artifact_kind: str, expected_config_hash: str | None = None,
                                    require_manifest: bool = True) -> dict[str, Any]:
    p = Path(path)
    if not p.exists(): raise ArtifactCompatibilityError(f"Artifact not found: {p}")
    mp = artifact_manifest_path(p)
    if not mp.exists():
        if require_manifest: raise ArtifactCompatibilityError(f"Artifact manifest missing for {p}. Retrain the artifact with the current tooling.")
        return {}
    try: m = json.loads(mp.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc: raise ArtifactCompatibilityError(f"Invalid artifact manifest: {mp}") from exc
    if m.get("artifact_kind") != artifact_kind: raise ArtifactCompatibilityError(f"Artifact kind mismatch: expected {artifact_kind}, got {m.get('artifact_kind')}")
    if expected_config_hash and m.get("config_hash") != expected_config_hash: raise ArtifactCompatibilityError("Artifact configuration fingerprint does not match the current configuration")
    current = runtime_signature(); art = m.get("runtime", {})
    for k in ["python_major_minor","numpy","scipy","scikit_learn","joblib"]:
        if art.get(k) and current.get(k) != art.get(k):
            raise ArtifactCompatibilityError(f"Artifact runtime mismatch for {k}: artifact={art.get(k)!r}, current={current.get(k)!r}. Retrain the artifact in the pinned environment.")
    actual_sha = file_sha256(p)
    if m.get("artifact_sha256") and actual_sha != m["artifact_sha256"]: raise ArtifactCompatibilityError("Artifact checksum mismatch; file may be corrupted or modified")
    return m

def seed_everything(seed: int) -> None:
    random.seed(seed)
    if np is not None: np.random.seed(seed)

def _main() -> int:
    parser = argparse.ArgumentParser(description="Project foundation utility")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate-config")
    sub.add_parser("fingerprint")
    sub.add_parser("runtime")
    args = parser.parse_args()
    if args.cmd == "validate-config":
        c = load_configs(); validate_configs(c)
        p, m, pol = c["parameters"], c["models"], c["policy"]
        print("CONFIGURATION VALID")
        print(f"project_root={PROJECT_ROOT}")
        print(f"parameter_count={len(p['parameters'])}")
        print(f"context_parameter_count={len(p['context_parameters'])}")
        print(f"component_family_count={len(p['components'])}")
        print(f"anomaly_ensemble_components={len(m['anomaly']['ensemble']['components'])}")
        print(f"forecast_horizon_h={float(m['forecast']['target_horizon_policy']['default_horizon_h'])}")
        print("policy_decision_states=" + ",".join(pol["decision_states"]["allowed"]))
        print("configuration_sha256=" + stable_config_hash(c))
    elif args.cmd == "fingerprint": print(stable_config_hash(load_configs()))
    else: print(json.dumps(runtime_signature(), indent=2))
    return 0

if __name__ == "__main__": raise SystemExit(_main())


