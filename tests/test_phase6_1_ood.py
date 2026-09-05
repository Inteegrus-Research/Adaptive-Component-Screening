import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.safety import fit_ood_profile, assess_data_ood, load_ood_profile


def make_reference(n=90, seed=7):
    rng = np.random.default_rng(seed)
    rows=[]
    specs=[
        ("IC","quiescent_current",10.0,0.25),
        ("IC","drain_source_leakage",2.0,0.08),
        ("CAP","capacitance",100.0,1.5),
    ]
    for family,param,base,sigma in specs:
        for i in range(n):
            v0=base+rng.normal(0,sigma)
            v24=v0+rng.normal(0,sigma*0.20)
            v96=v24+rng.normal(0,sigma*0.20)
            v168=v96+rng.normal(0,sigma*0.25)
            rows.append({"part_id":f"{family}_{param}_{i}","lot_id":f"L{i//15}","component_family":family,"parameter":param,"value_0h":v0,"value_24h":v24,"value_96h":v96,"value_168h":v168})
    return pd.DataFrame(rows)


def fit(tmp_path):
    ref=make_reference()
    artifact=tmp_path/"ood.joblib"
    profile,_=fit_ood_profile(ref, artifact)
    return ref, load_ood_profile(artifact)


def test_normal_mixed_physical_parameters_are_low(tmp_path):
    ref,p=fit(tmp_path)
    r=assess_data_ood(p,ref.groupby("parameter",sort=False).head(5))
    assert r["status"] == "LOW"
    assert max(x["status"] for x in r["parts"]) == "LOW"


def test_alias_resolves_to_same_parameter_reference(tmp_path):
    ref,p=fit(tmp_path)
    d=ref[ref.parameter.eq("quiescent_current")].head(10).copy()
    d["parameter"]="Icc_q"
    r=assess_data_ood(p,d)
    assert r["parts"][0]["components"]["parameter_status"] == "KNOWN"
    assert r["parts"][0]["components"]["reference_level"] == "family+parameter"


def test_unit_scale_of_another_parameter_cannot_pollute_current(tmp_path):
    ref,p=fit(tmp_path)
    current=ref[ref.parameter.eq("quiescent_current")].head(10).copy()
    cap=ref[ref.parameter.eq("capacitance")].head(10).copy()
    cap["value_0h"] *= 1000.0
    cap["value_24h"] *= 1000.0
    cap["value_96h"] *= 1000.0
    cap["value_168h"] *= 1000.0
    mixed=pd.concat([current,cap],ignore_index=True)
    r=assess_data_ood(p,mixed)
    current_rows=[x for x in r["parts"] if x["part_id"].startswith("IC_quiescent_current")]
    assert all(x["status"] == "LOW" for x in current_rows)


def test_known_parameter_large_population_shift_is_not_called_semantic_novelty(tmp_path):
    ref,p=fit(tmp_path)
    d=ref[ref.parameter.eq("quiescent_current")].head(10).copy()
    d["value_0h"] += 3.0
    d["value_24h"] += 3.0
    r=assess_data_ood(p,d)
    x=r["parts"][0]
    assert x["components"]["parameter_status"] == "KNOWN"
    assert x["components"]["reference_level"] == "family+parameter"
    assert x["status"] in {"MODERATE","SEVERE"}


def test_unknown_parameter_is_severe_but_numeric_not_fabricated(tmp_path):
    ref,p=fit(tmp_path)
    d=ref[ref.parameter.eq("quiescent_current")].head(5).copy()
    d["parameter"]="unknown_physical_quantity"
    d["value_0h"]=999999.0
    r=assess_data_ood(p,d)
    x=r["parts"][0]
    assert x["status"] == "SEVERE"
    assert x["components"]["parameter_status"] == "NOVEL"
    assert x["components"]["reference_level"] == "none"
    assert "numeric_value_0h" not in x["components"]


def test_missing_context_is_not_novel(tmp_path):
    ref,p=fit(tmp_path)
    d=ref.head(5).copy()
    # Reference never observed test_method/stress_mode.
    d["test_method"]="MIL-STD-883"
    d["stress_mode"]="HTRB"
    r=assess_data_ood(p,d)
    x=r["parts"][0]
    assert x["components"]["test_method_status"] == "UNOBSERVED_IN_REFERENCE"
    assert x["components"]["stress_status"] == "UNOBSERVED_IN_REFERENCE"
    assert x["components"]["test_method_novelty"] == 0.0
    assert x["components"]["stress_novelty"] == 0.0


def test_novel_context_is_reported_when_reference_has_context(tmp_path):
    ref=make_reference()
    ref["test_method"]="HTOL"
    ref["stress_mode"]="burn_in"
    artifact=tmp_path/"ood.joblib"
    p,_=fit_ood_profile(ref,artifact)
    d=ref.head(5).copy(); d["test_method"]="NEW_TEST"; d["stress_mode"]="NEW_STRESS"
    r=assess_data_ood(p,d)
    x=r["parts"][0]
    assert x["components"]["test_method_status"] == "NOVEL"
    assert x["components"]["stress_status"] == "NOVEL"
    assert x["status"] == "MODERATE"


def test_optional_schema_missing_is_reported_but_not_forced_severe(tmp_path):
    ref,p=fit(tmp_path)
    d=ref.head(10).drop(columns=["lot_id"])
    r=assess_data_ood(p,d)
    assert r["schema"]["status"] == "DEGRADED"
    assert all(x["status"] == "LOW" for x in r["parts"])


def test_irregular_readpoints_use_available_trajectory(tmp_path):
    ref,p=fit(tmp_path)
    d=ref[ref.parameter.eq("quiescent_current")].head(10).copy()
    d=d.drop(columns=["value_96h"])
    r=assess_data_ood(p,d)
    assert r["parts"][0]["components"]["reference_level"] == "family+parameter"


def test_profile_is_versioned_and_portable(tmp_path):
    ref,p=fit(tmp_path)
    assert p.profile_version == "ood_profile_v2"
    assert p.group_profiles
    assert p.parameter_profiles
