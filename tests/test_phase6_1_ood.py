import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.safety import (
    assess_data_ood,
    fit_ood_profile,
    load_ood_profile,
)


def make_reference(n=90, seed=7):
    rng = np.random.default_rng(seed)

    rows = []

    specs = [
        ("IC", "quiescent_current", 10.0, 0.25),
        ("IC", "drain_source_leakage", 2.0, 0.08),
        ("CAP", "capacitance", 100.0, 1.5),
    ]

    for family, parameter, base, sigma in specs:

        for i in range(n):

            value_0h = (
                base
                + rng.normal(
                    0,
                    sigma,
                )
            )

            value_24h = (
                value_0h
                + rng.normal(
                    0,
                    sigma * 0.20,
                )
            )

            value_96h = (
                value_24h
                + rng.normal(
                    0,
                    sigma * 0.20,
                )
            )

            value_168h = (
                value_96h
                + rng.normal(
                    0,
                    sigma * 0.25,
                )
            )

            rows.append(
                {
                    "part_id": (
                        f"{family}_{parameter}_{i}"
                    ),
                    "lot_id": f"L{i // 15}",
                    "component_family": family,
                    "parameter": parameter,
                    "value_0h": value_0h,
                    "value_24h": value_24h,
                    "value_96h": value_96h,
                    "value_168h": value_168h,
                }
            )

    return pd.DataFrame(rows)


def fit(tmp_path):

    reference = make_reference()

    artifact = (
        tmp_path
        / "ood.joblib"
    )

    fit_ood_profile(
        reference,
        artifact,
    )

    return (
        reference,
        load_ood_profile(
            artifact
        ),
    )


def part_map(report):
    return {
        item["part_id"]: item
        for item in report["parts"]
    }


def test_normal_mixed_physical_parameters_are_low(
    tmp_path,
):

    reference, profile = fit(tmp_path)

    data = (
        reference
        .groupby(
            "parameter",
            sort=False,
        )
        .head(5)
    )

    report = assess_data_ood(
        profile,
        data,
    )

    assert report["status"] == "LOW"

    assert all(
        item["status"] == "LOW"
        for item in report["parts"]
    )


def test_alias_resolves_to_same_parameter_reference(
    tmp_path,
):

    reference, profile = fit(tmp_path)

    data = (
        reference[
            reference.parameter.eq(
                "quiescent_current"
            )
        ]
        .head(10)
        .copy()
    )

    data["parameter"] = "Icc_q"

    report = assess_data_ood(
        profile,
        data,
    )

    item = report["parts"][0]

    assert (
        item["components"]["parameter_status"]
        == "KNOWN"
    )

    assert (
        item["components"]["reference_level"]
        == "family+parameter"
    )


def test_unit_scale_of_another_parameter_cannot_pollute_current(
    tmp_path,
):

    reference, profile = fit(tmp_path)

    current = (
        reference[
            reference.parameter.eq(
                "quiescent_current"
            )
        ]
        .head(10)
        .copy()
    )

    capacitance = (
        reference[
            reference.parameter.eq(
                "capacitance"
            )
        ]
        .head(10)
        .copy()
    )

    # Baseline current-only assessment.
    baseline = assess_data_ood(
        profile,
        current,
    )

    baseline_parts = part_map(
        baseline
    )

    # Deliberately corrupt the capacitance scale.
    capacitance[
        [
            "value_0h",
            "value_24h",
            "value_96h",
            "value_168h",
        ]
    ] *= 1000.0

    mixed = pd.concat(
        [
            current,
            capacitance,
        ],
        ignore_index=True,
    )

    mixed_report = assess_data_ood(
        profile,
        mixed,
    )

    mixed_parts = part_map(
        mixed_report
    )

    # Every quiescent-current assessment must remain
    # exactly invariant to the unrelated capacitance shift.
    for part_id in baseline_parts:

        assert (
            mixed_parts[part_id]["status"]
            == baseline_parts[part_id]["status"]
        )

        assert np.isclose(
            mixed_parts[part_id]["score"],
            baseline_parts[part_id]["score"],
            atol=1e-12,
        )


def test_known_parameter_large_population_shift_is_not_called_semantic_novelty(
    tmp_path,
):

    reference, profile = fit(tmp_path)

    data = (
        reference[
            reference.parameter.eq(
                "quiescent_current"
            )
        ]
        .head(10)
        .copy()
    )

    data["value_0h"] += 3.0
    data["value_24h"] += 3.0

    report = assess_data_ood(
        profile,
        data,
    )

    item = report["parts"][0]

    assert (
        item["components"]["parameter_status"]
        == "KNOWN"
    )

    assert (
        item["components"]["reference_level"]
        == "family+parameter"
    )

    assert item["status"] in {
        "MODERATE",
        "SEVERE",
    }

    assert (
        item["components"]["parameter_novelty"]
        == 0.0
    )


def test_unknown_parameter_is_severe_but_numeric_not_fabricated(
    tmp_path,
):

    reference, profile = fit(tmp_path)

    data = (
        reference[
            reference.parameter.eq(
                "quiescent_current"
            )
        ]
        .head(5)
        .copy()
    )

    data["parameter"] = (
        "unknown_physical_quantity"
    )

    data["value_0h"] = 999999.0

    report = assess_data_ood(
        profile,
        data,
    )

    item = report["parts"][0]

    assert item["status"] == "SEVERE"

    assert (
        item["components"]["parameter_status"]
        == "NOVEL"
    )

    assert (
        item["components"]["reference_level"]
        == "none"
    )

    assert (
        "numeric_value_0h"
        not in item["components"]
    )


def test_missing_context_is_not_novel(
    tmp_path,
):

    reference, profile = fit(tmp_path)

    data = reference.head(5).copy()

    # Reference never observed these context fields.
    data["test_method"] = "MIL-STD-883"
    data["stress_mode"] = "HTRB"

    report = assess_data_ood(
        profile,
        data,
    )

    item = report["parts"][0]

    assert (
        item["components"]["test_method_status"]
        == "UNOBSERVED_IN_REFERENCE"
    )

    assert (
        item["components"]["stress_status"]
        == "UNOBSERVED_IN_REFERENCE"
    )

    assert (
        item["components"]["test_method_novelty"]
        == 0.0
    )

    assert (
        item["components"]["stress_novelty"]
        == 0.0
    )


def test_novel_context_is_reported_when_reference_has_context(
    tmp_path,
):

    reference = make_reference()

    reference["test_method"] = "HTOL"
    reference["stress_mode"] = "burn_in"

    artifact = (
        tmp_path
        / "ood.joblib"
    )

    fit_ood_profile(
        reference,
        artifact,
    )

    profile = load_ood_profile(
        artifact
    )

    data = reference.head(5).copy()

    data["test_method"] = "NEW_TEST"
    data["stress_mode"] = "NEW_STRESS"

    report = assess_data_ood(
        profile,
        data,
    )

    item = report["parts"][0]

    assert (
        item["components"]["test_method_status"]
        == "NOVEL"
    )

    assert (
        item["components"]["stress_status"]
        == "NOVEL"
    )

    assert item["status"] == "MODERATE"

    assert (
        item["components"]["context_score"]
        == 1.0
    )


def test_optional_schema_missing_is_reported_but_not_forced_severe(
    tmp_path,
):

    reference, profile = fit(tmp_path)

    data = reference.head(10).drop(
        columns=["lot_id"]
    )

    report = assess_data_ood(
        profile,
        data,
    )

    assert (
        report["schema"]["status"]
        == "DEGRADED"
    )

    assert "lot_id" in (
        report["schema"]["missing_fields"]
    )

    # Missing optional metadata must not fabricate physical OOD.
    assert all(
        item["status"] == "LOW"
        for item in report["parts"]
    )


def test_irregular_readpoints_use_available_trajectory(
    tmp_path,
):

    reference, profile = fit(tmp_path)

    data = (
        reference[
            reference.parameter.eq(
                "quiescent_current"
            )
        ]
        .head(10)
        .copy()
    )

    data = data.drop(
        columns=["value_96h"]
    )

    report = assess_data_ood(
        profile,
        data,
    )

    item = report["parts"][0]

    assert (
        item["components"]["reference_level"]
        == "family+parameter"
    )

    assert (
        item["components"]["trajectory_shift"]
        >= 0.0
    )


def test_profile_is_versioned_and_portable(
    tmp_path,
):

    reference, profile = fit(tmp_path)

    assert (
        profile.profile_version
        == "ood_profile_v4"
    )

    assert profile.conditional_groups
    assert profile.trajectory_groups

    assert (
        profile.reference_rows
        == len(reference)
    )

    assert (
        profile.reference_parts
        == reference["part_id"].nunique()
    )
