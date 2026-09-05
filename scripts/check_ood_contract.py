from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.safety import (
    assess_data_ood,
    fit_ood_profile,
    load_ood_profile,
)


def make_reference(n: int = 90, seed: int = 7) -> pd.DataFrame:

    rng = np.random.default_rng(seed)

    rows = []

    specs = [
        ("IC", "quiescent_current", 10.0, 0.25),
        ("IC", "drain_source_leakage", 2.0, 0.08),
        ("CAP", "capacitance", 100.0, 1.5),
    ]

    for family, parameter, base, sigma in specs:

        for index in range(n):

            v0 = base + rng.normal(0, sigma)
            v24 = v0 + rng.normal(0, sigma * 0.20)
            v96 = v24 + rng.normal(0, sigma * 0.20)
            v168 = v96 + rng.normal(0, sigma * 0.25)

            rows.append(
                {
                    "part_id": f"{family}_{parameter}_{index}",
                    "lot_id": f"L{index // 15}",
                    "component_family": family,
                    "parameter": parameter,
                    "value_0h": v0,
                    "value_24h": v24,
                    "value_96h": v96,
                    "value_168h": v168,
                }
            )

    return pd.DataFrame(rows)


def parts(report):
    return {
        x["part_id"]: x
        for x in report["parts"]
    }


def main():

    reference = make_reference()

    with tempfile.TemporaryDirectory() as directory:

        artifact = (
            Path(directory)
            / "ood.joblib"
        )

        fit_ood_profile(
            reference,
            artifact,
        )

        profile = load_ood_profile(
            artifact
        )

        print()
        print("=" * 72)
        print("OOD V4 CONTRACT CHECK")
        print("=" * 72)

        print(
            "Profile:",
            profile.profile_version,
        )

        print(
            "Reference rows:",
            profile.reference_rows,
        )

        print(
            "Reference parts:",
            profile.reference_parts,
        )

        # --------------------------------------------------------------
        # 1. Normal data
        # --------------------------------------------------------------
        normal = (
            reference
            .groupby(
                "parameter",
                sort=False,
            )
            .head(5)
        )

        normal_report = assess_data_ood(
            profile,
            normal,
        )

        normal_statuses = [
            x["status"]
            for x in normal_report["parts"]
        ]

        print()
        print("[1] Normal reference sample")
        print(
            "    Dataset status:",
            normal_report["status"],
        )
        print(
            "    Part statuses:",
            sorted(set(normal_statuses)),
        )

        assert normal_report["status"] == "LOW"
        assert all(
            status == "LOW"
            for status in normal_statuses
        )

        # --------------------------------------------------------------
        # 2. Strong known-parameter shift
        # --------------------------------------------------------------
        shifted = (
            reference[
                reference.parameter.eq(
                    "quiescent_current"
                )
            ]
            .head(10)
            .copy()
        )

        shifted["value_0h"] += 3.0
        shifted["value_24h"] += 3.0

        shifted_report = assess_data_ood(
            profile,
            shifted,
        )

        shifted_item = shifted_report["parts"][0]

        print()
        print("[2] Strong known-parameter shift")
        print(
            "    Dataset status:",
            shifted_report["status"],
        )
        print(
            "    First part status:",
            shifted_item["status"],
        )
        print(
            "    Parameter status:",
            shifted_item["components"][
                "parameter_status"
            ],
        )
        print(
            "    Reference level:",
            shifted_item["components"][
                "reference_level"
            ],
        )

        assert (
            shifted_item["components"][
                "parameter_status"
            ]
            == "KNOWN"
        )

        assert shifted_item["status"] in {
            "MODERATE",
            "SEVERE",
        }

        # --------------------------------------------------------------
        # 3. Unknown parameter
        # --------------------------------------------------------------
        unknown = (
            reference.head(5)
            .copy()
        )

        unknown["parameter"] = (
            "completely_unknown_quantity"
        )

        unknown["value_0h"] = 999999.0

        unknown_report = assess_data_ood(
            profile,
            unknown,
        )

        unknown_item = unknown_report["parts"][0]

        print()
        print("[3] Unknown physical parameter")
        print(
            "    Status:",
            unknown_item["status"],
        )
        print(
            "    Parameter status:",
            unknown_item["components"][
                "parameter_status"
            ],
        )
        print(
            "    Reference level:",
            unknown_item["components"][
                "reference_level"
            ],
        )

        assert (
            unknown_item["status"]
            == "SEVERE"
        )

        assert (
            unknown_item["components"][
                "parameter_status"
            ]
            == "NOVEL"
        )

        assert (
            unknown_item["components"][
                "reference_level"
            ]
            == "none"
        )

        # --------------------------------------------------------------
        # 4. Novel context
        # --------------------------------------------------------------
        contextual_reference = reference.copy()

        contextual_reference["test_method"] = "HTOL"
        contextual_reference["stress_mode"] = "burn_in"

        contextual_artifact = (
            Path(directory)
            / "ood_context.joblib"
        )

        fit_ood_profile(
            contextual_reference,
            contextual_artifact,
        )

        contextual_profile = load_ood_profile(
            contextual_artifact
        )

        contextual_input = (
            contextual_reference.head(5)
            .copy()
        )

        contextual_input["test_method"] = (
            "NEW_TEST"
        )

        contextual_input["stress_mode"] = (
            "NEW_STRESS"
        )

        contextual_report = assess_data_ood(
            contextual_profile,
            contextual_input,
        )

        contextual_item = (
            contextual_report["parts"][0]
        )

        print()
        print("[4] Novel context")
        print(
            "    Status:",
            contextual_item["status"],
        )
        print(
            "    Test method:",
            contextual_item["components"][
                "test_method_status"
            ],
        )
        print(
            "    Stress:",
            contextual_item["components"][
                "stress_status"
            ],
        )

        assert (
            contextual_item["status"]
            == "MODERATE"
        )

        # --------------------------------------------------------------
        # 5. Missing optional schema
        # --------------------------------------------------------------
        degraded = (
            reference.head(10)
            .drop(columns=["lot_id"])
        )

        degraded_report = assess_data_ood(
            profile,
            degraded,
        )

        print()
        print("[5] Missing optional schema")
        print(
            "    Schema status:",
            degraded_report["schema"][
                "status"
            ],
        )
        print(
            "    Dataset OOD status:",
            degraded_report["status"],
        )

        assert (
            degraded_report["schema"][
                "status"
            ]
            == "DEGRADED"
        )

        assert all(
            x["status"] == "LOW"
            for x in degraded_report["parts"]
        )

        # --------------------------------------------------------------
        # 6. Cross-parameter isolation
        # --------------------------------------------------------------
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

        baseline = assess_data_ood(
            profile,
            current,
        )

        baseline_map = parts(
            baseline
        )

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

        mixed_map = parts(
            mixed_report
        )

        print()
        print("[6] Cross-parameter unit isolation")

        for part_id in baseline_map:

            baseline_score = baseline_map[
                part_id
            ]["score"]

            mixed_score = mixed_map[
                part_id
            ]["score"]

            print(
                f"    {part_id}: "
                f"{baseline_score:.12f} -> "
                f"{mixed_score:.12f}"
            )

            assert (
                mixed_map[part_id]["status"]
                == baseline_map[part_id]["status"]
            )

            assert np.isclose(
                mixed_score,
                baseline_score,
                atol=1e-12,
            )

        print()
        print("=" * 72)
        print("OOD V4 CONTRACT: PASS")
        print("=" * 72)


if __name__ == "__main__":
    main()
