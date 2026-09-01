from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

import iopr.analysis as analysis


def _manifest() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "case_id": "case_001",
            "plane": "HR",
            "annotator": "A",
            "organ_a": "unused",
            "organ_b": "unused",
            "expert_mask": "unused",
        },
        {
            "case_id": "case_002",
            "plane": "HR",
            "annotator": "A",
            "organ_a": "unused",
            "organ_b": "unused",
            "expert_mask": "unused",
        },
    ])


def _load_masks(row, manifest_base):
    organ_a = np.zeros((9, 9, 9), dtype=bool)
    organ_b = np.zeros((9, 9, 9), dtype=bool)
    expert = np.zeros((9, 9, 9), dtype=bool)

    organ_a[4, 4, 2] = True
    organ_b[4, 4, 6] = True
    expert[4, 4, 4] = True

    return organ_a, organ_b, expert, (1.0, 1.0, 1.0)


def test_negative_controls_use_configured_legacy_radius(monkeypatch):
    """A zero configured radius must produce empty control predictions."""
    monkeypatch.setattr(
        analysis,
        "load_row_masks",
        _load_masks,
    )

    result = analysis.negative_controls(
        _manifest(),
        Path("."),
        {
            "negative_control_legacy_radius_voxels": 0,
            "translation_mm": 0.0,
            "axial_axis": 0,
        },
    )

    included = result[result["included"] == True]
    assert set(included["control"]) == {
        "translation",
        "wrong_case",
    }
    assert included["dice"].tolist() == [0.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize("invalid_radius", [-1, 1.5, True])
def test_negative_control_radius_must_be_nonnegative_integer(
    invalid_radius,
):
    with pytest.raises(
        ValueError,
        match="negative_control_legacy_radius_voxels",
    ):
        analysis.negative_controls(
            _manifest(),
            Path("."),
            {
                "negative_control_legacy_radius_voxels": invalid_radius,
                "translation_mm": 0.0,
                "axial_axis": 0,
            },
        )
