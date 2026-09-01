from __future__ import annotations

import gc
import sys
import weakref
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("SimpleITK")

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

import iopr.analysis as analysis


def test_expert_pair_analysis_does_not_retain_study_masks(monkeypatch):
    """Run many case groups and verify loaded expert arrays are released.

    The test tracks live NumPy arrays with weak references. A group contains
    three experts, so live expert-mask count must remain bounded rather than
    increasing with the number of cases.
    """
    manifest_rows = []
    for case_index in range(30):
        for annotator in ("A", "B", "C"):
            manifest_rows.append({
                "case_id": f"case_{case_index:03d}",
                "plane": "HR",
                "annotator": annotator,
                "organ_a": "unused",
                "organ_b": "unused",
                "expert_mask": "unused",
            })

    manifest = pd.DataFrame(manifest_rows)

    live_refs: list[weakref.ReferenceType] = []
    peak_live = 0

    def fake_load_row_masks(row, manifest_base):
        nonlocal peak_live

        organ_a = np.zeros((10, 10, 10), dtype=bool)
        organ_b = np.zeros((10, 10, 10), dtype=bool)
        expert = np.zeros((10, 10, 10), dtype=bool)
        expert[2:5, 2:5, 2:5] = True

        live_refs.append(weakref.ref(expert))
        gc.collect()
        peak_live = max(
            peak_live,
            sum(ref() is not None for ref in live_refs),
        )

        return organ_a, organ_b, expert, (1.0, 1.0, 1.0)

    monkeypatch.setattr(
        analysis,
        "load_row_masks",
        fake_load_row_masks,
    )

    config = {
        "nsd_tolerances_mm": [2.0],
        "axial_axis": 0,
    }

    result = analysis.expert_vs_expert(
        manifest,
        Path("."),
        config,
    )

    gc.collect()
    final_live = sum(
        ref() is not None
        for ref in live_refs
    )

    assert len(result) == 30 * 3
    assert peak_live <= 5  # three retained experts + transient loader/frame references
    assert final_live <= 1
