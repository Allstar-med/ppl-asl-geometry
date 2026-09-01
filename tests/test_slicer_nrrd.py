from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")
nrrd = pytest.importorskip("nrrd")

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from iopr.io import list_slicer_segments, load_binary_mask


def test_slicer_segment_lookup_preserves_exact_names(tmp_path):
    array_xyz = np.zeros((8, 8, 8), dtype=np.uint8)
    array_xyz[1:3, 1:3, 1:3] = 1
    array_xyz[4:6, 4:6, 4:6] = 2

    path = tmp_path / "two_segments.seg.nrrd"

    header = {
        "encoding": "raw",
        "Segment0_Name": "Hepatorenal Recess",
        "Segment0_LabelValue": "1",
        "Segment1_Name": "HR_plane\\n",
        "Segment1_LabelValue": "2",
    }

    nrrd.write(str(path), array_xyz, header=header, compression_level=0, index_order="C")

    segments = list_slicer_segments(path)
    names = [segment["name"] for segment in segments]

    assert "Hepatorenal Recess" in names
    assert "HR_plane\\n" in names


def test_slicer_exact_segment_selection(tmp_path):
    array_xyz = np.zeros((8, 8, 8), dtype=np.uint8)
    array_xyz[1:3, 1:3, 1:3] = 1
    array_xyz[4:6, 4:6, 4:6] = 2

    path = tmp_path / "two_segments.seg.nrrd"

    header = {
        "encoding": "raw",
        "Segment0_Name": "Hepatorenal Recess",
        "Segment0_LabelValue": "1",
        "Segment1_Name": "HR_plane\\n",
        "Segment1_LabelValue": "2",
    }

    nrrd.write(str(path), array_xyz, header=header, compression_level=0, index_order="C")

    mask, _, _ = load_binary_mask(
        path,
        segment_name="Hepatorenal Recess",
    )

    assert int(mask.sum()) == 8
