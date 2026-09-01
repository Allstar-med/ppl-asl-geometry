from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
sitk = pytest.importorskip('SimpleITK')

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from iopr.io import (
    geometry_from_image,
    geometries_match,
    resample_label_to_reference,
)


def make_image(
    shape_zyx=(6, 7, 8),
    spacing_xyz=(1.0, 1.0, 2.0),
    origin_xyz=(0.0, 0.0, 0.0),
    direction=(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ),
):
    array = np.zeros(shape_zyx, dtype=np.uint8)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing_xyz)
    image.SetOrigin(origin_xyz)
    image.SetDirection(direction)
    return image


def test_complete_geometry_match_checks_origin():
    image_a = make_image()
    image_b = make_image(
        origin_xyz=(10.0, 0.0, 0.0),
    )

    assert not geometries_match(
        geometry_from_image(image_a),
        geometry_from_image(image_b),
    )


def test_complete_geometry_match_checks_direction():
    image_a = make_image()
    image_b = make_image(
        direction=(
            -1.0, 0.0, 0.0,
            0.0, -1.0, 0.0,
            0.0, 0.0, 1.0,
        )
    )

    assert not geometries_match(
        geometry_from_image(image_a),
        geometry_from_image(image_b),
    )


def test_physical_resampling_hits_reference_grid():
    reference = make_image(
        shape_zyx=(8, 8, 8),
        spacing_xyz=(1.0, 1.0, 1.0),
    )

    moving_array = np.zeros(
        (4, 4, 4),
        dtype=np.uint8,
    )
    moving_array[1:3, 1:3, 1:3] = 1

    moving = sitk.GetImageFromArray(
        moving_array
    )
    moving.SetSpacing((2.0, 2.0, 2.0))
    moving.SetOrigin((0.0, 0.0, 0.0))

    output = resample_label_to_reference(
        moving,
        reference,
    )

    assert sitk.GetArrayFromImage(output).shape == (
        8,
        8,
        8,
    )
    assert output.GetSpacing() == reference.GetSpacing()
    assert output.GetOrigin() == reference.GetOrigin()
    assert output.GetDirection() == reference.GetDirection()
