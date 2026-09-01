from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

import iopr.core as core


def cube(
    shape=(16, 16, 16),
    start=(2, 2, 2),
    stop=(5, 5, 5),
):
    mask = np.zeros(shape, dtype=bool)
    mask[
        start[0]:stop[0],
        start[1]:stop[1],
        start[2]:stop[2],
    ] = True
    return mask


def test_dice_identity():
    mask = cube()
    assert core.dice(mask, mask) == pytest.approx(1.0)


def test_shape_mismatch_is_refused():
    with pytest.raises(ValueError):
        core.dice(
            np.zeros((2, 2, 2), dtype=bool),
            np.zeros((3, 2, 2), dtype=bool),
        )


def test_physical_iopr_excludes_organs():
    organ_a = cube(
        start=(2, 2, 2),
        stop=(4, 4, 4),
    )
    organ_b = cube(
        start=(7, 2, 2),
        stop=(9, 4, 4),
    )

    output = core.physical_iopr(
        organ_a,
        organ_b,
        (1.0, 1.0, 1.0),
        4.0,
    )

    assert not np.any(output & organ_a)
    assert not np.any(output & organ_b)


def test_spacing_changes_physical_result():
    organ_a = cube(
        start=(2, 2, 2),
        stop=(4, 4, 4),
    )
    organ_b = cube(
        start=(2, 2, 7),
        stop=(4, 4, 9),
    )

    isotropic = core.physical_iopr(
        organ_a,
        organ_b,
        (1.0, 1.0, 1.0),
        3.0,
    )
    anisotropic = core.physical_iopr(
        organ_a,
        organ_b,
        (1.0, 1.0, 5.0),
        3.0,
    )

    assert isotropic.sum() != anisotropic.sum()


def test_surface_metrics_identity():
    mask = cube()

    assert core.normalized_surface_dice(
        mask,
        mask,
        (1.0, 1.0, 1.0),
        2.0,
    ) == pytest.approx(1.0)

    assd, hd95 = core.assd_hd95(
        mask,
        mask,
        (1.0, 1.0, 1.0),
    )

    assert assd == pytest.approx(0.0)
    assert hd95 == pytest.approx(0.0)


def test_assd_is_surface_point_weighted(monkeypatch):
    directed_a_to_b = np.array([1.0, 1.0, 1.0, 1.0])
    directed_b_to_a = np.array([9.0])

    monkeypatch.setattr(
        core,
        "directed_surface_distances",
        lambda *args, **kwargs: (
            directed_a_to_b,
            directed_b_to_a,
        ),
    )

    dummy = np.zeros((2, 2, 2), dtype=bool)
    assd, _ = core.assd_hd95(
        dummy,
        dummy,
        (1.0, 1.0, 1.0),
    )

    expected = (4.0 + 9.0) / 5.0
    assert assd == pytest.approx(expected)


def test_hd95_is_max_of_directed_95th_percentiles(monkeypatch):
    directed_a_to_b = np.array(
        [0.0] * 95 + [10.0] * 5,
        dtype=float,
    )
    directed_b_to_a = np.array(
        [0.0] * 95 + [20.0] * 5,
        dtype=float,
    )

    monkeypatch.setattr(
        core,
        "directed_surface_distances",
        lambda *args, **kwargs: (
            directed_a_to_b,
            directed_b_to_a,
        ),
    )

    dummy = np.zeros((2, 2, 2), dtype=bool)
    _, hd95 = core.assd_hd95(
        dummy,
        dummy,
        (1.0, 1.0, 1.0),
    )

    expected = max(
        np.percentile(directed_a_to_b, 95),
        np.percentile(directed_b_to_a, 95),
    )

    assert hd95 == pytest.approx(expected)


def test_occupied_slice_overlap_identity():
    mask = cube()

    assert core.occupied_slice_overlap(
        mask,
        mask,
        axis=0,
    ) == pytest.approx(1.0)


def test_translation_uses_zero_padding():
    mask = cube(
        shape=(10, 10, 10),
        start=(1, 1, 1),
        stop=(3, 3, 3),
    )

    translated = core.translate_mask_mm(
        mask,
        (1.0, 1.0, 1.0),
        2.0,
        axis=0,
    )

    assert translated.sum() == mask.sum()
    assert not translated[:2].any()


def test_ceiling_is_bounded():
    support = cube()
    expert = cube(
        start=(3, 3, 3),
        stop=(6, 6, 6),
    )

    ceiling = core.construct_constrained_dice_ceiling(
        support,
        expert,
    )

    assert 0.0 <= ceiling <= 1.0


def test_legacy_zero_radius_is_empty():
    organ_a = cube()
    organ_b = cube(
        start=(8, 8, 8),
        stop=(10, 10, 10),
    )

    assert core.legacy_iopr(
        organ_a,
        organ_b,
        0,
    ).sum() == 0
