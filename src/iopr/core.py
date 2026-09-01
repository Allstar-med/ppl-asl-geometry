from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


Mask = np.ndarray
Spacing = Tuple[float, float, float]


def as_bool_3d(mask: Mask, name: str = "mask") -> Mask:
    """Validate and return a 3-D boolean mask."""
    arr = np.asarray(mask)
    if arr.ndim != 3:
        raise ValueError(f"{name} must be 3-D; got shape {arr.shape}.")
    return arr.astype(bool, copy=False)


def validate_pair(mask_a: Mask, mask_b: Mask) -> tuple[Mask, Mask]:
    """Validate that two masks are 3-D and share an array grid."""
    a = as_bool_3d(mask_a, "mask_a")
    b = as_bool_3d(mask_b, "mask_b")
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}.")
    return a, b


def validate_spacing(spacing_mm: Iterable[float]) -> np.ndarray:
    """Validate 3-D positive finite spacing in array-axis order."""
    spacing = np.asarray(tuple(spacing_mm), dtype=float)
    if spacing.shape != (3,):
        raise ValueError("spacing_mm must contain exactly three values.")
    if np.any(~np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("spacing_mm values must be finite and > 0.")
    return spacing


def legacy_iopr(mask_a: Mask, mask_b: Mask, radius_voxels: int) -> Mask:
    """Legacy six-connected repeated voxel dilation.

    This is retained only for audit/sensitivity analysis.
    In anisotropic images, radius_voxels has no unique physical interpretation.
    """
    a, b = validate_pair(mask_a, mask_b)
    radius = int(radius_voxels)
    if radius < 0:
        raise ValueError("radius_voxels must be >= 0.")
    if radius == 0:
        return np.zeros_like(a, dtype=bool)

    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    dilated_a = ndimage.binary_dilation(a, structure=structure, iterations=radius)
    dilated_b = ndimage.binary_dilation(b, structure=structure, iterations=radius)

    return dilated_a & dilated_b & ~a & ~b


def physical_iopr(
    mask_a: Mask,
    mask_b: Mask,
    spacing_mm: Spacing,
    radius_mm: float,
) -> Mask:
    """Spacing-aware inter-organ proximity representation.

    G_r(A,B) = {x outside A union B:
                d_mm(x,A) <= r and d_mm(x,B) <= r}
    """
    a, b = validate_pair(mask_a, mask_b)
    spacing = validate_spacing(spacing_mm)
    radius = float(radius_mm)

    if not np.isfinite(radius) or radius < 0:
        raise ValueError("radius_mm must be finite and >= 0.")

    distance_to_a = ndimage.distance_transform_edt(~a, sampling=spacing)
    distance_to_b = ndimage.distance_transform_edt(~b, sampling=spacing)

    return (
        (distance_to_a <= radius)
        & (distance_to_b <= radius)
        & ~a
        & ~b
    )


def dice(mask_a: Mask, mask_b: Mask) -> float:
    """Dice similarity coefficient.

    Returns NaN if both masks are empty.
    """
    a, b = validate_pair(mask_a, mask_b)
    denominator = int(a.sum() + b.sum())
    if denominator == 0:
        return float("nan")

    intersection = int(np.logical_and(a, b).sum())
    return float(2.0 * intersection / denominator)


def binary_surface(mask: Mask) -> Mask:
    """Return a one-voxel inner surface using six-connectivity."""
    binary = as_bool_3d(mask)

    if not binary.any():
        return np.zeros_like(binary, dtype=bool)

    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    eroded = ndimage.binary_erosion(
        binary,
        structure=structure,
        border_value=0,
    )
    return binary & ~eroded


def directed_surface_distances(
    mask_a: Mask,
    mask_b: Mask,
    spacing_mm: Spacing,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest-neighbour surface distances A→B and B→A in mm."""
    a, b = validate_pair(mask_a, mask_b)
    spacing = validate_spacing(spacing_mm)

    surface_a = np.argwhere(binary_surface(a))
    surface_b = np.argwhere(binary_surface(b))

    if len(surface_a) == 0 or len(surface_b) == 0:
        nan = np.asarray([np.nan], dtype=float)
        return nan, nan

    points_a = surface_a * spacing
    points_b = surface_b * spacing

    tree_a = cKDTree(points_a)
    tree_b = cKDTree(points_b)

    distances_a_to_b = tree_b.query(points_a, k=1)[0]
    distances_b_to_a = tree_a.query(points_b, k=1)[0]

    return (
        np.asarray(distances_a_to_b, dtype=float),
        np.asarray(distances_b_to_a, dtype=float),
    )


def normalized_surface_dice(
    mask_a: Mask,
    mask_b: Mask,
    spacing_mm: Spacing,
    tolerance_mm: float,
) -> float:
    """Symmetric normalized surface Dice at a physical tolerance."""
    tolerance = float(tolerance_mm)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance_mm must be finite and >= 0.")

    distances_a_to_b, distances_b_to_a = directed_surface_distances(
        mask_a,
        mask_b,
        spacing_mm,
    )

    if np.isnan(distances_a_to_b).any() or np.isnan(distances_b_to_a).any():
        return float("nan")

    matched = int((distances_a_to_b <= tolerance).sum())
    matched += int((distances_b_to_a <= tolerance).sum())

    total = len(distances_a_to_b) + len(distances_b_to_a)
    return float(matched / total)


def assd_hd95(
    mask_a: Mask,
    mask_b: Mask,
    spacing_mm: Spacing,
) -> tuple[float, float]:
    """Return ASSD and HD95 in millimetres.

    ASSD is the pooled surface-point-weighted mean:
        (sum d(A→B) + sum d(B→A)) / (N_A + N_B)

    HD95 is:
        max(P95[d(A→B)], P95[d(B→A)])
    """
    distances_a_to_b, distances_b_to_a = directed_surface_distances(
        mask_a,
        mask_b,
        spacing_mm,
    )

    if np.isnan(distances_a_to_b).any() or np.isnan(distances_b_to_a).any():
        return float("nan"), float("nan")

    pooled_sum = float(distances_a_to_b.sum() + distances_b_to_a.sum())
    pooled_count = len(distances_a_to_b) + len(distances_b_to_a)
    assd = pooled_sum / pooled_count

    hd95 = max(
        float(np.percentile(distances_a_to_b, 95)),
        float(np.percentile(distances_b_to_a, 95)),
    )

    return float(assd), float(hd95)


def occupied_slice_overlap(
    mask_a: Mask,
    mask_b: Mask,
    axis: int = 0,
) -> float:
    """Dice overlap of occupied slice levels along one array axis."""
    a, b = validate_pair(mask_a, mask_b)

    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2.")

    reduce_axes = tuple(i for i in range(3) if i != axis)
    occupied_a = np.any(a, axis=reduce_axes)
    occupied_b = np.any(b, axis=reduce_axes)

    denominator = int(occupied_a.sum() + occupied_b.sum())
    if denominator == 0:
        return float("nan")

    intersection = int(np.logical_and(occupied_a, occupied_b).sum())
    return float(2.0 * intersection / denominator)


def translate_mask_mm(
    mask: Mask,
    spacing_mm: Spacing,
    shift_mm: float,
    axis: int = 0,
) -> Mask:
    """Translate a mask along one axis using physical millimetres.

    Translation is converted to the nearest integer voxel offset and zero-padded.
    No wrap-around is used.
    """
    binary = as_bool_3d(mask)
    spacing = validate_spacing(spacing_mm)

    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2.")

    shift = float(shift_mm)
    if not np.isfinite(shift):
        raise ValueError("shift_mm must be finite.")

    voxel_shift = int(round(shift / spacing[axis]))
    output = np.zeros_like(binary, dtype=bool)

    if abs(voxel_shift) >= binary.shape[axis]:
        return output

    source = [slice(None)] * 3
    destination = [slice(None)] * 3

    if voxel_shift >= 0:
        source[axis] = slice(0, binary.shape[axis] - voxel_shift)
        destination[axis] = slice(voxel_shift, binary.shape[axis])
    else:
        source[axis] = slice(-voxel_shift, binary.shape[axis])
        destination[axis] = slice(0, binary.shape[axis] + voxel_shift)

    output[tuple(destination)] = binary[tuple(source)]
    return output


def construct_constrained_dice_ceiling(
    prediction_support: Mask,
    expert_mask: Mask,
) -> float:
    """Construct-constrained Dice ceiling.

    This is an oracle ceiling under the constraint that any selected output
    must be a subset of the geometric prediction support.

    It is a diagnostic ceiling, not a deployable algorithm.
    """
    support, expert = validate_pair(prediction_support, expert_mask)

    overlap = int(np.logical_and(support, expert).sum())
    if overlap == 0:
        return 0.0

    optimal_subset_size = overlap
    expert_size = int(expert.sum())

    return float(
        2.0 * overlap / (optimal_subset_size + expert_size)
    )
