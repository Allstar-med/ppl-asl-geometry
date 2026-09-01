from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import SimpleITK as sitk


@dataclass(frozen=True)
class Geometry:
    """Image geometry in SimpleITK xyz convention plus array zyx shape."""

    shape_zyx: tuple[int, int, int]
    spacing_xyz: tuple[float, float, float]
    origin_xyz: tuple[float, float, float]
    direction: tuple[float, ...]


def read_image(path: str | Path) -> sitk.Image:
    """Read an image with SimpleITK."""
    return sitk.ReadImage(str(path))


def geometry_from_image(image: sitk.Image) -> Geometry:
    """Extract geometry from a SimpleITK image."""
    array = sitk.GetArrayViewFromImage(image)
    if array.ndim != 3:
        raise ValueError(f"Expected 3-D image, got shape {array.shape}.")

    return Geometry(
        shape_zyx=tuple(int(x) for x in array.shape),
        spacing_xyz=tuple(float(x) for x in image.GetSpacing()),
        origin_xyz=tuple(float(x) for x in image.GetOrigin()),
        direction=tuple(float(x) for x in image.GetDirection()),
    )


def spacing_zyx(image: sitk.Image) -> tuple[float, float, float]:
    """Return spacing in NumPy array axis order z,y,x."""
    sx, sy, sz = image.GetSpacing()
    return float(sz), float(sy), float(sx)


def _is_nrrd(path: Path) -> bool:
    """Return True for .nrrd and .seg.nrrd files."""
    name = path.name.lower()
    return name.endswith(".nrrd") or name.endswith(".seg.nrrd")


def list_slicer_segments(path: str | Path) -> list[dict[str, object]]:
    """List Slicer segmentation metadata using pynrrd.

    SimpleITK may not expose Slicer custom SegmentN_* metadata reliably.
    pynrrd preserves the NRRD header and is therefore used for segment lookup.
    """
    path = Path(path)
    if not _is_nrrd(path):
        return []

    try:
        import nrrd
    except ImportError as exc:
        raise ImportError(
            "pynrrd is required to inspect Slicer .seg.nrrd metadata. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    _, header = nrrd.read(str(path))

    segments: list[dict[str, object]] = []
    index = 0

    while True:
        name_key = f"Segment{index}_Name"
        label_key = f"Segment{index}_LabelValue"

        if name_key not in header and label_key not in header:
            # Slicer segment indices are contiguous in normal exports.
            # Stop after the first absent pair once at least one segment exists.
            if segments:
                break
            index += 1
            if index > 256:
                break
            continue

        name = header.get(name_key)
        label_value = header.get(label_key)

        segments.append({
            "index": index,
            "name": "" if name is None else str(name),
            "label_value": None if label_value is None else int(label_value),
        })
        index += 1

    return segments


def _label_value_from_slicer_header(
    path: Path,
    segment_name: str,
) -> int:
    """Resolve an exact Slicer segment name to its label value.

    Matching is intentionally exact. Whitespace and literal escape sequences (for example ``\\n``) are not
    silently normalized because they are part of the source annotation metadata.
    """
    segments = list_slicer_segments(path)

    exact = [
        segment
        for segment in segments
        if segment["name"] == segment_name
    ]

    if len(exact) == 1 and exact[0]["label_value"] is not None:
        return int(exact[0]["label_value"])

    available = [repr(segment["name"]) for segment in segments]

    if not exact:
        raise ValueError(
            f"Segment name {segment_name!r} was not found exactly in {path}. "
            f"Available names: {available}. "
            "Use the exact stored name, including any whitespace/newline characters."
        )

    raise ValueError(
        f"Segment name {segment_name!r} is ambiguous in {path}."
    )


def _extract_named_segment(
    path: Path,
    image: sitk.Image,
    segment_name: str,
) -> sitk.Image:
    """Extract one named segment from a segmentation file.

    For NRRD/Slicer files, pynrrd is used to resolve SegmentN_* metadata and
    SimpleITK is retained for voxel data and physical geometry.
    """
    array = sitk.GetArrayFromImage(image)

    if array.ndim != 3:
        raise ValueError(
            "Only 3-D single-layer segmentations are supported directly. "
            "Normalize multi-layer Slicer segmentations before analysis."
        )

    if _is_nrrd(path):
        label_value = _label_value_from_slicer_header(
            path,
            segment_name,
        )
    else:
        label_value: Optional[int] = None

        for key in image.GetMetaDataKeys():
            if key.endswith("_Name") and image.GetMetaData(key) == segment_name:
                prefix = key[:-5]
                label_key = prefix + "_LabelValue"
                if image.HasMetaDataKey(label_key):
                    label_value = int(image.GetMetaData(label_key))
                    break

        nonzero_values = np.unique(array[array != 0])

        if label_value is None:
            if len(nonzero_values) == 1:
                label_value = int(nonzero_values[0])
            else:
                raise ValueError(
                    f"Cannot unambiguously identify segment {segment_name!r} "
                    f"in {path}. Observed non-zero labels: "
                    f"{nonzero_values.tolist()}."
                )

    binary = (array == int(label_value)).astype(np.uint8)
    output = sitk.GetImageFromArray(binary)
    output.CopyInformation(image)
    return output


def load_binary_mask(
    path: str | Path,
    segment_name: str | None = None,
) -> tuple[np.ndarray, Geometry, sitk.Image]:
    """Load a binary mask and return NumPy data, geometry, and SimpleITK image."""
    path = Path(path)

    if path.suffix.lower() == ".npy":
        array = np.load(path)
        if array.ndim != 3:
            raise ValueError(f"{path} must contain a 3-D array.")
        binary = array.astype(bool)
        image = sitk.GetImageFromArray(binary.astype(np.uint8))
        geom = geometry_from_image(image)
        return binary, geom, image

    image = read_image(path)

    if segment_name:
        image = _extract_named_segment(path, image, segment_name)

    array = sitk.GetArrayFromImage(image)
    if array.ndim != 3:
        raise ValueError(f"{path} must be 3-D after segment selection.")

    nonzero = np.unique(array[array != 0])
    if len(nonzero) > 1 and not segment_name:
        raise ValueError(
            f"{path} is multi-label; provide expert_segment_name."
        )

    binary = array != 0
    geom = geometry_from_image(image)
    return binary.astype(bool), geom, image


def geometries_match(
    geometry_a: Geometry,
    geometry_b: Geometry,
    tolerance: float = 1e-6,
) -> bool:
    """Check complete grid geometry: shape, spacing, origin, and direction."""
    if geometry_a.shape_zyx != geometry_b.shape_zyx:
        return False

    checks = (
        np.allclose(
            geometry_a.spacing_xyz,
            geometry_b.spacing_xyz,
            atol=tolerance,
            rtol=0,
        ),
        np.allclose(
            geometry_a.origin_xyz,
            geometry_b.origin_xyz,
            atol=tolerance,
            rtol=0,
        ),
        np.allclose(
            geometry_a.direction,
            geometry_b.direction,
            atol=tolerance,
            rtol=0,
        ),
    )
    return bool(all(checks))


def physical_bounding_box_mm(
    image: sitk.Image,
) -> tuple[float, float, float, float, float, float] | None:
    """Return physical bounding box of non-zero voxels as xyz min/max.

    Returns None for an empty mask.
    """
    array = sitk.GetArrayFromImage(image) != 0
    indices_zyx = np.argwhere(array)

    if len(indices_zyx) == 0:
        return None

    mins = indices_zyx.min(axis=0)
    maxs = indices_zyx.max(axis=0)

    corners = []

    for z in (mins[0], maxs[0]):
        for y in (mins[1], maxs[1]):
            for x in (mins[2], maxs[2]):
                point = image.TransformIndexToPhysicalPoint(
                    (int(x), int(y), int(z))
                )
                corners.append(point)

    corners = np.asarray(corners, dtype=float)
    xyz_min = corners.min(axis=0)
    xyz_max = corners.max(axis=0)

    return (
        float(xyz_min[0]),
        float(xyz_min[1]),
        float(xyz_min[2]),
        float(xyz_max[0]),
        float(xyz_max[1]),
        float(xyz_max[2]),
    )


def resample_label_to_reference(
    moving_label: sitk.Image,
    reference_image: sitk.Image,
) -> sitk.Image:
    """Resample a label image to a reference physical grid.

    Uses an identity transform in physical coordinates and nearest-neighbour
    interpolation. Use only when both images describe the same patient/world
    coordinate system.
    """
    return sitk.Resample(
        moving_label,
        reference_image,
        sitk.Transform(),
        sitk.sitkNearestNeighbor,
        0,
        sitk.sitkUInt8,
    )
