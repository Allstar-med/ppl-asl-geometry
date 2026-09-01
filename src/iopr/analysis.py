from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .core import (
    legacy_iopr,
    physical_iopr,
    dice,
    normalized_surface_dice,
    assd_hd95,
    occupied_slice_overlap,
    translate_mask_mm,
    construct_constrained_dice_ceiling,
)
from .io import (
    load_binary_mask,
    geometries_match,
    spacing_zyx,
)


def resolve_path(base: Path, value: str) -> Path:
    """Resolve a manifest path relative to the manifest directory."""
    path = Path(str(value))
    return path if path.is_absolute() else (base / path).resolve()


def evaluate_masks(
    prediction: np.ndarray,
    expert: np.ndarray,
    spacing_mm: tuple[float, float, float],
    nsd_tolerances_mm: Iterable[float],
    axial_axis: int,
) -> dict[str, float]:
    """Compute the manuscript validation metrics for one mask pair."""
    assd, hd95 = assd_hd95(prediction, expert, spacing_mm)

    result = {
        "dice": dice(prediction, expert),
        "assd_mm": assd,
        "hd95_mm": hd95,
        "occupied_slice_overlap": occupied_slice_overlap(
            prediction,
            expert,
            axis=axial_axis,
        ),
    }

    for tolerance in nsd_tolerances_mm:
        result[f"nsd_{float(tolerance):g}mm"] = normalized_surface_dice(
            prediction,
            expert,
            spacing_mm,
            float(tolerance),
        )

    return result


def load_row_masks(
    row: pd.Series,
    manifest_base: Path,
) -> tuple:
    """Load organ A, organ B and expert mask for one manifest row."""
    organ_a_path = resolve_path(manifest_base, row["organ_a"])
    organ_b_path = resolve_path(manifest_base, row["organ_b"])
    expert_path = resolve_path(manifest_base, row["expert_mask"])

    segment_name = None
    if "expert_segment_name" in row.index and pd.notna(row["expert_segment_name"]):
        segment_name = str(row["expert_segment_name"])

    organ_a, geometry_a, image_a = load_binary_mask(organ_a_path)
    organ_b, geometry_b, image_b = load_binary_mask(organ_b_path)
    expert, geometry_e, image_e = load_binary_mask(
        expert_path,
        segment_name=segment_name,
    )

    if not geometries_match(geometry_a, geometry_b):
        raise ValueError(
            "organ_a and organ_b do not share identical physical geometry."
        )

    if not geometries_match(geometry_a, geometry_e):
        raise ValueError(
            "expert mask does not share the organ reference geometry."
        )

    spacing = spacing_zyx(image_a)
    return organ_a, organ_b, expert, spacing


def algorithm_vs_expert(
    manifest: pd.DataFrame,
    manifest_base: Path,
    config: dict,
) -> pd.DataFrame:
    """Stream manifest rows and compute algorithm-to-expert case-level metrics."""
    output_rows: list[dict] = []

    for _, row in manifest.iterrows():
        common = {
            "case_id": row["case_id"],
            "plane": row["plane"],
            "annotator": row["annotator"],
        }

        try:
            organ_a, organ_b, expert, spacing = load_row_masks(
                row,
                manifest_base,
            )

            for radius_mm in config["physical_radius_mm"]:
                prediction = physical_iopr(
                    organ_a,
                    organ_b,
                    spacing,
                    float(radius_mm),
                )

                result = {
                    **common,
                    "included": True,
                    "reason": "",
                    "mode": "physical",
                    "radius": float(radius_mm),
                    "spacing_z_mm": spacing[0],
                    "spacing_y_mm": spacing[1],
                    "spacing_x_mm": spacing[2],
                }
                result.update(
                    evaluate_masks(
                        prediction,
                        expert,
                        spacing,
                        config["nsd_tolerances_mm"],
                        config["axial_axis"],
                    )
                )
                result["construct_ceiling_dsc"] = (
                    construct_constrained_dice_ceiling(
                        prediction,
                        expert,
                    )
                )
                output_rows.append(result)

            for radius_voxels in config["legacy_radius_voxels"]:
                prediction = legacy_iopr(
                    organ_a,
                    organ_b,
                    int(radius_voxels),
                )

                result = {
                    **common,
                    "included": True,
                    "reason": "",
                    "mode": "legacy",
                    "radius": int(radius_voxels),
                    "spacing_z_mm": spacing[0],
                    "spacing_y_mm": spacing[1],
                    "spacing_x_mm": spacing[2],
                }
                result.update(
                    evaluate_masks(
                        prediction,
                        expert,
                        spacing,
                        config["nsd_tolerances_mm"],
                        config["axial_axis"],
                    )
                )
                result["construct_ceiling_dsc"] = (
                    construct_constrained_dice_ceiling(
                        prediction,
                        expert,
                    )
                )
                output_rows.append(result)

        except Exception as exc:
            output_rows.append({
                **common,
                "included": False,
                "reason": str(exc),
                "mode": "QC",
                "radius": np.nan,
            })

    return pd.DataFrame(output_rows)


def expert_vs_expert(
    manifest: pd.DataFrame,
    manifest_base: Path,
    config: dict,
) -> pd.DataFrame:
    """Compute expert-expert metrics one case/plane group at a time.

    This design intentionally avoids caching the entire study.
    """
    output_rows: list[dict] = []

    grouped = manifest.groupby(["case_id", "plane"], sort=True)

    for (case_id, plane), group in grouped:
        loaded: list[tuple[str, np.ndarray, tuple[float, float, float]]] = []

        for _, row in group.iterrows():
            try:
                _, _, expert, spacing = load_row_masks(
                    row,
                    manifest_base,
                )
                loaded.append(
                    (str(row["annotator"]), expert, spacing)
                )
            except Exception as exc:
                output_rows.append({
                    "case_id": case_id,
                    "plane": plane,
                    "annotator_1": row["annotator"],
                    "annotator_2": "",
                    "included": False,
                    "reason": str(exc),
                })

        for i in range(len(loaded)):
            for j in range(i + 1, len(loaded)):
                annotator_1, expert_1, spacing_1 = loaded[i]
                annotator_2, expert_2, spacing_2 = loaded[j]

                if expert_1.shape != expert_2.shape or not np.allclose(
                    spacing_1,
                    spacing_2,
                    atol=1e-6,
                    rtol=0,
                ):
                    output_rows.append({
                        "case_id": case_id,
                        "plane": plane,
                        "annotator_1": annotator_1,
                        "annotator_2": annotator_2,
                        "included": False,
                        "reason": "expert masks are not on the same analysis grid",
                    })
                    continue

                result = {
                    "case_id": case_id,
                    "plane": plane,
                    "annotator_1": annotator_1,
                    "annotator_2": annotator_2,
                    "included": True,
                    "reason": "",
                }
                result.update(
                    evaluate_masks(
                        expert_1,
                        expert_2,
                        spacing_1,
                        config["nsd_tolerances_mm"],
                        config["axial_axis"],
                    )
                )
                output_rows.append(result)

        # `loaded` is released at the end of this group.

    return pd.DataFrame(output_rows)


def negative_controls(
    manifest: pd.DataFrame,
    manifest_base: Path,
    config: dict,
) -> pd.DataFrame:
    """Compute negative controls without holding all masks in memory."""
    radius_voxels = config.get(
        "negative_control_legacy_radius_voxels"
    )
    if (
        isinstance(radius_voxels, bool)
        or not isinstance(radius_voxels, int)
        or radius_voxels < 0
    ):
        raise ValueError(
            "negative_control_legacy_radius_voxels must be a "
            "non-negative integer."
        )

    output_rows: list[dict] = []

    # Translation and wrong-plane controls are computed within case.
    for (case_id, annotator), group in manifest.groupby(
        ["case_id", "annotator"],
        sort=True,
    ):
        plane_rows = {
            str(row["plane"]): row
            for _, row in group.iterrows()
        }

        for plane, row in plane_rows.items():
            try:
                organ_a, organ_b, expert, spacing = load_row_masks(
                    row,
                    manifest_base,
                )
                prediction = legacy_iopr(
                    organ_a,
                    organ_b,
                    radius_voxels,
                )

                translated = translate_mask_mm(
                    expert,
                    spacing,
                    config["translation_mm"],
                    axis=config["axial_axis"],
                )

                output_rows.append({
                    "case_id": case_id,
                    "plane": plane,
                    "annotator": annotator,
                    "control": "translation",
                    "included": True,
                    "reason": "",
                    "paired_case": "",
                    "dice": dice(prediction, translated),
                })

                other_plane = "SR" if plane == "HR" else "HR"

                if other_plane in plane_rows:
                    try:
                        _, _, other_expert, other_spacing = load_row_masks(
                            plane_rows[other_plane],
                            manifest_base,
                        )
                        if (
                            prediction.shape == other_expert.shape
                            and np.allclose(
                                spacing,
                                other_spacing,
                                atol=1e-6,
                                rtol=0,
                            )
                        ):
                            output_rows.append({
                                "case_id": case_id,
                                "plane": plane,
                                "annotator": annotator,
                                "control": "wrong_plane",
                                "included": True,
                                "reason": "",
                                "paired_case": "",
                                "dice": dice(
                                    prediction,
                                    other_expert,
                                ),
                            })
                    except Exception:
                        pass

            except Exception as exc:
                output_rows.append({
                    "case_id": case_id,
                    "plane": plane,
                    "annotator": annotator,
                    "control": "within_case_qc",
                    "included": False,
                    "reason": str(exc),
                    "paired_case": "",
                    "dice": np.nan,
                })

    # Wrong-case control:
    # for each source row, scan sorted compatible candidate rows and load
    # only one candidate at a time.
    sorted_manifest = manifest.sort_values(
        ["plane", "annotator", "case_id"]
    )

    for _, source_row in sorted_manifest.iterrows():
        try:
            organ_a, organ_b, _, spacing = load_row_masks(
                source_row,
                manifest_base,
            )
            prediction = legacy_iopr(
                organ_a,
                organ_b,
                radius_voxels,
            )
        except Exception:
            continue

        candidates = sorted_manifest[
            (sorted_manifest["plane"] == source_row["plane"])
            & (sorted_manifest["annotator"] == source_row["annotator"])
            & (sorted_manifest["case_id"] != source_row["case_id"])
        ]

        for _, candidate_row in candidates.iterrows():
            try:
                _, _, candidate_expert, candidate_spacing = load_row_masks(
                    candidate_row,
                    manifest_base,
                )

                if (
                    prediction.shape == candidate_expert.shape
                    and np.allclose(
                        spacing,
                        candidate_spacing,
                        atol=1e-6,
                        rtol=0,
                    )
                ):
                    output_rows.append({
                        "case_id": source_row["case_id"],
                        "plane": source_row["plane"],
                        "annotator": source_row["annotator"],
                        "control": "wrong_case",
                        "included": True,
                        "reason": "",
                        "paired_case": candidate_row["case_id"],
                        "dice": dice(
                            prediction,
                            candidate_expert,
                        ),
                    })
                    break

            except Exception:
                continue

    return pd.DataFrame(output_rows)
