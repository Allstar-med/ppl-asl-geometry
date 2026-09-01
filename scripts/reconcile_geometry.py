from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iopr.analysis import resolve_path
from iopr.io import (
    load_binary_mask,
    geometries_match,
    geometry_from_image,
    physical_bounding_box_mm,
    resample_label_to_reference,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--audit", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = pd.read_csv(manifest_path)
    base = manifest_path.parent
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audit_rows = []
    output_rows = []

    for _, row in manifest.iterrows():
        output_row = row.copy()

        try:
            segment_name = None
            if (
                "expert_segment_name" in row.index
                and pd.notna(row["expert_segment_name"])
            ):
                segment_name = str(row["expert_segment_name"])

            _, organ_geometry, organ_image = load_binary_mask(
                resolve_path(base, row["organ_a"])
            )

            expert_array, expert_geometry, expert_image = load_binary_mask(
                resolve_path(base, row["expert_mask"]),
                segment_name=segment_name,
            )

            before_voxels = int(expert_array.sum())
            before_bbox = physical_bounding_box_mm(expert_image)

            if geometries_match(organ_geometry, expert_geometry):
                reconciled_image = expert_image
                action = "already_matched"
            else:
                reconciled_image = resample_label_to_reference(
                    expert_image,
                    organ_image,
                )
                action = "nearest_neighbor_physical_resample"

            reconciled_array = (
                sitk.GetArrayFromImage(reconciled_image) != 0
            ).astype(np.uint8)

            after_voxels = int(reconciled_array.sum())
            after_bbox = physical_bounding_box_mm(reconciled_image)

            destination = (
                output_dir
                / str(row["annotator"])
                / str(row["plane"])
                / f"{row['case_id']}.nii.gz"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)

            output_image = sitk.GetImageFromArray(reconciled_array)
            output_image.CopyInformation(organ_image)
            sitk.WriteImage(output_image, str(destination))

            output_row["expert_mask"] = str(destination.resolve())
            output_row["expert_segment_name"] = ""

            new_geometry = geometry_from_image(output_image)

            audit_rows.append({
                "case_id": row["case_id"],
                "plane": row["plane"],
                "annotator": row["annotator"],
                "action": action,
                "before_shape": str(expert_geometry.shape_zyx),
                "after_shape": str(new_geometry.shape_zyx),
                "reference_shape": str(organ_geometry.shape_zyx),
                "before_spacing_xyz": str(expert_geometry.spacing_xyz),
                "after_spacing_xyz": str(new_geometry.spacing_xyz),
                "reference_spacing_xyz": str(organ_geometry.spacing_xyz),
                "before_origin_xyz": str(expert_geometry.origin_xyz),
                "after_origin_xyz": str(new_geometry.origin_xyz),
                "reference_origin_xyz": str(organ_geometry.origin_xyz),
                "before_direction": str(expert_geometry.direction),
                "after_direction": str(new_geometry.direction),
                "reference_direction": str(organ_geometry.direction),
                "before_voxels": before_voxels,
                "after_voxels": after_voxels,
                "voxel_count_ratio": (
                    np.nan
                    if before_voxels == 0
                    else after_voxels / before_voxels
                ),
                "before_bbox_mm": str(before_bbox),
                "after_bbox_mm": str(after_bbox),
                "output_path": str(destination.resolve()),
                "status": "ok",
                "reason": "",
            })

        except Exception as exc:
            audit_rows.append({
                "case_id": row["case_id"],
                "plane": row["plane"],
                "annotator": row["annotator"],
                "status": "failed",
                "reason": str(exc),
            })

        output_rows.append(output_row)

    pd.DataFrame(output_rows).to_csv(
        args.out_manifest,
        index=False,
    )
    pd.DataFrame(audit_rows).to_csv(
        args.audit,
        index=False,
    )


if __name__ == "__main__":
    main()
