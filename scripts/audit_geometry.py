from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iopr.analysis import resolve_path
from iopr.io import (
    load_binary_mask,
    geometries_match,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = pd.read_csv(manifest_path)
    base = manifest_path.parent

    rows = []

    for _, row in manifest.iterrows():
        record = {
            "case_id": row["case_id"],
            "plane": row["plane"],
            "annotator": row["annotator"],
        }

        try:
            segment_name = None
            if (
                "expert_segment_name" in row.index
                and pd.notna(row["expert_segment_name"])
            ):
                segment_name = str(row["expert_segment_name"])

            organ_a, geom_a, _ = load_binary_mask(
                resolve_path(base, row["organ_a"])
            )
            organ_b, geom_b, _ = load_binary_mask(
                resolve_path(base, row["organ_b"])
            )
            expert, geom_e, _ = load_binary_mask(
                resolve_path(base, row["expert_mask"]),
                segment_name=segment_name,
            )

            record.update({
                "organ_shape": str(geom_a.shape_zyx),
                "expert_shape": str(geom_e.shape_zyx),
                "organ_spacing_xyz": str(geom_a.spacing_xyz),
                "expert_spacing_xyz": str(geom_e.spacing_xyz),
                "organ_origin_xyz": str(geom_a.origin_xyz),
                "expert_origin_xyz": str(geom_e.origin_xyz),
                "organ_direction": str(geom_a.direction),
                "expert_direction": str(geom_e.direction),
                "organs_match": geometries_match(geom_a, geom_b),
                "expert_matches_organs": geometries_match(geom_a, geom_e),
                "reason": "",
            })

        except Exception as exc:
            record["reason"] = str(exc)

        rows.append(record)

    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
