from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import pandas as pd

from .analysis import (
    algorithm_vs_expert,
    expert_vs_expert,
    negative_controls,
)


REQUIRED_MANIFEST_COLUMNS = {
    "case_id",
    "plane",
    "annotator",
    "organ_a",
    "organ_b",
    "expert_mask",
}


def run_pipeline(
    manifest_path: str | Path,
    config_path: str | Path,
    output_directory: str | Path,
) -> Path:
    """Run all reproducible analyses and write case-level source data first."""
    manifest_path = Path(manifest_path)
    config_path = Path(config_path)
    output_directory = Path(output_directory)

    manifest = pd.read_csv(manifest_path)
    missing = REQUIRED_MANIFEST_COLUMNS - set(manifest.columns)

    if missing:
        raise ValueError(
            f"Manifest is missing required columns: {sorted(missing)}"
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))

    source_dir = output_directory / "source_data"
    table_dir = output_directory / "tables"
    qc_dir = output_directory / "qc"

    source_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    manifest_base = manifest_path.parent

    algorithm_df = algorithm_vs_expert(
        manifest,
        manifest_base,
        config,
    )
    algorithm_df.to_csv(
        source_dir / "algorithm_vs_expert_case_level.csv",
        index=False,
    )

    expert_df = expert_vs_expert(
        manifest,
        manifest_base,
        config,
    )
    expert_df.to_csv(
        source_dir / "expert_vs_expert_case_level.csv",
        index=False,
    )

    control_df = negative_controls(
        manifest,
        manifest_base,
        config,
    )
    control_df.to_csv(
        source_dir / "negative_controls_case_level.csv",
        index=False,
    )

    # QC logs are derived directly from case-level outputs.
    algorithm_df[algorithm_df["included"] == False].to_csv(
        qc_dir / "algorithm_qc_failures.csv",
        index=False,
    )

    if "included" in expert_df.columns:
        expert_df[expert_df["included"] == False].to_csv(
            qc_dir / "expert_pair_qc_failures.csv",
            index=False,
        )

    if "included" in control_df.columns:
        control_df[control_df["included"] == False].to_csv(
            qc_dir / "control_qc_failures.csv",
            index=False,
        )

    # Summary tables are generated only from exported case-level values.
    included_algorithm = algorithm_df[
        algorithm_df["included"] == True
    ].copy()

    metric_columns = [
        column
        for column in included_algorithm.columns
        if (
            column in {
                "dice",
                "assd_mm",
                "hd95_mm",
                "occupied_slice_overlap",
                "construct_ceiling_dsc",
            }
            or column.startswith("nsd_")
        )
    ]

    if len(included_algorithm):
        summary = (
            included_algorithm
            .groupby(
                ["plane", "annotator", "mode", "radius"],
                dropna=False,
            )[metric_columns]
            .agg(["count", "mean", "std"])
            .reset_index()
        )
        summary.to_csv(
            table_dir / "algorithm_vs_expert_summary.csv",
            index=False,
        )

    included_expert = expert_df[
        expert_df.get("included", False) == True
    ].copy()

    if len(included_expert):
        expert_metric_columns = [
            column
            for column in included_expert.columns
            if (
                column in {
                    "dice",
                    "assd_mm",
                    "hd95_mm",
                    "occupied_slice_overlap",
                }
                or column.startswith("nsd_")
            )
        ]

        (
            included_expert
            .groupby(
                ["plane", "annotator_1", "annotator_2"],
                dropna=False,
            )[expert_metric_columns]
            .agg(["count", "mean", "std"])
            .reset_index()
            .to_csv(
                table_dir / "expert_vs_expert_summary.csv",
                index=False,
            )
        )

    included_controls = control_df[
        control_df.get("included", False) == True
    ].copy()

    if len(included_controls):
        (
            included_controls
            .groupby("control")["dice"]
            .agg(["count", "mean", "std"])
            .reset_index()
            .to_csv(
                table_dir / "negative_controls_summary.csv",
                index=False,
            )
        )

    run_metadata = {
        "python_version": platform.python_version(),
        "manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "config_sha256": hashlib.sha256(
            config_path.read_bytes()
        ).hexdigest(),
        "configuration": config,
    }

    (output_directory / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2),
        encoding="utf-8",
    )

    return output_directory
