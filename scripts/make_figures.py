from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    args = parser.parse_args()

    results = Path(args.results)
    figure_dir = results / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    algorithm = pd.read_csv(
        results
        / "source_data"
        / "algorithm_vs_expert_case_level.csv"
    )

    included = algorithm[
        algorithm["included"] == True
    ].copy()

    legacy = included[
        included["mode"] == "legacy"
    ].copy()

    if len(legacy):
        summary = (
            legacy
            .groupby(["plane", "radius"])["dice"]
            .mean()
            .reset_index()
        )

        for plane, data in summary.groupby("plane"):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(data["radius"], data["dice"], marker="o")
            ax.set_xlabel("Legacy voxel radius")
            ax.set_ylabel("Mean DSC")
            ax.set_title(f"{plane} legacy radius sensitivity")
            fig.tight_layout()
            fig.savefig(
                figure_dir / f"{plane}_legacy_radius_sensitivity.png",
                dpi=300,
            )
            plt.close(fig)

    physical = included[
        included["mode"] == "physical"
    ].copy()

    if len(physical):
        summary = (
            physical
            .groupby(["plane", "radius"])["dice"]
            .mean()
            .reset_index()
        )

        for plane, data in summary.groupby("plane"):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(data["radius"], data["dice"], marker="o")
            ax.set_xlabel("Physical radius (mm)")
            ax.set_ylabel("Mean DSC")
            ax.set_title(f"{plane} physical-distance sensitivity")
            fig.tight_layout()
            fig.savefig(
                figure_dir / f"{plane}_physical_radius_sensitivity.png",
                dpi=300,
            )
            plt.close(fig)

    controls_path = (
        results
        / "source_data"
        / "negative_controls_case_level.csv"
    )

    if controls_path.exists():
        controls = pd.read_csv(controls_path)
        controls = controls[
            controls.get("included", False) == True
        ]

        if len(controls):
            summary = (
                controls.groupby("control")["dice"]
                .mean()
                .sort_values()
            )

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(summary.index, summary.values)
            ax.set_ylabel("Mean DSC")
            ax.set_title("Negative controls")
            fig.tight_layout()
            fig.savefig(
                figure_dir / "negative_controls.png",
                dpi=300,
            )
            plt.close(fig)


if __name__ == "__main__":
    main()
