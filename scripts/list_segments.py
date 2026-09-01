from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iopr.io import list_slicer_segments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("segmentation")
    args = parser.parse_args()

    segments = list_slicer_segments(args.segmentation)

    if not segments:
        print("No Slicer SegmentN_* metadata found.")
        return

    for segment in segments:
        print(
            f"index={segment['index']} "
            f"name={segment['name']!r} "
            f"label_value={segment['label_value']}"
        )


if __name__ == "__main__":
    main()
